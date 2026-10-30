import re
import copy
from dataclasses import dataclass, field

import optree

from typing import Optional

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.checkpoint import checkpoint as activation_checkpoint

from flash_attn.ops.triton.layer_norm import RMSNorm

from hnet.modules.block import create_block
from hnet.modules.utils import get_seq_idx, get_stage_cfg

from hnet.models.config_hnet import HNetConfig
from hnet.te import fp8_context, graph_capture_enabled


@dataclass
class IsotropicInferenceParams:
    """Inference parameters that are passed to the main model in order
    to efficienly calculate and store the context during inference."""

    max_seqlen: int
    max_batch_size: int
    seqlen_offset: int = 0
    batch_size_offset: int = 0
    key_value_memory_dict: dict = field(default_factory=dict)
    lengths_per_sample: Optional[torch.Tensor] = None
    # For the innermost MSAR stack, each recurrent loop keeps one bounded
    # history per source stream.  The history stores only the previous
    # ``max_kernel - 1`` states; the current token is supplied separately to
    # the MSAR mixer and appended after the layer boundary is produced.
    msar_history: Optional[list[list[torch.Tensor]]] = None

    def reset(self, max_seqlen, max_batch_size):
        self.max_seqlen = max_seqlen
        self.max_batch_size = max_batch_size
        self.seqlen_offset = 0
        if self.lengths_per_sample is not None:
            self.lengths_per_sample.zero_()

        optree.tree_map(
            lambda x: x.zero_() if isinstance(x, torch.Tensor) else x,
            self.key_value_memory_dict,
        )
        # MSAR keeps causal stream state outside the attention KV tensors.  A
        # reused cache must reset those histories as well, otherwise a new
        # prompt reads the previous sequence's tail during its first decode
        # tokens.
        optree.tree_map(
            lambda x: x.zero_() if isinstance(x, torch.Tensor) else x,
            self.msar_history,
        )


class Isotropic(nn.Module):
    def __init__(
        self,
        config: HNetConfig,
        pos_idx: int,
        stage_idx: int,
        device=None,
        dtype=None,
    ):
        factory_kwargs = {"device": device, "dtype": dtype}
        super().__init__()

        self.stage_idx = stage_idx
        # Only the deepest main network is looped. Encoder/decoder modules
        # around H-Net chunk boundaries remain single-pass, preserving the
        # routing and dechunking identity path.
        self.loop_count = (
            int(getattr(config, "inner_loops", 1))
            if stage_idx == len(config.d_model) - 1
            else 1
        )
        self.msar_enabled = bool(
            getattr(config, "msar", False)
            and stage_idx == len(config.d_model) - 1
        )
        self.msar_checkpointing = bool(
            getattr(config, "msar_activation_checkpointing", False)
        )
        self.mamba_checkpointing = bool(
            getattr(config, "mamba_activation_checkpointing", False)
        )
        self.msar_kernels = tuple(getattr(config, "msar_stream_kernels", [1, 4, 8, 12]))
        self.msar_block_size = int(getattr(config, "msar_block_size", 4))
        self.d_model = config.d_model[self.stage_idx]
        self.ssm_cfg = get_stage_cfg(config.ssm_cfg, stage_idx)
        self.attn_cfg = get_stage_cfg(config.attn_cfg, stage_idx)

        arch_layout = config.arch_layout
        for _ in range(stage_idx):
            arch_layout = arch_layout[1]
        arch_layout = arch_layout[pos_idx]
        layout_parse = re.findall(r"([mMsStT])(\d+)", arch_layout)

        layers = []
        layer_idx = 0
        self.arch_full = []

        # self.height counts the number of things that get added to the residual stream
        self.height = 0
        for arch, n_layer in layout_parse:
            assert arch in ("m", "M", "s", "S", "t", "T")
            assert n_layer.isdigit()
            block_attn_cfg = dict(self.attn_cfg)
            if arch in ("s", "S"):
                block_attn_cfg["window_size"] = int(getattr(config, "swa_window_size", 128))
                block_attn_cfg["croissattn_aux"] = False
                # VHA is the requested FA KV-cache compression.  SWA keeps
                # its ordinary same-head local attention path and is not part
                # of the compressed global-cache budget.
                block_attn_cfg["vha_enabled"] = False
            layers += [
                create_block(
                    arch,
                    self.d_model,
                    d_intermediate=config.d_intermediate[self.stage_idx],
                    ssm_cfg=self.ssm_cfg,
                    attn_cfg=block_attn_cfg,
                    layer_idx=(layer_idx + i),
                    moe_cfg=config,
                    **factory_kwargs,
                )
                for i in range(int(n_layer))
            ]
            if arch.islower():
                self.height += int(n_layer)
            else:
                self.height += 2 * int(n_layer)
            self.arch_full.extend([arch for _ in range(int(n_layer))])
            layer_idx += int(n_layer)

        self.layers = nn.ModuleList(layers)

        self.rmsnorm = RMSNorm(self.d_model, eps=1e-5, **factory_kwargs)
        if self.msar_enabled:
            n_streams = len(self.msar_kernels)
            n_layers = len(self.layers)
            self.msar_queries = nn.Parameter(
                torch.zeros(n_layers, n_streams, self.d_model, device=device, dtype=dtype)
            )
            self.msar_pre = nn.Parameter(
                torch.zeros(n_layers, n_streams * self.d_model, n_streams, device=device, dtype=dtype)
            )
            self.msar_bias = nn.Parameter(
                torch.tensor(
                    [4.0] + [-4.0] * (n_streams - 1),
                    device=device,
                    dtype=dtype,
                ).expand(n_layers, -1).clone()
            )
            self.msar_scale = nn.Parameter(torch.full((n_layers,), 0.1, device=device, dtype=dtype))
            self.msar_conv = nn.ParameterList(
                [
                    nn.Parameter(
                        torch.zeros(self.d_model, kernel, device=device, dtype=dtype)
                    )
                    for kernel in self.msar_kernels[1:]
                ]
            )

    def _msar_expand(self, source, kernel, weight, cu_seqlens=None):
        if kernel == 1:
            return source
        original_2d = source.dim() == 2
        x = source.unsqueeze(0) if original_2d else source
        if cu_seqlens is None:
            y = F.conv1d(
                F.pad(x.transpose(1, 2), (kernel - 1, 0)),
                weight.to(dtype=x.dtype).unsqueeze(1),
                groups=self.d_model,
            ).transpose(1, 2)
            return y.squeeze(0) if original_2d else y
        # Packed streams need causal resets at document boundaries. Express
        # the depthwise convolution as tensor shifts masked by document IDs;
        # this removes the Python ``tolist``/per-document launch loop and keeps
        # the packed MSAR path graph-safe for a fixed token bound.
        from hnet.modules.utils import get_seq_idx
        seq_idx = get_seq_idx(
            cu_seqlens, device=x.device, total_length=x.shape[1]
        )
        y = x * weight[:, kernel - 1].to(dtype=x.dtype)
        for lag in range(1, kernel):
            shifted = torch.cat((torch.zeros_like(x[:, :lag]), x[:, :-lag]), dim=1)
            same_doc = torch.cat(
                (torch.zeros_like(seq_idx[:, :lag], dtype=torch.bool),
                 seq_idx[:, lag:] == seq_idx[:, :-lag]),
                dim=1,
            )
            # Accumulate in-place after masking and scaling the temporary
            # shift.  The previous expression materialized two additional
            # [tokens, d_model] tensors per lag; at the full 16-sample batch
            # that transient peak was enough to exhaust a 275-GiB B300.
            shifted.mul_(same_doc.unsqueeze(-1).to(dtype=x.dtype))
            shifted.mul_(weight[:, kernel - 1 - lag].to(dtype=x.dtype))
            y.add_(shifted)
        return y.squeeze(0) if original_2d else y

    def _msar_mix(self, sources, layer_idx, cu_seqlens=None):
        # Build and reduce one source at a time.  The former implementation
        # stacked every source and every kernel stream before normalization,
        # which briefly kept both BF16 and FP32 copies of a full [sources,
        # streams, batch, length, width] activation bank.  Keeping per-source
        # stream lists preserves the exact MSAR equations while removing that
        # peak allocation and making larger microbatches feasible.
        source_streams = []
        source_scores = []
        query = self.msar_queries[layer_idx].float()
        for source in sources:
            streams = [source]
            for kernel_idx, kernel in enumerate(self.msar_kernels[1:]):
                streams.append(
                    self._msar_expand(
                        source, kernel, self.msar_conv[kernel_idx], cu_seqlens
                    )
                )
            scores = []
            for stream, stream_query in zip(streams, query, strict=True):
                stream_f = stream.float()
                normed = stream_f * torch.rsqrt(
                    stream_f.square().mean(dim=-1, keepdim=True).add(1e-6)
                )
                scores.append(
                    torch.einsum("...d,d->...", normed, stream_query)
                    / (self.d_model**0.5)
                )
            source_streams.append(streams)
            source_scores.append(torch.stack(scores, dim=0))
        weights = torch.softmax(torch.stack(source_scores, dim=0), dim=0)
        retrieved_streams = []
        for stream_index in range(len(self.msar_kernels)):
            retrieved_streams.append(
                sum(
                    weights[source_index, stream_index].unsqueeze(-1)
                    * source_streams[source_index][stream_index]
                    for source_index in range(len(source_streams))
                )
            )
        # Apply the MSAR pre-projection one stream at a time. Concatenating
        # all streams into [batch, length, streams*width] creates a large
        # transient copy; splitting the learned matrix gives the identical
        # result without that allocation.
        pre = self.msar_pre[layer_idx].float().view(
            len(self.msar_kernels), self.d_model, len(self.msar_kernels)
        )
        mix_logits = self.msar_bias[layer_idx].float()
        for stream_index, stream in enumerate(retrieved_streams):
            mix_logits = mix_logits + stream.float() @ pre[stream_index]
        mix = torch.sigmoid(mix_logits)
        # Accumulate each MSAR stream in-place.  Materializing the full
        # ``mix[..., None] * retrieved_last`` product doubles the largest
        # [batch, length, streams, width] activation and was the final memory
        # cliff for the high-throughput graph batch.
        fused = torch.zeros_like(retrieved_streams[0])
        for stream_index, stream in enumerate(retrieved_streams):
            fused.addcmul_(
                mix[..., stream_index].unsqueeze(-1),
                stream,
            )
        return fused.to(sources[-1].dtype)

    def _msar_mix_recompute(self, sources, layer_idx, cu_seqlens=None):
        """Run MSAR with backward recomputation to bound activation memory."""
        if not self.training or not self.msar_checkpointing:
            return self._msar_mix(sources, layer_idx, cu_seqlens=cu_seqlens)

        def run(*source_tensors):
            return self._msar_mix(
                list(source_tensors), layer_idx, cu_seqlens=cu_seqlens
            )

        return activation_checkpoint(
            run, *sources, use_reentrant=False, preserve_rng_state=False
        )

    def _run_mamba_block(
        self,
        layer,
        hidden_states,
        residual,
        *,
        inference_params,
        mixer_kwargs,
        active_mask,
    ):
        def run(h, r=None):
            return layer(
                h,
                r,
                inference_params=inference_params,
                mixer_kwargs=mixer_kwargs,
                active_mask=active_mask,
            )

        if self.training and self.mamba_checkpointing and inference_params is None:
            if residual is None:
                return activation_checkpoint(
                    lambda h: run(h, None),
                    hidden_states,
                    use_reentrant=False,
                    preserve_rng_state=False,
                )
            return activation_checkpoint(
                run,
                hidden_states,
                residual,
                use_reentrant=False,
                preserve_rng_state=False,
            )
        return run(hidden_states, residual)

    def allocate_inference_cache(self, batch_size, max_seqlen, dtype=None):
        """
        Allocate the inference cache for the Isotropic module.

        Arguments:
            batch_size: int. The number of sequences in the batch.
            max_seqlen: int. The maximum sequence length in the batch, not used for this module.
            dtype: torch.dtype. The dtype of the inference cache.

        The inference cache contains a list of inference caches, one for each block.
        """
        key_value_memory_dict = {}
        cache_device = self.rmsnorm.weight.device
        for i, layer in enumerate(self.layers):
            key_value_memory_dict[i] = layer.allocate_inference_cache(
                batch_size, max_seqlen, dtype=dtype
            )
        msar_history = None
        if self.msar_enabled:
            # Keep one recurrent stream for the input and for the output of
            # every layer.  A layer's current state is itself an MSAR source
            # on the following layer, so boundary-only storage would lose the
            # causal history of transient (non-boundary) states at decode.
            source_count = len(self.layers) + 1
            history_len = max(self.msar_kernels) - 1
            msar_history = [
                [
                    torch.zeros(
                        batch_size,
                        history_len,
                        self.d_model,
                        device=cache_device,
                        dtype=dtype,
                    )
                    for _ in range(source_count)
                ]
                for _ in range(self.loop_count)
            ]
        return IsotropicInferenceParams(
            key_value_memory_dict=key_value_memory_dict,
            max_seqlen=max_seqlen,
            max_batch_size=batch_size,
            msar_history=msar_history,
        )

    @staticmethod
    def _append_msar_history(history, current):
        """Append one token state to a bounded per-stream history."""
        if history.shape[1] == 0:
            return
        history.copy_(torch.cat((history[:, 1:], current.unsqueeze(1)), dim=1))

    @staticmethod
    def _store_msar_history(history, sequence):
        """Store the final bounded stream tail after cached prefill."""
        if sequence.dim() != 3:
            raise ValueError("MSAR cache prefill requires an unpacked (B,L,D) sequence")
        history.zero_()
        take = min(history.shape[1], sequence.shape[1])
        if take:
            history[:, -take:].copy_(sequence[:, -take:].to(dtype=history.dtype))

    def _msar_mix_step(self, source_currents, history, layer_idx, history_indices):
        """Compute one MSAR token using recurrent stream histories.

        This is the recurrent equivalent of :meth:`_msar_mix`.  It uses the
        exact same depthwise causal coefficients and source/query mixing, but
        never materializes a fake sequence dimension or reads host lengths.
        """
        streams = []
        for source_index, current in enumerate(source_currents):
            source_history = history[history_indices[source_index]]
            per_kernel = [current]
            for kernel_index, kernel in enumerate(self.msar_kernels[1:]):
                weight = self.msar_conv[kernel_index].to(dtype=current.dtype)
                value = current * weight[:, kernel - 1]
                for lag in range(1, kernel):
                    value = value + source_history[:, -lag] * weight[:, kernel - 1 - lag]
                per_kernel.append(value)
            streams.append(torch.stack(per_kernel, dim=0))
        stream_sources = torch.stack(streams, dim=0)  # [sources, streams, batch, D]
        normed = stream_sources.float() * torch.rsqrt(
            stream_sources.float().square().mean(dim=-1, keepdim=True).add(1e-6)
        )
        query = self.msar_queries[layer_idx].float()
        scores = torch.einsum("smbd,md->smb", normed, query) / (self.d_model**0.5)
        weights = torch.softmax(scores, dim=0)
        retrieved = (weights.unsqueeze(-1) * stream_sources).sum(dim=0)
        retrieved_last = retrieved.movedim(0, -2)  # [batch, streams, D]
        flat = retrieved_last.reshape(retrieved_last.shape[0], -1)
        mix = torch.sigmoid(
            flat.float() @ self.msar_pre[layer_idx].float()
            + self.msar_bias[layer_idx].float()
        )
        fused = (mix.unsqueeze(-1) * retrieved_last).sum(dim=-2)
        return fused.to(source_currents[-1].dtype)

    def forward(
        self,
        hidden_states,
        cu_seqlens=None,
        max_seqlen=None,
        mask=None,
        inference_params=None,
        graph_dense_mask=False,
        **mixer_kwargs,
    ):
        assert (mask is not None) or (
            cu_seqlens is not None and max_seqlen is not None
        ), "Either mask or cu_seqlens and max_seqlen must be provided"

        # Cached prefill is token-indexed and cannot use packed variable-length
        # metadata.  The top-level H-Net trims its static latent padding before
        # calling this stack; force the same dense contract here for the
        # one-chunk case as well.
        if inference_params is not None and mask is not None:
            if not bool(mask.all()):
                raise ValueError("inference-cache prefill requires an all-true dense mask")
            cu_seqlens = None
            max_seqlen = None

        attn_mixer_kwargs = copy.deepcopy(mixer_kwargs)
        ssm_mixer_kwargs = copy.deepcopy(mixer_kwargs)
        unpack_mask = None
        if mask is not None:
            packed = False
            assert (
                hidden_states.dim() == 3
            ), "Hidden states must be (B, L, D) in unpacked mode"
            # Static chunking left-aligns valid chunks and pads the remainder.
            # Pack those rows before attention so FlashAttention never
            # normalizes over zero-valued padding keys. The result is scattered
            # back to the fixed outer shape after all inner layers finish.
            # Router chunking left-aligns selected boundaries and leaves all
            # padding at the end of each row. During graph capture, keep this
            # fixed padded shape instead of reading ``mask.all()`` on the host:
            # causal attention never lets a real query attend to the trailing
            # padding, while the active mask zeros padded residual/MLP rows.
            if graph_capture_enabled():
                active = mask.unsqueeze(-1)
                hidden_states = hidden_states * active.to(hidden_states.dtype)
            elif (
                inference_params is None
                and hidden_states.shape[1] == 1
                and bool(mask.all())
            ):
                # The pinned TE BSHD CroissAttn kernel requires at least two
                # query positions.  A learned router can legitimately emit a
                # single latent boundary for a short/easy prompt; route that
                # singleton through the equivalent packed THD interface so it
                # still uses fused TE attention and preserves the outer shape.
                unpack_mask = mask
                counts = mask.sum(dim=-1, dtype=torch.int32)
                cu_seqlens = F.pad(counts.cumsum(0), (1, 0)).to(torch.int32)
                max_seqlen = 1
                hidden_states = hidden_states[mask]
                attn_mixer_kwargs.update(
                    {"cu_seqlens": cu_seqlens, "max_seqlen": max_seqlen}
                )
                ssm_mixer_kwargs.update(
                    {"seq_idx": get_seq_idx(
                        cu_seqlens,
                        device=hidden_states.device,
                        total_length=hidden_states.shape[0],
                    )}
                )
                packed = True
                mask = None
            elif not bool(mask.all()):
                if inference_params is not None:
                    raise ValueError("padded static chunking cannot use an inference cache")
                unpack_mask = mask
                counts = mask.sum(dim=-1, dtype=torch.int32)
                cu_seqlens = F.pad(counts.cumsum(0), (1, 0)).to(torch.int32)
                max_seqlen = int(counts.max())
                hidden_states = hidden_states[mask]
                attn_mixer_kwargs.update(
                    {"cu_seqlens": cu_seqlens, "max_seqlen": max_seqlen}
                )
                ssm_mixer_kwargs.update(
                    {
                        "seq_idx": get_seq_idx(
                            cu_seqlens,
                            device=hidden_states.device,
                            total_length=(
                                hidden_states.shape[0]
                                if hidden_states.dim() == 2
                                else hidden_states.shape[1]
                            ),
                        )
                    }
                )
                packed = True
                mask = None
            else:
                active = mask.unsqueeze(-1)
                hidden_states = hidden_states * active.to(hidden_states.dtype)
        else:
            attn_mixer_kwargs.update(
                {"cu_seqlens": cu_seqlens.int(), "max_seqlen": max_seqlen}
            )
            ssm_mixer_kwargs.update(
                {
                    "seq_idx": get_seq_idx(
                        cu_seqlens,
                        device=hidden_states.device,
                        total_length=(
                            hidden_states.shape[0]
                            if hidden_states.dim() == 2
                            else hidden_states.shape[1]
                        ),
                    )
                }
            )
            packed = True

        if self.loop_count > 1 and inference_params is not None and any(
            arch in ("m", "M") for arch in self.arch_full
        ):
            raise RuntimeError("inner recurrent loops with Mamba require per-loop SSM state; use attention-only inner blocks")

        for _loop_idx in range(self.loop_count):
            # Transformer Engine delayed-scaling modules keep per-call
            # quantizer state in the active autocast region.  The canonical
            # middle stack reuses every layer for two recurrent executions;
            # give each execution its own region so a layer is invoked once
            # per region (the TE-supported contract).  This also keeps the
            # two FBT passes independent when HNet calls us repeatedly.
            with fp8_context():
                if self.msar_enabled:
                    msar_state = hidden_states
                    msar_sources = [hidden_states]
                    msar_stream_outputs = [hidden_states]
                    for layer_idx, (layer, arch) in enumerate(zip(self.layers, self.arch_full)):
                        layer_input = self._msar_mix_recompute(
                            msar_sources + [msar_state], layer_idx,
                            cu_seqlens=cu_seqlens if packed else None,
                        )
                        if arch in ("m", "M") and layer_input.dim() == 2:
                            layer_input = layer_input.unsqueeze(0)
                        elif arch in ("s", "S", "t", "T") and packed and layer_input.dim() == 3:
                            layer_input = layer_input.squeeze(0)
                        # MSAR retrieves in FP32 for stable cosine/softmax math;
                        # restore the block's parameter dtype before its fused
                        # mixer projection.
                        layer_input = layer_input.to(dtype=layer.norm1.weight.dtype)
                        if arch in ("m", "M"):
                            msar_out, msar_residual = self._run_mamba_block(
                                layer,
                                layer_input,
                                None,
                                inference_params=inference_params,
                                mixer_kwargs=ssm_mixer_kwargs,
                                active_mask=mask,
                            )
                        else:
                            msar_out, msar_residual = layer(
                                layer_input,
                                None,
                                inference_params=inference_params,
                                mixer_kwargs=attn_mixer_kwargs,
                                active_mask=mask,
                            )
                        if mask is not None:
                            active_dtype = active.to(msar_out.dtype)
                            msar_out = msar_out * active_dtype
                            if msar_residual is not None:
                                msar_residual = msar_residual * active_dtype
                        raw_state = msar_out if msar_residual is None else msar_out + msar_residual
                        # Gate each layer's multi-scale residual independently.
                        # A single mean scale would force all layers to share one
                        # scalar gradient and waste the per-layer parameters.
                        layer_scale = torch.tanh(self.msar_scale[layer_idx]).to(raw_state.dtype)
                        msar_state = layer_input + layer_scale * (raw_state - layer_input)
                        if packed and msar_state.dim() == 3:
                            msar_state = msar_state.squeeze(0)
                        if (layer_idx + 1) % self.msar_block_size == 0:
                            msar_sources.append(msar_state)
                        msar_stream_outputs.append(msar_state)
                    hidden_states = msar_state
                    hidden_states = self.rmsnorm(
                        hidden_states, residual=None, prenorm=False, residual_in_fp32=True
                    )
                    if inference_params is not None:
                        loop_history = inference_params.msar_history[_loop_idx]
                        for source_idx, source in enumerate(msar_stream_outputs):
                            self._store_msar_history(loop_history[source_idx], source)
                    residual = None
                    continue
                residual = None
                # Reusing the same layer objects shares weights. CausalMHA uses a
                # loop-indexed cache slot so each recurrent depth has exact
                # autoregressive context; rotary offsets advance once per token,
                # after all loops finish.
                for layer, arch in zip(self.layers, self.arch_full):
                    # Dechunk/residual paths may promote activations to FP32;
                    # fused Mamba/FlashAttention projections are BF16 in the
                    # production model.
                    hidden_states = hidden_states.to(dtype=layer.norm1.weight.dtype)
                    if arch in ("m", "M"):
                        layer_mixer_kwargs = ssm_mixer_kwargs
                        if hidden_states.dim() == 2:
                            hidden_states = hidden_states.unsqueeze(0)
                            residual = None if residual is None else residual.unsqueeze(0)
                    elif arch in ("s", "S", "t", "T"):
                        layer_mixer_kwargs = attn_mixer_kwargs
                        if hidden_states.dim() == 3 and packed:
                            hidden_states = hidden_states.squeeze(0)
                            residual = None if residual is None else residual.squeeze(0)
                    else:
                        # Currently supporting only Mamba2 and MHA
                        raise NotImplementedError
                    if arch in ("m", "M"):
                        hidden_states, residual = self._run_mamba_block(
                            layer,
                            hidden_states,
                            residual,
                            inference_params=inference_params,
                            mixer_kwargs=layer_mixer_kwargs,
                            active_mask=mask,
                        )
                    else:
                        hidden_states, residual = layer(
                            hidden_states,
                            residual,
                            inference_params=inference_params,
                            mixer_kwargs=layer_mixer_kwargs,
                            active_mask=mask,
                        )
                    if mask is not None:
                        active_dtype = active.to(hidden_states.dtype)
                        hidden_states = hidden_states * active_dtype
                        if residual is not None:
                            residual = residual * active_dtype

                # Setting prenorm=False ignores the residual.
                hidden_states = self.rmsnorm(
                    hidden_states, residual=residual, prenorm=False, residual_in_fp32=True
                )

        if hidden_states.dim() == 3 and packed:
            hidden_states = hidden_states.squeeze(0)

        if unpack_mask is not None:
            scattered = hidden_states.new_zeros(
                unpack_mask.shape + (hidden_states.shape[-1],)
            )
            hidden_states = scattered.masked_scatter(
                unpack_mask.unsqueeze(-1).expand_as(scattered), hidden_states
            )

        if inference_params is not None:
            # here we also explicitly assume the mask is all True
            assert hidden_states.shape[0] == 1, "seqlen_offset handling assumes batch size 1"
            inference_params.seqlen_offset += hidden_states.shape[1]

        return hidden_states

    def step(self, hidden_states, inference_params):
        """
        Assumes hidden_states is (B, 1, D). Steps each of the layers in order, and then steps the main model.
        """
        if self.msar_enabled:
            if inference_params.msar_history is None:
                raise RuntimeError("MSAR inference cache was not allocated for this stack")
            # The canonical deepest stack is attention-only, so its two loops
            # can share each layer's KV slot while MSAR keeps a separate
            # bounded stream history for each recurrent execution.
            for loop_idx in range(self.loop_count):
                history = inference_params.msar_history[loop_idx]
                base_state = hidden_states[:, 0]
                state = base_state
                boundary_currents = []
                boundary_indices = []
                layer_outputs = []
                for layer_idx, layer in enumerate(self.layers):
                    source_currents = [base_state, *boundary_currents, state]
                    history_indices = [0, *boundary_indices, layer_idx]
                    layer_input = self._msar_mix_step(
                        source_currents, history, layer_idx, history_indices
                    )
                    # MSAR retrieval performs its normalization and mixing in
                    # FP32.  Restore the block's parameter dtype before the
                    # TE projection, matching the training/prefill path.
                    block_input = layer_input.to(dtype=layer.norm1.weight.dtype).unsqueeze(1)
                    # Decode must use the same TE FP8 projection path as
                    # cached prefill; otherwise generation silently switches
                    # to BF16 while training/prefill use delayed-scaling FP8.
                    with fp8_context():
                        block_out, block_residual = layer.step(
                            block_input,
                            inference_params,
                            residual=None,
                        )
                    raw_state = block_out[:, 0]
                    if block_residual is not None:
                        raw_state = raw_state + block_residual[:, 0]
                    scale = torch.tanh(self.msar_scale[layer_idx]).to(raw_state.dtype)
                    # Training casts the MSAR retrieval to the block's
                    # parameter dtype before applying the gated residual. Use
                    # that same rounded value in decode; retaining the FP32
                    # pre-cast retrieval here makes cached generation follow
                    # a subtly different recurrence from prefill/training.
                    block_input_state = block_input[:, 0]
                    state = block_input_state + scale * (raw_state - block_input_state)
                    layer_outputs.append(state)
                    if (layer_idx + 1) % self.msar_block_size == 0:
                        boundary_currents.append(state)
                        boundary_indices.append(layer_idx + 1)
                # Histories represent source streams from prior tokens. Append
                # only after the complete current token has traversed the
                # stack; appending at a boundary would make the next layer's
                # causal kernel read the current token as lag-1.
                self._append_msar_history(history[0], base_state)
                for layer_index, current in enumerate(layer_outputs, start=1):
                    self._append_msar_history(history[layer_index], current)
                hidden_states = self.rmsnorm(
                    state.unsqueeze(1),
                    residual=None,
                    prenorm=False,
                    residual_in_fp32=True,
                )
            inference_params.seqlen_offset += 1
            return hidden_states
        if self.loop_count > 1 and any(arch in ("m", "M") for arch in self.arch_full):
            # Mamba2Wrapper owns one recurrent state per layer; unlike
            # CausalMHA it does not allocate loop-indexed state slots.
            raise RuntimeError(
                "inner recurrent loops with Mamba require per-loop SSM state; "
                "use attention-only inner blocks"
            )
        for _loop_idx in range(self.loop_count):
            residual = None
            for layer in self.layers:
                hidden_states = hidden_states.to(dtype=layer.norm1.weight.dtype)
                with fp8_context():
                    hidden_states, residual = layer.step(
                        hidden_states, inference_params, residual=residual
                    )
            hidden_states = self.rmsnorm(
                hidden_states, residual=residual, prenorm=False, residual_in_fp32=True
            )
        inference_params.seqlen_offset += 1

        return hidden_states
