import re
import copy
from dataclasses import dataclass, field

import optree

from typing import Optional

import torch
import torch.nn as nn
import torch.nn.functional as F

from flash_attn.ops.triton.layer_norm import RMSNorm

from hnet.modules.block import create_block
from hnet.modules.utils import get_seq_idx, get_stage_cfg

from hnet.models.config_hnet import HNetConfig


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
        self.msar_kernels = tuple(getattr(config, "msar_stream_kernels", [1, 4, 8, 12]))
        self.msar_block_size = int(getattr(config, "msar_block_size", 4))
        self.d_model = config.d_model[self.stage_idx]
        self.ssm_cfg = get_stage_cfg(config.ssm_cfg, stage_idx)
        self.attn_cfg = get_stage_cfg(config.attn_cfg, stage_idx)

        arch_layout = config.arch_layout
        for _ in range(stage_idx):
            arch_layout = arch_layout[1]
        arch_layout = arch_layout[pos_idx]
        layout_parse = re.findall(r"([mMtT])(\d+)", arch_layout)

        layers = []
        layer_idx = 0
        self.arch_full = []

        # self.height counts the number of things that get added to the residual stream
        self.height = 0
        for arch, n_layer in layout_parse:
            assert arch in ("m", "M", "t", "T")
            assert n_layer.isdigit()
            layers += [
                create_block(
                    arch,
                    self.d_model,
                    d_intermediate=config.d_intermediate[self.stage_idx],
                    ssm_cfg=self.ssm_cfg,
                    attn_cfg=self.attn_cfg,
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
        # Packed streams need causal resets at each document boundary.
        pieces = []
        for start, end in zip(cu_seqlens[:-1].tolist(), cu_seqlens[1:].tolist()):
            seg = x[:, start:end]
            pieces.append(
                F.conv1d(
                    F.pad(seg.transpose(1, 2), (kernel - 1, 0)),
                    weight.to(dtype=x.dtype).unsqueeze(1),
                    groups=self.d_model,
                ).transpose(1, 2)
            )
        y = torch.cat(pieces, dim=1)
        return y.squeeze(0) if original_2d else y

    def _msar_mix(self, sources, layer_idx, cu_seqlens=None):
        expanded = []
        conv_idx = 0
        for source in sources:
            streams = [source]
            for kernel in self.msar_kernels[1:]:
                streams.append(
                    self._msar_expand(source, kernel, self.msar_conv[conv_idx], cu_seqlens)
                )
                conv_idx += 1
            expanded.append(torch.stack(streams, dim=0))
            conv_idx = 0
        stream_sources = torch.stack(expanded, dim=0)  # [N,m,...,D]
        normed = stream_sources.float() * torch.rsqrt(
            stream_sources.float().square().mean(dim=-1, keepdim=True).add(1e-6)
        )
        query = self.msar_queries[layer_idx].float()
        scores = torch.einsum("sm...d,md->sm...", normed, query) / (self.d_model**0.5)
        weights = torch.softmax(scores, dim=0)
        retrieved = (weights.unsqueeze(-1) * stream_sources).sum(dim=0)
        retrieved_last = retrieved.movedim(0, -2)
        flat = retrieved_last.reshape(*retrieved.shape[1:-1], -1)
        mix = torch.sigmoid(
            flat.float() @ self.msar_pre[layer_idx].float() + self.msar_bias[layer_idx].float()
        )
        fused = (mix.unsqueeze(-1) * retrieved_last).sum(dim=-2)
        return fused.to(sources[-1].dtype)

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
        for i, layer in enumerate(self.layers):
            key_value_memory_dict[i] = layer.allocate_inference_cache(
                batch_size, max_seqlen, dtype=dtype
            )
        return IsotropicInferenceParams(
            key_value_memory_dict=key_value_memory_dict,
            max_seqlen=max_seqlen,
            max_batch_size=batch_size,
        )

    def forward(
        self,
        hidden_states,
        cu_seqlens=None,
        max_seqlen=None,
        mask=None,
        inference_params=None,
        **mixer_kwargs,
    ):
        assert (mask is not None) or (
            cu_seqlens is not None and max_seqlen is not None
        ), "Either mask or cu_seqlens and max_seqlen must be provided"

        attn_mixer_kwargs = copy.deepcopy(mixer_kwargs)
        ssm_mixer_kwargs = copy.deepcopy(mixer_kwargs)
        if mask is not None:
            packed = False
            assert (
                hidden_states.dim() == 3
            ), "Hidden states must be (B, L, D) in unpacked mode"
        else:
            attn_mixer_kwargs.update(
                {"cu_seqlens": cu_seqlens.int(), "max_seqlen": max_seqlen}
            )
            ssm_mixer_kwargs.update(
                {"seq_idx": get_seq_idx(cu_seqlens, device=hidden_states.device)}
            )
            packed = True

        if self.loop_count > 1 and inference_params is not None and any(
            arch in ("m", "M") for arch in self.arch_full
        ):
            raise RuntimeError("inner recurrent loops with Mamba require per-loop SSM state; use attention-only inner blocks")

        for _loop_idx in range(self.loop_count):
            if self.msar_enabled:
                msar_state = hidden_states
                msar_sources = [hidden_states]
                for layer_idx, (layer, arch) in enumerate(zip(self.layers, self.arch_full)):
                    layer_input = self._msar_mix(
                        msar_sources + [msar_state], layer_idx,
                        cu_seqlens=cu_seqlens if packed else None,
                    )
                    if arch in ("m", "M") and layer_input.dim() == 2:
                        layer_input = layer_input.unsqueeze(0)
                    elif arch in ("t", "T") and packed and layer_input.dim() == 3:
                        layer_input = layer_input.squeeze(0)
                    # MSAR retrieves in FP32 for stable cosine/softmax math;
                    # restore the block's parameter dtype before its fused
                    # mixer projection.
                    layer_input = layer_input.to(dtype=layer.norm1.weight.dtype)
                    msar_out, msar_residual = layer(
                        layer_input,
                        None,
                        inference_params=inference_params,
                        mixer_kwargs=(
                            ssm_mixer_kwargs if arch in ("m", "M")
                            else {**attn_mixer_kwargs, "loop_idx": _loop_idx}
                        ),
                        active_mask=mask,
                    )
                    msar_state = msar_out if msar_residual is None else msar_out + msar_residual
                    if packed and msar_state.dim() == 3:
                        msar_state = msar_state.squeeze(0)
                    if (layer_idx + 1) % self.msar_block_size == 0:
                        msar_sources.append(msar_state)
                hidden_states = hidden_states + torch.tanh(self.msar_scale.mean()) * (
                    msar_state - hidden_states
                )
                hidden_states = self.rmsnorm(
                    hidden_states, residual=None, prenorm=False, residual_in_fp32=True
                )
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
                elif arch in ("t", "T"):
                    layer_mixer_kwargs = {**attn_mixer_kwargs, "loop_idx": _loop_idx}
                    if hidden_states.dim() == 3 and packed:
                        hidden_states = hidden_states.squeeze(0)
                        residual = None if residual is None else residual.squeeze(0)
                else:
                    # Currently supporting only Mamba2 and MHA
                    raise NotImplementedError
                hidden_states, residual = layer(
                    hidden_states,
                    residual,
                    inference_params=inference_params,
                    mixer_kwargs=layer_mixer_kwargs,
                    active_mask=mask,
                )

        # Setting prenorm=False ignores the residual
            hidden_states = self.rmsnorm(
                hidden_states, residual=residual, prenorm=False, residual_in_fp32=True
            )

        if hidden_states.dim() == 3 and packed:
            hidden_states = hidden_states.squeeze(0)

        if inference_params is not None:
            # here we also explicitly assume the mask is all True
            assert mask.shape[0] == 1, "seqlen_offset handling assumes batch size 1"
            inference_params.seqlen_offset += hidden_states.shape[1]

        return hidden_states

    def step(self, hidden_states, inference_params):
        """
        Assumes hidden_states is (B, 1, D). Steps each of the layers in order, and then steps the main model.
        """
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
                hidden_states, residual = layer.step(
                    hidden_states, inference_params, residual=residual, loop_idx=_loop_idx
                )
            hidden_states = self.rmsnorm(
                hidden_states, residual=residual, prenorm=False, residual_in_fp32=True
            )
        inference_params.seqlen_offset += 1

        return hidden_states
