from dataclasses import dataclass
from typing import Union, Optional

import torch
import torch.nn as nn
import torch.nn.functional as F
from flash_attn.ops.triton.layer_norm import RMSNorm

from hnet.modules.isotropic import Isotropic, IsotropicInferenceParams
from hnet.modules.dc import (
    RoutingModule,
    ChunkLayer,
    DeChunkLayer,
    RoutingModuleState,
    DeChunkState,
)
from hnet.modules.utils import apply_optimization_params
from hnet.te import fp8_context, make_linear

from .config_hnet import HNetConfig


class ResidualSTE(torch.autograd.Function):
    """Residual gate with a straight-through boundary gradient.

    The old implementation materialized a full ``ones_like(p)`` tensor and
    multiplied it into the decoded activation.  The forward value is exactly
    ``out + residual``; only the backward pass needs the gate derivative.  A
    custom function keeps that derivative while eliminating the wide ones and
    multiply kernels from every FBT pass.
    """

    @staticmethod
    def forward(ctx, out, residual, p):
        ctx.save_for_backward(out, p)
        return out + residual

    @staticmethod
    def backward(ctx, grad_output):
        out, p = ctx.saved_tensors
        grad_p = (grad_output * out).sum_to_size(p.shape)
        return grad_output, grad_output, grad_p


@dataclass
class HNetState:
    encoder_state: Optional[IsotropicInferenceParams] = None
    routing_module_state: Optional[RoutingModuleState] = None
    main_network_state: Optional[Union["HNetState", IsotropicInferenceParams]] = None
    dechunk_state: Optional[DeChunkState] = None
    decoder_state: Optional[IsotropicInferenceParams] = None
    # Outer-stage FBT state. The first decoded token uses the plain byte
    # feature; subsequent tokens fuse this previous top-level state with the
    # current byte feature through the learned gated cross.
    feedback_hidden_state: Optional[torch.Tensor] = None
    feedback_initialized: Optional[torch.Tensor] = None


class HNet(nn.Module):
    def __init__(
        self,
        config: HNetConfig,
        stage_idx: int,
        device=None,
        dtype=None,
        rotary_emb_shared=None,
    ) -> None:
        super().__init__()

        self.config = config
        factory_kwargs = {"device": device, "dtype": dtype}

        self.stage_idx = stage_idx
        self.d_model = config.d_model[stage_idx]

        arch_layout = config.arch_layout
        for _ in range(stage_idx):
            arch_layout = arch_layout[1]

        assert isinstance(arch_layout, list), f"Wrong arch_layout: {arch_layout}"
        if len(arch_layout) == 3:
            sub_model_names = ["encoder", "main_network", "decoder"]
            self.is_innermost = False
        elif len(arch_layout) == 1:
            sub_model_names = ["main_network"]
            self.is_innermost = True
        else:
            raise NotImplementedError

        for _name, _layout in zip(sub_model_names, arch_layout):
            if self.is_innermost or _name in ("encoder", "decoder"):
                SubModel = Isotropic
                _stage_idx = stage_idx
                _pos_idx = None
                if _name == "encoder":
                    _pos_idx = 0
                elif self.is_innermost:
                    # if innermost, then len(layer_layout) == 1
                    _pos_idx = 0
                elif _name == "decoder":
                    _pos_idx = 2
                _pos_idx_dict = {"pos_idx": _pos_idx}
            else:
                SubModel = HNet
                _stage_idx = stage_idx + 1
                _pos_idx_dict = {}

            _sub_model = SubModel(
                config=config,
                stage_idx=_stage_idx,
                **_pos_idx_dict,
                rotary_emb_shared=rotary_emb_shared,
                **factory_kwargs,
            )
            self.add_module(_name, _sub_model)

        if not self.is_innermost:
            self.routing_module = RoutingModule(self.d_model, **factory_kwargs)
            self.chunk_layer = ChunkLayer(
                static_max_seqlen=(
                    min(
                        int(getattr(config, "max_seq_len", 4096)),
                        int(getattr(config, "chunk_capacity_tokens", 1024)),
                    )
                    if getattr(config, "static_chunking", False)
                    else None
                )
            )
            self.dechunk_layer = DeChunkLayer(self.d_model)

            # do the residual in fp32
            self.residual_proj = make_linear(
                self.d_model,
                self.d_model,
                device=device,
                dtype=torch.float32,
                fp8=False,
            )
            nn.init.zeros_(self.residual_proj.weight)
            nn.init.zeros_(self.residual_proj.bias)
            self.residual_proj.weight._no_reinit = True
            self.residual_proj.bias._no_reinit = True

            self.residual_func = ResidualSTE.apply
            # Full-bandwidth feedback is attached only at the outer boundary
            # before the deepest main network. It never changes router masks,
            # chunk counts, or the encoder/decoder residual path.
            self.fbt_enabled = bool(getattr(config, "fbt", False) and stage_idx == 0)
            if self.fbt_enabled:
                # The recursive main network returns its state projected back
                # to the outer width D0. Feedback therefore stays at D0 and
                # never changes chunk boundaries or inner dimensions.
                self.fbt_up = make_linear(self.d_model, self.d_model, bias=False, **factory_kwargs)
                self.fbt_gate = make_linear(self.d_model, self.d_model, bias=False, **factory_kwargs)
                # FBT uses the paper's shared input RMSNorm (Appendix Fig. 9)
                # on the token gate input and again on the fused input before
                # the shared stack. Sharing this norm keeps the two input
                # paths in one learned basis.
                self.fbt_input_norm_1 = RMSNorm(self.d_model, eps=1e-5, **factory_kwargs)
                nn.init.zeros_(self.fbt_up.weight)
                nn.init.zeros_(self.fbt_gate.weight)
                self.fbt_up.weight._no_reinit = True
                self.fbt_gate.weight._no_reinit = True
        else:
            self.fbt_enabled = False

        if stage_idx > 0 and self.d_model - config.d_model[stage_idx - 1] > 0:
            self.pad_dimension = nn.Parameter(
                torch.zeros(
                    self.d_model - config.d_model[stage_idx - 1], **factory_kwargs
                )
            )
        else:
            self.pad_dimension = None
    

    def _init_weights(self, initializer_range: float = 0.02, parent_residuals: int = 0) -> None:
        n_residuals = parent_residuals

        def is_reinitializable_projection(module: nn.Module) -> bool:
            # Transformer Engine's GroupedLinear is not an nn.Linear, but its
            # stacked expert weights are still projections on the residual
            # path.  Include both TE class spellings so routed experts receive
            # the same depth scaling as dense/shared fc2 projections.
            return isinstance(module, nn.Linear) or module.__class__.__name__ in (
                "Linear",
                "AlignedTELinear",
                "GroupedLinear",
                "TEGroupedLinear",
            )

        def init_projection(name: str, module: nn.Module) -> None:
            if module.__class__.__name__ in ("GroupedLinear", "TEGroupedLinear"):
                # TE stores one parameter per expert (weight0, weight1, ...)
                # rather than exposing a single ``weight`` attribute.
                for parameter_name, weight in module.named_parameters(recurse=False):
                    if not parameter_name.startswith("weight") or getattr(
                        weight, "_no_reinit", False
                    ):
                        continue
                    std = (
                        initializer_range / (n_residuals ** 0.5)
                        if "out_proj" in name or "fc2" in name
                        else initializer_range
                    )
                    nn.init.normal_(weight, mean=0.0, std=std)
                return
            weight = getattr(module, "weight", None)
            if weight is None or getattr(weight, "_no_reinit", False):
                return
            if "out_proj" in name or "fc2" in name:
                nn.init.normal_(
                    weight, mean=0.0, std=initializer_range / (n_residuals ** 0.5)
                )
            else:
                nn.init.normal_(weight, mean=0.0, std=initializer_range)

        if self.is_innermost:
            n_residuals += self.main_network.height
            for name, m in self.main_network.named_modules():
                if is_reinitializable_projection(m):
                    init_projection(name, m)

        else:
            n_residuals += self.encoder.height + self.decoder.height
            for name, m in self.encoder.named_modules():
                if is_reinitializable_projection(m):
                    init_projection(name, m)
            for name, m in self.decoder.named_modules():
                if is_reinitializable_projection(m):
                    init_projection(name, m)
                    
            self.main_network._init_weights(initializer_range, n_residuals)

        # TE SwiGLU consumes gate/value FC1 rows. Swap freshly initialized
        # rows once after recursive projection initialization so the fused
        # activation preserves the historical value/gate function.
        for module in self.modules():
            reorder = getattr(module, "reorder_glu_parameters", None)
            if reorder is not None:
                reorder()
    

    def _apply_lr_multiplier(self, lr_multiplier: list[float]) -> None:
        """
        Applies the learning rate multipliers to the parameters of the model.
        """
        # a little stupid: we apply lr_multiplier to all parameters, and then for the main stage (which may have another hierarchy), we just apply it again there.
        for param in self.parameters():
            apply_optimization_params(param, lr_multiplier=lr_multiplier[self.stage_idx])
        
        if not self.is_innermost:
            self.main_network._apply_lr_multiplier(lr_multiplier)

    def allocate_inference_cache(self, batch_size, max_seqlen, dtype=None):
        """
        Allocate the inference cache for the HNet.

        Arguments:
            batch_size: int. The number of sequences in the batch.
            max_seqlen: int. The maximum sequence length in the batch.
            dtype: torch.dtype. The dtype of the inference cache.

        The structure of the inference cache is as follows:
            - [encoder state]
            - [routing module state]
            - [main network state]
            - [dechunk state]
            - [decoder state]
        It is thus a list of length 5.
        """
        if self.is_innermost:
            return HNetState(
                main_network_state=self.main_network.allocate_inference_cache(
                    batch_size, max_seqlen, dtype=dtype
                )
            )
        else:
            device = self.residual_proj.weight.device
            return HNetState(
                encoder_state=self.encoder.allocate_inference_cache(
                    batch_size, max_seqlen, dtype=dtype
                ),
                routing_module_state=self.routing_module.allocate_inference_cache(
                    batch_size, max_seqlen, device, dtype=dtype
                ),
                main_network_state=self.main_network.allocate_inference_cache(
                    batch_size, max_seqlen, dtype=dtype
                ),
                dechunk_state=self.dechunk_layer.allocate_inference_cache(
                    batch_size, max_seqlen, device, dtype=dtype
                ),
                decoder_state=self.decoder.allocate_inference_cache(
                    batch_size, max_seqlen, dtype=dtype
                ),
                feedback_hidden_state=(
                    torch.zeros(
                        batch_size,
                        self.d_model,
                        device=device,
                        dtype=dtype,
                    )
                    if self.fbt_enabled
                    else None
                ),
                feedback_initialized=(
                    torch.zeros(batch_size, device=device, dtype=torch.bool)
                    if self.fbt_enabled
                    else None
                ),
            )

    def forward(
        self,
        hidden_states,
        cu_seqlens=None,
        max_seqlen=None,
        mask=None,
        inference_params=None,
        fbt_passes=1,
        return_feedback=False,
        **mixer_kwargs,
    ):
        assert mask is not None or (
            cu_seqlens is not None and max_seqlen is not None
        ), "Either mask or cu_seqlens and max_seqlen must be provided"
        if fbt_passes > 1 and inference_params is not None:
            raise ValueError("FBT multi-pass training requires fresh caches; do not combine it with inference_params")

        # The model creates an empty state container for ordinary training so
        # nested modules can share one call signature.  Only an object supplied
        # by the caller represents a requested autoregressive cache; keep that
        # distinction explicit for the prefill-only trimming below.
        cache_requested = inference_params is not None
        if inference_params is None:
            inference_params = HNetState(main_network_state=None)
        else:
            assert (
                mask is not None
            ), "Mask must be provided if inference_params is provided"
            # The recurrent cache stores one state per real token.  A padded
            # outer prompt would make the encoder/router offsets ambiguous and
            # cannot be represented by the scalar ``seqlen_offset`` fields.
            # Require a contiguous, unpadded prefill instead of silently
            # caching padding as context.
            if mask.dtype is not torch.bool or not bool(mask.all()):
                raise ValueError(
                    "inference-cache prefill requires a contiguous unpadded mask"
                )

        D = hidden_states.shape[-1]
        EARLY_DIMS = hidden_states.shape[:-1]
        # The trainer's static byte mask is dense, but masks produced by the
        # router/chunk layer are data-dependent.  Keep this marker local to
        # encoder/decoder calls so an inner recurrent H-Net cannot mistake its
        # padded chunk mask for a dense sequence during graph capture.
        graph_dense_mask = bool(mixer_kwargs.pop("graph_dense_mask", False))

        if self.pad_dimension is not None:
            hidden_states = torch.cat(
                (hidden_states, self.pad_dimension.expand(EARLY_DIMS + (-1,))), dim=-1
            )

        if self.is_innermost:
            hidden_states = self.main_network(
                hidden_states,
                cu_seqlens=cu_seqlens,
                max_seqlen=max_seqlen,
                mask=mask,
                inference_params=inference_params.main_network_state,
                **mixer_kwargs,
            )
            hidden_states = hidden_states[..., :D]
            return hidden_states, []

        hidden_states = self.encoder(
            hidden_states,
            cu_seqlens=cu_seqlens,
            max_seqlen=max_seqlen,
            mask=mask,
            inference_params=inference_params.encoder_state,
            graph_dense_mask=graph_dense_mask,
            **mixer_kwargs,
        )

        hidden_states_for_residual = hidden_states.to(
            dtype=self.residual_proj.weight.dtype
        )
        # Residual and boundary controls stay outside FP8 autocast. Their
        # projections are initialized at zero and feed the chunk mask/identity
        # path; quantizing them would perturb the hard routing decision and
        # defeat the FP32 safety contract used by the CE/router fixes.
        residual = self.residual_proj(hidden_states_for_residual)
        bpred_output = self.routing_module(
            hidden_states,
            cu_seqlens=cu_seqlens,
            mask=mask,
            inference_params=inference_params.routing_module_state,
        )
        (
            hidden_states,
            next_cu_seqlens,
            next_max_seqlen,
            next_mask,
            effective_boundary_mask,
        ) = self.chunk_layer(
            hidden_states,
            bpred_output.boundary_mask,
            cu_seqlens,
            mask=mask,
            boundary_prob=bpred_output.boundary_prob,
        )
        # Capacity clipping is causal and can discard later router decisions.
        # Preserve the raw mask for diagnostics, but expose the retained mask
        # to the objective so router balancing follows the actual chunk set
        # consumed by the inner stack.
        bpred_output.effective_boundary_mask = effective_boundary_mask
        # Dechunking must use the same capacity-trimmed boundary set that was
        # gathered for the inner stack.  Keep the router's original mask in
        # ``bpred_output`` so the routing loss and diagnostics measure the
        # learned decision rather than the static capacity overflow policy.
        effective_selected_probs = torch.where(
            effective_boundary_mask,
            bpred_output.boundary_prob[..., 1],
            bpred_output.boundary_prob[..., 0],
        ).unsqueeze(-1)

        # ChunkLayer's static path returns a left-aligned padded latent tensor
        # so the training graph has stable addresses.  Inference caches are
        # token indexed, therefore remove that padding before the deepest
        # stack.  This is exact for batch size one (the supported generation
        # path) because there is a single active latent count and the valid
        # rows are already left aligned by ChunkLayer.
        if cache_requested and next_mask is not None:
            if next_mask.shape[0] != 1:
                raise ValueError(
                    "inference-cache H-Net prefill currently requires batch size 1"
                )
            latent_count = int(next_mask.sum().item())
            if latent_count <= 0:
                raise ValueError("inference-cache prefill produced no latent boundaries")
            hidden_states = hidden_states[:, :latent_count]
            next_mask = torch.ones(
                (1, latent_count), device=hidden_states.device, dtype=torch.bool
            )
            next_max_seqlen = None
            if inference_params.routing_module_state.boundary_count is not None:
                inference_params.routing_module_state.boundary_count.copy_(
                    effective_boundary_mask.sum(dim=-1).to(dtype=torch.int32)
                )

        n_passes = max(1, int(fbt_passes)) if self.fbt_enabled else 1
        # A normal one-pass prefill must populate the allocated recurrent/KV
        # caches for autoregressive ``step``. FBT uses fresh inner passes and
        # is explicitly disallowed with inference caches above.
        cached_main_state = (
            inference_params.main_network_state if n_passes == 1 else None
        )
        cached_dechunk_state = (
            inference_params.dechunk_state if n_passes == 1 else None
        )
        cached_decoder_state = (
            inference_params.decoder_state if n_passes == 1 else None
        )
        feedback_logits = []
        base_chunk_states = hidden_states
        feedback_token = (
            self.fbt_input_norm_1(base_chunk_states)
            if self.fbt_enabled and n_passes > 1
            else None
        )
        previous_inner = None
        final_prev_boundary_predictions = []
        for pass_idx in range(n_passes):
            pass_input = base_chunk_states
            if pass_idx > 0:
                # Shift within each packed document; no cross-document latent
                # state is allowed to enter a new byte stream.
                feedback_source = previous_inner
                if self.training and getattr(self.config, "fbt_jitter", 0.0) > 0:
                    # The paper perturbs the previous hidden state before the
                    # causal shift. This keeps every document's first feedback
                    # token exactly zero instead of injecting noise at a reset.
                    feedback_source = feedback_source + torch.empty_like(
                        feedback_source
                    ).uniform_(
                        -float(self.config.fbt_jitter),
                        float(self.config.fbt_jitter),
                    )
                if next_cu_seqlens is None:
                    # The padded graph path has a fixed dense sequence.  F.pad
                    # emits one contiguous copy for the causal shift, whereas
                    # concatenating a separately allocated zero column causes
                    # two full activation copies and an extra fill launch.
                    shifted = F.pad(feedback_source[:, :-1], (0, 0, 1, 0))
                else:
                    # Packed feedback is shifted within each document.  Use
                    # differentiable tensor operations instead of indexed
                    # writes into a ``zeros_like`` destination.  Indexed
                    # writes there detach the previous pass and silently
                    # freeze both FBT projection matrices at zero.
                    # Keep the shift as one destination allocation and two
                    # slice copies. This preserves CopySlices autograd while
                    # avoiding the full-token cat temporary and its fill.
                    shifted = torch.empty_like(feedback_source)
                    shifted[:1].zero_()
                    shifted[1:].copy_(feedback_source[:-1])
                    starts = next_cu_seqlens[:-1]
                    is_start = torch.zeros(
                        (shifted.shape[0],), device=shifted.device, dtype=torch.bool
                    )
                    is_start.scatter_(0, starts, True)
                    shifted.masked_fill_(is_start[:, None], 0)
                # Quantize the learned feedback projections with the same FP8
                # recipe as the main stack, then evaluate the sigmoid in FP32
                # so the multiplicative gate keeps stable gradients.
                # FBT's fused input is the gated projection of the previous
                # pass (Eq. 4 in arXiv:2608.08888).  Do not add the current
                # pass input here: that shortcut changes the objective's
                # conditioning path and lets the second pass bypass feedback.
                shifted = shifted.to(dtype=self.fbt_up.weight.dtype)
                with fp8_context():
                    gate_logits = self.fbt_gate(feedback_token)
                    feedback_value = self.fbt_up(shifted)
                gate = torch.sigmoid(gate_logits.float()).to(pass_input.dtype)
                pass_input = feedback_value.to(pass_input.dtype) * gate
                if getattr(self.config, "fbt_prefix_mixin", False):
                    # Keep a random plain prefix during training so the model
                    # sees both prompt-prefill and feedback-conditioned rows.
                    # Evaluation must be deterministic: use the fully
                    # feedback-conditioned path rather than sampling a new
                    # prefix on every forward.
                    plain = torch.zeros(
                        pass_input.shape[:-1], dtype=torch.bool, device=pass_input.device
                    )
                    if self.training and next_cu_seqlens is None:
                        # ChunkLayer left-aligns real chunks and pads to a
                        # static bound. Draw one prefix length per batch row
                        # from the real count; a fixed first-half prefix would
                        # make feedback conditioning correlate with position
                        # and leave the same rows permanently unconditioned.
                        real_counts = next_mask.sum(dim=-1)
                        prefixes = (
                            torch.rand(real_counts.shape, device=pass_input.device)
                            * (real_counts + 1).to(torch.float32)
                        ).to(real_counts.dtype)
                        token_idx = torch.arange(
                            pass_input.shape[1], device=pass_input.device
                        )[None, :]
                        plain = token_idx < prefixes[:, None]
                        plain = plain & next_mask
                    elif self.training:
                        # Draw one prefix length per packed document on the
                        # device and map each token to its document without
                        # Python ``tolist``/``item`` synchronization.
                        starts = next_cu_seqlens[:-1]
                        ends = next_cu_seqlens[1:]
                        lengths = ends - starts
                        prefixes = (
                            torch.rand(lengths.shape, device=pass_input.device)
                            * (lengths + 1).to(torch.float32)
                        ).to(lengths.dtype)
                        token_idx = torch.arange(
                            pass_input.shape[0],
                            device=pass_input.device,
                            dtype=starts.dtype,
                        )
                        doc_idx = torch.bucketize(token_idx, ends[:-1], right=True)
                        plain = (token_idx - starts[doc_idx]) < prefixes[doc_idx]
                    # Cached prefill feeds the plain latent directly to the
                    # stack. Normalize only feedback-conditioned rows here so
                    # the random prefix mixin has the same input distribution
                    # as prefill once this shared norm learns non-unit scale.
                    fused_input = self.fbt_input_norm_1(pass_input)
                    pass_input = torch.where(
                        plain.unsqueeze(-1), base_chunk_states, fused_input
                    )
                else:
                    pass_input = self.fbt_input_norm_1(pass_input)
            inner_out, prev_boundary_predictions = self.main_network(
                pass_input,
                cu_seqlens=next_cu_seqlens,
                max_seqlen=next_max_seqlen,
                mask=next_mask,
                inference_params=cached_main_state,
                **mixer_kwargs,
            )
            previous_inner = inner_out
            final_prev_boundary_predictions = prev_boundary_predictions

            decoded = self.dechunk_layer(
                inner_out,
                effective_boundary_mask,
                bpred_output.boundary_prob,
                next_cu_seqlens,
                mask=mask,
                inference_params=cached_dechunk_state,
            )
            decoded = self.residual_func(
                decoded.to(dtype=residual.dtype), residual, effective_selected_probs
            ).to(decoded.dtype)
            decoded = self.decoder(
                decoded,
                cu_seqlens=cu_seqlens,
                max_seqlen=max_seqlen,
                mask=mask,
                inference_params=cached_decoder_state,
                graph_dense_mask=graph_dense_mask,
                **mixer_kwargs,
            )
            if return_feedback:
                feedback_logits.append(decoded)
        hidden_states = decoded

        hidden_states = hidden_states[..., :D]
        if (
            cache_requested
            and self.fbt_enabled
            and inference_params.feedback_hidden_state is not None
        ):
            # Prefill is the plain-input pass.  FBT is attached to the
            # chunk-level main network, so the recurrence state is the final
            # valid latent produced by that network rather than the decoded
            # byte-level state.  Decode applies exactly the same GLU to the
            # next selected latent boundary.
            if previous_inner is None or previous_inner.shape[1] == 0:
                raise RuntimeError("FBT prefill produced no latent state")
            inference_params.feedback_hidden_state.copy_(
                previous_inner[:, -1].to(dtype=inference_params.feedback_hidden_state.dtype)
            )
            inference_params.feedback_initialized.fill_(True)
        result = [bpred_output, *final_prev_boundary_predictions]
        if return_feedback:
            return hidden_states, result, feedback_logits
        return hidden_states, result

    def step(self, hidden_states, inference_params, moe_sequence_ids=None):
        D = hidden_states.shape[-1]

        if moe_sequence_ids is None:
            moe_sequence_ids = torch.arange(
                hidden_states.shape[0], device=hidden_states.device, dtype=torch.long
            )
        else:
            moe_sequence_ids = moe_sequence_ids.reshape(-1).to(
                device=hidden_states.device, dtype=torch.long
            )
            if moe_sequence_ids.numel() != hidden_states.shape[0]:
                raise ValueError("MoE sequence ids must match H-Net step batch rows")

        if self.pad_dimension is not None:
            hidden_states = torch.cat(
                (
                    hidden_states,
                    self.pad_dimension.expand(hidden_states.shape[:-1] + (-1,)),
                ),
                dim=-1,
            )

        if self.is_innermost:
            hidden_states = self.main_network.step(
                hidden_states,
                inference_params.main_network_state,
                moe_sequence_ids=moe_sequence_ids,
            )
            hidden_states = hidden_states[..., :D]
            return hidden_states, []

        hidden_states = self.encoder.step(hidden_states, inference_params.encoder_state)
        hidden_states_for_residual = hidden_states.to(
            dtype=self.residual_proj.weight.dtype
        )
        residual = self.residual_proj(hidden_states_for_residual)

        bpred_output = self.routing_module.step(
            hidden_states, inference_params.routing_module_state
        )
        # Decode must obey the same bounded latent contract as static
        # chunking.  Once the prefill has retained the configured maximum,
        # additional router boundaries are treated as ordinary bytes; this
        # keeps the online function within the capacity used for training.
        effective_boundary_mask = bpred_output.boundary_mask
        boundary_count = inference_params.routing_module_state.boundary_count
        if boundary_count is not None:
            capacity = int(getattr(self.config, "chunk_capacity_tokens", 0))
            if capacity > 0:
                effective_boundary_mask = effective_boundary_mask & (
                    boundary_count < capacity
                )
                boundary_count.add_(effective_boundary_mask.to(dtype=boundary_count.dtype))
        effective_selected_probs = torch.where(
            effective_boundary_mask,
            bpred_output.boundary_prob[..., 1],
            bpred_output.boundary_prob[..., 0],
        ).unsqueeze(-1)
        hidden_states_inner = self.chunk_layer.step(
            hidden_states, effective_boundary_mask
        )

        if hidden_states_inner.shape[0] > 0:
            active_rows = effective_boundary_mask.nonzero(as_tuple=True)[0]
            inner_sequence_ids = moe_sequence_ids.index_select(0, active_rows)
            if self.fbt_enabled:
                if (
                    inference_params.feedback_hidden_state is None
                    or inference_params.feedback_initialized is None
                ):
                    raise RuntimeError("FBT inference cache was not allocated for the outer H-Net")
                # The training objective fuses ``shift_right(h)`` with the
                # current chunk input ``e`` before rerunning the shared main
                # network.  Apply the same recurrence at decode time, after
                # routing selects a latent boundary.  Keeping this at the
                # chunk level is essential: the training pass does not feed
                # the decoded byte stream back into the encoder/router.
                active_rows = effective_boundary_mask.nonzero(as_tuple=True)[0]
                previous = inference_params.feedback_hidden_state.index_select(
                    0, active_rows
                ).unsqueeze(1)
                initialized = inference_params.feedback_initialized.index_select(
                    0, active_rows
                )[:, None, None]
                token_path = self.fbt_input_norm_1(hidden_states_inner)
                with fp8_context():
                    feedback_value = self.fbt_up(previous)
                    gate_logits = self.fbt_gate(token_path)
                fused = self.fbt_input_norm_1(
                    feedback_value
                    * torch.sigmoid(gate_logits.float()).to(feedback_value.dtype)
                )
                # The first selected boundary after a plain prefill remains a
                # normal input only when no prefill state exists.  This also
                # keeps direct one-token calls well-defined before prefill.
                hidden_states_inner = torch.where(
                    initialized, fused, hidden_states_inner
                )
            hidden_states_inner, prev_boundary_predictions = self.main_network.step(
                hidden_states_inner,
                inference_params.main_network_state,
                moe_sequence_ids=inner_sequence_ids,
            )
            if self.fbt_enabled:
                # Store the new top-layer latent only for streams that emitted
                # a boundary.  Non-boundary bytes update the dechunker but do
                # not advance the chunk-level FBT recurrence.
                inference_params.feedback_hidden_state.index_copy_(
                    0,
                    active_rows,
                    hidden_states_inner[:, -1].to(
                        dtype=inference_params.feedback_hidden_state.dtype
                    ),
                )
                inference_params.feedback_initialized.index_fill_(
                    0, active_rows, True
                )
        else:
            prev_boundary_predictions = []

        hidden_states = self.dechunk_layer.step(
            hidden_states_inner,
            effective_boundary_mask,
            bpred_output.boundary_prob,
            inference_params.dechunk_state,
        )

        hidden_states = self.residual_func(
            hidden_states.to(dtype=residual.dtype), residual, effective_selected_probs
        ).to(hidden_states.dtype)

        hidden_states = self.decoder.step(hidden_states, inference_params.decoder_state)
        hidden_states = hidden_states[..., :D]

        return hidden_states, [bpred_output, *prev_boundary_predictions]
