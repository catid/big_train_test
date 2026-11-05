from dataclasses import dataclass
from typing import Optional

import torch
import torch.nn as nn

from hnet.modules.dc import (
    RoutingModule,
    ChunkLayer,
    DeChunkLayer,
    RoutingModuleState,
    DeChunkState,
)
from hnet.modules.utils import apply_optimization_params
from .config_hnet import HNetConfig


class STE(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x):
        return torch.ones_like(x)

    @staticmethod
    def backward(ctx, grad_output):
        grad_x = grad_output
        return grad_x

def ste_func(x):
    return STE.apply(x)


@dataclass
class HNetState:
    encoder_state: Optional[object] = None
    routing_module_state: Optional[RoutingModuleState] = None
    main_network_state: Optional[object] = None
    dechunk_state: Optional[DeChunkState] = None
    decoder_state: Optional[object] = None


class HNet(nn.Module):
    def __init__(
        self,
        config: HNetConfig,
        stage_idx: int,
        device=None,
        dtype=None,
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
                # Every production stack is built from the explicit GDN2,
                # LoL, and FA library.  There is no alternate mixer factory.
                from hnet.modules.hybrid import HybridStack as SubModel
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
                **factory_kwargs,
            )
            self.add_module(_name, _sub_model)

        if not self.is_innermost:
            # Named-library stages intentionally use a 3072-channel byte
            # stream and a 1280-channel latent trunk.  The original H-Net
            # implementation only padded upward, which would feed a wider
            # tensor into a narrower inner stack and fail at its first norm.
            # Make the hierarchy boundary explicit and reversible.
            inner_width = int(self.main_network.d_model)
            if inner_width != self.d_model:
                self.main_network_input_proj = nn.Linear(
                    self.d_model, inner_width, bias=False, **factory_kwargs
                )
                self.main_network_output_proj = nn.Linear(
                    inner_width, self.d_model, bias=False, **factory_kwargs
                )
            else:
                self.main_network_input_proj = nn.Identity()
                self.main_network_output_proj = nn.Identity()

            self.routing_module = RoutingModule(self.d_model, **factory_kwargs)
            # ``max_seq_len`` is the byte-stream capacity.  The inner FA must
            # use the separately configured latent capacity; conflating these
            # two values silently turns an 8K byte batch into an 8K latent
            # quadratic pass.
            static_latent_seqlen = None
            if getattr(config, "static_chunking", False):
                static_latent_seqlen = getattr(config, "latent_max_seq_len", None)
                if static_latent_seqlen is None:
                    static_latent_seqlen = config.max_seq_len
                static_latent_seqlen = int(static_latent_seqlen)
                if static_latent_seqlen <= 0 or static_latent_seqlen > int(config.max_seq_len):
                    raise ValueError(
                        "latent_max_seq_len must be positive and no larger than max_seq_len "
                        f"({static_latent_seqlen}, {config.max_seq_len})"
                    )
            self.chunk_layer = ChunkLayer(static_max_seqlen=static_latent_seqlen)
            self.dechunk_layer = DeChunkLayer(
                self.d_model,
                static_max_seqlen=static_latent_seqlen,
                device=device,
            )

            # do the residual in fp32
            self.residual_proj = nn.Linear(
                self.d_model, self.d_model, device=device, dtype=torch.float32
            )
            nn.init.zeros_(self.residual_proj.weight)
            self.residual_proj.weight._no_reinit = True

            self.residual_func = lambda out, residual, p: out * ste_func(p) + residual
            # Full-bandwidth feedback is attached only at the outer boundary
            # before the deepest main network. It never changes router masks,
            # chunk counts, or the encoder/decoder residual path.
            self.fbt_enabled = bool(getattr(config, "fbt", False) and stage_idx == 0)
            if self.fbt_enabled:
                # The recursive H-Net trims its output back to the caller's
                # width after the padded inner stage, so feedback remains at
                # the outer stream width.
                self.fbt_up = nn.Linear(self.d_model, self.d_model, bias=False, **factory_kwargs)
                self.fbt_gate = nn.Linear(self.d_model, self.d_model, bias=False, **factory_kwargs)
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
        def init_linear_modules(root: nn.Module, residuals: int) -> None:
            # Transformer Engine's Linear/GroupedLinear classes are not
            # subclasses of torch.nn.Linear, so the old isinstance-only pass
            # silently left every FP8 MoE projection at TE's default scale.
            # Initialize those modules with the same residual-depth rule while
            # leaving Mamba state/conv parameters under their kernel defaults.
            for name, module in root.named_modules():
                module_name = module.__class__.__module__
                is_te_linear = (
                    module_name.startswith("transformer_engine")
                    and "Linear" in module.__class__.__name__
                )
                if not (isinstance(module, nn.Linear) or is_te_linear):
                    continue
                if "out_proj" in name or "fc2" in name:
                    std = initializer_range / (residuals ** 0.5)
                else:
                    std = initializer_range
                if isinstance(module, nn.Linear):
                    weights = (module.weight,)
                else:
                    # TE GroupedLinear stores discrete expert matrices as
                    # weight0, weight1, ... rather than exposing one
                    # ``weight`` attribute.  Initialize every registered
                    # expert parameter through the regular PyTorch API so TE
                    # metadata stays attached to the same Parameter objects.
                    weights = tuple(
                        parameter
                        for parameter_name, parameter in module.named_parameters(recurse=False)
                        if "weight" in parameter_name
                    )
                with torch.no_grad():
                    for weight in weights:
                        if not getattr(weight, "_no_reinit", False):
                            nn.init.normal_(weight, mean=0.0, std=std)

        n_residuals = parent_residuals
        if self.is_innermost:
            n_residuals += self.main_network.height
            init_linear_modules(self.main_network, n_residuals)

        else:
            n_residuals += self.encoder.height + self.decoder.height
            init_linear_modules(self.encoder, n_residuals)
            init_linear_modules(self.decoder, n_residuals)
                    
            self.main_network._init_weights(initializer_range, n_residuals)
            init_linear_modules(self.main_network_input_proj, n_residuals)
            init_linear_modules(self.main_network_output_proj, n_residuals)
    

    def _apply_lr_multiplier(self, lr_multiplier: list[float]) -> None:
        """
        Applies the learning rate multipliers to the parameters of the model.
        """
        # a little stupid: we apply lr_multiplier to all parameters, and then for the main stage (which may have another hierarchy), we just apply it again there.
        for param in self.parameters():
            apply_optimization_params(param, lr_multiplier=lr_multiplier[self.stage_idx])
        
        if not self.is_innermost:
            self.main_network._apply_lr_multiplier(lr_multiplier)

    def compile_blocks(self, *, mode="max-autotune-no-cudagraphs"):
        """Compile unique static LoL/FA blocks before DDP/graph capture."""
        count = 0
        for module in self.modules():
            if module is self:
                continue
            compile_fn = getattr(module, "compile_blocks", None)
            if compile_fn is not None and module.__class__.__name__ == "HybridStack":
                count += int(compile_fn(mode=mode))
        return count

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

        if inference_params is None:
            inference_params = HNetState(main_network_state=None)
        else:
            assert (
                mask is not None
            ), "Mask must be provided if inference_params is provided"

        D = hidden_states.shape[-1]
        EARLY_DIMS = hidden_states.shape[:-1]

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
            **mixer_kwargs,
        )

        hidden_states_for_residual = hidden_states.to(
            dtype=self.residual_proj.weight.dtype
        )
        residual = self.residual_proj(hidden_states_for_residual)

        bpred_output = self.routing_module(
            hidden_states,
            cu_seqlens=cu_seqlens,
            mask=mask,
            inference_params=inference_params.routing_module_state,
        )
        hidden_states, next_cu_seqlens, next_max_seqlen, next_mask = self.chunk_layer(
            hidden_states, bpred_output.boundary_mask, cu_seqlens, mask=mask
        )
        # ChunkLayer may enforce the canonical static latent capacity.  The
        # dechunker must consume the same effective boundary set, otherwise it
        # would index a shorter latent stream with the uncapped route mask.
        bpred_output.boundary_mask = self.chunk_layer.effective_boundary_mask

        n_passes = max(1, int(fbt_passes)) if self.fbt_enabled else 1
        # A normal one-pass prefill must populate the allocated recurrent/KV
        # caches for autoregressive ``step``.  FBT uses fresh inner passes and
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
        previous_inner = None
        final_prev_boundary_predictions = []
        for pass_idx in range(n_passes):
            pass_input = base_chunk_states
            if pass_idx > 0:
                # Shift within each packed document; no cross-document latent
                # state is allowed to enter a new byte stream.
                if next_cu_seqlens is None:
                    shifted = torch.zeros_like(previous_inner)
                    shifted[:, 1:] = previous_inner[:, :-1]
                else:
                    # Packed feedback is one flat token stream.  A single
                    # roll followed by zeroing document starts is equivalent
                    # to the old Python loop, but avoids synchronizing
                    # ``cu_seqlens`` to the host and launching one slice copy
                    # per document.
                    shifted = torch.roll(previous_inner, shifts=1, dims=0)
                    starts = torch.zeros(
                        previous_inner.shape[0], dtype=torch.bool,
                        device=previous_inner.device,
                    )
                    starts[next_cu_seqlens[:-1]] = True
                    shifted = shifted.masked_fill(starts[:, None], 0)
                if self.training and getattr(self.config, "fbt_jitter", 0.0) > 0:
                    shifted = shifted + torch.empty_like(shifted).uniform_(
                        -float(self.config.fbt_jitter), float(self.config.fbt_jitter)
                    )
                # The projection is kept outside any FP8 region and the gate
                # is evaluated in FP32 for stable sigmoid gradients.
                gate = torch.sigmoid(self.fbt_gate(pass_input).float()).to(pass_input.dtype)
                pass_input = pass_input + self.fbt_up(shifted).to(pass_input.dtype) * gate
                if getattr(self.config, "fbt_prefix_mixin", False):
                    # Keep a plain prefix so training matches prompt-prefill
                    # followed by feedback-conditioned generation.
                    plain = torch.zeros(pass_input.shape[:-1], dtype=torch.bool, device=pass_input.device)
                    if next_cu_seqlens is None:
                        # ChunkLayer packs valid latent tokens at the front and
                        # pads the remainder when static chunking is enabled.
                        # Derive the prefix from the per-row valid count so a
                        # short latent sequence does not make the whole valid
                        # region plain (which would turn FBT into a no-op).
                        # All operations stay on device and are graph-safe.
                        if next_mask is not None:
                            valid = next_mask.sum(dim=-1)
                            prefix = (valid + 1) // 2
                            positions = torch.arange(
                                pass_input.shape[1], device=pass_input.device
                            )[None, :]
                            plain = (positions < prefix[:, None]) & next_mask
                        else:
                            # A fixed midpoint is used when no valid-count mask
                            # is available for an eager packed input.
                            plain[:, : pass_input.shape[1] // 2] = True
                    else:
                        # Draw one uniformly random cutoff per document on the
                        # device.  Searchsorted maps each flat token to its
                        # document, avoiding Python ``.item()`` calls and
                        # preserving document-local prefix boundaries.
                        doc_starts = next_cu_seqlens[:-1]
                        doc_ends = next_cu_seqlens[1:]
                        lengths = doc_ends - doc_starts
                        prefix = torch.floor(
                            torch.rand(lengths.shape, device=pass_input.device)
                            * (lengths.float() + 1.0)
                        ).to(dtype=lengths.dtype)
                        positions = torch.arange(
                            pass_input.shape[0], device=pass_input.device,
                            dtype=next_cu_seqlens.dtype,
                        )
                        doc_idx = torch.searchsorted(doc_ends, positions, right=True)
                        offsets = positions - doc_starts.index_select(0, doc_idx)
                        plain = offsets < prefix.index_select(0, doc_idx)
                    pass_input = torch.where(plain.unsqueeze(-1), base_chunk_states, pass_input)
            inner_input = self.main_network_input_proj(pass_input)
            inner_out, prev_boundary_predictions = self.main_network(
                inner_input,
                cu_seqlens=next_cu_seqlens,
                max_seqlen=next_max_seqlen,
                mask=next_mask,
                inference_params=cached_main_state,
                **mixer_kwargs,
            )
            inner_out = self.main_network_output_proj(inner_out)
            previous_inner = inner_out
            final_prev_boundary_predictions = prev_boundary_predictions

            decoded = self.dechunk_layer(
                inner_out,
                bpred_output.boundary_mask,
                bpred_output.boundary_prob,
                next_cu_seqlens,
                mask=mask,
                inference_params=cached_dechunk_state,
            )
            decoded = self.residual_func(
                decoded.to(dtype=residual.dtype), residual, bpred_output.selected_probs
            ).to(decoded.dtype)
            decoded = self.decoder(
                decoded,
                cu_seqlens=cu_seqlens,
                max_seqlen=max_seqlen,
                mask=mask,
                inference_params=cached_decoder_state,
                **mixer_kwargs,
            )
            if return_feedback:
                feedback_logits.append(decoded)
        hidden_states = decoded

        hidden_states = hidden_states[..., :D]
        result = [bpred_output, *final_prev_boundary_predictions]
        if return_feedback:
            return hidden_states, result, feedback_logits
        return hidden_states, result
