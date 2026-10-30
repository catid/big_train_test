from dataclasses import dataclass
from typing import Union, Optional

import torch
import torch.nn as nn

from hnet.modules.isotropic import Isotropic, IsotropicInferenceParams
from hnet.modules.dc import (
    RoutingModule,
    ChunkLayer,
    DeChunkLayer,
    RoutingModuleState,
    DeChunkState,
)
from hnet.modules.utils import apply_optimization_params
from hnet.te import make_linear

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
    encoder_state: Optional[IsotropicInferenceParams] = None
    routing_module_state: Optional[RoutingModuleState] = None
    main_network_state: Optional[Union["HNetState", IsotropicInferenceParams]] = None
    dechunk_state: Optional[DeChunkState] = None
    decoder_state: Optional[IsotropicInferenceParams] = None


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
                **factory_kwargs,
            )
            self.add_module(_name, _sub_model)

        if not self.is_innermost:
            self.routing_module = RoutingModule(self.d_model, **factory_kwargs)
            self.chunk_layer = ChunkLayer(
                static_max_seqlen=(config.max_seq_len if getattr(config, "static_chunking", False) else None)
            )
            self.dechunk_layer = DeChunkLayer(self.d_model)

            # do the residual in fp32
            self.residual_proj = make_linear(
                self.d_model, self.d_model, device=device, dtype=torch.float32
            )
            nn.init.zeros_(self.residual_proj.weight)
            nn.init.zeros_(self.residual_proj.bias)
            self.residual_proj.weight._no_reinit = True
            self.residual_proj.bias._no_reinit = True

            self.residual_func = lambda out, residual, p: out * ste_func(p) + residual
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
        if self.is_innermost:
            n_residuals += self.main_network.height
            for name, m in self.main_network.named_modules():
                if (isinstance(m, nn.Linear) or m.__class__.__name__ in ("Linear", "AlignedTELinear")) and not getattr(m.weight, "_no_reinit", False):
                    if "out_proj" in name or "fc2" in name:
                        nn.init.normal_(m.weight, mean=0.0, std=initializer_range / (n_residuals ** 0.5))
                    else:
                        nn.init.normal_(m.weight, mean=0.0, std=initializer_range)

        else:
            n_residuals += self.encoder.height + self.decoder.height
            for name, m in self.encoder.named_modules():
                if (isinstance(m, nn.Linear) or m.__class__.__name__ in ("Linear", "AlignedTELinear")) and not getattr(m.weight, "_no_reinit", False):
                    if "out_proj" in name or "fc2" in name:
                        nn.init.normal_(m.weight, mean=0.0, std=initializer_range / (n_residuals ** 0.5))
                    else:
                        nn.init.normal_(m.weight, mean=0.0, std=initializer_range)
            for name, m in self.decoder.named_modules():
                if (isinstance(m, nn.Linear) or m.__class__.__name__ in ("Linear", "AlignedTELinear")) and not getattr(m.weight, "_no_reinit", False):
                    if "out_proj" in name or "fc2" in name:
                        nn.init.normal_(m.weight, mean=0.0, std=initializer_range / (n_residuals ** 0.5))
                    else:
                        nn.init.normal_(m.weight, mean=0.0, std=initializer_range)
                    
            self.main_network._init_weights(initializer_range, n_residuals)
    

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
        previous_inner = None
        final_prev_boundary_predictions = []
        for pass_idx in range(n_passes):
            pass_input = base_chunk_states
            if pass_idx > 0:
                # Shift within each packed document; no cross-document latent
                # state is allowed to enter a new byte stream.
                if next_cu_seqlens is None:
                    shifted = torch.cat(
                        (torch.zeros_like(previous_inner[:, :1]), previous_inner[:, :-1]),
                        dim=1,
                    )
                else:
                    # Packed feedback is shifted within each document.  Use
                    # differentiable tensor operations instead of indexed
                    # writes into a ``zeros_like`` destination.  Indexed
                    # writes there detach the previous pass and silently
                    # freeze both FBT projection matrices at zero.
                    shifted = torch.cat(
                        (torch.zeros_like(previous_inner[:1]), previous_inner[:-1]),
                        dim=0,
                    )
                    starts = next_cu_seqlens[:-1]
                    is_start = torch.zeros(
                        (shifted.shape[0],), device=shifted.device, dtype=torch.bool
                    )
                    is_start.scatter_(0, starts, True)
                    shifted = torch.where(is_start[:, None], torch.zeros_like(shifted), shifted)
                if self.training and getattr(self.config, "fbt_jitter", 0.0) > 0:
                    shifted = shifted + torch.empty_like(shifted).uniform_(
                        -float(self.config.fbt_jitter), float(self.config.fbt_jitter)
                    )
                # The projection is kept outside any FP8 region and the gate
                # is evaluated in FP32 for stable sigmoid gradients.
                gate = torch.sigmoid(self.fbt_gate(pass_input).float()).to(pass_input.dtype)
                shifted = shifted.to(dtype=self.fbt_up.weight.dtype)
                pass_input = pass_input + self.fbt_up(shifted).to(pass_input.dtype) * gate
                if getattr(self.config, "fbt_prefix_mixin", False):
                    # Keep a plain prefix so training matches prompt-prefill
                    # followed by feedback-conditioned generation.
                    plain = torch.zeros(pass_input.shape[:-1], dtype=torch.bool, device=pass_input.device)
                    if next_cu_seqlens is None:
                        # ChunkLayer left-aligns real chunks and pads to a
                        # static bound. Choose the prefix from the real count,
                        # rather than the padded width; otherwise short
                        # sequences make every real chunk plain and FBT never
                        # trains its feedback projections.
                        real_counts = next_mask.sum(dim=-1)
                        token_idx = torch.arange(
                            pass_input.shape[1], device=pass_input.device
                        )[None, :]
                        plain = token_idx < (real_counts[:, None] // 2)
                        plain = plain & next_mask
                    else:
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
                    pass_input = torch.where(plain.unsqueeze(-1), base_chunk_states, pass_input)
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

    def step(self, hidden_states, inference_params):
        D = hidden_states.shape[-1]

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
                hidden_states, inference_params.main_network_state
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
        hidden_states_inner = self.chunk_layer.step(
            hidden_states, bpred_output.boundary_mask
        )

        if hidden_states_inner.shape[0] > 0:
            hidden_states_inner, prev_boundary_predictions = self.main_network.step(
                hidden_states_inner, inference_params.main_network_state
            )
        else:
            prev_boundary_predictions = []

        hidden_states = self.dechunk_layer.step(
            hidden_states_inner,
            bpred_output.boundary_mask,
            bpred_output.boundary_prob,
            inference_params.dechunk_state,
        )

        hidden_states = self.residual_func(
            hidden_states.to(dtype=residual.dtype), residual, bpred_output.selected_probs
        ).to(hidden_states.dtype)

        hidden_states = self.decoder.step(hidden_states, inference_params.decoder_state)
        hidden_states = hidden_states[..., :D]

        return hidden_states, [bpred_output, *prev_boundary_predictions]
