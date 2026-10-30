from dataclasses import dataclass

import torch
import torch.nn as nn
from torch.utils._pytree import register_pytree_node

from flash_attn.utils.generation import GenerationMixin

from .hnet import HNet, HNetState
from .config_hnet import HNetConfig

from hnet.modules.dc import RoutingModuleOutput
from hnet.te import make_linear, graph_capture_enabled


class FixedByteFeatures(nn.Module):
    """Deterministic byte-to-width lifting with no trainable lookup table.

    A seeded, row-normalized Gaussian codebook is a persistent buffer and is
    never updated. Independent rows keep router boundaries informative at step
    zero while preserving a strictly fixed byte input path.
    """

    def __init__(self, d_model: int, device=None, seed: int = 1729):
        super().__init__()
        generator = torch.Generator(device="cpu")
        generator.manual_seed(seed)
        codes = torch.randn(256, d_model, generator=generator, dtype=torch.float32)
        codes = codes / codes.norm(dim=-1, keepdim=True).clamp_min(1e-6)
        self.register_buffer("codes", codes.to(device=device), persistent=True)
        self.d_model = d_model

    def forward(self, input_ids: torch.Tensor, dtype: torch.dtype) -> torch.Tensor:
        return self.codes.index_select(0, input_ids.long().reshape(-1)).view(
            *input_ids.shape, self.d_model
        ).to(dtype=dtype)

@dataclass
class CausalLMOutput:
    logits: torch.Tensor
    bpred_output: list[RoutingModuleOutput]
    inference_params: HNetState
    feedback_logits: list[torch.Tensor] | None = None
    moe_aux_loss: torch.Tensor | None = None


def _flatten_causal_output(value):
    return (
        value.logits,
        value.bpred_output,
        value.inference_params,
        value.feedback_logits,
        value.moe_aux_loss,
    ), None


def _unflatten_causal_output(values, _context):
    return CausalLMOutput(*values)


register_pytree_node(
    CausalLMOutput,
    _flatten_causal_output,
    _unflatten_causal_output,
)


class HNetForByteCausalLM(nn.Module, GenerationMixin):
    """Production raw-byte H-Net language model.

    The input path has no ``nn.Embedding`` or trainable byte/token lookup.
    Bytes are lifted with :class:`FixedByteFeatures`, routed through the H-Net
    hierarchy, and decoded by an FP32 output head for stable byte-level CE.
    """

    def __init__(self, config: HNetConfig, device=None, dtype=None):
        super().__init__()
        config.validate_canonical()
        self.config = config
        self.input_dtype = dtype or torch.get_default_dtype()
        self.byte_features = FixedByteFeatures(config.d_model[0], device=device)
        self.backbone = HNet(config=config, stage_idx=0, device=device, dtype=dtype)
        # Keep the CE output path in FP32, matching the TE model's stable head.
        self.lm_head = make_linear(
            config.d_model[0],
            config.vocab_size,
            bias=False,
            device=device,
            dtype=torch.float32,
            fp8=False,
        )
        nn.init.normal_(self.lm_head.weight, mean=0.0, std=0.02)
        self.backbone._init_weights(0.02)

    def forward(self, input_ids, mask=None, inference_params=None, fbt_passes=None, **mixer_kwargs):
        required_passes = int(getattr(self.config, "fbt_passes", 2))
        if not bool(getattr(self.config, "fbt", True)) or required_passes != 2:
            raise ValueError("the canonical model requires two-pass FBT in its configuration")
        if fbt_passes is None:
            fbt_passes = 1 if inference_params is not None else required_passes
        if inference_params is None and int(fbt_passes) != required_passes:
            raise ValueError(f"the canonical model requires fbt_passes={required_passes}")
        if inference_params is not None and int(fbt_passes) != 1:
            raise ValueError("inference-cache prefill uses exactly one pass")
        integer_dtypes = (torch.uint8, torch.int8, torch.int16, torch.int32, torch.int64)
        if input_ids.dtype not in integer_dtypes:
            raise TypeError("H-Net byte inputs must be integer byte IDs")
        if input_ids.ndim != 2:
            raise ValueError("canonical H-Net byte input must have shape (batch, sequence)")
        if input_ids.shape[1] > int(self.config.max_seq_len):
            raise ValueError(
                f"sequence length {input_ids.shape[1]} exceeds canonical max_seq_len "
                f"{self.config.max_seq_len}"
            )
        # The trainer supplies uint8 directly, so the hot path has no range
        # reduction or host synchronization.  Public callers may use signed
        # integer tensors; validate those explicitly instead of letting -1
        # silently wrap to byte 255 through index_select.
        if input_ids.dtype != torch.uint8 and input_ids.numel():
            low = int(input_ids.detach().amin().item())
            high = int(input_ids.detach().amax().item())
            if low < 0 or high > 255:
                raise ValueError(f"H-Net byte IDs must be in [0, 255], got [{low}, {high}]")
        # MoE routing statistics are accumulated across recurrent inner loops
        # and FBT passes.  Reset once at the outer model boundary so a stale
        # value from the previous minibatch cannot leak into this loss.
        for module in self.backbone.modules():
            reset_aux_loss = getattr(module, "reset_aux_loss", None)
            if reset_aux_loss is not None:
                reset_aux_loss()
        hidden_dtype = self.input_dtype
        hidden_states = self.byte_features(input_ids, hidden_dtype)
        B, L, D = hidden_states.shape
        if mask is None:
            raise ValueError("canonical H-Net training requires an explicit static mask")
        if mask.shape != input_ids.shape or mask.dtype != torch.bool:
            raise ValueError("canonical H-Net mask must be bool with shape (batch, sequence)")
        result = self.backbone(
            hidden_states,
            mask=mask,
            inference_params=inference_params,
            fbt_passes=fbt_passes,
            return_feedback=int(fbt_passes) > 1,
            graph_dense_mask=graph_capture_enabled(),
            **mixer_kwargs,
        )
        if len(result) == 3:
            hidden_states, bpred_output, feedback_hidden = result
        else:
            hidden_states, bpred_output = result
            feedback_hidden = []
        hidden_states = hidden_states.view(B, L, D)
        logits = self.lm_head(hidden_states.float())
        # The final feedback pass is identical to ``hidden_states`` and its
        # logits are already available as ``logits``.  Keep only earlier pass
        # logits here so the LM head is evaluated once per distinct state.
        feedback_logits = [
            self.lm_head(h.view(B, L, D).float()) for h in feedback_hidden[:-1]
        ]
        moe_losses = [m.aux_loss for m in self.backbone.modules()
                      if getattr(m, "aux_loss", None) is not None]
        moe_aux_loss = torch.stack(moe_losses).mean() if moe_losses else None
        return CausalLMOutput(
            logits=logits,
            bpred_output=bpred_output,
            inference_params=inference_params,
            feedback_logits=feedback_logits,
            moe_aux_loss=moe_aux_loss,
        )

    def allocate_inference_cache(self, batch_size, max_seqlen, dtype=None):
        return self.backbone.allocate_inference_cache(
            batch_size, max_seqlen, dtype=self.input_dtype if dtype is None else dtype
        )

    def step(self, input_ids, inference_params):
        if input_ids.shape[0] != 1:
            raise ValueError("H-Net byte generation currently requires batch size 1")
        hidden_states = self.byte_features(input_ids, self.input_dtype)
        hidden_states, bpred = self.backbone.step(hidden_states, inference_params)
        logits = self.lm_head(hidden_states.float())
        return CausalLMOutput(logits=logits, bpred_output=bpred, inference_params=inference_params, feedback_logits=[])
