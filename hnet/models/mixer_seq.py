from dataclasses import dataclass

import torch
import torch.nn as nn

from flash_attn.utils.generation import GenerationMixin

from .hnet import HNet, HNetState
from .config_hnet import HNetConfig

from hnet.modules.dc import RoutingModuleOutput


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


class HNetForByteCausalLM(nn.Module, GenerationMixin):
    """Production raw-byte H-Net language model.

    The input path has no ``nn.Embedding`` or trainable byte/token lookup.
    Bytes are lifted with :class:`FixedByteFeatures`, routed through the H-Net
    hierarchy, and decoded by an FP32 output head for stable byte-level CE.
    """

    def __init__(self, config: HNetConfig, device=None, dtype=None):
        super().__init__()
        self.config = config
        self.input_dtype = dtype or torch.get_default_dtype()
        self.byte_features = FixedByteFeatures(config.d_model[0], device=device)
        self.backbone = HNet(config=config, stage_idx=0, device=device, dtype=dtype)
        # Keep the CE output path in FP32, matching the TE model's stable head.
        self.lm_head = nn.Linear(
            config.d_model[0], config.vocab_size, bias=False, device=device, dtype=torch.float32
        )
        nn.init.normal_(self.lm_head.weight, mean=0.0, std=0.02)
        self.backbone._init_weights(0.02)

    def forward(self, input_ids, mask=None, inference_params=None, fbt_passes=1, **mixer_kwargs):
        integer_dtypes = (torch.uint8, torch.int8, torch.int16, torch.int32, torch.int64)
        if input_ids.dtype not in integer_dtypes:
            raise TypeError("H-Net byte inputs must be integer byte IDs")
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
            if inference_params is not None:
                raise ValueError("inference_params requires an explicit mask")
            flat = hidden_states.flatten(0, 1)
            cu_seqlens = torch.arange(B + 1, device=hidden_states.device, dtype=torch.int32) * L
            result = self.backbone(
                flat,
                cu_seqlens=cu_seqlens,
                max_seqlen=L,
                inference_params=None,
                fbt_passes=fbt_passes,
                return_feedback=int(fbt_passes) > 1,
                **mixer_kwargs,
            )
        else:
            result = self.backbone(
                hidden_states,
                mask=mask,
                inference_params=inference_params,
                fbt_passes=fbt_passes,
                return_feedback=int(fbt_passes) > 1,
                **mixer_kwargs,
            )
        if len(result) == 3:
            hidden_states, bpred_output, feedback_hidden = result
        else:
            hidden_states, bpred_output = result
            feedback_hidden = []
        hidden_states = hidden_states.view(B, L, D)
        logits = self.lm_head(hidden_states.float())
        feedback_logits = [self.lm_head(h.view(B, L, D).float()) for h in feedback_hidden]
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
