from dataclasses import dataclass

import torch
import torch.nn as nn
import torch.nn.functional as F

from .hnet import HNet
from .config_hnet import HNetConfig

from hnet.modules.dc import RoutingModuleOutput


class FixedByteFeatures(nn.Module):
    """Deterministic byte-to-width lifting with no trainable lookup table.

    The published checkpoint uses a learned 256-row embedding.  This trainer
    deliberately removes those trainable parameters while retaining the
    categorical byte geometry: a seeded, row-normalized Gaussian codebook is a
    persistent buffer and is never updated.  Independent rows keep the router
    boundaries informative at step zero; a smooth sinusoid code would make
    adjacent bytes nearly collinear and initially collapse chunking.
    """

    def __init__(self, d_model: int, device=None, dtype=None, seed: int = 1729):
        super().__init__()
        generator = torch.Generator(device="cpu")
        generator.manual_seed(seed)
        codes = torch.randn(256, d_model, generator=generator, dtype=torch.float32)
        codes = codes / codes.norm(dim=-1, keepdim=True).clamp_min(1e-6)
        # The codebook is immutable; materialize it directly in the model
        # activation dtype so every batch avoids an FP32 gather followed by a
        # full [B,L,D] cast. Keep normalization in FP32 for deterministic
        # initialization before the one-time conversion.
        target_dtype = torch.float32 if dtype is None else dtype
        self.register_buffer("codes", codes.to(device=device, dtype=target_dtype), persistent=True)
        self.d_model = d_model

    def forward(self, input_ids: torch.Tensor, dtype: torch.dtype) -> torch.Tensor:
        # F.embedding selects the same fixed rows as index_select but uses the
        # fused embedding lookup kernel on CUDA, avoiding the flatten/view
        # bookkeeping in the byte hot path.
        # CUDA's embedding kernel accepts int32 indices.  ByteStream batches
        # arrive as uint8; promoting to int64 doubles the index-transfer and
        # gather bandwidth for no benefit and creates an avoidable temporary
        # before every graph replay.  Keep the narrow int32 conversion.
        indices = input_ids.to(dtype=torch.int32)
        output = F.embedding(indices, self.codes)
        return output if output.dtype == dtype else output.to(dtype=dtype)

@dataclass
class CausalLMOutput:
    logits: torch.Tensor
    bpred_output: list[RoutingModuleOutput]
    feedback_logits: list[torch.Tensor] | None = None
    # LatentMoE router regularizers are returned separately so the trainer can
    # include them in the FP32 objective without touching CE numerics.
    moe_aux_loss: torch.Tensor | None = None
    moe_z_loss: torch.Tensor | None = None


class HNetForByteCausalLM(nn.Module):
    """Production raw-byte H-Net language model.

    The input path has no ``nn.Embedding`` or trainable byte/token lookup.
    Bytes are lifted with :class:`FixedByteFeatures`, routed through the H-Net
    hierarchy, and decoded by an FP32 output head for stable byte-level CE.
    """

    def __init__(self, config: HNetConfig, device=None, dtype=None):
        super().__init__()
        self.config = config
        self.input_dtype = dtype or torch.get_default_dtype()
        self.byte_features = FixedByteFeatures(config.d_model[0], device=device, dtype=dtype)
        self.backbone = HNet(
            config=config,
            stage_idx=0,
            device=device,
            dtype=dtype,
        )
        # The MoE regularizers are emitted by a fixed set of LatentMoE
        # modules. Cache the tuple once at construction instead of walking
        # the complete recursive backbone on every training forward (which
        # otherwise adds Python traversal and attribute checks to each graph
        # warmup/replay dispatch).
        self._moe_modules = tuple(
            module
            for module in self.backbone.modules()
            if hasattr(module, "aux_loss") and hasattr(module, "z_loss")
        )
        # Keep the CE output path in FP32, matching the TE model's stable head.
        self.lm_head = nn.Linear(
            config.d_model[0], config.vocab_size, bias=False, device=device, dtype=torch.float32
        )
        nn.init.normal_(self.lm_head.weight, mean=0.0, std=0.02)
        self.backbone._init_weights(0.02)

    def forward(self, input_ids, mask=None, inference_params=None, fbt_passes=1, **mixer_kwargs):
        if inference_params is not None:
            raise RuntimeError("the canonical fixed-shape trainer does not support inference caches")
        if input_ids.dtype not in (torch.uint8, torch.int8, torch.int16, torch.int32, torch.int64):
            raise TypeError("H-Net byte inputs must be integer byte IDs")
        hidden_dtype = self.input_dtype
        hidden_states = self.byte_features(input_ids, hidden_dtype)
        B, L, D = hidden_states.shape
        if mask is None:
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
                inference_params=None,
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
        # ``HNet.forward`` appends every feedback pass, including the final
        # decoded pass.  The old path projected that final activation twice:
        # once for ``logits`` and again for ``feedback_logits[-1]``.  Reuse
        # the final feedback projection as the primary logits tensor.  This
        # removes one full output-head GEMM from every FBT training step and
        # keeps the returned tensors and loss ordering unchanged.
        if feedback_hidden:
            # FBT produces a fixed number of same-shaped decoded states.  A
            # Python loop here launches one output-head GEMM per pass, which
            # leaves the small per-pass matrices under-occupied and adds
            # graph nodes.  Stack the states along a temporary leading
            # dimension and project them with one larger GEMM.  The reshape
            # keeps each pass contiguous and the split is a view, so logits
            # retain the exact per-pass ordering expected by the loss.
            feedback_states = torch.stack(feedback_hidden, dim=0).reshape(
                -1, D
            ).float()
            feedback_projected = self.lm_head(feedback_states).view(
                len(feedback_hidden), B, L, self.config.vocab_size
            )
            feedback_logits = list(feedback_projected.unbind(0))
            logits = feedback_logits[-1]
        else:
            logits = self.lm_head(hidden_states.float())
            feedback_logits = []
        moe_aux_loss = logits.new_zeros((), dtype=torch.float32)
        moe_z_loss = logits.new_zeros((), dtype=torch.float32)
        # Keep each LatentMoE's FP32 router losses on-device.  The cached
        # module tuple is fixed after construction, so this remains safe for
        # CUDA graph capture while avoiding a recursive Python walk per step.
        # Regularizers come from the final FBT pass; earlier passes contribute
        # auxiliary CE only and do not silently double-count router penalties.
        for module in self._moe_modules:
            moe_aux_loss = moe_aux_loss + module.aux_loss.float()
            moe_z_loss = moe_z_loss + module.z_loss.float()
        return CausalLMOutput(
            logits=logits,
            bpred_output=bpred_output,
            feedback_logits=feedback_logits,
            moe_aux_loss=moe_aux_loss,
            moe_z_loss=moe_z_loss,
        )

    def compile_blocks(self, *, mode="max-autotune-no-cudagraphs"):
        """Compile fixed LoL/FA blocks before DDP and CUDA graph capture."""
        return self.backbone.compile_blocks(mode=mode)
