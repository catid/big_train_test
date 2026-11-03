from collections import namedtuple
from dataclasses import dataclass
from contextlib import nullcontext

import torch
import torch.nn as nn
import torch.nn.functional as F

from flash_attn.utils.generation import GenerationMixin

from .hnet import HNet, HNetState
from .config_hnet import HNetConfig

from hnet.modules.dc import RoutingModuleOutput
from hnet.modules.utils import apply_optimization_params
from hnet.modules.rotary import LeRoPERegistry


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
    inference_params: HNetState
    feedback_logits: list[torch.Tensor] | None = None
    # LatentMoE router regularizers are returned separately so the trainer can
    # include them in the FP32 objective without touching CE numerics.
    moe_aux_loss: torch.Tensor | None = None
    moe_z_loss: torch.Tensor | None = None


class HNetForCausalLM(nn.Module, GenerationMixin):
    def __init__(
        self,
        config: HNetConfig,
        device=None,
        dtype=None,
    ) -> None:
        self.config = config

        vocab_size = self.config.vocab_size
        d_embed = self.config.d_model[0]
        factory_kwargs = {"device": device, "dtype": dtype}

        super().__init__()

        # We consider the HNet as a map (B, L, D[0]) -> (B, L, D[0])
        # Thus, the embedding is defined outside of the HNet.
        self.embeddings = nn.Embedding(vocab_size, d_embed, **factory_kwargs)

        self.lerope_registry = LeRoPERegistry()
        self.backbone = HNet(
            config=config,
            # We pass in the stage_idx as an HNet needs to know what
            # depth of the hierarchy it is in.
            stage_idx=0,
            **factory_kwargs,
            lerope_registry=self.lerope_registry,
        )
        self.lm_head = nn.Linear(d_embed, vocab_size, bias=False, **factory_kwargs)
        self.tie_weights()

    def tie_weights(self):
        if self.config.tie_embeddings:
            self.lm_head.weight = self.embeddings.weight
    
    def init_weights(self, initializer_range: float = 0.02) -> None:
        """
        Initializes the weights of the model.
        """
        nn.init.normal_(self.lm_head.weight, mean=0.0, std=initializer_range)
        # embeddings are initialized differently from linears
        nn.init.normal_(self.embeddings.weight, mean=0.0, std=1.0)
        self.backbone._init_weights(initializer_range)

    def apply_lr_multiplier(self, lr_multiplier: list[float]) -> None:
        """
        Applies the learning rate multipliers to the parameters of the model.
        NOTE: Must be ran before creating parameter groups, see hnet.utils.train.group_params for an example on how to run parameter groups.

        Inputs:
            lr_multiplier: A list of learning rate multipliers, one for each stage of the hierarchy, with the outer stages first (e.g. [3.0, 1.7, 0.9]).
        """
        for param in self.embeddings.parameters():
            apply_optimization_params(param, lr_multiplier=lr_multiplier[0])
        for param in self.lm_head.parameters():
            apply_optimization_params(param, lr_multiplier=lr_multiplier[0])
        self.backbone._apply_lr_multiplier(lr_multiplier)

    def allocate_inference_cache(self, batch_size, max_seqlen, dtype=None, **kwargs):
        return self.backbone.allocate_inference_cache(
            batch_size, max_seqlen, dtype=dtype, **kwargs
        )

    def forward(
        self,
        input_ids,
        mask=None,
        position_ids=None,
        inference_params=None,
        num_last_tokens=0,
        **mixer_kwargs,
    ):
        """
        num_last_tokens: if > 0, only return the logits for the last n tokens
        """
        hidden_states = self.embeddings(input_ids)

        B, L, D = hidden_states.shape

        assert (
            position_ids is None
        ), "Position ids are not supported for HNet due to the subsampling hierarchical structure"

        if mask is None:
            # Absent a mask, we assume we are running in packed mode
            assert (
                inference_params is None
            ), "Inference params are not supported in packed mode"
            hidden_states = hidden_states.flatten(0, 1)
            # FlashAttention's varlen kernels require int32 cumulative lengths.
            cu_seqlens = torch.arange(
                B + 1, device=hidden_states.device, dtype=torch.int32
            ) * L
            # FlashAttention's varlen API takes a Python int.  Passing a CUDA
            # scalar here works in some versions and fails in others.
            max_seqlen = L
        else:
            cu_seqlens = None
            max_seqlen = None

        hidden_states, bpred_output = self.backbone(
            hidden_states,
            cu_seqlens=cu_seqlens,
            max_seqlen=max_seqlen,
            mask=mask,
            inference_params=inference_params,
            **mixer_kwargs,
        )

        hidden_states = hidden_states.view(B, L, D)

        if num_last_tokens > 0:
            hidden_states = hidden_states[:, -num_last_tokens:]
        lm_logits = self.lm_head(hidden_states)

        CausalLMOutput = namedtuple(
            "CausalLMOutput", ["logits", "bpred_output", "inference_params"]
        )
        return CausalLMOutput(
            logits=lm_logits,
            bpred_output=bpred_output,
            inference_params=inference_params,
        )

    def step(self, input_ids, inference_params):
        B = input_ids.shape[0]
        assert B == 1, (
            "HNetForCausalLM step currently only supports batch size 1 -- "
            "need to handle different-size lengths for each sample"
        )
        hidden_states = self.embeddings(input_ids)
        hidden_states, bpred_output = self.backbone.step(hidden_states, inference_params)
        logits = self.lm_head(hidden_states)
        return CausalLMOutput(
            logits=logits, bpred_output=bpred_output, inference_params=inference_params
        )


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
        self.byte_features = FixedByteFeatures(config.d_model[0], device=device, dtype=dtype)
        self.lerope_registry = LeRoPERegistry()
        self.backbone = HNet(
            config=config,
            stage_idx=0,
            device=device,
            dtype=dtype,
            lerope_registry=self.lerope_registry,
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
        if input_ids.dtype not in (torch.uint8, torch.int8, torch.int16, torch.int32, torch.int64):
            raise TypeError("H-Net byte inputs must be integer byte IDs")
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
            inference_params=inference_params,
            feedback_logits=feedback_logits,
            moe_aux_loss=moe_aux_loss,
            moe_z_loss=moe_z_loss,
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
