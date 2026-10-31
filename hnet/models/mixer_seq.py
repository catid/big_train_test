from collections import namedtuple
from dataclasses import dataclass
from contextlib import nullcontext

import torch
import torch.nn as nn

from flash_attn.utils.generation import GenerationMixin

from .hnet import HNet, HNetState
from .config_hnet import HNetConfig

from hnet.modules.dc import RoutingModuleOutput
from hnet.modules.utils import apply_optimization_params


class FixedByteFeatures(nn.Module):
    """Deterministic byte-to-width lifting with no trainable lookup table.

    The published checkpoint uses a learned 256-row embedding.  This trainer
    deliberately removes those trainable parameters while retaining the
    categorical byte geometry: a seeded, row-normalized Gaussian codebook is a
    persistent buffer and is never updated.  Independent rows keep the router
    boundaries informative at step zero; a smooth sinusoid code would make
    adjacent bytes nearly collinear and initially collapse chunking.
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

        self.backbone = HNet(
            config=config,
            # We pass in the stage_idx as an HNet needs to know what
            # depth of the hierarchy it is in.
            stage_idx=0,
            **factory_kwargs,
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
        self.byte_features = FixedByteFeatures(config.d_model[0], device=device)
        self.backbone = HNet(config=config, stage_idx=0, device=device, dtype=dtype)
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
        logits = self.lm_head(hidden_states.float())
        feedback_logits = [self.lm_head(h.view(B, L, D).float()) for h in feedback_hidden]
        moe_aux_loss = logits.new_zeros((), dtype=torch.float32)
        moe_z_loss = logits.new_zeros((), dtype=torch.float32)
        # The production graph has a fixed module set; traversing it here is
        # capture-safe and keeps each LatentMoE's FP32 router losses on-device.
        # LatentMoE exposes the regularizers from the final FBT pass, matching
        # the primary final-pass CE objective; earlier passes are auxiliary CE
        # supervision only and do not silently double-count router penalties.
        for module in self.backbone.modules():
            if hasattr(module, "aux_loss") and hasattr(module, "z_loss"):
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
