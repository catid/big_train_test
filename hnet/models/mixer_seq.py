from dataclasses import dataclass

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils._pytree import register_pytree_node

from flash_attn.utils.generation import GenerationMixin

from .hnet import HNet, HNetState
from .config_hnet import HNetConfig

from hnet.modules.dc import RoutingModuleOutput
from hnet.te import make_linear, graph_capture_enabled
from hnet.modules.rotary import LeRoPEFrequencies


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
        # F.embedding accepts int32 indices on CUDA.  The trainer's byte
        # stream is uint8, so promoting it to int64 needlessly doubles index
        # bandwidth and creates a larger temporary before the lookup.  Keep
        # the fixed, non-trainable byte codebook and use the narrow CUDA
        # integer index type directly.
        output = F.embedding(input_ids.to(dtype=torch.int32), self.codes)
        return output if output.dtype == dtype else output.to(dtype=dtype)

@dataclass
class CausalLMOutput:
    logits: torch.Tensor
    bpred_output: list[RoutingModuleOutput]
    inference_params: HNetState
    feedback_logits: list[torch.Tensor] | None = None
    moe_aux_loss: torch.Tensor | None = None
    moe_balance_loss: torch.Tensor | None = None
    # Mean accepted top-1 route fraction per expert for the current forward.
    # This is a graph-owned tensor so the trainer can report expert balance
    # without running a second routing pass.
    moe_expert_load: torch.Tensor | None = None


def _flatten_causal_output(value):
    return (
        value.logits,
        value.bpred_output,
        value.inference_params,
        value.feedback_logits,
        value.moe_aux_loss,
        value.moe_balance_loss,
        value.moe_expert_load,
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
        rotary_dim = int(config.attn_cfg.rotary_emb_dim[0])
        if not bool(getattr(config, "lerope_enabled", False)):
            raise ValueError("the canonical model requires LeRoPE for SWA")
        self.lerope = LeRoPEFrequencies(
            rotary_dim,
            base=10000.0,
            device=device,
        )
        self.backbone = HNet(
            config=config,
            stage_idx=0,
            device=device,
            dtype=dtype,
            rotary_emb_shared=self.lerope,
        )
        # The production stack has a fixed module topology.  Cache the small
        # set of LatentMoE modules once instead of walking the entire nested
        # H-Net hierarchy on every eager forward to reset and collect routing
        # statistics.  CUDA-graph replay does not execute this Python path,
        # while calibration/eager validation avoid repeated ``modules()``
        # generators and attribute probes.
        self._moe_modules = tuple(
            module
            for module in self.backbone.modules()
            if callable(getattr(module, "reset_aux_loss", None))
        )
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
        # The shared LeRoPE table is attached to the current autograd graph.
        # Clear only the Python cache between model forwards; CUDA graph replay
        # reruns the captured table construction directly from log_scale.
        self.lerope.begin_forward()
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
        for module in self._moe_modules:
            module.reset_aux_loss()
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
        if feedback_hidden:
            # FBT returns same-shaped decoded states for every pass, with the
            # final state equal to ``hidden_states``.  Stack them and run one
            # larger output-head GEMM instead of one GEMM for the final pass
            # plus one per auxiliary pass.  Unbinding the result preserves
            # the existing pass ordering and loss semantics.
            feedback_states = torch.stack(feedback_hidden, dim=0).reshape(
                -1, D
            ).float()
            feedback_projected = self.lm_head(feedback_states).view(
                len(feedback_hidden), B, L, self.config.vocab_size
            )
            feedback_all = list(feedback_projected.unbind(0))
            logits = feedback_all[-1]
            feedback_logits = feedback_all[:-1]
        else:
            logits = self.lm_head(hidden_states.float())
            feedback_logits = []
        # The router's z-loss is part of the same FP32 routing regularizer as
        # load balancing.  Keep it attached to the returned objective; merely
        # storing it on LatentMoE leaves the computed term dead during training.
        moe_losses = []
        moe_balance_losses = []
        expert_loads = []
        for module in self._moe_modules:
            aux_loss = getattr(module, "aux_loss", None)
            if aux_loss is None:
                continue
            z_loss = getattr(module, "z_loss", None)
            if z_loss is not None:
                aux_loss = aux_loss + z_loss.to(dtype=aux_loss.dtype)
            moe_losses.append(aux_loss)
            balance = getattr(module, "aux_loss", None)
            if balance is not None:
                moe_balance_losses.append(balance.float())
            load = getattr(module, "expert_load", None)
            if load is not None:
                expert_loads.append(load.float())
        moe_aux_loss = torch.stack(moe_losses).mean() if moe_losses else None
        moe_balance_loss = (
            torch.stack(moe_balance_losses).mean()
            if moe_balance_losses else logits.new_zeros((), dtype=torch.float32)
        )
        moe_expert_load = (
            torch.stack(expert_loads).mean(dim=0)
            if expert_loads
            else logits.new_zeros((8,), dtype=torch.float32)
        )
        return CausalLMOutput(
            logits=logits,
            bpred_output=bpred_output,
            inference_params=inference_params,
            feedback_logits=feedback_logits,
            moe_aux_loss=moe_aux_loss,
            moe_balance_loss=moe_balance_loss,
            moe_expert_load=moe_expert_load,
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
        return CausalLMOutput(
            logits=logits,
            bpred_output=bpred,
            inference_params=inference_params,
            feedback_logits=[],
            moe_balance_loss=logits.new_zeros((), dtype=torch.float32),
            moe_expert_load=logits.new_zeros((8,), dtype=torch.float32),
        )
