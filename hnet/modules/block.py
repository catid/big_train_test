# Base code imported from
# https://github.com/state-spaces/mamba
from functools import partial
from typing import Optional

from torch import nn, Tensor

from flash_attn.ops.triton.layer_norm import RMSNorm
from mamba_ssm.modules.mamba3 import Mamba3

from .mha import CausalMHA
from .latent_moe import LatentMoE


class Mamba3Wrapper(Mamba3):
    """Mamba-3 with H-Net's cache and block-step interface.

    Mamba-3's public ``forward`` path calls its low-level ``step`` with a
    ``(B, L, D)`` tensor during cached decode, while the fused Cute kernel
    expects ``(B, D)``.  H-Net also owns the per-layer cache container and
    advances the outer sequence offset.  This adapter keeps those contracts
    explicit and has no Mamba-2 compatibility path.
    """

    supports_activation_checkpointing = False

    def forward(self, u, seq_idx=None, cu_seqlens=None, inference_params=None):
        if inference_params is not None and inference_params.seqlen_offset > 0:
            if u.ndim != 3 or u.shape[1] != 1:
                raise ValueError("Mamba-3 cached decode requires a single token")
            return self.step(u, inference_params)
        return super().forward(
            u,
            seq_idx=seq_idx,
            cu_seqlens=cu_seqlens,
            inference_params=inference_params,
        )

    def step(self, hidden_states, inference_or_angle, ssm_state=None,
              k_state=None, v_state=None):
        # The native Mamba-3 call shape is retained for its forward helper;
        # H-Net's Block calls the two-argument cache form.
        if ssm_state is not None:
            u = hidden_states[:, 0] if hidden_states.ndim == 3 else hidden_states
            out, angle, state, key, value = super().step(
                u, inference_or_angle, ssm_state, k_state, v_state
            )
            return (
                (out.unsqueeze(1) if hidden_states.ndim == 3 else out),
                angle,
                state,
                key,
                value,
            )

        inference_params = inference_or_angle
        if hidden_states.ndim == 3:
            if hidden_states.shape[1] != 1:
                raise ValueError("Mamba-3 cached decode requires a single token")
            u = hidden_states[:, 0]
        else:
            u = hidden_states
        try:
            angle, state, key, value = inference_params.key_value_memory_dict[
                self.layer_idx
            ]
        except (AttributeError, KeyError) as exc:
            raise RuntimeError(
                f"missing Mamba-3 cache for layer {self.layer_idx}; "
                "allocate the complete canonical inference cache"
            ) from exc
        out, next_angle, next_state, next_key, next_value = super().step(
            u, angle, state, key, value
        )
        # Cute updates the recurrent SSM tensor in place, while rotary/K/V
        # outputs are separate tensors. Keep the stable H-Net cache storage
        # for those three streams and avoid a redundant fourth device copy.
        angle.copy_(next_angle)
        key.copy_(next_key)
        value.copy_(next_value)
        return out.unsqueeze(1)


def create_block(
    arch,
    d_model,
    d_intermediate=None,
    ssm_cfg=dict(),
    attn_cfg=dict(),
    norm_epsilon=1e-5,
    layer_idx=None,
    residual_in_fp32=True,
    device=None,
    dtype=None,
    moe_cfg=None,
):
    factory_kwargs = {"device": device, "dtype": dtype}

    # Mixer
    if arch in ("s", "S", "t", "T"):
        mixer_cls = partial(
            CausalMHA, **attn_cfg, **factory_kwargs, layer_idx=layer_idx
        )
    elif arch in ("m", "M"):
        mamba_cfg = dict(ssm_cfg)
        # Mamba-3's fused SISO path is tuned for 64-token chunks.  Its
        # constructor absorbs the legacy d_conv field, which is not part of
        # Mamba-3's recurrence.
        mamba_cfg["chunk_size"] = 64
        mixer_cls = partial(
            Mamba3Wrapper, **mamba_cfg, **factory_kwargs, layer_idx=layer_idx
        )
    else:
        raise NotImplementedError

    # MLP
    if arch in ("S", "T", "M"):
        if moe_cfg is None or not bool(getattr(moe_cfg, "moe_enabled", False)):
            raise ValueError(
                "uppercase H-Net layers require the configured sparse MoE/shared expert"
            )
        mlp_cls = partial(
            LatentMoE,
            d_intermediate=d_intermediate,
            num_experts=int(getattr(moe_cfg, "moe_num_experts", 8)),
            top_k=int(getattr(moe_cfg, "moe_top_k", 1)),
            expert_intermediate=int(getattr(moe_cfg, "moe_expert_intermediate", 0)),
            shared_intermediate=int(getattr(moe_cfg, "moe_shared_intermediate", 0)),
            capacity_factor=float(getattr(moe_cfg, "moe_capacity_factor", 1.25)),
            capacity_tokens=int(getattr(moe_cfg, "moe_capacity_tokens", 0)),
            aux_loss_weight=float(getattr(moe_cfg, "moe_aux_loss_weight", 0.01)),
            **factory_kwargs,
        )
    elif arch in ("s", "t", "m"):
        mlp_cls = nn.Identity
    else:
        raise NotImplementedError

    # Normalization
    norm_cls = partial(RMSNorm, eps=norm_epsilon, **factory_kwargs)

    block = Block(
        d_model,
        mixer_cls,
        mlp_cls,
        norm_cls=norm_cls,
        residual_in_fp32=residual_in_fp32,
    )
    return block


class Block(nn.Module):
    def __init__(
        self,
        d_model,
        mixer_cls=None,
        mlp_cls=None,
        norm_cls=None,
        residual_in_fp32=True,
    ):
        super().__init__()
        self.residual_in_fp32 = residual_in_fp32
        self.norm1 = norm_cls(d_model)
        self.mixer = mixer_cls(d_model)
        if mlp_cls is not nn.Identity:
            self.norm2 = norm_cls(d_model)
            self.mlp = mlp_cls(d_model)
        else:
            self.mlp = None

        assert RMSNorm is not None, "Triton is not installed"
        assert isinstance(self.norm1, RMSNorm), "Only RMSNorm is supported"

    def forward(
        self,
        hidden_states: Tensor,
        residual: Optional[Tensor] = None,
        inference_params=None,
        mixer_kwargs=None,
        active_mask=None,
        moe_sequence_ids=None,
        moe_route_counts=None,
        moe_loop_idx=0,
        moe_layer_idx=0,
    ):
        hidden_states, residual = self.norm1(
            hidden_states,
            residual=residual,
            prenorm=True,
            residual_in_fp32=self.residual_in_fp32,
        )

        if mixer_kwargs is None:
            mixer_kwargs = {}
        hidden_states = self.mixer(
            hidden_states, inference_params=inference_params, **mixer_kwargs
        )

        if self.mlp is not None:
            hidden_states, residual = self.norm2(
                hidden_states,
                residual=residual,
                prenorm=True,
                residual_in_fp32=self.residual_in_fp32,
            )
            hidden_states = self.mlp(
                hidden_states,
                active_mask=active_mask,
                sequence_ids=moe_sequence_ids,
                route_counts=moe_route_counts,
                loop_idx=moe_loop_idx,
                layer_idx=moe_layer_idx,
            )

        return hidden_states, residual

    def allocate_inference_cache(self, batch_size, max_seqlen, dtype=None, **kwargs):
        return self.mixer.allocate_inference_cache(
            batch_size, max_seqlen, dtype=dtype, **kwargs
        )

    def step(
        self,
        hidden_states,
        inference_params,
        residual=None,
        moe_sequence_ids=None,
        moe_route_counts=None,
        moe_loop_idx=0,
        moe_layer_idx=0,
    ):
        hidden_states, residual = self.norm1(
            hidden_states,
            residual=residual,
            prenorm=True,
            residual_in_fp32=self.residual_in_fp32,
        )
        hidden_states = self.mixer.step(hidden_states, inference_params)
        if self.mlp is not None:
            hidden_states, residual = self.norm2(
                hidden_states,
                residual=residual,
                prenorm=True,
                residual_in_fp32=self.residual_in_fp32,
            )
            hidden_states = self.mlp(
                hidden_states,
                sequence_ids=moe_sequence_ids,
                route_counts=moe_route_counts,
                loop_idx=moe_loop_idx,
                layer_idx=moe_layer_idx,
                decode=True,
            )

        return hidden_states, residual
