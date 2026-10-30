# Base code imported from
# https://github.com/state-spaces/mamba
from functools import partial
from typing import Optional

from torch import nn, Tensor

from flash_attn.ops.triton.layer_norm import RMSNorm
from mamba_ssm.modules.mamba2 import Mamba2

from .mha import CausalMHA
from .mlp import SparseMoESwiGLU


class Mamba2Wrapper(Mamba2):
    """
    Mamba2 wrapper class that has the same inference interface as the CausalMHA class.
    """

    def step(self, hidden_states, conv_state_or_inference, ssm_state=None):
        """Support Mamba's native cache call and H-Net's cache interface.

        ``Mamba2.forward`` calls ``self.step(u, conv_state, ssm_state)`` once
        its cache offset is nonzero, while :class:`Block` calls the mixer as
        ``step(x, inference_params)``.  The old two-argument override treated
        the former convolution tensor as an H-Net cache object and raised a
        ``TypeError`` during outer-stack decode.
        """
        if ssm_state is not None:
            return super().step(
                hidden_states, conv_state_or_inference, ssm_state
            )

        inference_params = conv_state_or_inference
        try:
            conv_state, ssm_state = inference_params.key_value_memory_dict[
                self.layer_idx
            ]
        except (AttributeError, KeyError) as exc:
            raise RuntimeError(
                f"missing Mamba cache for layer {self.layer_idx}; "
                "allocate the complete canonical inference cache"
            ) from exc
        result, conv_state, ssm_state = super().step(
            hidden_states, conv_state, ssm_state
        )

        # Update the state cache in-place so repeated H-Net decode calls
        # observe the same state objects allocated during prefill.
        inference_params.key_value_memory_dict[self.layer_idx][0].copy_(conv_state)
        inference_params.key_value_memory_dict[self.layer_idx][1].copy_(ssm_state)
        return result


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
        mixer_cls = partial(
            Mamba2Wrapper, **ssm_cfg, **factory_kwargs, layer_idx=layer_idx
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
            SparseMoESwiGLU,
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
