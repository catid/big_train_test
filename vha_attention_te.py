#!/usr/bin/env python3
"""Transformer Engine implementation of the VHA training-time attention core.

This follows Virtual Head Attention (arXiv:2610.17959): a physical query
projection with half the eventual query heads, two KV heads, per-KV-group
near-identity query Premix, and low-rank identity-residual Postmix.  The
resulting query/key/value tensors are handed to TE DotProductAttention so its
delayed FP8 DPA and CroissAttn backward correction remain available.

Inputs and outputs use TE's ``sbhd`` convention: ``[sequence, batch, hidden]``
at the module boundary and ``[sequence, batch, heads, head_dim]`` inside DPA.
"""
from __future__ import annotations

import math

import torch
import torch.nn as nn
import transformer_engine.pytorch as te


class VHAAttention(nn.Module):
    """VHA attention with a standard GQA-2 DPA kernel after Premix."""

    def __init__(
        self,
        hidden_size: int,
        num_attention_heads: int,
        num_key_value_heads: int = 2,
        head_dim: int | None = None,
        postmix_rank: int | None = None,
        window_size: tuple[int, int] | None = None,
        params_dtype: torch.dtype = torch.bfloat16,
        device: torch.device | str = "cuda",
        seq_length: int | None = None,
        micro_batch_size: int | None = None,
        layer_number: int = 1,
    ) -> None:
        super().__init__()
        if head_dim is None:
            if hidden_size % num_attention_heads:
                raise ValueError("hidden_size must divide evenly into attention heads")
            head_dim = hidden_size // num_attention_heads
        if num_attention_heads % num_key_value_heads:
            raise ValueError("num_attention_heads must be divisible by num_key_value_heads")
        if postmix_rank is None:
            postmix_rank = num_attention_heads // 4
        if postmix_rank <= 0:
            raise ValueError("postmix_rank must be positive")

        self.hidden_size = hidden_size
        self.num_attention_heads = num_attention_heads
        self.num_key_value_heads = num_key_value_heads
        self.physical_query_heads = num_attention_heads // num_key_value_heads
        self.head_dim = head_dim
        self.postmix_rank = postmix_rank

        common = dict(
            bias=False,
            params_dtype=params_dtype,
            device=device,
        )
        self.q_proj = te.Linear(
            hidden_size, self.physical_query_heads * head_dim, **common
        )
        self.k_proj = te.Linear(hidden_size, num_key_value_heads * head_dim, **common)
        self.v_proj = te.Linear(hidden_size, num_key_value_heads * head_dim, **common)
        self.dpa = te.DotProductAttention(
            num_attention_heads=num_attention_heads,
            kv_channels=head_dim,
            num_gqa_groups=num_key_value_heads,
            attention_dropout=0.0,
            qkv_format="sbhd",
            attn_mask_type="causal",
            window_size=window_size,
            layer_number=layer_number,
        )
        self.o_proj = te.Linear(num_attention_heads * head_dim, hidden_size, **common)

        eye = torch.eye(head_dim, device=device, dtype=params_dtype)
        noise = torch.randn(
            num_key_value_heads,
            head_dim,
            head_dim,
            device=device,
            dtype=params_dtype,
        )
        premix = eye.unsqueeze(0) + (0.1 / math.sqrt(head_dim)) * noise
        self.premix_weight = nn.Parameter(premix)
        self.postmix_u = nn.Parameter(
            torch.randn(
                num_attention_heads,
                postmix_rank,
                device=device,
                dtype=params_dtype,
            )
            * 0.01
        )
        self.postmix_v = nn.Parameter(
            torch.zeros(
                num_attention_heads,
                postmix_rank,
                device=device,
                dtype=params_dtype,
            )
        )

    def apply_premix(self, query: torch.Tensor) -> torch.Tensor:
        """Expand [S,B,Hq',D] physical queries into [S,B,Hq,D] virtual heads."""
        if query.shape[-2:] != (self.physical_query_heads, self.head_dim):
            raise ValueError(
                f"expected physical query tail {(self.physical_query_heads, self.head_dim)}, "
                f"got {tuple(query.shape[-2:])}"
            )
        expanded = torch.einsum("sbir,grd->sbgid", query, self.premix_weight)
        return expanded.reshape(
            query.shape[0], query.shape[1], self.num_attention_heads, self.head_dim
        )

    def apply_postmix(self, output: torch.Tensor) -> torch.Tensor:
        """Apply Y + V(U^T Y) along the head axis, preserving head shape."""
        z = torch.einsum("sbhd,hr->sbrd", output, self.postmix_u)
        delta = torch.einsum("sbrd,hr->sbhd", z, self.postmix_v)
        return output + delta

    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        query = self.q_proj(hidden_states).reshape(
            hidden_states.shape[0],
            hidden_states.shape[1],
            self.physical_query_heads,
            self.head_dim,
        )
        query = self.apply_premix(query)
        key = self.k_proj(hidden_states).reshape(
            hidden_states.shape[0], hidden_states.shape[1], self.num_key_value_heads, self.head_dim
        )
        value = self.v_proj(hidden_states).reshape(
            hidden_states.shape[0], hidden_states.shape[1], self.num_key_value_heads, self.head_dim
        )
        output = self.dpa(query, key, value, qkv_format="sbhd")
        output = output.reshape(
            hidden_states.shape[0], hidden_states.shape[1], self.num_attention_heads, self.head_dim
        )
        output = self.apply_postmix(output)
        return self.o_proj(output.reshape(hidden_states.shape[0], hidden_states.shape[1], -1))


__all__ = ["VHAAttention"]
