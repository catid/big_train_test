# Copyright (c) 2023, Tri Dao.

import torch
import torch.nn as nn
from einops import rearrange

from flash_attn import (
    flash_attn_qkvpacked_func,
    flash_attn_varlen_qkvpacked_func,
    flash_attn_with_kvcache,
)

from .rotary import RotaryEmbedding
from hnet.te import TEDotProductAttention, assert_fused_attention_selected, make_linear



class FlashCausalSelfAttention(nn.Module):
    """Implement the scaled dot product attention with softmax.
    Arguments
    ---------
        softmax_scale: The temperature to use for the softmax attention.
                      (default: 1/sqrt(d_keys) where d_keys is computed at
                      runtime)
    """

    def __init__(
        self,
        softmax_scale=None,
        window_size=(-1, -1),
    ):
        super().__init__()
        assert (
            flash_attn_varlen_qkvpacked_func is not None
        ), "FlashAttention is not installed"
        self.softmax_scale = softmax_scale
        self.window_size = window_size

    def _window(self):
        # FlashAttention expects one ``(left, right)`` tuple.  Accept both
        # the scalar shorthand used by H-Net configs and an explicit tuple;
        # nesting a tuple here would produce an invalid SWA shape.
        if isinstance(self.window_size, (tuple, list)):
            if len(self.window_size) != 2:
                raise ValueError("window_size must be a scalar or (left, right) pair")
            return tuple(self.window_size)
        return (self.window_size, -1)

    def forward(self, qkv, cu_seqlens=None, max_seqlen=None):
        """Implements the multihead softmax attention.
        Arguments
        ---------
            qkv: The tensor containing the query, key, and value.
                If cu_seqlens is None and max_seqlen is None, then qkv has shape (B, S, 3, H, D).
                If cu_seqlens is not None and max_seqlen is not None, then qkv has shape
                (total, 3, H, D), where total is the sum of the sequence lengths in the batch.
            cu_seqlens: (batch_size + 1,), dtype torch.int32. The cumulative sequence lengths
                of the sequences in the batch, used to index into qkv.
            max_seqlen: int. Maximum sequence length in the batch.
        Returns:
        --------
            out: (total, H, D) if cu_seqlens is not None and max_seqlen is not None,
                else (B, S, H, D).
        """
        assert qkv.dtype in [torch.float16, torch.bfloat16]
        assert qkv.is_cuda
        if cu_seqlens is not None:
            assert cu_seqlens.dtype == torch.int32
            assert max_seqlen is not None
            assert isinstance(max_seqlen, int)
            return flash_attn_varlen_qkvpacked_func(
                qkv,
                cu_seqlens,
                max_seqlen,
                softmax_scale=self.softmax_scale,
                causal=True,
                window_size=self._window(),
            )
        else:
            return flash_attn_qkvpacked_func(
                qkv,
                softmax_scale=self.softmax_scale,
                causal=True,
                window_size=self._window(),
            )


class CausalMHA(nn.Module):
    def __init__(
        self,
        d_model,
        num_heads,
        qkv_proj_bias=False,
        out_proj_bias=False,
        window_size=-1,  # -1 for global Attention
        softmax_scale=None,
        layer_idx=None,
        rotary_emb_dim=0,
        rotary_emb_base=10000.0,
        rotary_emb_interleaved=False,
        croissattn_aux=False,
        device=None,
        dtype=None,
    ) -> None:
        """
        return_residual: whether to return the input x along with the output. This is for
            performance reason: for post-norm architecture, returning the input allows us
            to fuse the backward of nn.Linear with the residual connection.
        """
        factory_kwargs = {"device": device, "dtype": dtype}
        super().__init__()
        self.d_model = d_model
        self.layer_idx = layer_idx
        self.softmax_scale = softmax_scale
        self.rotary_emb_dim = rotary_emb_dim
        self.window_size = window_size
        self.croissattn_aux = bool(croissattn_aux)
        if flash_attn_with_kvcache is None:
            raise RuntimeError(
                "H-Net requires FlashAttention's fused KV-cache kernel; "
                "portable decode fallbacks are unsupported"
            )

        self.num_heads = num_heads
        assert self.d_model % num_heads == 0, "d_model must be divisible by num_heads"
        self.head_dim = self.d_model // num_heads
        qkv_dim = self.head_dim * (3 * self.num_heads)

        if self.rotary_emb_dim > 0:
            self.rotary_emb = RotaryEmbedding(
                self.rotary_emb_dim,
                base=rotary_emb_base,
                interleaved=rotary_emb_interleaved,
                device=device,
            )

        self.Wqkv = make_linear(d_model, qkv_dim, bias=qkv_proj_bias, **factory_kwargs)
        self.inner_attn = FlashCausalSelfAttention(
            softmax_scale=softmax_scale,
            window_size=window_size,
        )
        self.out_proj = make_linear(d_model, d_model, bias=out_proj_bias, **factory_kwargs)
        self.te_attn = None
        if self.croissattn_aux:
            self.te_attn = TEDotProductAttention(
                num_attention_heads=num_heads,
                kv_channels=self.head_dim,
                qkv_format="bshd",
                # Use the ordinary causal policy for padded ``bshd`` calls;
                # packed ``thd`` calls override it with ``padding_causal``.
                attn_mask_type="causal",
                attention_dropout=0.0,
                softmax_scale=softmax_scale,
            )

    def allocate_inference_cache(self, batch_size, max_seqlen, dtype=None):
        dtype = self.out_proj.weight.dtype if dtype is None else dtype
        device = self.out_proj.weight.device
        return torch.empty(
            batch_size,
            max_seqlen,
            2,
            self.num_heads,
            self.head_dim,
            dtype=dtype,
            device=device,
        )

    def _ensure_loop_cache(self, inference_params, cache_layer_idx):
        if cache_layer_idx not in inference_params.key_value_memory_dict:
            base = inference_params.key_value_memory_dict[self.layer_idx]
            inference_params.key_value_memory_dict[cache_layer_idx] = torch.empty_like(base)

    def _apply_rotary_update_kvcache_attention(self, q, kv, inference_params, cache_layer_idx=None):
        """
        Fast path that combine 3 steps: apply rotary to Q and K, update kv cache, and apply attention.
        q: (batch_size, seqlen_q, nheads, head_dim)
        kv: (batch_size, seqlen_k, 2, nheads_kv, head_dim)
        """
        assert inference_params is not None and inference_params.seqlen_offset > 0
        if self.rotary_emb_dim > 0:
            assert self.rotary_emb.scale is None, "This code path does not support xPos"
            self.rotary_emb._update_cos_sin_cache(
                inference_params.max_seqlen, device=q.device, dtype=q.dtype
            )
            rotary_cos, rotary_sin = (
                self.rotary_emb._cos_cached,
                self.rotary_emb._sin_cached,
            )
        else:
            rotary_cos, rotary_sin = None, None
        batch = q.shape[0]
        cache_layer_idx = self.layer_idx if cache_layer_idx is None else cache_layer_idx
        self._ensure_loop_cache(inference_params, cache_layer_idx)
        kv_cache = inference_params.key_value_memory_dict[cache_layer_idx][:batch]
        cache_seqlens = (
            inference_params.lengths_per_sample[:batch]
            if inference_params.lengths_per_sample is not None
            else inference_params.seqlen_offset
        )
        context = flash_attn_with_kvcache(
            q,
            kv_cache[:, :, 0],
            kv_cache[:, :, 1],
            kv[:, :, 0],
            kv[:, :, 1],
            rotary_cos=rotary_cos,
            rotary_sin=rotary_sin,
            cache_seqlens=cache_seqlens,
            softmax_scale=self.softmax_scale,
            causal=True,
            rotary_interleaved=(
                self.rotary_emb.interleaved if self.rotary_emb_dim > 0 else False
            ),
            window_size=self.inner_attn._window(),
        )
        return context

    def _update_kvcache_attention(self, q, kv, inference_params, cache_layer_idx=None):
        """Write kv to inference_params, then do attention"""
        batch = q.shape[0]
        cache_layer_idx = self.layer_idx if cache_layer_idx is None else cache_layer_idx
        self._ensure_loop_cache(inference_params, cache_layer_idx)
        kv_cache = inference_params.key_value_memory_dict[cache_layer_idx][:batch]
        cache_seqlens = (
            inference_params.lengths_per_sample[:batch]
            if inference_params.lengths_per_sample is not None
            else inference_params.seqlen_offset
        )
        return flash_attn_with_kvcache(
            q,
            kv_cache[:, :, 0],
            kv_cache[:, :, 1],
            kv[:, :, 0],
            kv[:, :, 1],
            cache_seqlens=cache_seqlens,
            softmax_scale=self.softmax_scale,
            causal=True,
            window_size=self.inner_attn._window()
        )

    def forward(
        self,
        x,
        cu_seqlens=None,
        max_seqlen=None,
        inference_params=None,
        **kwargs,
    ):
        """
        Arguments:
            x: (batch, seqlen, hidden_dim) (where hidden_dim = num heads * head dim) if
                cu_seqlens is None and max_seqlen is None, else (total, hidden_dim) where total
                is the is the sum of the sequence lengths in the batch.
            cu_seqlens: (batch_size + 1,), dtype torch.int32. The cumulative sequence lengths
                of the sequences in the batch, used to index into x. Only applicable when using
                FlashAttention.
            max_seqlen: int. Maximum sequence length in the batch.
            inference_params: for generation. Adapted from Megatron-LM (and Apex)
            https://github.com/NVIDIA/apex/blob/3ff1a10f72ec07067c4e44759442329804ac5162/apex/transformer/testing/standalone_transformer_lm.py#L470
        """
        if cu_seqlens is not None:
            assert max_seqlen is not None
            # assert self.rotary_emb_dim == 0
        if inference_params is not None:
            assert cu_seqlens is None and max_seqlen is None

        # The recurrent executions share one KV slot. Each loop rewrites the
        # current token at the same sequence offset, so cache memory stays
        # constant while the second pass attends to the refined history.
        kwargs.pop("loop_idx", None)
        cache_layer_idx = self.layer_idx

        kwargs = {"cu_seqlens": cu_seqlens, "max_seqlen": max_seqlen, **kwargs}
        seqlen_offset = (
            0
            if inference_params is None
            else (
                inference_params.lengths_per_sample
                if inference_params.lengths_per_sample is not None
                else inference_params.seqlen_offset
            )
        )
        rotary_max_seqlen = (
            inference_params.max_seqlen if inference_params is not None else None
        )

        qkv = self.Wqkv(x)
        qkv = rearrange(
            qkv, "... (three h d) -> ... three h d", three=3, d=self.head_dim
        )
        if (
            inference_params is None
            or inference_params.seqlen_offset == 0
            or (self.rotary_emb_dim == 0 or self.rotary_emb_dim % 16 != 0)
        ):
            if self.rotary_emb_dim > 0:
                # qkv = self.rotary_emb(
                #     qkv, seqlen_offset=seqlen_offset, max_seqlen=rotary_max_seqlen
                # )
                qkv = self.rotary_emb(qkv, seqlen_offset=seqlen_offset, **kwargs)
            if inference_params is None and self.te_attn is not None:
                # TE's patched DPA selects its CroissAttn backward correction
                # under the delayed-scaling FP8 recipe.  Keep the packed and
                # padded layouts explicit so no reference BF16 bmm is used.
                q, k, v = qkv.unbind(dim=1 if cu_seqlens is not None else 2)
                if cu_seqlens is None:
                    context = self.te_attn(
                        query_layer=q, key_layer=k, value_layer=v,
                        qkv_format="bshd",
                    )
                else:
                    context = self.te_attn(
                        query_layer=q, key_layer=k, value_layer=v,
                        qkv_format="thd",
                        attn_mask_type="padding_causal",
                        cu_seqlens_q=cu_seqlens,
                        cu_seqlens_kv=cu_seqlens,
                        max_seqlen_q=max_seqlen,
                        max_seqlen_kv=max_seqlen,
                    )
                assert_fused_attention_selected()
            elif inference_params is None:
                context = self.inner_attn(qkv, **kwargs)
            else:
                context = self._update_kvcache_attention(
                    qkv[:, :, 0], qkv[:, :, 1:], inference_params, cache_layer_idx=cache_layer_idx
                )
        else:
            context = self._apply_rotary_update_kvcache_attention(
                qkv[:, :, 0], qkv[:, :, 1:], inference_params,
                cache_layer_idx=cache_layer_idx,
            )
        # TE DPA returns the head dimension flattened for its packed and
        # padded APIs, while FlashAttention returns ``[..., H, D]``.
        context_flat = (
            context
            if context.shape[-1] == self.d_model
            else rearrange(context, "... h d -> ... (h d)")
        )
        out = self.out_proj(context_flat)
        return out

    def step(self, x, inference_params, loop_idx=0):
        # Recurrent inner-loop execution intentionally reuses the same cache
        # slot and sequence offset. The later loop refines the current token's
        # K/V in place instead of appending a duplicate position.
        return self.forward(x, inference_params=inference_params, loop_idx=loop_idx)
