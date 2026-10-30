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


def _fp8_qdq(x: torch.Tensor, backward: bool = False) -> torch.Tensor:
    """Round an activation to FP8 and decode it without a host-side fallback.

    CroissAttn's correction is applied to FP8 training products.  The B300
    PyTorch build does not expose a batched ``_scaled_mm`` API, so products
    use vectorized ``bmm`` on decoded BF16 operands; this preserves the
    algorithm and avoids a Python loop over heads or tokens.  Transformer
    Engine's fused path is provided by the production training environment.
    """
    dtype = torch.float8_e5m2 if backward else torch.float8_e4m3fn
    limit = 57344.0 if backward else 448.0
    scale = x.detach().float().abs().amax().div(limit).clamp_min(torch.finfo(torch.float32).tiny)
    return (x / scale).to(dtype).to(x.dtype) * scale.to(x.dtype)


class _CroissAttn(torch.autograd.Function):
    """Vectorized CroissAttn reference with the corrected auxiliary O'.

    This opt-in path is used only for regular (unpacked) training attention.
    It follows ~/fp8's six-product schedule and zeros the strictly upper
    causal triangle before dQ/dK, preventing masked logits from contributing
    gradients.  There is no Python per-token fallback.
    """

    @staticmethod
    def forward(ctx, q, k, v, scale, lengths=None):
        # q/k/v are flattened to [batch * heads, S, D].  A non-empty lengths
        # tensor marks padded rows in the packed-document path.
        if lengths is None:
            lengths = q.new_empty((0,), dtype=torch.int32)
        has_lengths = lengths.numel() > 0
        if has_lengths:
            if q.shape[0] % lengths.numel() != 0:
                raise ValueError("CroissAttn lengths must divide batch*heads")
            lengths_expanded = lengths.to(device=q.device, dtype=torch.int64).repeat_interleave(
                q.shape[0] // lengths.numel()
            )
        else:
            lengths_expanded = None
        qf = _fp8_qdq(q)
        kf = _fp8_qdq(k)
        vf = _fp8_qdq(v)
        seqlen = qf.shape[1]
        score_bytes = qf.shape[0] * seqlen * seqlen * 4
        # Keep the fused-sized path for short chunks. Use tiled CroissAttn for
        # long sequences so batch scaling does not retain an O(S^2) graph.
        # Keep the saved probability/auxiliary tensors bounded.  The
        # canonical 4096-byte inner sequence otherwise retains a dense
        # [batch * heads, S, S] graph for every FA layer and FBT pass.
        if score_bytes <= 256 * 1024 * 1024:
            scores = torch.bmm(qf, kf.transpose(1, 2)) * scale
            causal = torch.ones((seqlen, seqlen), device=scores.device, dtype=torch.bool).triu(1)
            if has_lengths:
                key_invalid = torch.arange(seqlen, device=scores.device)[None, :] >= lengths_expanded[:, None]
                causal = causal[None, :, :] | key_invalid[:, None, :]
            scores = scores.masked_fill(causal, float("-inf"))
            p = torch.softmax(scores.float(), dim=-1).to(q.dtype)
            out = torch.bmm(_fp8_qdq(p), vf)
            aux = torch.bmm(p.float(), vf.float()).to(q.dtype)
            if has_lengths:
                query_valid = torch.arange(seqlen, device=scores.device)[None, :] < lengths_expanded[:, None]
                out = torch.where(query_valid[..., None], out, torch.zeros_like(out))
            # Keep the original activation operands so each backward product can
            # independently recast both operands to E5M2.  Reusing the E4M3
            # forward operands here would violate CroissAttn's six-product
            # schedule and noticeably bias dP/dQ/dK.
            ctx.save_for_backward(q, k, v, p, aux, lengths)
            ctx.tiled = False
            ctx.scale = scale
            ctx.has_lengths = has_lengths
            return out
        # Materialize attention probabilities in query tiles.  The previous
        # implementation allocated [B*H,S,S], which made batch scaling fail
        # at S=4096 even on 256 GiB GPUs.  A 512-row tile preserves the exact
        # CroissAttn arithmetic while bounding peak score memory.
        block = min(512, seqlen)
        outputs = []
        for start in range(0, seqlen, block):
            end = min(start + block, seqlen)
            scores = torch.bmm(qf[:, start:end], kf[:, :end].transpose(1, 2)) * scale
            causal = torch.ones((end - start, end), device=scores.device, dtype=torch.bool).triu(1 + start)
            if has_lengths:
                key_invalid = torch.arange(end, device=scores.device)[None, :] >= lengths_expanded[:, None]
                causal = causal[None, :, :] | key_invalid[:, None, :]
            scores = scores.masked_fill(causal, float("-inf"))
            p = torch.softmax(scores.float(), dim=-1).to(q.dtype)
            tile_out = torch.bmm(_fp8_qdq(p), vf[:, :end])
            if has_lengths:
                query_valid = torch.arange(start, end, device=scores.device)[None, :] < lengths_expanded[:, None]
                tile_out = torch.where(query_valid[..., None], tile_out, torch.zeros_like(tile_out))
            outputs.append(tile_out)
        out = torch.cat(outputs, dim=1)
        # Recompute P and the unrounded auxiliary O'=P@V_f in backward to
        # avoid retaining the full
        # [batch*heads, seqlen, seqlen] probability and auxiliary tensors for
        # every recurrent/FBT invocation.  This is essential for batch scaling
        # while preserving the CroissAttn correction (O' is never the rounded
        # FP8 PV output).
        # Save original operands; backward reconstructs both E4M3 forward
        # values and independent E5M2 backward operands without retaining two
        # decoded copies of every Q/K/V tensor.
        ctx.save_for_backward(q, k, v, lengths)
        ctx.scale = scale
        ctx.seqlen = seqlen
        ctx.block = block
        ctx.tiled = True
        ctx.has_lengths = has_lengths
        return out

    @staticmethod
    def backward(ctx, dout):
        if not ctx.tiled:
            q, k, v, p, aux, lengths = ctx.saved_tensors
            # Each backward GEMM independently quantizes both operands to
            # E5M2.  The auxiliary O' above intentionally remains P@V_f,
            # where V_f is the E4M3 forward value.
            qb = _fp8_qdq(q, True)
            kb = _fp8_qdq(k, True)
            vb = _fp8_qdq(v, True)
            dpf = torch.bmm(_fp8_qdq(dout, True), vb.transpose(1, 2))
            dv = torch.bmm(_fp8_qdq(p.transpose(1, 2), True), _fp8_qdq(dout, True))
            dstar = (dout.float() * aux.float()).sum(dim=-1, keepdim=True)
            causal = torch.ones((p.shape[-2], p.shape[-1]), device=p.device, dtype=torch.bool).triu(1)
            if ctx.has_lengths:
                lengths_expanded = lengths.to(device=p.device, dtype=torch.int64).repeat_interleave(
                    q.shape[0] // lengths.numel()
                )
                key_invalid = torch.arange(p.shape[-1], device=p.device)[None, :] >= lengths_expanded[:, None]
                causal = causal[None, :, :] | key_invalid[:, None, :]
            ds = p * (dpf.float() - dstar).to(p.dtype)
            ds = ds.masked_fill(causal, 0)
            if ctx.has_lengths:
                query_valid = torch.arange(p.shape[-2], device=p.device)[None, :] < lengths_expanded[:, None]
                ds = torch.where(query_valid[..., None], ds, torch.zeros_like(ds))
            dq = torch.bmm(_fp8_qdq(ds, True), kb) * ctx.scale
            dk = torch.bmm(_fp8_qdq(ds.transpose(1, 2), True), qb) * ctx.scale
            return dq, dk, dv, None, None
        q, k, v, lengths = ctx.saved_tensors
        qf = _fp8_qdq(q)
        kf = _fp8_qdq(k)
        vf = _fp8_qdq(v)
        qb = _fp8_qdq(q, True)
        kb = _fp8_qdq(k, True)
        vb = _fp8_qdq(v, True)
        seqlen, block = ctx.seqlen, ctx.block
        lengths_expanded = None
        if ctx.has_lengths:
            lengths_expanded = lengths.to(device=q.device, dtype=torch.int64).repeat_interleave(
                q.shape[0] // lengths.numel()
            )
        dq = torch.zeros_like(qf)
        dk = torch.zeros_like(kf)
        dv = torch.zeros_like(vf)
        for start in range(0, seqlen, block):
            end = min(start + block, seqlen)
            key_len = end
            scores = torch.bmm(qf[:, start:end], kf[:, :key_len].transpose(1, 2)) * ctx.scale
            causal = torch.ones((end - start, key_len), device=qf.device, dtype=torch.bool).triu(1 + start)
            if ctx.has_lengths:
                key_invalid = torch.arange(key_len, device=qf.device)[None, :] >= lengths_expanded[:, None]
                causal = causal[None, :, :] | key_invalid[:, None, :]
            scores = scores.masked_fill(causal, float("-inf"))
            p = torch.softmax(scores.float(), dim=-1).to(qf.dtype)
            d_out = dout[:, start:end]
            dpf = torch.bmm(
                _fp8_qdq(d_out, True), vb[:, :key_len].transpose(1, 2)
            )
            dv[:, :key_len] += torch.bmm(
                _fp8_qdq(p.transpose(1, 2), True), _fp8_qdq(d_out, True)
            )
            # CroissAttn's corrected auxiliary O'=P@V_f uses the unrounded
            # probability/value product in the softmax derivative.
            aux = torch.bmm(p.float(), vf[:, :key_len].float()).to(qf.dtype)
            dstar = (d_out.float() * aux.float()).sum(dim=-1, keepdim=True)
            ds = p * (dpf.float() - dstar).to(p.dtype)
            ds = ds.masked_fill(causal, 0)
            if ctx.has_lengths:
                query_valid = torch.arange(start, end, device=qf.device)[None, :] < lengths_expanded[:, None]
                ds = torch.where(query_valid[..., None], ds, torch.zeros_like(ds))
            dq[:, start:end] = torch.bmm(
                _fp8_qdq(ds, True), kb[:, :key_len]
            ) * ctx.scale
            dk[:, :key_len] += torch.bmm(
                _fp8_qdq(ds.transpose(1, 2), True), qb[:, start:end]
            ) * ctx.scale
        return dq, dk, dv, None, None


def croissattn_fp8(q, k, v, scale):
    """Apply corrected CroissAttn to [B,H,S,D] tensors."""
    b, h, s, d = q.shape
    out = _CroissAttn.apply(
        q.reshape(b * h, s, d),
        k.reshape(b * h, s, d),
        v.reshape(b * h, s, d),
        scale,
        None,
    )
    return out.reshape(b, h, s, d)


def croissattn_varlen_fp8(q, k, v, cu_seqlens, max_seqlen, scale):
    """Corrected CroissAttn for packed, independent causal documents."""
    if q.dim() != 3 or k.shape != q.shape or v.shape != q.shape:
        raise ValueError("varlen CroissAttn expects q/k/v with shape [T, H, D]")
    if cu_seqlens.dtype != torch.int32:
        cu_seqlens = cu_seqlens.to(torch.int32)
    lengths = (cu_seqlens[1:] - cu_seqlens[:-1]).to(torch.int32)
    batch = int(lengths.numel())
    total, heads, dim = q.shape
    if int(cu_seqlens[-1]) != total:
        raise ValueError("cu_seqlens[-1] must equal packed token count")
    if max_seqlen < 1 or bool((lengths > max_seqlen).any()):
        raise ValueError("max_seqlen is smaller than a packed document")
    row = torch.arange(total, device=q.device, dtype=torch.long)
    doc = torch.bucketize(row, cu_seqlens[1:-1].to(torch.long), right=True)
    pos = row - cu_seqlens[:-1].to(torch.long)[doc]
    dst = doc * int(max_seqlen) + pos
    padded_shape = (batch * int(max_seqlen), heads, dim)
    q_pad = q.new_zeros(padded_shape).index_copy(0, dst, q)
    k_pad = k.new_zeros(padded_shape).index_copy(0, dst, k)
    v_pad = v.new_zeros(padded_shape).index_copy(0, dst, v)
    q_pad = q_pad.view(batch, max_seqlen, heads, dim).transpose(1, 2)
    k_pad = k_pad.view(batch, max_seqlen, heads, dim).transpose(1, 2)
    v_pad = v_pad.view(batch, max_seqlen, heads, dim).transpose(1, 2)
    out_pad = _CroissAttn.apply(
        q_pad.reshape(batch * heads, max_seqlen, dim),
        k_pad.reshape(batch * heads, max_seqlen, dim),
        v_pad.reshape(batch * heads, max_seqlen, dim),
        scale,
        lengths,
    )
    out_pad = out_pad.view(batch, heads, max_seqlen, dim).transpose(1, 2)
    return out_pad.reshape(batch * max_seqlen, heads, dim).index_select(0, dst)


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

        self.Wqkv = nn.Linear(d_model, qkv_dim, bias=qkv_proj_bias, **factory_kwargs)
        self.inner_attn = FlashCausalSelfAttention(
            softmax_scale=softmax_scale,
            window_size=window_size,
        )
        self.out_proj = nn.Linear(
            d_model, d_model, bias=out_proj_bias, **factory_kwargs
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
            if (inference_params is None and self.croissattn_aux
                    and self.inner_attn._window() == (-1, -1)):
                # Padded qkv is [B,S,3,H,D], while packed varlen qkv is
                # [T,3,H,D].
                q, k, v = qkv.unbind(dim=1 if cu_seqlens is not None else 2)
                if cu_seqlens is None:
                    context = croissattn_fp8(
                        q.permute(0, 2, 1, 3),
                        k.permute(0, 2, 1, 3),
                        v.permute(0, 2, 1, 3),
                        self.softmax_scale or (self.head_dim ** -0.5),
                    ).permute(0, 2, 1, 3)
                else:
                    context = croissattn_varlen_fp8(
                        q, k, v, cu_seqlens, max_seqlen,
                        self.softmax_scale or (self.head_dim ** -0.5),
                    )
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
        out = self.out_proj(rearrange(context, "... h d -> ... (h d)"))
        return out

    def step(self, x, inference_params, loop_idx=0):
        # Recurrent inner-loop execution intentionally reuses the same cache
        # slot and sequence offset. The later loop refines the current token's
        # K/V in place instead of appending a duplicate position.
        return self.forward(x, inference_params=inference_params, loop_idx=loop_idx)
