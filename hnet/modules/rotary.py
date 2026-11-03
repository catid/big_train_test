# Copyright (c) 2023, Tri Dao.

from typing import Optional, Tuple, Union

import torch
from einops import rearrange, repeat
from flash_attn.ops.triton.rotary import apply_rotary


def _lerope_grad_neox_impl(y, dy, positions, freq_scale):
    """FP32 LeRoPE log-frequency reduction for the canonical NeoX pairing.

    Keeping this contraction in one compiled kernel avoids the separate
    elementwise dangle, cast, broadcast, and reduction launches that otherwise
    occur once per SWA layer during backward.
    """
    bands = y.shape[-1] // 2
    y1 = y[..., :bands]
    y2 = y[..., bands : 2 * bands]
    dy1 = dy[..., :bands]
    dy2 = dy[..., bands : 2 * bands]
    dangle = (-dy1 * y2 + dy2 * y1).float()
    if dangle.ndim == 4:
        weighted = dangle * positions.to(dangle.dtype).view(1, -1, 1, 1)
        return (weighted * freq_scale.view(1, 1, 1, -1)).sum(dim=(0, 1, 2))
    weighted = dangle * positions.to(dangle.dtype).view(-1, 1, 1)
    return (weighted * freq_scale.view(1, 1, -1)).sum(dim=(0, 1))


_lerope_grad_neox = torch.compile(
    _lerope_grad_neox_impl,
    fullgraph=True,
    dynamic=True,
    options={"triton.cudagraphs": False},
)



def rotate_half(x, interleaved=False):
    if not interleaved:
        x1, x2 = x.chunk(2, dim=-1)
        return torch.cat((-x2, x1), dim=-1)
    else:
        x1, x2 = x[..., ::2], x[..., 1::2]
        return rearrange(
            torch.stack((-x2, x1), dim=-1), "... d two -> ... (d two)", two=2
        )


def apply_rotary_emb_torch(x, cos, sin, interleaved=False):
    """
    x: (batch_size, seqlen, nheads, headdim)
    cos, sin: (seqlen, rotary_dim / 2) or (batch_size, seqlen, rotary_dim / 2)
    """
    ro_dim = cos.shape[-1] * 2
    assert ro_dim <= x.shape[-1]
    cos = repeat(
        cos, "... d -> ... 1 (2 d)" if not interleaved else "... d -> ... 1 (d 2)"
    )
    sin = repeat(
        sin, "... d -> ... 1 (2 d)" if not interleaved else "... d -> ... 1 (d 2)"
    )
    return torch.cat(
        [
            x[..., :ro_dim] * cos + rotate_half(x[..., :ro_dim], interleaved) * sin,
            x[..., ro_dim:],
        ],
        dim=-1,
    )


class ApplyRotaryEmb(torch.autograd.Function):
    @staticmethod
    def forward(
        ctx,
        x,
        cos,
        sin,
        interleaved=False,
        inplace=False,
        seqlen_offsets: Union[int, torch.Tensor] = 0,
        cu_seqlens: Optional[torch.Tensor] = None,
        max_seqlen: Optional[int] = None,
    ):
        out = apply_rotary(
            x,
            cos,
            sin,
            seqlen_offsets=seqlen_offsets,
            cu_seqlens=cu_seqlens,
            max_seqlen=max_seqlen,
            interleaved=interleaved,
            inplace=inplace,
        )
        if isinstance(seqlen_offsets, int):
            ctx.save_for_backward(
                cos, sin, cu_seqlens
            )  # Can't save int with save_for_backward
            ctx.seqlen_offsets = seqlen_offsets
        else:
            ctx.save_for_backward(cos, sin, cu_seqlens, seqlen_offsets)
            ctx.seqlen_offsets = None
        ctx.interleaved = interleaved
        ctx.inplace = inplace
        ctx.max_seqlen = max_seqlen
        return out if not inplace else x

    @staticmethod
    def backward(ctx, do):
        seqlen_offsets = ctx.seqlen_offsets
        if seqlen_offsets is None:
            cos, sin, cu_seqlens, seqlen_offsets = ctx.saved_tensors
        else:
            cos, sin, cu_seqlens = ctx.saved_tensors
        # TD [2023-09-02]: For some reason Triton (2.0.0.post1) errors with
        # "[CUDA]: invalid device context", and cloning makes it work. Idk why. Triton 2.1.0 works.
        if not ctx.interleaved and not ctx.inplace:
            do = do.clone()
        dx = apply_rotary(
            do,
            cos,
            sin,
            seqlen_offsets=seqlen_offsets,
            cu_seqlens=cu_seqlens,
            max_seqlen=ctx.max_seqlen,
            interleaved=ctx.interleaved,
            inplace=ctx.inplace,
            conjugate=True,
        )
        return dx, None, None, None, None, None, None, None


def apply_rotary_emb(
    x,
    cos,
    sin,
    interleaved=False,
    inplace=False,
    seqlen_offsets: Union[int, torch.Tensor] = 0,
    cu_seqlens: Optional[torch.Tensor] = None,
    max_seqlen: Optional[int] = None,
):
    """
    Arguments:
        x: (batch_size, seqlen, nheads, headdim) if cu_seqlens is None
            else (total_seqlen, nheads, headdim)
        cos, sin: (seqlen_rotary, rotary_dim / 2)
        interleaved: if True, rotate pairs of even and odd dimensions (GPT-J style) instead
            of 1st half and 2nd half (GPT-NeoX style).
        inplace: if True, apply rotary embedding in-place.
        seqlen_offsets: (batch_size,) or int. Each sequence in x is shifted by this amount.
            Most commonly used in inference when we have KV cache.
        cu_seqlens: (batch + 1,) or None
        max_seqlen: int
    Return:
        out: (batch_size, seqlen, nheads, headdim) if cu_seqlens is None
            else (total_seqlen, nheads, headdim)
    rotary_dim must be <= headdim
    Apply rotary embedding to the first rotary_dim of x.
    """
    return ApplyRotaryEmb.apply(
        x, cos, sin, interleaved, inplace, seqlen_offsets, cu_seqlens, max_seqlen
    )


# For backward compatibility
apply_rotary_emb_func = apply_rotary_emb


class ApplyRotaryEmbQKV_(torch.autograd.Function):
    @staticmethod
    def forward(
        ctx,
        qkv,
        cos,
        sin,
        cos_k=None,
        sin_k=None,
        interleaved=False,
        seqlen_offsets: Union[int, torch.Tensor] = 0,
        num_heads_q: Union[int] = None,
    ):
        if cos_k is None and sin_k is None and qkv.is_contiguous():
            # Call 1 kernel instead of 2 kernels
            # We need qkv to be contiguous so that when we reshape to combine (3, nheads)
            # dimensions, we get the same tensor
            if qkv.dim() == 5:
                batch, seqlen, three, nheads, headdim = qkv.shape
                assert three == 3
                # qk = rearrange(qkv[:, :, :2], "b s t h d -> b s (t h) d")
                qk = qkv[:, :, :2].reshape(batch, seqlen, -1, headdim)
            else:
                assert qkv.dim() == 4
                assert num_heads_q is not None
                num_heads_k = (qkv.shape[2] - num_heads_q) // 2
                assert qkv.shape[2] == num_heads_q + 2 * num_heads_k
                qk = qkv[:, :, : num_heads_q + num_heads_k]
            apply_rotary(
                qk,
                cos,
                sin,
                seqlen_offsets=seqlen_offsets,
                interleaved=interleaved,
                inplace=True,
            )
        else:
            cos_k = cos if cos_k is None else cos_k
            sin_k = sin if sin_k is None else sin_k
            if qkv.dim() == 5:
                q, k = qkv[:, :, 0], qkv[:, :, 1]
            else:
                assert qkv.dim() == 4
                assert num_heads_q is not None
                num_heads_k = (qkv.shape[2] - num_heads_q) // 2
                assert qkv.shape[2] == num_heads_q + 2 * num_heads_k
                q, k = (
                    qkv[:, :, :num_heads_q],
                    qkv[:, :, num_heads_q : num_heads_q + num_heads_k],
                )
            apply_rotary(
                q, cos, sin, seqlen_offsets, interleaved=interleaved, inplace=True
            )
            apply_rotary(
                k, cos_k, sin_k, seqlen_offsets, interleaved=interleaved, inplace=True
            )
            ctx.save_for_backward(cos, sin, cos_k, sin_k)
        if isinstance(seqlen_offsets, int):
            ctx.save_for_backward(cos, sin, cos_k, sin_k)
            ctx.seqlen_offsets = seqlen_offsets
        else:
            ctx.save_for_backward(cos, sin, cos_k, sin_k, seqlen_offsets)
            ctx.seqlen_offsets = None
        ctx.interleaved = interleaved
        ctx.num_heads_q = num_heads_q
        return qkv

    @staticmethod
    def backward(ctx, dqkv):
        seqlen_offsets = ctx.seqlen_offsets
        if seqlen_offsets is None:
            cos, sin, cos_k, sin_k, seqlen_offsets = ctx.saved_tensors
        else:
            cos, sin, cos_k, sin_k = ctx.saved_tensors
        if cos_k is None and sin_k is None and dqkv.is_contiguous():
            # Call 1 kernel instead of 2 kernels
            # We need dqkv to be contiguous so that when we reshape to combine (3, nheads)
            # dimensions, we get the same tensor
            if dqkv.dim() == 5:
                dqk = rearrange(dqkv[:, :, :2], "b s t h d -> b s (t h) d")
            else:
                assert dqkv.dim() == 4
                assert ctx.num_heads_q is not None
                num_heads_k = (dqkv.shape[2] - ctx.num_heads_q) // 2
                assert dqkv.shape[2] == ctx.num_heads_q + 2 * num_heads_k
                dqk = dqkv[:, :, : ctx.num_heads_q + num_heads_k]
            apply_rotary(
                dqk,
                cos,
                sin,
                seqlen_offsets=seqlen_offsets,
                interleaved=ctx.interleaved,
                inplace=True,
                conjugate=True,
            )
        else:
            cos_k = cos if cos_k is None else cos_k
            sin_k = sin if sin_k is None else sin_k
            if dqkv.dim() == 5:
                dq, dk = dqkv[:, :, 0], dqkv[:, :, 1]
            else:
                assert dqkv.dim() == 4
                assert ctx.num_heads_q is not None
                num_heads_k = (dqkv.shape[2] - ctx.num_heads_q) // 2
                assert dqkv.shape[2] == ctx.num_heads_q + 2 * num_heads_k
                dq = dqkv[:, :, : ctx.num_heads_q]
                dk = dqkv[:, :, ctx.num_heads_q : ctx.num_heads_q + num_heads_k]
            apply_rotary(
                dq,
                cos,
                sin,
                seqlen_offsets,
                interleaved=ctx.interleaved,
                inplace=True,
                conjugate=True,
            )
            apply_rotary(
                dk,
                cos_k,
                sin_k,
                seqlen_offsets,
                interleaved=ctx.interleaved,
                inplace=True,
                conjugate=True,
            )
        return dqkv, None, None, None, None, None, None, None


def apply_rotary_emb_qkv_(
    qkv,
    cos,
    sin,
    cos_k=None,
    sin_k=None,
    interleaved=False,
    seqlen_offsets: Union[int, torch.Tensor] = 0,
    num_heads_q: Optional[int] = None,
):
    """
    Arguments:
        qkv: (batch_size, seqlen, 3, nheads, headdim) or (batch_size, seqlen, num_heads_q + 2 * num_heads_k, headdim).
            If qkv has shape (batch_size, seqlen, num_heads_q + 2 * num_heads_k, headdim) (e.g. MQA / GQA),
            then num_heads_q must be provided.
        cos, sin: (seqlen, rotary_dim / 2)
        cos_k, sin_k: (seqlen, rotary_dim / 2), optional
        interleaved: if True, rotate pairs of even and odd dimensions (GPT-J style) instead of
            1st half and 2nd half (GPT-NeoX style).
        seqlen_offsets: (batch_size,) or int. Each sequence in Q and K is shifted by this amount.
            Most commonly used in inference when we have KV cache.
    Return:
        qkv: (batch_size, seqlen, 3, nheads, headdim) or (batch_size, seqlen, num_heads_q + 2 * num_heads_k, headdim)
    rotary_dim must be <= headdim
    Apply rotary embedding *inplace* to the first rotary_dim of Q and K.
    """
    return ApplyRotaryEmbQKV_.apply(
        qkv, cos, sin, cos_k, sin_k, interleaved, seqlen_offsets, num_heads_q
    )


class ApplyLearnableRotaryEmbQKV_(torch.autograd.Function):
    """FlashAttention rotary kernel with a differentiable frequency table.

    ``flash_attn``'s normal rotary autograd function deliberately returns no
    gradient for its cosine/sine tables.  That is correct for ordinary RoPE,
    whose table is a constant, but would silently turn LeRoPE's frequency
    parameters into dead parameters.  This wrapper keeps the same fused
    in-place forward/backward kernel and supplies the table gradients from the
    rotated output.  For a rotation ``y = R(theta)x``,
    ``d y / d theta = J y``; therefore the angle derivative needs only the
    output saved by the in-place kernel, avoiding a second copy of the input.
    """

    @staticmethod
    def _positions(qkv, cu_seqlens, max_seqlen, seqlen_offsets):
        if cu_seqlens is not None:
            lengths = cu_seqlens[1:] - cu_seqlens[:-1]
            # ``output_size`` avoids the host synchronization that the
            # default repeat_interleave path performs for CUDA tensors.
            starts = torch.repeat_interleave(
                cu_seqlens[:-1], lengths, output_size=qkv.shape[0]
            )
            return torch.arange(
                qkv.shape[0], device=qkv.device, dtype=torch.long
            ) - starts
        seqlen = qkv.shape[1]
        local = torch.arange(seqlen, device=qkv.device, dtype=torch.long)
        if isinstance(seqlen_offsets, int):
            return local + int(seqlen_offsets)
        # Tensor offsets are used only by the cache/decode API and have one
        # offset per batch row.  The returned shape broadcasts over heads.
        return local[None, :] + seqlen_offsets.to(dtype=torch.long)[:, None]

    @staticmethod
    def forward(
        ctx,
        qkv,
        cos,
        sin,
        log_scale=None,
        theta=None,
        interleaved=False,
        seqlen_offsets=0,
        cu_seqlens=None,
        max_seqlen=None,
        num_heads_q=None,
        positions=None,
    ):
        packed = cu_seqlens is not None
        if qkv.dim() == 5:
            qk = qkv[:, :, :2].reshape(qkv.shape[0], qkv.shape[1], -1, qkv.shape[-1])
        elif qkv.dim() == 3:
            qk = qkv
        elif packed:
            qk = qkv[:, :2].reshape(qkv.shape[0], -1, qkv.shape[-1])
        else:
            if num_heads_q is None:
                # A standalone Q tensor (used by the VHA projection path) has
                # no packed K/V heads to split.
                qk = qkv
            else:
                num_heads_k = (qkv.shape[2] - num_heads_q) // 2
                qk = qkv[:, :, : num_heads_q + num_heads_k]
        apply_rotary(
            qk,
            cos,
            sin,
            seqlen_offsets=seqlen_offsets,
            cu_seqlens=cu_seqlens,
            max_seqlen=max_seqlen,
            interleaved=interleaved,
            inplace=True,
        )
        if positions is None:
            positions = ApplyLearnableRotaryEmbQKV_._positions(
                qkv, cu_seqlens, max_seqlen, seqlen_offsets
            )
        elif positions.device != qkv.device or positions.dtype != torch.long:
            raise ValueError("cached rotary positions must be device-local int64")
        # Packed QKV is flattened across documents.  Its position vector must
        # therefore contain one document-local index per packed token; a
        # padded [B, S] cache (or a cache sized from the head dimension) would
        # broadcast incorrectly and fail only when autograd gathers the
        # frequency gradients.  Keep this check here as a guard for callers
        # that invoke the low-level autograd function directly, in addition to
        # the high-level RotaryEmbedding path which deliberately does not pass
        # its padded static cache for packed inputs.
        if cu_seqlens is not None:
            if positions.ndim != 1 or positions.shape[0] != qkv.shape[0]:
                raise ValueError(
                    "packed rotary positions must be a 1D document-local vector "
                    f"with length {qkv.shape[0]}, got {tuple(positions.shape)}"
                )
        # Save the post-rotation tensor by reference.  The forward kernel is
        # in-place, so this is the output ``y`` and costs no extra activation
        # storage.  It is enough to reconstruct ``x`` via R(-theta)y.
        # ``cos``/``sin`` are detached in the direct-gradient path below.  We
        # save the shared log-frequency parameter and base frequencies so the
        # backward can reduce straight to d(log_scale), avoiding a full
        # dcos/dsin tensor and index_add for every SWA layer.
        if log_scale is None or theta is None:
            ctx.save_for_backward(qkv, cos, sin, positions)
            ctx.direct_frequency_grad = False
        else:
            ctx.save_for_backward(qkv, cos, sin, positions, log_scale, theta)
            ctx.direct_frequency_grad = True
        ctx.interleaved = bool(interleaved)
        ctx.cu_seqlens = cu_seqlens
        ctx.max_seqlen = max_seqlen
        ctx.seqlen_offsets = seqlen_offsets
        ctx.num_heads_q = num_heads_q
        return qkv

    @staticmethod
    def backward(ctx, dqkv):
        if ctx.direct_frequency_grad:
            qkv, cos, sin, positions, log_scale, theta = ctx.saved_tensors
        else:
            qkv, cos, sin, positions = ctx.saved_tensors
        seqlen_offsets = ctx.seqlen_offsets
        packed = ctx.cu_seqlens is not None
        if qkv.dim() == 5:
            y = qkv[:, :, :2].reshape(qkv.shape[0], qkv.shape[1], -1, qkv.shape[-1])
            dy = dqkv[:, :, :2].reshape(dqkv.shape[0], dqkv.shape[1], -1, dqkv.shape[-1])
        elif qkv.dim() == 3:
            y, dy = qkv, dqkv
        elif packed:
            y = qkv[:, :2].reshape(qkv.shape[0], -1, qkv.shape[-1])
            dy = dqkv[:, :2].reshape(dqkv.shape[0], -1, dqkv.shape[-1])
        else:
            if ctx.num_heads_q is None:
                y, dy = qkv, dqkv
            else:
                num_heads_k = (qkv.shape[2] - ctx.num_heads_q) // 2
                y = qkv[:, :, : ctx.num_heads_q + num_heads_k]
                dy = dqkv[:, :, : ctx.num_heads_q + num_heads_k]
        rotary_dim = cos.shape[-1] * 2
        # The direct LeRoPE path only needs the angle derivative.  Computing
        # it from the rotated output (dy1 * -y2 + dy2 * y1) avoids rebuilding
        # the pre-rotation tensor and keeps the large activation-sized
        # contractions in BF16.  The final tiny frequency vector is reduced
        # in FP32.  The legacy cosine-table-gradient path below remains
        # available for callers that do not provide log_scale/theta.
        if ctx.direct_frequency_grad:
            y = y[..., :rotary_dim]
            dy = dy[..., :rotary_dim]
            freq_scale = (theta * log_scale.exp()).to(dtype=torch.float32)
            if ctx.interleaved:
                raise RuntimeError("compiled LeRoPE path requires NeoX pairing")
            if positions.ndim != 1:
                # Cached decode is outside the canonical training graph; keep
                # the explicit fallback for its per-row offsets.
                y1, y2 = y[..., : rotary_dim // 2], y[..., rotary_dim // 2 :]
                dy1, dy2 = dy[..., : rotary_dim // 2], dy[..., rotary_dim // 2 :]
                dangle = (-dy1 * y2 + dy2 * y1).float()
                pos = positions.to(dangle.dtype).unsqueeze(-1).unsqueeze(-1)
                grad_log_scale = (dangle * pos * freq_scale.view(1, 1, 1, -1)).sum(
                    dim=tuple(range(dangle.ndim - 1))
                )
            else:
                grad_log_scale = _lerope_grad_neox(y, dy, positions, freq_scale)
        else:
            y = y[..., :rotary_dim].float()
            dy = dy[..., :rotary_dim].float()
        if not ctx.direct_frequency_grad:
            if positions.ndim == 1:
                c = cos.index_select(0, positions).to(dtype=y.dtype)
                s = sin.index_select(0, positions).to(dtype=y.dtype)
                view_shape = (1, positions.shape[0], 1, -1)
                if y.dim() == 3:
                    view_shape = (positions.shape[0], 1, -1)
                c, s = c.view(view_shape), s.view(view_shape)
            else:
                flat_pos = positions.reshape(-1)
                c = cos.index_select(0, flat_pos).view(*positions.shape, -1).to(dtype=y.dtype)
                s = sin.index_select(0, flat_pos).view(*positions.shape, -1).to(dtype=y.dtype)
                c, s = c.unsqueeze(-2), s.unsqueeze(-2)
            if ctx.interleaved:
                y1, y2 = y[..., 0::2], y[..., 1::2]
                dy1, dy2 = dy[..., 0::2], dy[..., 1::2]
                x1 = y1 * c + y2 * s
                x2 = -y1 * s + y2 * c
                dcos = dy1 * x1 + dy2 * x2
                dsin = -dy1 * x2 + dy2 * x1
            else:
                y1, y2 = y[..., :rotary_dim // 2], y[..., rotary_dim // 2:]
                dy1, dy2 = dy[..., :rotary_dim // 2], dy[..., rotary_dim // 2:]
                x1 = y1 * c + y2 * s
                x2 = -y1 * s + y2 * c
                dcos = dy1 * x1 + dy2 * x2
                dsin = -dy1 * x2 + dy2 * x1
            if positions.ndim == 1:
                if y.dim() == 4:
                    dcos = dcos.sum(dim=(0, 2))
                    dsin = dsin.sum(dim=(0, 2))
                else:
                    dcos = dcos.sum(dim=1)
                    dsin = dsin.sum(dim=1)
                grad_cos = torch.zeros(cos.shape, device=cos.device, dtype=dcos.dtype).index_add(0, positions, dcos).to(dtype=cos.dtype)
                grad_sin = torch.zeros(sin.shape, device=sin.device, dtype=dsin.dtype).index_add(0, positions, dsin).to(dtype=sin.dtype)
            else:
                grad_cos = torch.zeros(cos.shape, device=cos.device, dtype=dcos.dtype).index_add(0, positions.reshape(-1), dcos.sum(dim=2).reshape(-1, rotary_dim // 2)).to(dtype=cos.dtype)
                grad_sin = torch.zeros(sin.shape, device=sin.device, dtype=dsin.dtype).index_add(0, positions.reshape(-1), dsin.sum(dim=2).reshape(-1, rotary_dim // 2)).to(dtype=sin.dtype)
        # FlashAttention's rotary backward mutates the Q/K gradient in-place.
        # Keep that fast path here too: cloning the complete [B,S,3,H,D] tensor
        # for every SWA layer costs a full activation-sized copy and showed up
        # as ~40 ms/step at the canonical 8-GPU shape.  Autograd gives us an
        # ephemeral gradient buffer, so it is safe to rotate its Q/K view and
        # leave V untouched.
        dq = dqkv
        if dq.dim() == 5:
            dqk = dq[:, :, :2].reshape(dq.shape[0], dq.shape[1], -1, dq.shape[-1])
        elif dq.dim() == 3:
            dqk = dq
        elif ctx.cu_seqlens is not None:
            dqk = dq[:, :2].reshape(dq.shape[0], -1, dq.shape[-1])
        else:
            if ctx.num_heads_q is None:
                dqk = dq
            else:
                num_heads_k = (dq.shape[2] - ctx.num_heads_q) // 2
                dqk = dq[:, :, : ctx.num_heads_q + num_heads_k]
        apply_rotary(
            dqk,
            cos,
            sin,
            seqlen_offsets=seqlen_offsets,
            cu_seqlens=ctx.cu_seqlens,
            max_seqlen=ctx.max_seqlen,
            interleaved=ctx.interleaved,
            inplace=True,
            conjugate=True,
        )
        if ctx.direct_frequency_grad:
            return dq, None, None, grad_log_scale, None, None, None, None, None, None, None
        return dq, grad_cos, grad_sin, None, None, None, None, None, None, None, None


def apply_learnable_rotary_emb_qkv_(
    qkv,
    cos,
    sin,
    interleaved=False,
    seqlen_offsets=0,
    cu_seqlens=None,
    max_seqlen=None,
    num_heads_q=None,
    log_scale=None,
    theta=None,
    positions=None,
):
    return ApplyLearnableRotaryEmbQKV_.apply(
        qkv,
        cos,
        sin,
        log_scale,
        theta,
        interleaved,
        seqlen_offsets,
        cu_seqlens,
        max_seqlen,
        num_heads_q,
        positions,
    )


class ApplyRotaryEmbKV_(torch.autograd.Function):
    @staticmethod
    def forward(
        ctx,
        kv,
        cos,
        sin,
        interleaved=False,
        seqlen_offsets: Union[int, torch.Tensor] = 0,
    ):
        batch, seqlen, two, nheads, headdim = kv.shape
        assert two == 2
        k = kv[:, :, 0]
        apply_rotary(
            k,
            cos,
            sin,
            seqlen_offsets=seqlen_offsets,
            interleaved=interleaved,
            inplace=True,
        )
        if isinstance(seqlen_offsets, int):
            ctx.save_for_backward(cos, sin)  # Can't save int with save_for_backward
            ctx.seqlen_offsets = seqlen_offsets
        else:
            ctx.save_for_backward(cos, sin, seqlen_offsets)
            ctx.seqlen_offsets = None
        ctx.interleaved = interleaved
        return kv

    @staticmethod
    def backward(ctx, dkv):
        seqlen_offsets = ctx.seqlen_offsets
        if seqlen_offsets is None:
            cos, sin, seqlen_offsets = ctx.saved_tensors
        else:
            cos, sin = ctx.saved_tensors
        apply_rotary(
            dkv[:, :, 0],
            cos,
            sin,
            seqlen_offsets=seqlen_offsets,
            interleaved=ctx.interleaved,
            inplace=True,
            conjugate=True,
        )
        return dkv, None, None, None, None


apply_rotary_emb_kv_ = ApplyRotaryEmbKV_.apply


def apply_rotary_emb_kv_(
    kv,
    cos,
    sin,
    interleaved=False,
    seqlen_offsets: Union[int, torch.Tensor] = 0,
):
    """
    Arguments:
        kv: (batch_size, seqlen, 2, nheads, headdim)
        cos, sin: (seqlen, rotary_dim / 2)
        interleaved: if True, rotate pairs of even and odd dimensions (GPT-J style) instead of
            1st half and 2nd half (GPT-NeoX style).
        seqlen_offsets: (batch_size,) or int. Each sequence in Q and K is shifted by this amount.
            Most commonly used in inference when we have KV cache.
    Return:
        kv: (batch_size, seqlen, 2, nheads, headdim)
    rotary_dim must be <= headdim
    Apply rotary embedding *inplace* to the first rotary_dim of K.
    """
    return ApplyRotaryEmbKV_.apply(kv, cos, sin, interleaved, seqlen_offsets)


class LeRoPEFrequencies(torch.nn.Module):
    """One shared learnable log-frequency scale per rotary band.

    The parameter is intentionally FP32 even when the model activations are
    BF16.  This keeps the frequency update well-conditioned at 4K context and
    lets the angle table be evaluated in FP32 before it is narrowed for the
    FlashAttention rotary kernel.
    """

    def __init__(self, dim: int, base: float = 10000.0, device=None):
        super().__init__()
        if int(dim) <= 0 or int(dim) % 2:
            raise ValueError("LeRoPE dimension must be a positive even number")
        self.dim = int(dim)
        self.base = float(base)
        bands = torch.arange(0, self.dim, 2, device=device, dtype=torch.float32)
        self.register_buffer(
            "theta", self.base ** (-bands / float(self.dim)), persistent=False
        )
        self.log_scale = torch.nn.Parameter(
            torch.zeros(self.dim // 2, device=device, dtype=torch.float32)
        )
        # The optimizer and trainer use this marker to keep LeRoPE out of
        # weight decay and to apply its separate gradient clipping norm.
        self.log_scale._lerope_frequency = True
        self.log_scale._no_weight_decay = True
        self.log_scale._no_soda = True
        self._table_cache = {}

    def begin_forward(self):
        """Drop graph-connected tables from the previous model forward."""
        self._table_cache.clear()

    def frequencies(self, device=None):
        theta = self.theta if device is None else self.theta.to(device=device)
        return theta * self.log_scale.to(device=theta.device).exp()

    def table(self, seqlen: int, *, device, dtype):
        key = (int(seqlen), str(device), dtype)
        table = self._table_cache.get(key)
        if table is None:
            t = torch.arange(seqlen, device=device, dtype=torch.float32)
            angle = torch.outer(t, self.frequencies(device=device))
            table = (torch.cos(angle).to(dtype), torch.sin(angle).to(dtype))
            self._table_cache[key] = table
        return table


class RotaryEmbedding(torch.nn.Module):
    """
    The rotary position embeddings from RoFormer_ (Su et. al).
    A crucial insight from the method is that the query and keys are
    transformed by rotation matrices which depend on the relative positions.

    Other implementations are available in the Rotary Transformer repo_ and in
    GPT-NeoX_, GPT-NeoX was an inspiration

    .. _RoFormer: https://arxiv.org/abs/2104.09864
    .. _repo: https://github.com/ZhuiyiTechnology/roformer
    .. _GPT-NeoX: https://github.com/EleutherAI/gpt-neox

    If scale_base is not None, this implements XPos (Sun et al., https://arxiv.org/abs/2212.10554).
    A recommended value for scale_base is 512: https://github.com/HazyResearch/flash-attention/issues/96
    Reference: https://github.com/sunyt32/torchscale/blob/main/torchscale/component/xpos_relative_position.py
    """

    def __init__(
        self,
        dim: int,
        base=10000.0,
        interleaved=False,
        scale_base=None,
        pos_idx_in_fp32=True,
        device=None,
        lerope: Optional[LeRoPEFrequencies] = None,
    ):
        """
        interleaved: if True, rotate pairs of even and odd dimensions (GPT-J style) instead
            of 1st half and 2nd half (GPT-NeoX style).
        pos_idx_in_fp32: if True, the position indices [0.0, ..., seqlen - 1] are in fp32,
            otherwise they might be in lower precision.
            This option was added because previously (before 2023-07-02), when we construct
            the position indices, we use the dtype of self.inv_freq. In most cases this would
            be fp32, but if the model is trained in pure bf16 (not mixed precision), then
            self.inv_freq would be bf16, and the position indices are also in bf16.
            Because of the limited precision of bf16 (e.g. 1995.0 is rounded to 2000.0), the
            embeddings for some positions will coincide.
            To maintain compatibility with models previously trained in pure bf16,
            we add this option.
        """
        super().__init__()
        self.dim = dim
        self.base = float(base)
        self.pos_idx_in_fp32 = pos_idx_in_fp32
        self.lerope = lerope
        if self.lerope is not None and int(self.lerope.dim) != int(dim):
            raise ValueError(
                f"shared LeRoPE dimension {self.lerope.dim} does not match rotary dim {dim}"
            )
        # Generate and save the inverse frequency buffer (non trainable)
        inv_freq = self._compute_inv_freq(device)
        self.register_buffer("inv_freq", inv_freq, persistent=False)
        self.interleaved = interleaved
        self.scale_base = scale_base
        scale = (
            (torch.arange(0, dim, 2, device=device, dtype=torch.float32) + 0.4 * dim)
            / (1.4 * dim)
            if scale_base is not None
            else None
        )
        self.register_buffer("scale", scale, persistent=False)

        self._seq_len_cached = 0
        self._cos_cached = None
        self._sin_cached = None
        self._cos_k_cached = None
        self._sin_k_cached = None
        # Static padded training repeatedly rotates the same sequence lengths.
        # Keep the position indices as persistent device tensors so every SWA
        # layer reuses them instead of launching a fresh arange on each graph
        # replay. Packed sequences and tensor-offset decode retain the dynamic
        # path below because their positions depend on document boundaries or
        # cache state.
        self._position_cache = {}

    def _compute_inv_freq(self, device=None):
        return 1.0 / (
            self.base
            ** (
                torch.arange(0, self.dim, 2, device=device, dtype=torch.float32)
                / self.dim
            )
        )

    def _update_cos_sin_cache(self, seqlen, device=None, dtype=None):
        # LeRoPE's table is owned by the model-level shared frequency module,
        # rather than by each RotaryEmbedding instance.  Always resolve it
        # through that owner here as well (the inference/cache API can call
        # this helper directly); otherwise a same-length cached table could
        # retain a graph from an earlier forward or stale log-scale values.
        if self.lerope is not None:
            self._seq_len_cached = seqlen
            self._cos_cached, self._sin_cached = self.lerope.table(
                seqlen, device=device, dtype=dtype
            )
            return
        # Reset the tables if the sequence length has changed,
        # if we're on a new device (possibly due to tracing for instance),
        # or if we're switching from inference mode to training
        if (
            seqlen > self._seq_len_cached
            or self._cos_cached is None
            or self._cos_cached.device != device
            or self._cos_cached.dtype != dtype
            or (self.training and self._cos_cached.is_inference())
        ):
            self._seq_len_cached = seqlen
            # We want fp32 here, not self.inv_freq.dtype, since the model could be loaded in bf16
            # And the output of arange can be quite large, so bf16 would lose a lot of precision.
            # However, for compatibility reason, we add an option to use the dtype of self.inv_freq.
            if self.pos_idx_in_fp32:
                t = torch.arange(seqlen, device=device, dtype=torch.float32)
                # We want fp32 here as well since inv_freq will be multiplied with t, and the output
                # will be large. Having it in bf16 will lose a lot of precision and cause the
                # cos & sin output to change significantly.
                # We want to recompute self.inv_freq if it was not loaded in fp32
                if self.inv_freq.dtype != torch.float32:
                    inv_freq = self._compute_inv_freq(device=device)
                else:
                    inv_freq = self.inv_freq
            else:
                t = torch.arange(seqlen, device=device, dtype=self.inv_freq.dtype)
                inv_freq = self.inv_freq
            # Don't do einsum, it converts fp32 to fp16 under AMP
            # freqs = torch.einsum("i,j->ij", t, self.inv_freq)
            freqs = torch.outer(t, inv_freq)
            if self.scale is None:
                self._cos_cached = torch.cos(freqs).to(dtype)
                self._sin_cached = torch.sin(freqs).to(dtype)
            else:
                power = (
                    torch.arange(
                        seqlen, dtype=self.scale.dtype, device=self.scale.device
                    )
                    - seqlen // 2
                ) / self.scale_base
                scale = self.scale.to(device=power.device) ** rearrange(
                    power, "s -> s 1"
                )
                # We want the multiplication by scale to happen in fp32
                self._cos_cached = (torch.cos(freqs) * scale).to(dtype)
                self._sin_cached = (torch.sin(freqs) * scale).to(dtype)
                self._cos_k_cached = (torch.cos(freqs) / scale).to(dtype)
                self._sin_k_cached = (torch.sin(freqs) / scale).to(dtype)

    def _compute_lerope_cos_sin(self, seqlen, *, device, dtype):
        """Build the current shared frequency table in FP32 angle space."""
        return self.lerope.table(seqlen, device=device, dtype=dtype)

    def _lerope_cos_sin(self, seqlen, *, device, dtype):
        # Unlike ordinary RoPE, no Python-side cache is used.  CUDA graph
        # replay must observe the updated log_scale after every optimizer step.
        return self._compute_lerope_cos_sin(seqlen, device=device, dtype=dtype)

    def _static_positions(self, seqlen, *, device):
        key = (int(seqlen), device.type, device.index)
        positions = self._position_cache.get(key)
        if positions is None:
            positions = torch.arange(int(seqlen), device=device, dtype=torch.long)
            self._position_cache[key] = positions
        return positions

    def forward(
        self,
        qkv: torch.Tensor,
        kv: Optional[torch.Tensor] = None,
        seqlen_offset: Union[int, torch.Tensor] = 0,
        cu_seqlens: Optional[torch.Tensor] = None,
        max_seqlen: Optional[int] = None,
        num_heads_q: Optional[int] = None,
    ) -> Union[torch.Tensor, Tuple[torch.Tensor, torch.Tensor]]:
        """
        qkv: (batch, seqlen, 3, nheads, headdim) or (batch, seqlen, num_heads_q + 2 * num_heads_k, headdim)
            if kv is none, else it's just q of shape (batch, seqlen, nheads, headdim).
            If qkv has shape (batch, seqlen, num_heads_q + 2 * num_heads_k, headdim) (e.g. MQA / GQA),
            then num_heads_q must be provided.
        kv: (batch, seqlen, 2, nheads, headdim)
        seqlen_offset: (batch_size,) or int. Each sequence in x is shifted by this amount.
            Most commonly used in inference when we have KV cache.
            If it's a tensor of shape (batch_size,), then to update the cos / sin cache, one
            should pass in max_seqlen, which will update the cos / sin cache up to that length.
        Apply rotary embedding *inplace* to qkv and / or kv.
        """
        if cu_seqlens is not None:
            assert max_seqlen is not None
            assert self.scale is None
            if self.lerope is not None:
                rotary_cos, rotary_sin = self._lerope_cos_sin(
                    max_seqlen, device=qkv.device, dtype=qkv.dtype
                )
                if kv is not None:
                    raise RuntimeError(
                        "LeRoPE is only supported on the canonical SWA packed-QKV path; "
                        "VHA q/KV rotary is unsupported and must use NoPE"
                    )
                return apply_learnable_rotary_emb_qkv_(
                    qkv,
                    rotary_cos.detach(),
                    rotary_sin.detach(),
                    interleaved=self.interleaved,
                    seqlen_offsets=0,
                    cu_seqlens=cu_seqlens,
                    max_seqlen=max_seqlen,
                    num_heads_q=num_heads_q,
                    log_scale=self.lerope.log_scale,
                    theta=self.lerope.theta,
                    positions=None,
                )
            self._update_cos_sin_cache(max_seqlen, device=qkv.device, dtype=qkv.dtype)
            return apply_pad_packed_rotary_emb_qkv_(
                qkv,
                self._cos_cached,
                self._sin_cached,
                cu_seqlens,
                max_seqlen,
                interleaved=self.interleaved,
                seqlen_offsets=0,
                num_heads_q=num_heads_q,
            )

        seqlen = qkv.shape[1]
        if self.lerope is not None:
            rotary_len = (
                int(max_seqlen)
                if max_seqlen is not None
                else qkv.shape[1] + int(seqlen_offset)
                if isinstance(seqlen_offset, int)
                else qkv.shape[1]
            )
            rotary_cos, rotary_sin = self._lerope_cos_sin(
                rotary_len, device=qkv.device, dtype=qkv.dtype
            )
        elif max_seqlen is not None:
            self._update_cos_sin_cache(max_seqlen, device=qkv.device, dtype=qkv.dtype)
        elif isinstance(seqlen_offset, int):
            self._update_cos_sin_cache(
                seqlen + seqlen_offset, device=qkv.device, dtype=qkv.dtype
            )
        if kv is None:
            if self.lerope is not None:
                return apply_learnable_rotary_emb_qkv_(
                    qkv,
                    rotary_cos.detach(),
                    rotary_sin.detach(),
                    interleaved=self.interleaved,
                    seqlen_offsets=seqlen_offset,
                    num_heads_q=num_heads_q,
                    log_scale=self.lerope.log_scale,
                    theta=self.lerope.theta,
                    positions=(
                        self._static_positions(qkv.shape[1], device=qkv.device)
                        if isinstance(seqlen_offset, int) and seqlen_offset == 0
                        else None
                    ),
                )
            if self.scale is None:
                return apply_rotary_emb_qkv_(
                    qkv,
                    self._cos_cached,
                    self._sin_cached,
                    interleaved=self.interleaved,
                    seqlen_offsets=seqlen_offset,
                    num_heads_q=num_heads_q,
                )
            else:
                return apply_rotary_emb_qkv_(
                    qkv,
                    self._cos_cached,
                    self._sin_cached,
                    self._cos_k_cached,
                    self._sin_k_cached,
                    interleaved=self.interleaved,
                    seqlen_offsets=seqlen_offset,
                    num_heads_q=num_heads_q,
                )
        else:
            q = qkv
            if self.lerope is not None:
                raise RuntimeError(
                    "LeRoPE is only supported on the canonical SWA packed-QKV path; "
                    "VHA q/KV rotary is unsupported and must use NoPE"
                )
            else:
                q = apply_rotary_emb_func(
                    q,
                    self._cos_cached,
                    self._sin_cached,
                    interleaved=self.interleaved,
                    inplace=True,
                    seqlen_offsets=seqlen_offset,
                )
            if self.lerope is not None:
                return q, kv
            if self.scale is None:
                kv = apply_rotary_emb_kv_(
                    kv,
                    rotary_cos if self.lerope is not None else self._cos_cached,
                    rotary_sin if self.lerope is not None else self._sin_cached,
                    interleaved=self.interleaved,
                    seqlen_offsets=seqlen_offset,
                )
            else:
                kv = apply_rotary_emb_kv_(
                    kv,
                    self._cos_k_cached,
                    self._sin_k_cached,
                    interleaved=self.interleaved,
                    seqlen_offsets=seqlen_offset,
                )
            return q, kv


class ApplyPadPackedRotaryEmbQKV_(torch.autograd.Function):
    @staticmethod
    def forward(
        ctx,
        qkv,
        cos,
        sin,
        cu_seqlens,
        max_seqlen,
        interleaved=False,
        seqlen_offsets: Union[int, torch.Tensor] = 0,
        num_heads_q: Union[int] = None,
    ):
        assert qkv.is_contiguous()
        assert qkv.dim() == 4
        total_seqlen, three, nheads, headdim = qkv.shape
        assert three == 3
        # qk = rearrange(qkv[:, :2], "s t h d -> s (t h) d")
        qk = qkv[:, :2].reshape(total_seqlen, -1, headdim)

        apply_rotary(
            qk,
            cos,
            sin,
            seqlen_offsets=seqlen_offsets,
            cu_seqlens=cu_seqlens,
            max_seqlen=max_seqlen,
            interleaved=interleaved,
            inplace=True,
        )
        if isinstance(seqlen_offsets, int):
            ctx.save_for_backward(cos, sin, cu_seqlens)
            ctx.seqlen_offsets = seqlen_offsets
        else:
            ctx.save_for_backward(cos, sin, cu_seqlens, seqlen_offsets)
            ctx.seqlen_offsets = None
        ctx.interleaved = interleaved
        ctx.num_heads_q = num_heads_q
        ctx.max_seqlen = max_seqlen
        return qkv

    @staticmethod
    def backward(ctx, dqkv):
        assert dqkv.is_contiguous()
        seqlen_offsets = ctx.seqlen_offsets
        if seqlen_offsets is None:
            cos, sin, cu_seqlens, seqlen_offsets = ctx.saved_tensors
        else:
            cos, sin, cu_seqlens = ctx.saved_tensors
        dqk = rearrange(dqkv[:, :2], "s t h d -> s (t h) d")
        apply_rotary(
            dqk,
            cos,
            sin,
            seqlen_offsets=seqlen_offsets,
            cu_seqlens=cu_seqlens,
            max_seqlen=ctx.max_seqlen,
            interleaved=ctx.interleaved,
            inplace=True,
            conjugate=True,
        )
        return dqkv, None, None, None, None, None, None, None


def apply_pad_packed_rotary_emb_qkv_(
    qkv,
    cos,
    sin,
    cu_seqlens,
    max_seqlen,
    interleaved=False,
    seqlen_offsets: Union[int, torch.Tensor] = 0,
    num_heads_q: Optional[int] = None,
):
    """
    Arguments:
        qkv: (batch_size, seqlen, 3, nheads, headdim) or (batch_size, seqlen, num_heads_q + 2 * num_heads_k, headdim).
            If qkv has shape (batch_size, seqlen, num_heads_q + 2 * num_heads_k, headdim) (e.g. MQA / GQA),
            then num_heads_q must be provided.
        cos, sin: (seqlen, rotary_dim / 2)
        interleaved: if True, rotate pairs of even and odd dimensions (GPT-J style) instead of
            1st half and 2nd half (GPT-NeoX style).
        seqlen_offsets: (batch_size,) or int. Each sequence in Q and K is shifted by this amount.
            Most commonly used in inference when we have KV cache.
    Return:
        qkv: (batch_size, seqlen, 3, nheads, headdim) or (batch_size, seqlen, num_heads_q + 2 * num_heads_k, headdim)
    rotary_dim must be <= headdim
    Apply rotary embedding *inplace* to the first rotary_dim of Q and K.
    """
    return ApplyPadPackedRotaryEmbQKV_.apply(
        qkv, cos, sin, cu_seqlens, max_seqlen, interleaved, seqlen_offsets, num_heads_q
    )
