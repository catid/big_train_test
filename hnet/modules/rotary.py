# Copyright (c) 2023, Tri Dao.

from typing import Optional, Tuple, Union

import torch
from einops import rearrange, repeat
from flash_attn.ops.triton.rotary import apply_rotary


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


class LeRoPERegistry(torch.nn.Module):
    """Shared bank of learnable RoPE frequencies.

    LeRoPE uses one log-frequency scale per rotary channel, shared by every
    layer and head.  H-Net has different head/rotary widths at different
    hierarchy stages, so the bank is keyed by the rotary width and base.  A
    registry owned by the top-level model makes each key a single Parameter
    while still allowing the attention blocks to retain their normal
    ``rotary_emb`` interface.
    """

    def __init__(self):
        super().__init__()
        self._embeddings = torch.nn.ModuleDict()

    def get(
        self,
        dim: int,
        *,
        base: float = 10000.0,
        interleaved: bool = False,
        device=None,
    ) -> "RotaryEmbedding":
        key = f"d{int(dim)}_b{float(base):g}_i{int(bool(interleaved))}"
        if key not in self._embeddings:
            self._embeddings[key] = RotaryEmbedding(
                dim,
                base=base,
                interleaved=interleaved,
                device=device,
                learnable=True,
            )
        return self._embeddings[key]


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
        learnable=True,
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
        # Generate and save the inverse frequency buffer (non trainable)
        inv_freq = self._compute_inv_freq(device)
        self.register_buffer("inv_freq", inv_freq, persistent=False)
        # LeRoPE learns one log scale per frequency band.  A zero
        # initialization is exactly ordinary RoPE, while the log-space
        # parameterization keeps the frequencies positive and gives stable
        # optimization dynamics over long contexts.
        self.learnable = bool(learnable)
        if self.learnable:
            self.log_scale = torch.nn.Parameter(
                torch.zeros(dim // 2, device=device, dtype=torch.float32)
            )
            self.log_scale._no_weight_decay = True
            self.log_scale._no_soda = True
            self.log_scale._lerope_frequency = True
        else:
            self.register_parameter("log_scale", None)
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
        self._cache_log_scale_version = -1

    def _current_inv_freq(self, device=None):
        """Return learned frequencies in FP32 without mutating buffers."""
        if self.learnable:
            # Keep the multiplication in FP32 even when the model is BF16 or
            # FP8.  The resulting trigonometric tables are cast only after
            # angle evaluation.
            return self.inv_freq.to(device=device, dtype=torch.float32) * self.log_scale.exp()
        return self.inv_freq.to(device=device, dtype=torch.float32)

    @staticmethod
    def _rotate(x, cos, sin, rotary_dim, interleaved):
        """Differentiable LeRoPE rotation used by training and graph capture."""
        x_rot = x[..., :rotary_dim]
        x_tail = x[..., rotary_dim:]
        # Trigonometric values are computed in FP32 but applying them in the
        # activation dtype avoids widening the entire attention projection.
        cos = cos.to(dtype=x.dtype)
        sin = sin.to(dtype=x.dtype)
        if interleaved:
            x_even = x_rot[..., 0::2]
            x_odd = x_rot[..., 1::2]
            out = torch.stack(
                (x_even * cos - x_odd * sin, x_even * sin + x_odd * cos), dim=-1
            ).flatten(-2)
        else:
            half = rotary_dim // 2
            x1, x2 = x_rot[..., :half], x_rot[..., half:]
            out = torch.cat((x1 * cos - x2 * sin, x1 * sin + x2 * cos), dim=-1)
        return torch.cat((out, x_tail), dim=-1)

    def _angles(self, positions: torch.Tensor, *, dtype=None):
        frequencies = self._current_inv_freq(device=positions.device)
        # Position multiplication and trigonometry intentionally remain FP32;
        # BF16 position indices lose integer precision well before 4K tokens.
        angle = positions.to(torch.float32).unsqueeze(-1) * frequencies
        cos = angle.cos()
        sin = angle.sin()
        if dtype is not None:
            cos = cos.to(dtype=dtype)
            sin = sin.to(dtype=dtype)
        return cos, sin

    def _packed_positions(self, cu_seqlens, total, device):
        lengths = (cu_seqlens[1:] - cu_seqlens[:-1]).to(torch.long)
        # Supplying output_size keeps the varlen metadata allocation static
        # during CUDA graph capture.  The production trainer uses the masked
        # path today, but packed inference and future packed training should
        # remain graph-safe as well.
        starts = torch.repeat_interleave(
            cu_seqlens[:-1].to(torch.long), lengths, output_size=int(total)
        )
        return torch.arange(total, device=device, dtype=torch.long) - starts

    def _apply_le_rope(
        self, qkv, positions, *, num_heads_q=None, packed=False, kv_stream=False
    ):
        rotary_dim = self.dim
        if qkv.dim() == 5:
            # (B, S, 3, H, D), position table is (B, S, R/2) or (S, R/2).
            if positions.dim() == 1:
                cos, sin = self._angles(positions, dtype=qkv.dtype)
                cos = cos[None, :, None, None, :]
                sin = sin[None, :, None, None, :]
            else:
                cos, sin = self._angles(positions, dtype=qkv.dtype)
                cos = cos[:, :, None, None, :]
                sin = sin[:, :, None, None, :]
            # RoPE is applied to Q and K only.  A combined QKV tensor has
            # three slices; a separate KV stream has two, of which only K is
            # rotated.
            n_rot = 1 if kv_stream else 2
            rotated = self._rotate(
                qkv[:, :, :n_rot], cos, sin, rotary_dim, self.interleaved
            )
            return torch.cat((rotated, qkv[:, :, n_rot:]), dim=2)
        if qkv.dim() == 4:
            if packed:
                # Packed varlen (total, 3, H, D), position table is
                # (total, R/2).  The explicit flag avoids confusing a
                # separate Q tensor whose sequence length happens to be 3.
                cos, sin = self._angles(positions, dtype=qkv.dtype)
                n_rot = 1 if kv_stream else 2
                rotated = self._rotate(
                    qkv[:, :n_rot],
                    cos[:, None, None, :],
                    sin[:, None, None, :],
                    rotary_dim,
                    self.interleaved,
                )
                return torch.cat((rotated, qkv[:, n_rot:]), dim=1)
            # A separate Q or KV stream (B, S, H, D), used by cross
            # attention and some KV-cache callers.
            cos, sin = self._angles(positions, dtype=qkv.dtype)
            if positions.dim() == 1:
                cos, sin = cos[None, :, None, :], sin[None, :, None, :]
            else:
                cos, sin = cos[:, :, None, :], sin[:, :, None, :]
            return self._rotate(qkv, cos, sin, rotary_dim, self.interleaved)
        raise ValueError(f"unsupported LeRoPE input rank: {qkv.dim()}")

    def _compute_inv_freq(self, device=None):
        return 1.0 / (
            self.base
            ** (
                torch.arange(0, self.dim, 2, device=device, dtype=torch.float32)
                / self.dim
            )
        )

    def _update_cos_sin_cache(self, seqlen, device=None, dtype=None):
        # Reset the tables if the sequence length has changed,
        # if we're on a new device (possibly due to tracing for instance),
        # or if we're switching from inference mode to training
        if (
            seqlen > self._seq_len_cached
            or self._cos_cached is None
            or self._cos_cached.device != device
            or self._cos_cached.dtype != dtype
            or (self.training and self._cos_cached.is_inference())
            or (
                self.learnable
                and self._cache_log_scale_version != self.log_scale._version
            )
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
            if self.learnable:
                inv_freq = self._current_inv_freq(device=device)
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
            self._cache_log_scale_version = self.log_scale._version if self.learnable else -1

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
            if self.learnable:
                # In eval/inference the learned scales are fixed, so retain
                # FlashAttention's fused Triton rotary kernel.  Training uses
                # the differentiable path below to propagate gradients into
                # log_scale.
                if not self.training and not torch.is_grad_enabled():
                    self._update_cos_sin_cache(
                        max_seqlen, device=qkv.device, dtype=qkv.dtype
                    )
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
                positions = self._packed_positions(cu_seqlens, qkv.shape[0], qkv.device)
                return self._apply_le_rope(
                    qkv, positions, num_heads_q=num_heads_q, packed=True
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
        if self.learnable:
            if not self.training and not torch.is_grad_enabled():
                self._update_cos_sin_cache(
                    seqlen + (int(seqlen_offset) if isinstance(seqlen_offset, int) else 0),
                    device=qkv.device,
                    dtype=qkv.dtype,
                )
                if kv is None:
                    return apply_rotary_emb_qkv_(
                        qkv,
                        self._cos_cached,
                        self._sin_cached,
                        interleaved=self.interleaved,
                        seqlen_offsets=seqlen_offset,
                        num_heads_q=num_heads_q,
                    )
                q = apply_rotary_emb_func(
                    qkv,
                    self._cos_cached,
                    self._sin_cached,
                    interleaved=self.interleaved,
                    inplace=False,
                    seqlen_offsets=seqlen_offset,
                )
                kv = apply_rotary_emb_kv_(
                    kv,
                    self._cos_cached,
                    self._sin_cached,
                    interleaved=self.interleaved,
                    seqlen_offsets=seqlen_offset,
                )
                return q, kv
            if isinstance(seqlen_offset, int):
                positions = torch.arange(
                    seqlen, device=qkv.device, dtype=torch.long
                ) + int(seqlen_offset)
            elif torch.is_tensor(seqlen_offset):
                offsets = seqlen_offset.to(device=qkv.device, dtype=torch.long)
                positions = torch.arange(seqlen, device=qkv.device, dtype=torch.long)
                positions = positions[None, :] + offsets.reshape(-1, 1)
            else:
                raise TypeError("seqlen_offset must be an int or tensor")
            if kv is None:
                return self._apply_le_rope(qkv, positions, num_heads_q=num_heads_q)
            # Cross attention supplies q and kv separately; use the same
            # positions for both streams (the current H-Net path has equal
            # query/key offsets for its KV cache).
            q = self._apply_le_rope(qkv, positions, num_heads_q=num_heads_q)
            if kv.shape[1] != qkv.shape[1]:
                if isinstance(seqlen_offset, int):
                    kv_positions = torch.arange(
                        kv.shape[1], device=kv.device, dtype=torch.long
                    ) + int(seqlen_offset)
                else:
                    kv_positions = torch.arange(
                        kv.shape[1], device=kv.device, dtype=torch.long
                    )[None, :] + seqlen_offset.to(
                        device=kv.device, dtype=torch.long
                    ).reshape(-1, 1)
            else:
                kv_positions = positions
            kv = self._apply_le_rope(
                kv, kv_positions, num_heads_q=num_heads_q, kv_stream=True
            )
            return q, kv
        if max_seqlen is not None:
            self._update_cos_sin_cache(max_seqlen, device=qkv.device, dtype=qkv.dtype)
        elif isinstance(seqlen_offset, int):
            self._update_cos_sin_cache(
                seqlen + seqlen_offset, device=qkv.device, dtype=qkv.dtype
            )
        if kv is None:
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
            q = apply_rotary_emb_func(
                q,
                self._cos_cached,
                self._sin_cached,
                interleaved=self.interleaved,
                inplace=True,
                seqlen_offsets=seqlen_offset,
            )
            if self.scale is None:
                kv = apply_rotary_emb_kv_(
                    kv,
                    self._cos_cached,
                    self._sin_cached,
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
