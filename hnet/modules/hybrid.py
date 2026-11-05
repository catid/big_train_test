"""The canonical GDN2/LoL/FA H-Net mixer stack.

This module keeps the hierarchy and FBT machinery in :mod:`hnet.models.hnet`,
but gives its encoder, middle, and decoder stacks a single explicit block
library.  GDN2 runs at 3072 channels, while LoL and FA run at 1280 channels;
the width changes are represented by real projection matrices so no activation
is silently truncated.  LoL uses fixed shifted 128-token windows and native
GQA, which keeps all tensors static for compilation and CUDA graph capture.

The external GDN2 implementation is loaded by ``GDN2Adapter``.  There is no
in-tree recurrent fallback: a production construction fails if the inspected
GDN2/cuDNN dependency is unavailable.
"""

from __future__ import annotations

import re
import torch
from torch import nn
from torch.nn import functional as F

from .gdn2_adapter import GDN2Adapter
from .latent_moe import LatentMoE


class _LearnedRoPE(nn.Module):
    """Differentiable RoPE table for fixed-shape local attention."""

    def __init__(self, head_dim: int, max_seq_len: int, device=None):
        super().__init__()
        if head_dim % 2:
            raise ValueError("attention head dimension must be even for RoPE")
        inv = 1.0 / (
            10000.0
            ** (torch.arange(0, head_dim, 2, device=device, dtype=torch.float32) / head_dim)
        )
        self.register_buffer("inv_freq", inv, persistent=False)
        self.log_scale = nn.Parameter(torch.zeros(head_dim // 2, device=device))
        self.log_scale._no_weight_decay = True

    def cos_sin(self, positions: torch.Tensor, dtype: torch.dtype, device) -> tuple[torch.Tensor, torch.Tensor]:
        """Compute one fixed-position rotation table for both Q and K."""
        freq = self.inv_freq.to(device=device) * self.log_scale.exp()
        angle = positions.to(torch.float32).unsqueeze(-1) * freq
        return angle.cos().to(dtype), angle.sin().to(dtype)

    @staticmethod
    def apply(x: torch.Tensor, cos: torch.Tensor, sin: torch.Tensor) -> torch.Tensor:
        """Apply a shared rotation table to a Q/K tensor."""
        # Tables are [T, D/2]; make them broadcast over batch and heads.
        cos = cos.view(1, -1, 1, cos.shape[-1])
        sin = sin.view(1, -1, 1, sin.shape[-1])
        half = x.shape[-1] // 2
        first, second = x[..., :half], x[..., half:]
        return torch.cat((first * cos - second * sin, first * sin + second * cos), dim=-1)

    def forward(self, x: torch.Tensor, positions: torch.Tensor) -> torch.Tensor:
        # Keep angle evaluation in FP32, then return in the projection dtype.
        cos, sin = self.cos_sin(positions, x.dtype, x.device)
        if positions.dim() == 1:
            return self.apply(x, cos, sin)
        else:
            cos = cos.unsqueeze(-2)
            sin = sin.unsqueeze(-2)
        half = x.shape[-1] // 2
        first, second = x[..., :half], x[..., half:]
        return torch.cat((first * cos - second * sin, first * sin + second * cos), dim=-1)


class _GDN2Block(nn.Module):
    """Pre-norm GDN2 mixer-only residual block."""

    def __init__(self, width: int, config, *, device=None, dtype=None):
        super().__init__()
        self.norm = nn.RMSNorm(width, eps=1e-5, device=device, dtype=dtype)
        self.mixer = GDN2Adapter(width, config, device=device, dtype=dtype)

    def forward(self, x, *, mask=None, cu_seqlens=None, max_seqlen=None):
        kwargs = {}
        if cu_seqlens is not None:
            kwargs.update(cu_seqlens=cu_seqlens, max_seqlen=max_seqlen)
        # Named-library training uses fixed [B, latent_cap, D] buffers.  The
        # latent validity mask only marks trailing padding; passing it into
        # FLA/FROST would unpad to a data-dependent length and break graph
        # capture. GDN2 is causal, so padded suffix rows cannot affect the
        # valid prefix consumed by dechunking.
        del mask
        return x + self.mixer(self.norm(x), mask=None, **kwargs)


class _AttentionBlock(nn.Module):
    """GQA attention plus the production eight-expert LatentMoE FFN.

    ``local=True`` is the LoL shifted-window primitive.  ``local=False`` is
    the full-attention (FA) primitive and deliberately uses NoPE.
    """

    def __init__(
        self,
        width: int,
        heads: int,
        kv_heads: int,
        ff_width: int,
        *,
        local: bool,
        window_size: int,
        shift: int,
        max_seq_len: int,
        moe_num_experts: int,
        moe_top_k: int,
        moe_latent_alpha: int,
        device=None,
        dtype=None,
    ):
        super().__init__()
        if width % heads:
            raise ValueError(f"width {width} must divide heads {heads}")
        if heads % kv_heads:
            raise ValueError(f"query heads {heads} must divide KV heads {kv_heads}")
        self.width = int(width)
        self.heads = int(heads)
        self.kv_heads = int(kv_heads)
        self.head_dim = width // heads
        self.local = bool(local)
        self.window_size = int(window_size)
        self.shift = int(shift) if local else 0
        self.static_seq_len = int(max_seq_len)
        qkv_width = (heads + 2 * kv_heads) * self.head_dim
        self.norm1 = nn.RMSNorm(width, eps=1e-5, device=device, dtype=dtype)
        self.qkv = nn.Linear(width, qkv_width, bias=False, device=device, dtype=dtype)
        self.q_norm = nn.RMSNorm(self.head_dim, eps=1e-5, device=device, dtype=dtype)
        self.k_norm = nn.RMSNorm(self.head_dim, eps=1e-5, device=device, dtype=dtype)
        self.rope = (
            _LearnedRoPE(self.head_dim, max_seq_len + window_size, device=device)
            if local
            else None
        )
        self.register_buffer(
            "_positions",
            torch.arange(max_seq_len + window_size, device=device, dtype=torch.float32),
            persistent=False,
        )
        # Precompute the fixed shifted-window causal mask.  The canonical
        # outer and latent streams use static lengths, so replay can reuse
        # this device-resident tensor instead of rebuilding position grids.
        if self.local and self.shift % self.window_size and max_seq_len % self.window_size == 0:
            shift_value = self.shift % self.window_size
            right = (self.window_size - ((max_seq_len + shift_value) % self.window_size)) % self.window_size
            padded = max_seq_len + shift_value + right
            n_windows = padded // self.window_size
            positions = torch.arange(padded, device=device, dtype=torch.long) - shift_value
            pos = positions.view(n_windows, self.window_size)
            valid = (pos >= 0) & (pos < max_seq_len)
            pos_q = pos.view(1, n_windows, self.window_size, 1)
            pos_k = pos.view(1, n_windows, 1, self.window_size)
            allowed = (pos_q >= pos_k) & valid.view(1, n_windows, 1, self.window_size)
            self.register_buffer(
                "_static_allowed",
                allowed.reshape(n_windows, 1, self.window_size, self.window_size),
                persistent=False,
            )
        else:
            self.register_buffer(
                "_static_allowed",
                torch.empty(0, dtype=torch.bool, device=device),
                persistent=False,
            )
        self.out = nn.Linear(width, width, bias=False, device=device, dtype=dtype)
        self.norm2 = nn.RMSNorm(width, eps=1e-5, device=device, dtype=dtype)
        self.mlp = LatentMoE(
            width,
            ff_width,
            num_experts=int(moe_num_experts),
            top_k=int(moe_top_k),
            latent_alpha=int(moe_latent_alpha),
            device=device,
            dtype=dtype,
        )
        # Compile only attention and its residual projection.  LatentMoE owns
        # CUDA-graph-aware routing branches that inspect capture state; folding
        # that Python dispatch into an Inductor graph could specialize the
        # eager (non-capturing) branch and make graph replay use host reads.
        self._compiled_attention = None

    def _attention(self, x, mask):
        batch, length, _ = x.shape
        qkv = self.qkv(x)
        q_width = self.heads * self.head_dim
        kv_width = self.kv_heads * self.head_dim
        q, k, v = qkv.split((q_width, kv_width, kv_width), dim=-1)
        q = self.q_norm(q.view(batch, length, self.heads, self.head_dim))
        k = self.k_norm(k.view(batch, length, self.kv_heads, self.head_dim))
        v = v.view(batch, length, self.kv_heads, self.head_dim)
        if self.rope is not None:
            positions = self._positions[:length]
            # Q and K use the same positions and learned frequencies. Compute
            # the trigonometric table once instead of repeating two identical
            # angle/exp/cos/sin paths for every local-attention block.
            cos, sin = self.rope.cos_sin(positions, q.dtype, q.device)
            q = self.rope.apply(q, cos, sin)
            k = self.rope.apply(k, cos, sin)

        if not self.local:
            # Causal attention ensures trailing static padding cannot affect
            # valid tokens; the caller masks those trailing outputs in the
            # chunk/dechunk path. Native GQA avoids materializing repeated KV.
            q = q.transpose(1, 2)
            k = k.transpose(1, 2)
            v = v.transpose(1, 2)
            return F.scaled_dot_product_attention(
                q, k, v, is_causal=True, enable_gqa=self.kv_heads != self.heads
            ).transpose(1, 2).reshape(batch, length, self.width)

        window = self.window_size
        shift = self.shift % window
        right = (window - ((length + shift) % window)) % window
        q = F.pad(q, (0, 0, 0, 0, shift, right))
        k = F.pad(k, (0, 0, 0, 0, shift, right))
        v = F.pad(v, (0, 0, 0, 0, shift, right))
        padded = q.shape[1]
        n_windows = padded // window
        q = q.view(batch * n_windows, window, self.heads, self.head_dim).transpose(1, 2)
        k = k.view(batch * n_windows, window, self.kv_heads, self.head_dim).transpose(1, 2)
        v = v.view(batch * n_windows, window, self.kv_heads, self.head_dim).transpose(1, 2)

        # A packed fixed-shape training batch has no padding/document holes.
        # In that case the left/right shift padding is already zero and causal
        # SDPA excludes the right padding from every real query.  Avoiding the
        # [B*n_windows, 1, W, W] boolean mask lets PyTorch select the flash
        # kernel; the masked path below remains for explicit boundary masks.
        if mask is None and shift == 0:
            z = F.scaled_dot_product_attention(
                q, k, v, is_causal=True,
                enable_gqa=self.kv_heads != self.heads,
            )
            z = z.transpose(1, 2).reshape(batch, padded, self.width)
            return z[:, shift : shift + length]

        if (
            mask is None
            and shift
            and length == self.static_seq_len
            and self._static_allowed.numel()
        ):
            allowed = self._static_allowed.expand(
                batch, -1, -1, -1, -1
            ).reshape(batch * n_windows, 1, window, window)
            z = F.scaled_dot_product_attention(
                q, k, v, attn_mask=allowed, is_causal=False,
                enable_gqa=self.kv_heads != self.heads,
            )
            z = z.transpose(1, 2).reshape(batch, padded, self.width)
            return z[:, shift : shift + length]

        positions = self._positions[: length + shift + right].to(torch.long) - shift
        pos = positions.view(n_windows, window)
        valid = (pos >= 0) & (pos < length)
        if mask is None:
            token_valid = torch.ones(
                (batch, length), device=x.device, dtype=torch.bool
            )
        else:
            token_valid = mask[:, :length].to(torch.bool)
        token_valid = F.pad(token_valid, (shift, right), value=False)
        valid = valid.unsqueeze(0) & token_valid.view(batch, n_windows, window)
        valid = valid.reshape(batch * n_windows, window)
        pos_q = pos.view(1, n_windows, window, 1).expand(batch, -1, -1, -1)
        pos_k = pos.view(1, n_windows, 1, window).expand(batch, -1, -1, -1)
        allowed = (pos_q >= pos_k) & valid.view(batch, n_windows, 1, window)
        allowed = allowed.reshape(batch * n_windows, 1, window, window)
        z = F.scaled_dot_product_attention(
            q, k, v, attn_mask=allowed, is_causal=False,
            enable_gqa=self.kv_heads != self.heads,
        )
        z = z.transpose(1, 2).reshape(batch, padded, self.width)
        return z[:, shift : shift + length]

    def _attention_residual(self, x, *, mask=None):
        return x + self.out(self._attention(self.norm1(x), mask))

    def _forward_impl(self, x, *, mask=None, **kwargs):
        del kwargs
        if self._compiled_attention is None:
            x = self._attention_residual(x, mask=mask)
        else:
            x = self._compiled_attention(x, mask=mask)
        return x + self.mlp(self.norm2(x), token_mask=mask)

    def compile_static(self, *, mode="max-autotune-no-cudagraphs"):
        """Compile this fixed-shape attention/FFN block once."""
        if self._compiled_attention is None:
            self._compiled_attention = torch.compile(
                self._attention_residual, dynamic=False, mode=mode
            )
        return self

    def forward(self, x, *, mask=None, **kwargs):
        return self._forward_impl(x, mask=mask, **kwargs)


class _HybridStep(nn.Module):
    def __init__(self, in_width, target_width, core, *, device=None, dtype=None):
        super().__init__()
        self.proj = (
            nn.Linear(in_width, target_width, bias=False, device=device, dtype=dtype)
            if in_width != target_width else nn.Identity()
        )
        self.core = core

    def forward(self, x, *, mask=None, cu_seqlens=None, max_seqlen=None):
        x = self.proj(x)
        return self.core(mask=mask, cu_seqlens=cu_seqlens, max_seqlen=max_seqlen, x=x)


class _HybridSequence(nn.Module):
    def __init__(self, steps, output_proj, *, final_width):
        super().__init__()
        self.steps = nn.ModuleList(steps)
        self.output_proj = output_proj
        self.final_width = final_width

    def forward(self, x, *, mask=None, cu_seqlens=None, max_seqlen=None):
        for step in self.steps:
            x = step(x, mask=mask, cu_seqlens=cu_seqlens, max_seqlen=max_seqlen)
        return self.output_proj(x)


class HybridStack(nn.Module):
    """Static heterogeneous stack used by the canonical H-Net config."""

    _TOKEN_RE = re.compile(r"([GLF])(?:x?(\d+))?")

    def __init__(self, config, pos_idx: int, stage_idx: int, *, device=None, dtype=None):
        super().__init__()
        self.config = config
        self.stage_idx = int(stage_idx)
        self.d_model = int(config.d_model[stage_idx])
        layout = config.arch_layout
        for _ in range(stage_idx):
            layout = layout[1]
        layout = layout[pos_idx] if len(layout) == 3 else layout[0]
        readable = str(layout).replace("GDN2", "G").replace("LoL", "L")
        readable = readable.replace("FA", "F").replace("x", "")
        tokens = [(kind, int(count or 1)) for kind, count in self._TOKEN_RE.findall(readable)]
        compact = re.sub(r"[^GLF0-9]", "", readable)
        compact_tokens = "".join(f"{k}{n}" for k, n in tokens)
        if not tokens or compact_tokens.replace("G1", "G").replace("L1", "L").replace("F1", "F") != compact:
            raise ValueError(f"invalid canonical block layout {layout!r}; use compact G/L/F codes")

        self.gdn_width = int(config.gdn2_hidden_size)
        self.lol_width = int(getattr(config, "lol_hidden_size", 1280))
        self.ff_width = int(getattr(config, "lol_ffn_width", 1536))
        self.inner_loops = int(config.inner_loops) if self.stage_idx > 0 else 1
        self.static_full_sequence = bool(
            getattr(config, "static_full_sequence", False) and self.stage_idx == 0
        )
        attention_seq_len = int(config.max_seq_len)
        if self.stage_idx > 0 and getattr(config, "latent_max_seq_len", None) is not None:
            attention_seq_len = int(config.latent_max_seq_len)
        moe_kwargs = dict(
            moe_num_experts=int(getattr(config, "moe_num_experts", 8)),
            moe_top_k=int(getattr(config, "moe_top_k", 4)),
            moe_latent_alpha=int(getattr(config, "moe_latent_alpha", 2)),
        )
        if self.inner_loops < 1:
            raise ValueError("inner_loops must be positive")
        self.height = sum(count for _, count in tokens) * self.inner_loops
        self._shared_gdn = None
        self._shared_fa = None
        if self.stage_idx > 0:
            self._shared_gdn = _GDN2Block(self.gdn_width, config, device=device, dtype=dtype)
            self._shared_fa = _AttentionBlock(
                self.lol_width, int(config.fa_num_heads), int(config.fa_kv_heads),
                self.ff_width, local=False, window_size=int(config.lol_window_size),
                shift=0, max_seq_len=attention_seq_len, **moe_kwargs,
                device=device, dtype=dtype,
            )

        sequences = []
        for loop_idx in range(self.inner_loops):
            steps = []
            current = self.d_model
            local_seen = 0
            for kind, count in tokens:
                for _ in range(count):
                    if kind == "G":
                        target = self.gdn_width
                        core = self._shared_gdn if self._shared_gdn is not None else _GDN2Block(target, config, device=device, dtype=dtype)
                    elif kind == "L":
                        target = self.lol_width
                        core = _AttentionBlock(
                            target, int(config.lol_num_heads), int(config.lol_kv_heads),
                            self.ff_width, local=True, window_size=int(config.lol_window_size),
                            shift=(local_seen % 2) * (int(config.lol_window_size) // 2),
                            max_seq_len=attention_seq_len, **moe_kwargs,
                            device=device, dtype=dtype,
                        )
                        local_seen += 1
                    elif kind == "F":
                        target = self.lol_width
                        core = self._shared_fa if self._shared_fa is not None else _AttentionBlock(
                            target, int(config.fa_num_heads), int(config.fa_kv_heads),
                            self.ff_width, local=False, window_size=int(config.lol_window_size),
                            shift=0, max_seq_len=attention_seq_len, **moe_kwargs,
                            device=device, dtype=dtype,
                        )
                    else:
                        raise AssertionError(kind)
                    steps.append(_HybridStep(current, target, core, device=device, dtype=dtype))
                    current = target
            output_proj = (
                nn.Linear(current, self.d_model, bias=False, device=device, dtype=dtype)
                if current != self.d_model else nn.Identity()
            )
            sequences.append(_HybridSequence(steps, output_proj, final_width=self.d_model))
        self.sequences = nn.ModuleList(sequences)
        self.final_norm = nn.RMSNorm(self.d_model, eps=1e-5, device=device, dtype=dtype)

    def _init_weights(self, *args, **kwargs):
        del args, kwargs

    def _apply_lr_multiplier(self, lr_multiplier):
        multiplier = float(lr_multiplier[self.stage_idx])
        for parameter in self.parameters():
            parameter._lr_multiplier = multiplier

    def compile_blocks(self, *, mode="max-autotune-no-cudagraphs"):
        """Compile each unique LoL/FA block, preserving middle weight ties."""
        seen = set()
        count = 0
        for module in self.modules():
            if isinstance(module, _AttentionBlock) and id(module) not in seen:
                seen.add(id(module))
                module.compile_static(mode=mode)
                count += 1
        return count

    def allocate_inference_cache(self, *args, **kwargs):
        raise RuntimeError("Hybrid GDN2/LoL stacks require an explicit recurrent cache bridge for inference")

    def forward(self, hidden_states, *, mask=None, cu_seqlens=None, max_seqlen=None, inference_params=None, **kwargs):
        del kwargs
        if inference_params is not None:
            raise RuntimeError("Hybrid GDN2/LoL training stack has no implicit inference cache")
        if hidden_states.dim() == 2:
            raise ValueError(
                "canonical GDN2/LoL/FA training requires a fixed-shape [B,S,D] "
                "input with an explicit mask"
            )
        # The byte-side outer stacks are always full fixed-length sequences in
        # the canonical trainer. Dropping the all-true mask activates fused
        # causal/window attention and lets GDN2 use its dense batch path. The
        # latent trunk keeps its validity mask because its 2048-capacity
        # buffer may contain padded rows.
        inner_mask = None if self.static_full_sequence else mask
        inner_cu_seqlens = None if self.static_full_sequence else cu_seqlens
        inner_max_seqlen = None if self.static_full_sequence else max_seqlen
        for sequence in self.sequences:
            hidden_states = sequence(
                hidden_states, mask=inner_mask, cu_seqlens=inner_cu_seqlens,
                max_seqlen=inner_max_seqlen
            )
        hidden_states = self.final_norm(hidden_states)
        return hidden_states

    def step(self, *args, **kwargs):
        raise RuntimeError("Hybrid GDN2/LoL stacks are training-only until cache adaptation is implemented")


__all__ = ["HybridStack"]
