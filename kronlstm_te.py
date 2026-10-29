#!/usr/bin/env python3
"""KronLSTM (Kronecker Delta Rule) sequence mixer.

This is a small, dependency-light reference implementation of the mixer from
``Kronecker Delta Rule for Long Context Linear Attention`` (arXiv:2610.26834).
It uses the sequence-first convention used by Transformer Engine
(``[sequence, batch, hidden]``).  The recurrent state is deliberately kept
local to a forward call, so it can be used as a drop-in causal attention block
without exposing a cache format to the trainer.

The paper's production implementation uses chunkwise parallel kernels.  The
reference path below processes short chunks sequentially, which keeps the
update exact and makes the module useful for correctness probes and small
training runs.  ``chunk_size`` bounds activation graph depth; each chunk is
recomputed during backward when checkpointing is enabled.
"""
from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.utils.checkpoint as checkpoint

try:  # Transformer Engine is present in the target training environment.
    import transformer_engine.pytorch as te
except ImportError:  # pragma: no cover - allows CPU shape/import checks.
    te = None


class _Linear(nn.Module):
    def __init__(self, in_features: int, out_features: int, *, device, dtype):
        super().__init__()
        if te is not None:
            self.impl = te.Linear(
                in_features, out_features, bias=False, params_dtype=dtype, device=device
            )
        else:  # pragma: no cover
            self.impl = nn.Linear(in_features, out_features, bias=False, device=device, dtype=dtype)

    def forward(self, x):
        return self.impl(x)


class KronLSTMAttention(nn.Module):
    """Gated Kronecker-delta linear attention.

    Args follow :class:`vha_attention_te.VHAAttention`: ``hidden_states`` are
    ``[S, B, D]`` and the result has the same shape.  ``num_heads`` should
    divide ``hidden_size``.  The state uses one ``d x d`` matrix per head and
    batch element, where ``d = hidden_size / num_heads``.
    """

    def __init__(
        self,
        hidden_size: int,
        num_heads: int,
        *,
        params_dtype: torch.dtype = torch.bfloat16,
        device: torch.device | str = "cuda",
        chunk_size: int = 64,
        checkpoint_chunks: bool = True,
    ) -> None:
        super().__init__()
        if hidden_size % num_heads:
            raise ValueError("hidden_size must be divisible by num_heads")
        if chunk_size <= 0:
            raise ValueError("chunk_size must be positive")
        self.hidden_size = hidden_size
        self.num_heads = num_heads
        self.head_dim = hidden_size // num_heads
        self.chunk_size = chunk_size
        self.checkpoint_chunks = checkpoint_chunks

        d = self.head_dim
        # q, k, v and the three gates.  Keeping these projections separate
        # makes it possible to disable individual gates in ablations while
        # still allowing TE to fuse their GEMMs under FP8 autocast.
        self.q_proj = _Linear(hidden_size, hidden_size, device=device, dtype=params_dtype)
        self.k_proj = _Linear(hidden_size, hidden_size, device=device, dtype=params_dtype)
        self.v_proj = _Linear(hidden_size, hidden_size, device=device, dtype=params_dtype)
        self.beta_proj = _Linear(hidden_size, num_heads, device=device, dtype=params_dtype)
        self.write_proj = _Linear(hidden_size, num_heads, device=device, dtype=params_dtype)
        self.forget_proj = _Linear(hidden_size, hidden_size, device=device, dtype=params_dtype)
        self.out_gate_proj = _Linear(hidden_size, hidden_size, device=device, dtype=params_dtype)
        self.o_proj = _Linear(hidden_size, hidden_size, device=device, dtype=params_dtype)

        # A positive, learned per-head exponent controls each head's retention
        # horizon.  Initialize close to one as in the paper.
        self.a_log = nn.Parameter(torch.zeros(num_heads, device=device, dtype=params_dtype))
        self.rms_weight = nn.Parameter(torch.ones(hidden_size, device=device, dtype=params_dtype))

    def _rms_norm(self, x: torch.Tensor) -> torch.Tensor:
        # Accumulate in FP32; this is particularly helpful when the recurrent
        # state is BF16 and a long context is used.
        rms = x.float().square().mean(dim=-1, keepdim=True).add(1e-6).rsqrt()
        return (x.float() * rms).to(dtype=x.dtype) * self.rms_weight

    def _chunk(
        self,
        q: torch.Tensor,
        k: torch.Tensor,
        v: torch.Tensor,
        beta: torch.Tensor,
        write: torch.Tensor,
        forget: torch.Tensor,
        state: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Run an exact KDR recurrence over one ``[T,B,H,D]`` chunk."""
        outputs = []
        # The scalar contraction is vᵀ (S F) k.  Applying F on the right is
        # just a broadcast multiply over the key dimension.
        for i in range(q.shape[0]):
            f = forget[i]
            decayed = state * f.unsqueeze(-2)
            vk = torch.einsum("bhd,bhm->bhdm", v[i], k[i])
            retrieval = torch.einsum("bhdm,bhm,bhd->bh", decayed, k[i], v[i])
            # KronLSTM: S <- S F - beta*(vᵀ S F k) v kᵀ + c v kᵀ.
            coeff = (write[i] - beta[i] * retrieval).unsqueeze(-1).unsqueeze(-1)
            state = decayed + coeff * vk
            outputs.append(torch.einsum("bhdm,bhm->bhd", state, q[i]))
        return torch.stack(outputs), state

    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        if hidden_states.ndim != 3:
            raise ValueError("KronLSTMAttention expects [sequence,batch,hidden]")
        s, b, d_model = hidden_states.shape
        if d_model != self.hidden_size:
            raise ValueError(f"expected hidden size {self.hidden_size}, got {d_model}")
        d = self.head_dim
        shape = (s, b, self.num_heads, d)
        q = F.normalize(self.q_proj(hidden_states).reshape(shape), dim=-1, eps=1e-6)
        k = F.normalize(self.k_proj(hidden_states).reshape(shape), dim=-1, eps=1e-6)
        v = F.normalize(self.v_proj(hidden_states).reshape(shape), dim=-1, eps=1e-6)
        beta = torch.sigmoid(self.beta_proj(hidden_states))
        write = torch.sigmoid(self.write_proj(hidden_states))
        forget_logits = self.forget_proj(hidden_states).reshape(shape)
        # f=(1-sigmoid(r))**a, with a=exp(a_log), as specified by Eq. (12).
        exponent = self.a_log.exp().view(1, 1, self.num_heads, 1)
        forget = (1.0 - torch.sigmoid(forget_logits)).clamp_min(1e-4).pow(exponent)
        output_gate = torch.sigmoid(self.out_gate_proj(hidden_states).reshape(shape))

        state = torch.zeros(
            b, self.num_heads, d, d, dtype=hidden_states.dtype, device=hidden_states.device
        )
        pieces = []
        use_ckpt = self.checkpoint_chunks and self.training and torch.is_grad_enabled()
        for start in range(0, s, self.chunk_size):
            end = min(start + self.chunk_size, s)
            args = (q[start:end], k[start:end], v[start:end], beta[start:end], write[start:end], forget[start:end], state)
            if use_ckpt:
                # Preserve RNG state only when a caller adds stochastic gates;
                # this module itself is deterministic and does not need it.
                out, state = checkpoint.checkpoint(
                    self._chunk, *args, use_reentrant=False, preserve_rng_state=False
                )
            else:
                out, state = self._chunk(*args)
            pieces.append(out)
        output = torch.cat(pieces, dim=0) * output_gate
        output = self._rms_norm(output.reshape(s, b, -1))
        return self.o_proj(output)


__all__ = ["KronLSTMAttention"]
