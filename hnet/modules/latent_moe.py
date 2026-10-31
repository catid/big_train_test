"""LatentMoE feed-forward block used by every production FFN layer.

The implementation follows arXiv:2601.18089.  Routing is performed from the
full residual width, while routed experts operate on a shared latent projection.
The router and the full-width shared expert are launched independently on CUDA;
the selected routed experts are dispatched only after their routing decisions
are available.  Router math and expert accumulation stay in FP32 so FP8
rounding cannot leak into the gate normalization or CroissAttn-style auxiliary
corrections.
"""

from __future__ import annotations

import math
from contextlib import nullcontext
from typing import Optional

import torch
from torch import Tensor, nn
import torch.nn.functional as F


def _round_multiple(value: int, multiple: int = 128) -> int:
    return max(multiple, (int(value) + multiple - 1) // multiple * multiple)


def matched_expert_width(
    d_model: int,
    dense_intermediate: int,
    *,
    num_experts: int = 8,
    latent_alpha: int = 2,
    multiple: int = 64,
) -> int:
    """Return a shared hidden width under the dense SwiGLU budget.

    A dense SwiGLU with hidden width ``m`` has ``3*d*m`` matrix parameters.
    LatentMoE has a ``d -> d/a -> d`` pair, ``E`` latent experts and one full
    width shared expert.  Giving every expert the same hidden width ``n``
    makes the budget

      ``2*d*(d/a) + 3*E*(d/a)*n + 3*d*n + E*d``.

    The final term is the bias-free router.  We solve this expression for n
    and round down to a GEMM-friendly multiple so the replacement never
    exceeds the configured dense layer.  Because there are nine expert
    modules, the total neuron count is still larger than the original single
    FFN even though each latent matrix is half width.
    """
    d = int(d_model)
    m = int(dense_intermediate)
    e = int(num_experts)
    a = int(latent_alpha)
    if d <= 0 or m <= 0 or e <= 0 or a <= 0 or d % a:
        raise ValueError("invalid dimensions for matched LatentMoE width")
    dense_budget = 3 * d * m
    latent_dim = d // a
    fixed = 2 * d * latent_dim + e * d
    per_width = 3 * (e * latent_dim + d)
    width = (dense_budget - fixed) // per_width
    width = (width // multiple) * multiple
    if width < multiple:
        raise ValueError(
            f"dense FFN ({d}x{m}) is too small for a matched LatentMoE "
            f"with latent_alpha={a}, experts={e}"
        )
    return int(width)


class _ExpertMLP(nn.Module):
    """SwiGLU expert with optional Transformer Engine FP8 linears."""

    def __init__(self, d_in: int, d_hidden: int, *, bias: bool, device, dtype, fp8: bool):
        super().__init__()
        if fp8:
            try:
                import transformer_engine.pytorch as te
            except ImportError as exc:  # explicit configuration error, never silent fallback
                raise ImportError("LatentMoE fp8=True requires Transformer Engine") from exc
            self.fc1 = te.Linear(d_in, 2 * d_hidden, bias=bias, device=device, params_dtype=dtype)
            self.fc2 = te.Linear(d_hidden, d_in, bias=bias, device=device, params_dtype=dtype)
        else:
            self.fc1 = nn.Linear(d_in, 2 * d_hidden, bias=bias, device=device, dtype=dtype)
            self.fc2 = nn.Linear(d_hidden, d_in, bias=bias, device=device, dtype=dtype)

    def forward(self, x: Tensor) -> Tensor:
        a, gate = self.fc1(x).chunk(2, dim=-1)
        # Keep the nonlinear reduction in the input dtype; the caller performs
        # the routed weighted sum in FP32 before converting back to model dtype.
        return self.fc2(F.silu(gate) * a)


class LatentMoE(nn.Module):
    """Eight-expert LatentMoE replacement for a normal SwiGLU FFN.

    The width is solved from the former dense SwiGLU parameter budget. With
    the default alpha=2, each routed expert operates on a latent width half of
    the residual stream, while eight routed experts and one shared expert expose
    nine times the matched neuron width.
    """

    num_experts: int = 8
    latent_alpha: int = 2
    top_k: int = 4
    num_shared_experts: int = 1

    def __init__(
        self,
        d_model: int,
        d_intermediate: int,
        *,
        num_experts: int = 8,
        top_k: int = 4,
    latent_alpha: int = 2,
        num_shared_experts: int = 1,
        bias: bool = False,
        device=None,
        dtype=None,
        fp8: bool = False,
        aux_loss_coef: float = 1e-2,
        z_loss_coef: float = 1e-3,
    ) -> None:
        super().__init__()
        if num_experts != 8:
            raise ValueError("The production architecture requires exactly 8 routed experts")
        if latent_alpha < 1 or d_model % latent_alpha:
            raise ValueError("latent_alpha must be a positive divisor of d_model")
        if not 1 <= top_k <= num_experts:
            raise ValueError("top_k must be in [1, 8]")
        if num_shared_experts != 1:
            raise ValueError("The production architecture requires one shared expert")
        self.d_model = int(d_model)
        self.latent_dim = d_model // latent_alpha
        self.num_experts = num_experts
        self.top_k = top_k
        self.latent_alpha = latent_alpha
        self.fp8 = bool(fp8)
        self.aux_loss_coef = float(aux_loss_coef)
        self.z_loss_coef = float(z_loss_coef)
        # Match the complete dense SwiGLU matrix budget while retaining the
        # half-width latent projection.  Routed and shared experts use one
        # common width; nine expert modules therefore expose more total
        # neurons than the replaced dense FFN without increasing parameters.
        self.routed_intermediate = matched_expert_width(
            d_model,
            d_intermediate,
            num_experts=num_experts,
            latent_alpha=latent_alpha,
        )
        self.shared_intermediate = self.routed_intermediate
        factory = {"device": device, "dtype": dtype}
        self.router = nn.Linear(d_model, num_experts, bias=False, **factory)
        self.down = nn.Linear(d_model, self.latent_dim, bias=bias, **factory)
        self.up = nn.Linear(self.latent_dim, d_model, bias=bias, **factory)
        self.experts = nn.ModuleList(
            _ExpertMLP(self.latent_dim, self.routed_intermediate, bias=bias,
                       device=device, dtype=dtype, fp8=fp8)
            for _ in range(num_experts)
        )
        self.shared_expert = _ExpertMLP(
            d_model, self.shared_intermediate, bias=bias,
            device=device, dtype=dtype, fp8=fp8,
        )
        self.register_buffer("last_aux_loss", torch.zeros((), dtype=torch.float32), persistent=False)
        self.register_buffer("last_z_loss", torch.zeros((), dtype=torch.float32), persistent=False)
        self._shared_stream: Optional[torch.cuda.Stream] = None

    def _fp8_context(self):
        if not self.fp8:
            return nullcontext()
        import transformer_engine.pytorch as te
        # Delayed scaling gives the MoE GEMMs amax history and keeps all gate
        # reductions outside FP8.  The caller may nest a project-wide TE context;
        # TE permits the nested context and reuses its metadata tensors.
        recipe = te.common.recipe.DelayedScaling(fp8_dpa=False, amax_history_len=16)
        return te.fp8_autocast(enabled=True, fp8_recipe=recipe)

    def _route(self, x: Tensor):
        logits = self.router(x.float())
        probs = F.softmax(logits, dim=-1)
        weights, indices = probs.topk(self.top_k, dim=-1)
        weights = weights / weights.sum(dim=-1, keepdim=True).clamp_min(1e-8)
        slots = torch.zeros_like(probs)
        slots.scatter_(1, indices, 1.0)
        fraction = slots.mean(dim=0)
        mean_prob = probs.mean(dim=0)
        aux = self.aux_loss_coef * self.num_experts * (fraction * mean_prob).sum()
        z = self.z_loss_coef * torch.logsumexp(logits, dim=-1).square().mean()
        return indices, weights, aux, z

    def _dispatch(self, latent: Tensor, indices: Tensor, weights: Tensor) -> Tensor:
        # FP32 accumulation is essential: summing several rounded FP8 expert
        # outputs in model dtype introduces a biased gate gradient.
        result = torch.zeros_like(latent, dtype=torch.float32)
        token_ids = torch.arange(latent.shape[0], device=latent.device).repeat_interleave(self.top_k)
        expert_ids = indices.reshape(-1)
        slot_weights = weights.reshape(-1).float()
        order = torch.argsort(expert_ids)
        sorted_experts = expert_ids[order]
        sorted_tokens = token_ids[order]
        sorted_weights = slot_weights[order]
        counts = torch.bincount(sorted_experts, minlength=self.num_experts)
        start = 0
        with self._fp8_context():
            for expert_id, count in enumerate(counts.tolist()):
                end = start + count
                if count:
                    out = self.experts[expert_id](latent[sorted_tokens[start:end]])
                    result.index_add_(0, sorted_tokens[start:end], out.float() * sorted_weights[start:end, None])
                start = end
        return result

    def _shared_forward(self, x: Tensor) -> Tensor:
        with self._fp8_context():
            return self.shared_expert(x)

    def forward(self, x: Tensor) -> Tensor:
        shape = x.shape
        if shape[-1] != self.d_model:
            raise ValueError(f"LatentMoE expected width {self.d_model}, got {shape[-1]}")
        flat = x.reshape(-1, self.d_model)
        shared = None
        event = None
        # Shared expert is independent of routing. Launch it on a separate CUDA
        # stream so router latency is hidden; event ordering keeps autograd safe.
        if flat.is_cuda:
            if self._shared_stream is None or self._shared_stream.device != flat.device:
                self._shared_stream = torch.cuda.Stream(device=flat.device)
            stream = self._shared_stream
            event = torch.cuda.Event()
            with torch.cuda.stream(stream):
                shared = self._shared_forward(flat)
                event.record(stream)
        else:
            shared = self._shared_forward(flat)

        indices, weights, aux, z = self._route(flat)
        with self._fp8_context():
            latent = self.down(flat)
        routed = self._dispatch(latent, indices, weights)
        if event is not None:
            torch.cuda.current_stream(flat.device).wait_event(event)
        with self._fp8_context():
            output = self.up(routed.to(dtype=flat.dtype))
        output = output.float() + shared.float()
        self.last_aux_loss = aux
        self.last_z_loss = z
        return output.to(dtype=x.dtype).reshape(shape)

    @property
    def aux_loss(self) -> Tensor:
        return self.last_aux_loss

    @property
    def z_loss(self) -> Tensor:
        return self.last_z_loss

    @property
    def parameter_budget(self) -> int:
        """Matrix parameter count, including router, for audit logs."""
        d = self.d_model
        latent = self.latent_dim
        return (
            2 * d * latent
            + self.num_experts * 3 * latent * self.routed_intermediate
            + 3 * d * self.shared_intermediate
            + self.num_experts * d
        )

    def extra_repr(self) -> str:
        return (f"d_model={self.d_model}, latent={self.latent_dim}, experts={self.num_experts}, "
                f"top_k={self.top_k}, routed_m={self.routed_intermediate}, "
                f"shared_m={self.shared_intermediate}, fp8={self.fp8}")
