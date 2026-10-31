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
        self.fp8 = bool(fp8)
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
        # TE's FP8 GEMM requires the flattened token dimension to be a
        # multiple of eight.  Routing produces variable per-expert counts,
        # and document-aware chunking can produce an arbitrary shared-expert
        # count, so pad only the row dimension for the GEMM and discard those
        # rows afterward.  This keeps the full production FP8 path enabled
        # without changing routing probabilities or sequence boundaries.
        rows = x.shape[0]
        pad_rows = (-rows) % 8 if self.fp8 else 0
        if pad_rows:
            x = F.pad(x, (0, 0, 0, pad_rows))
        a, gate = self.fc1(x).chunk(2, dim=-1)
        # Center the post-activation features per token.  The activation-bias
        # study (arXiv:2610.14500) uses this adaptive shift in its scale-up
        # language-model experiments: it removes the current feature mean
        # instead of relying on a fixed initialization-time scalar.  Compute
        # the statistic in FP32 so BF16/FP8 rounding cannot leave a systematic
        # offset, then return to the TE linear's input dtype.
        activated = F.silu(gate) * a
        activated = activated - activated.float().mean(dim=-1, keepdim=True).to(
            dtype=activated.dtype
        )
        # The caller performs the routed weighted sum in FP32 before converting
        # back to model dtype.
        output = self.fc2(activated)
        return output[:rows] if pad_rows else output


class _GroupedExpertMLP(nn.Module):
    """SwiGLU experts backed by two Transformer Engine grouped GEMMs.

    The split metadata is deliberately supplied as a CPU tensor.  The pinned
    TE implementation converts splits to a Python tuple internally; keeping
    that tensor on the host makes the conversion capture-safe while retaining
    the static device input/output buffers used by CUDA graphs.
    """

    def __init__(self, num_experts: int, d_in: int, d_hidden: int, *, device, dtype):
        super().__init__()
        try:
            import transformer_engine.pytorch as te
        except ImportError as exc:  # explicit production dependency
            raise ImportError("grouped FP8 experts require Transformer Engine") from exc
        self.fc1 = te.GroupedLinear(
            num_experts, d_in, 2 * d_hidden, bias=False,
            device=device, params_dtype=dtype,
        )
        self.fc2 = te.GroupedLinear(
            num_experts, d_hidden, d_in, bias=False,
            device=device, params_dtype=dtype,
        )

    def forward(self, x: Tensor, m_splits: Tensor) -> Tensor:
        a, gate = self.fc1(x, m_splits).chunk(2, dim=-1)
        activated = F.silu(gate) * a
        activated = activated - activated.float().mean(dim=-1, keepdim=True).to(
            dtype=activated.dtype
        )
        return self.fc2(activated, m_splits)


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
        if fp8:
            if bias:
                raise ValueError("grouped FP8 LatentMoE experts require bias=False")
            self.grouped_experts = _GroupedExpertMLP(
                num_experts, self.latent_dim, self.routed_intermediate,
                device=device, dtype=dtype,
            )
            self.experts = None
        else:
            self.grouped_experts = None
            self.experts = nn.ModuleList(
                _ExpertMLP(self.latent_dim, self.routed_intermediate, bias=bias,
                           device=device, dtype=dtype, fp8=False)
                for _ in range(num_experts)
            )
        self.shared_expert = _ExpertMLP(
            d_model, self.shared_intermediate, bias=bias,
            device=device, dtype=dtype, fp8=fp8,
        )
        self.register_buffer("last_aux_loss", torch.zeros((), dtype=torch.float32), persistent=False)
        self.register_buffer("last_z_loss", torch.zeros((), dtype=torch.float32), persistent=False)
        self._shared_stream: Optional[torch.cuda.Stream] = None
        self._graph_grouped_splits: dict[int, Tensor] = {}
        # CUDA graph training sets this on all MoE modules before warmup so
        # autograd never retains dependencies on an uncaptured side stream.
        # Eager training leaves it false to overlap the shared expert with
        # routing and the selected experts.
        self.disable_async_shared = False

    def _fp8_context(self):
        if not self.fp8:
            return nullcontext()
        import transformer_engine.pytorch as te
        # ``common`` is a separate package in current Transformer Engine
        # releases (including the patched TE used by ``~/fp8``); older
        # releases happened to re-export it from ``transformer_engine.pytorch``.
        # Importing the public recipe module directly keeps the production FP8
        # path compatible without silently dropping to a non-FP8 implementation.
        from transformer_engine.common.recipe import DelayedScaling
        # Delayed scaling gives the MoE GEMMs amax history and keeps all gate
        # reductions outside FP8.  The caller may nest a project-wide TE context;
        # TE permits the nested context and reuses its metadata tensors.
        recipe = DelayedScaling(fp8_dpa=False, amax_history_len=16)
        return te.fp8_autocast(enabled=True, fp8_recipe=recipe)

    def _route(self, x: Tensor):
        # Router reductions are intentionally FP32 for stable top-k decisions.
        # The production model stores parameters in BF16, so calling the
        # module directly with ``x.float()`` would trigger a dtype mismatch.
        # Make the tiny router matmul explicitly homogeneous instead of
        # weakening the FP32 routing path or relying on implicit casts.
        router_weight = self.router.weight.float()
        router_bias = None if self.router.bias is None else self.router.bias.float()
        logits = F.linear(x.float(), router_weight, router_bias)
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

    def _dispatch(
        self, latent: Tensor, indices: Tensor, weights: Tensor, *, fp8_context: bool = True
    ) -> Tensor:
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
        capturing = latent.is_cuda and torch.cuda.is_current_stream_capturing()
        if not capturing:
            # Eager execution can use the exact variable-size dispatch.  This
            # avoids padding work and, unlike a fixed-capacity graph buffer,
            # never drops assignments when the router is temporarily skewed.
            counts = torch.bincount(sorted_experts, minlength=self.num_experts)
            if self.grouped_experts is None:
                result = torch.zeros_like(latent, dtype=torch.float32)
                start = 0
                context = self._fp8_context() if fp8_context else nullcontext()
                with context:
                    for expert_id, count in enumerate(counts.tolist()):
                        end = start + count
                        if count:
                            out = self.experts[expert_id](latent[sorted_tokens[start:end]])
                            result.index_add_(
                                0,
                                sorted_tokens[start:end],
                                out.float() * sorted_weights[start:end, None],
                            )
                        start = end
                return result
            # Eager grouped dispatch keeps exact variable expert counts, so it
            # does not pay graph-capacity padding.  The host read is outside
            # capture and is already required by TE's grouped metadata API.
            counts_cpu = counts.cpu().tolist()
            rounded = [((int(c) + 7) // 8) * 8 for c in counts_cpu]
            offsets = []
            total = 0
            for c in rounded:
                offsets.append(total)
                total += c
            starts = counts.cumsum(0) - counts
            ranks = torch.arange(sorted_experts.numel(), device=latent.device, dtype=torch.int64)
            ranks = ranks - starts.index_select(0, sorted_experts)
            destinations = torch.tensor(offsets, device=latent.device, dtype=torch.int64).index_select(
                0, sorted_experts
            ) + ranks
            packed = latent.new_zeros((total, latent.shape[-1]))
            packed.index_copy_(0, destinations, latent.index_select(0, sorted_tokens))
            m_splits = torch.tensor(rounded, dtype=torch.int64)
            context = self._fp8_context() if fp8_context else nullcontext()
            with context:
                packed_out = self.grouped_experts(packed, m_splits)
            selected = packed_out.index_select(0, destinations)
            result = torch.zeros_like(latent, dtype=torch.float32)
            result.index_add_(0, sorted_tokens, selected.float() * sorted_weights[:, None])
            return result
        # Keep routing entirely on-device. ``torch.bincount`` and converting
        # its result to Python integers are forbidden during CUDA graph
        # capture.  The fixed-size histogram below has static shape and uses
        # only capture-safe tensor operations.
        counts = torch.zeros(
            (self.num_experts,), dtype=torch.int64, device=latent.device
        )
        counts.scatter_add_(
            0, expert_ids,
            torch.ones_like(expert_ids, dtype=torch.int64),
        )
        # Give every expert a fixed, alignment-padded segment.  The segment
        # count depends only on the static activation shape, never on routing
        # data, so this remains valid during CUDA graph capture.
        align = 8 if self.fp8 else 1
        slots = sorted_experts.numel()
        # Graph capture has no variable-shape allocation.  Reserve the exact
        # worst-case capacity (all top-k assignments may select one expert) so
        # no routing assignment is silently dropped.
        capacity = ((latent.shape[0] + align - 1) // align) * align
        expert_starts = counts.cumsum(0) - counts
        ranks = torch.arange(slots, device=latent.device, dtype=torch.int64)
        ranks = ranks - expert_starts.index_select(0, sorted_experts)
        destinations = sorted_experts * capacity + ranks
        packed = latent.new_zeros((self.num_experts * capacity, latent.shape[-1]))
        values = latent.index_select(0, sorted_tokens)
        packed.index_copy_(0, destinations, values)
        context = self._fp8_context() if fp8_context else nullcontext()
        with context:
            if self.grouped_experts is not None:
                # A CPU split tensor is safe for TE's internal ``tolist``
                # conversion during capture and remains constant on replay.
                m_splits = self._graph_grouped_splits.get(capacity)
                if m_splits is None:
                    m_splits = torch.full(
                        (self.num_experts,), capacity, dtype=torch.int64
                    )
                    self._graph_grouped_splits[capacity] = m_splits
                packed_out = self.grouped_experts(packed, m_splits)
            else:
                packed_out = torch.empty_like(packed)
                for expert_id in range(self.num_experts):
                    start = expert_id * capacity
                    packed_out[start : start + capacity] = self.experts[expert_id](
                        packed[start : start + capacity]
                    )
            selected = packed_out.index_select(0, destinations)
        result.index_add_(0, sorted_tokens, selected.float() * sorted_weights[:, None])
        return result

    def _shared_forward(self, x: Tensor, *, fp8_context: bool = True) -> Tensor:
        context = self._fp8_context() if fp8_context else nullcontext()
        with context:
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
        # CUDA graph capture cannot import an event recorded on a stream that
        # was not itself captured.  Keep the eager latency-hiding stream, but
        # execute the shared expert on the capture stream so graph replay has
        # no cross-capture dependency.
        capture = flat.is_cuda and torch.cuda.is_current_stream_capturing()
        if flat.is_cuda and not capture and not self.disable_async_shared:
            if self._shared_stream is None or self._shared_stream.device != flat.device:
                self._shared_stream = torch.cuda.Stream(device=flat.device)
            stream = self._shared_stream
            event = torch.cuda.Event()
            with torch.cuda.stream(stream):
                shared = self._shared_forward(flat)
                event.record(stream)
        else:
            shared = None

        indices, weights, aux, z = self._route(flat)
        # Keep all main-stream MoE projections under one delayed-scaling
        # context. Re-entering it for down/dispatch/up forces repeated amax
        # reductions and metadata setup on every layer.
        main_context = self._fp8_context()
        with main_context:
            if shared is None:
                shared = self.shared_expert(flat)
            latent = self.down(flat)
            routed = self._dispatch(latent, indices, weights, fp8_context=False)
            output = self.up(routed.to(dtype=flat.dtype))
        if event is not None:
            torch.cuda.current_stream(flat.device).wait_event(event)
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
