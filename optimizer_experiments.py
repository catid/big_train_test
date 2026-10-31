"""Standalone AnchorMuon optimizer.

This file is intentionally self-contained and suitable for copying into another
project. It implements only the repo-wide winning optimizer family for the next
language-modeling round:

    SODA + row-only PMuonEq + Gram Newton-Schulz + NorMuon

The implementation deliberately omits the matrix-path research ablation
switches:

    no AMUSE / schedule-free train-eval sequence
    no MiMuon branch
    no full/stale PMuon dense covariance path
    no column PMuonEq scaling
    no post-NorMuon aspect multiplier
    no optional NorMuon disable path

Scalar/vector tensors use the AdamATan2 fallback. No alternate optimizer or
fallback mode is part of the canonical trainer.

The current best-supported recipe comes from a CIFAR-10 train/validation/test
split with the official test split evaluated only once at the end:

    model:    vit5_micro, 458,858 trainable parameters
    data:     CIFAR-10 45k train / 5k validation / official 10k test
    training: 50 epochs, 4,350 steps, batch size 512
    seeds:    123, 456, 789, 101112, 131415
    schedule: trainer-side WSD with 80-step warmup

    lr=0.014, fallback_lr=0.007, row_gamma=0.35,
    pmuoneq_beta=0.90, normuon_beta2=0.93

    official test loss: 0.4147 +/- 0.0152
    official test acc:  87.57% +/- 0.39
    step time:          17.51 +/- 0.16 ms

A 5-seed CIFAR-10 follow-up found that the fixed AdamATan2 fallback,
``lr=0.014``, and ``fallback_lr=0.007`` slightly improved final validation
accuracy/loss and was essentially tied with RMS on official test accuracy
within seed variance. AdamATan2 is now the constructor default for fallback
parameters so new users get that stronger validation-loss recipe by default.

Language-model transfer note: bounded byte-level 49.4M GPT checks now have two
useful starting points. WikiText-103 selected ``lr=0.0015``,
``fallback_lr=0.00075``, ``row_gamma=0.55``, and
``soda_lambda_scale=0.01`` after a 213-candidate 1200-step HPO and 1600-step
family replay. FineWeb-Edu needed its own tuning: a 255-candidate 1600-step HPO
with 10k-step family replay selected ``lr=0.0012``, ``fallback_lr=0.0012``,
``row_gamma=0.45``, ``soda_lambda_scale=0.003``,
``pmuoneq_beta=0.90``, and ``normuon_beta2=0.93``. That FineWeb point beat
tuned plain Muon by 1.1755 vs 1.1798 validation loss while running faster per
step. These are real-text but byte-level, single-seed optimizer-transfer
signals; treat them as starting points rather than standard LM perplexity
claims.

The aspect-scaled variant was close and sometimes won on other CIFAR proxies,
but no-aspect won the cleanest official-test protocol. This root file keeps
the no-aspect variant only. If aspect or column gamma is needed for an ablation,
use the older ``soda_pmuoneq_normuon.py`` research file instead.

Matrix update math
==================

For a matrix parameter ``W_t`` with gradient ``G_t``:

1. Momentum / Nesterov-style source:

       M_t = beta_m M_{t-1} + (1 - beta_m) G_t
       U_t = (1 - beta_m) G_t + beta_m M_t

2. Row-only PMuonEq:

       r_t = beta_p r_{t-1} + (1 - beta_p) mean_cols(G_t^2)
       a_t = normalize(r_t^{-gamma_row})
       A_t = a_t[:, None] * U_t

   This is the fast diagonal/row approximation to PMuon. It keeps the useful
   row reliability scaling before orthogonalization without dense covariance
   matrices, QR/eigendecomposition, or inverse-power matrix products.

3. Gram Newton-Schulz:

       P_t = GramNS(A_t)
       D_t = 0.2 * sqrt(max(rows, cols)) * P_t

4. NorMuon after GramNS:

       s_t = beta_n s_{t-1} + (1 - beta_n) mean_cols(D_t^2)
       N_t = D_t / sqrt(s_t + eps)
       N_t = N_t * ||D_t||_F / (||N_t||_F + eps)

   NorMuon changes row allocation after the spectral direction is produced,
   while preserving the whole-matrix Frobenius norm. There is no aspect-ratio
   multiplier in this standalone version.

5. SODA anchor pull and update:

       lambda_t = min(1, soda_lambda_scale / (t + 1)^soda_lambda_power)
       W_t <- (1 - lambda_t) W_t + lambda_t W_0
       W_{t+1} = W_t - lr_t N_t

Fallback parameters
===================

Scalar/vector tensors and any parameters explicitly placed in a fallback group
use the same SODA anchor plus AdamATan2. Automatic matrix eligibility is based
only on effective shape after ignoring singleton dimensions. Two-dimensional
weights use the spectral matrix path. Stacked expert banks use a batched
spectral path, keeping each expert's matrix independent; names are never used
for routing in this standalone file.

The fixed AdamATan2 fallback keeps first/second moments but uses a bounded
angular update:

       m_t = beta_1 m_{t-1} + (1 - beta_1) g_t
       v_t = beta_2 v_{t-1} + (1 - beta_2) g_t^2
       update = atan2(m_hat_t, sqrt(v_hat_t) + eps)

AnchorMuon applies decoupled matrix and fallback weight decay; SODA is an
additional anchor regularizer controlled by the trainer.

Recommended starting defaults
=============================

The constructor defaults are the recommended starting point. They are based on
the CIFAR-10 ViT-5 sweeps above plus cross-worker comparisons against AdamW and
nearby aspect/column/MiMuon ablations. New projects should start with:

    lr              = 8e-3
    fallback_lr     = lr
    momentum        = 0.95
    pmuoneq_beta    = 0.90
    row_gamma       = 0.35
    normuon_beta2   = 0.93
    trainer schedule = 80-step warmup + constant LR

Do not tune everything at once. Treat the knobs in three tiers:

    Tier 1, tune first:
        lr
            Main quality/speed knob for matrix weights. Try
            {0.0010, 0.0012, 0.0015} for 50M-class byte-level language
            models, or {0.004, 0.006, 0.008} for smaller ViTs. The current
            FineWeb-Edu 50M byte-level point used 0.0012; the WikiText-103
            point used 0.0015.

        row_gamma
            Strength of row-wise PMuonEq reliability scaling before GramNS.
            Default 0.35 was the best clean CIFAR setting. Try
            {0.35, 0.45, 0.55}; lower it if training is noisy or unstable.

    Tier 2, tune only after Tier 1:
        fallback_lr
            LR for scalars, vectors, and any tensors explicitly
            routed to fallback.
            Default matches lr because this is what the strongest
            reproduced CIFAR and FineWeb-Edu recipes used. WikiText-103 used
            half-rate AdamATan2 fallback, so tune this separately when moving
            datasets.

        normuon_beta2
            Row second-moment smoothing after GramNS. Default 0.93. Try
            {0.90, 0.93, 0.95}; higher is smoother, lower adapts faster.

        learning-rate schedule
            AnchorMuon intentionally does not own this. Training code should
            update param-group ``lr`` values for warmup, WSD, cosine, or any
            other schedule. The strongest reproduced CIFAR recipe used an
            80-step warmup plus constant LR.

        min_matrix_dim
            Safety threshold for the spectral path. A 2D parameter must have
            both flattened matrix dimensions at least this large. This keeps
            tiny projections out of GramNS by default.

    Tier 3, normally leave fixed:
        momentum = 0.95
            Momentum feeding the Nesterov-style matrix source.

        pmuoneq_beta = 0.90
            EMA coefficient for row gradient-power estimates.

        AdamATan2 fallback
            Scalar/vector AdamATan2 update; this is the only fallback path.

        fallback_betas = (0.9, 0.95)
            AdamATan2 first/second moment defaults.

        soda_lambda_scale = 1.0, soda_lambda_power = 1.0
            SODA anchor pull schedule. Changing these changes the regularizer,
            not just the step size.

        eps, pmuoneq_eps, normuon_eps, ns_epsilon
            Numerical-stability constants.

        ns_compute_dtype
            Leave None unless profiling shows the GramNS dtype choice is a
            bottleneck on the target hardware.

Language-modeling tuning order:

    1. Verify shape-based parameter grouping with ``optimizer.group_summary()``.
    2. Compare against the actual LM baseline optimizer, not only AdamW.
    3. Tune lr around the scale above; the WikiText-103 byte-level 49.4M GPT
       harder HPO selected 0.0015.
    4. Tune row_gamma in {0.35, 0.45, 0.55}.
    5. Tune normuon_beta2 in {0.90, 0.93, 0.95}.
    6. Tune pmuoneq_beta in {0.90, 0.95}.
    7. Tune soda_lambda_scale in {0.01, 0.03, 0.07, 0.1}; the LM check
       preferred much weaker SODA than CIFAR.
    8. Only then revisit fallback LR/weight decay.

DDP and batching
================

The optimizer does not perform distributed communication. Use normal PyTorch
DDP, which all-reduces gradients before ``optimizer.step()``. Optimizer state is
local and deterministic across ranks when gradients are synchronized. For GPU
efficiency, same-shape matrices in a parameter group are bucketed and processed
as a batch through PMuonEq, Gram Newton-Schulz, and NorMuon.

References and lineage
======================

This file combines ideas from several optimizer lines, but the implementation is
not a verbatim copy of any one paper:

* SODA anchor regularization follows the initialization-anchor/optimistic dual
  averaging idea from ``Optimistic Dual Averaging Unifies Modern Optimizers``
  (arXiv:2605.11172). Here SODA is always enabled alongside decoupled weight
  decay.
* PMuonEq is this repo's cheap row-only approximation to PMuon-style
  preconditioned Muon directions. Dense PMuon would use
  ``polar(L^-gamma @ momentum @ R^-gamma)``; AnchorMuon keeps only a row EMA
  scale before the polar approximation.
* The polar direction uses the Muon/Gram Newton-Schulz family of matrix
  orthogonalization, with coefficients from the Gram Newton-Schulz reference
  implementation at ``github.com/Dao-AILab/gram-newton-schulz``.
* NorMuon-style post-polar row normalization is from the HTMuon/NorMuon line,
  especially ``HTMuon: Improving Muon via Heavy-Tailed Spectral Correction``
  (arXiv:2603.10067). AnchorMuon applies it after GramNS and preserves the
  whole-matrix Frobenius norm.
* The optional AdamATan2 fallback follows Appendix C.5 of
  ``Scaling Exponents Across Parameterizations and Optimizers``
  (Everett et al., arXiv:2407.05872), which replaces Adam's
  ``m / sqrt(v)`` update with ``atan2(m, sqrt(v))`` to remove the additive
  epsilon sensitivity and make the update scale-invariant up to precision
  limits.
* Learning-rate schedules such as warmup+constant, WSD, linear decay, and
  cosine decay are intentionally trainer-side policies. AnchorMuon consumes the
  current param-group ``lr`` and does not implement a scheduler internally.
"""

from __future__ import annotations

import math
from collections import defaultdict
from collections.abc import Callable, Iterable
from typing import Any, TypeAlias

import torch


# Full 623M profiling on B300: 32-entry GramNS banks are about 21% faster
# than the previous eight-entry banks, while 64-entry banks add memory without
# improving the optimizer tail. Keep this as one canonical setting rather than
# exposing an unvalidated optimizer ablation through the production trainer.
MATRIX_BUCKET_CHUNK_SIZE = 32


__version__ = "0.5.1"

try:
    from torch.optim.optimizer import ParamsT
except ImportError:  # pragma: no cover
    ParamsT: TypeAlias = Iterable[torch.Tensor] | Iterable[dict[str, Any]]


POLAR_EXPRESS_UNSCALED: tuple[tuple[float, float, float], ...] = (
    (8.28721201814563, -23.595886519098837, 17.300387312530933),
    (4.107059111542203, -2.9478499167379106, 0.5448431082926601),
    (3.9486908534822946, -2.908902115962949, 0.5518191394370137),
    (3.3184196573706015, -2.488488024314874, 0.51004894012372),
    (2.300652019954817, -1.6689039845747493, 0.4188073119525673),
)
POLAR_EXPRESS_SAFETY_FACTOR = 1.05
POLAR_EXPRESS_COEFFICIENTS: tuple[tuple[float, float, float], ...] = tuple(
    (
        a / POLAR_EXPRESS_SAFETY_FACTOR,
        b / POLAR_EXPRESS_SAFETY_FACTOR**3,
        c / POLAR_EXPRESS_SAFETY_FACTOR**5,
    )
    for a, b, c in POLAR_EXPRESS_UNSCALED
)

@torch.no_grad()
def _matrix_view(x: torch.Tensor) -> torch.Tensor:
    effective_shape = tuple(int(dim) for dim in x.shape if int(dim) > 1)
    if len(effective_shape) < 2:
        raise ValueError("matrix update requires at least two non-singleton dimensions")
    if len(effective_shape) == 2:
        return x.reshape(effective_shape)
    # Preserve the final two matrix dimensions and flatten only leading bank
    # dimensions. This makes [experts, out, in] a batch of independent Muon
    # matrices instead of one matrix that couples all experts.
    rows, cols = effective_shape[-2:]
    banks = math.prod(effective_shape[:-2])
    return x.reshape(banks, rows, cols)


@torch.no_grad()
def _diag_inverse_power(values: torch.Tensor, *, gamma: float, eps: float) -> torch.Tensor:
    x = values.to(torch.float32)
    # Scale the ridge relative to the observed row energy.  Clamping this
    # mean to one makes the fixed 1e-6 epsilon dominate the tiny gradients of
    # a BF16 transformer and collapses PMuonEq to an almost uniform factor.
    scale = (
        x.mean(dim=-1, keepdim=True).clamp_min(eps)
        if x.ndim > 1
        else x.mean().clamp_min(eps)
    )
    lam = (x + eps * scale).clamp_min(eps * scale)
    out = lam.pow(-float(gamma))
    norm = out.norm(dim=-1, keepdim=True).clamp_min(eps) if out.ndim > 1 else out.norm().clamp_min(eps)
    return out * (math.sqrt(out.size(-1)) / norm)


@torch.no_grad()
def _normuon_row_normalize(
    update: torch.Tensor,
    second_momentum: torch.Tensor,
    *,
    beta2: float,
    eps: float,
) -> torch.Tensor:
    if tuple(second_momentum.shape[-2:]) != (update.shape[-2], 1):
        raise ValueError(
            f"NorMuon row second momentum shape {tuple(second_momentum.shape[-2:])} "
            f"!= {(update.shape[-2], 1)}"
        )
    dtype = update.dtype
    eps_t = torch.tensor(eps, dtype=dtype, device=update.device)
    old_norm = update.norm(dim=(-2, -1), keepdim=True)
    try:
        row_power = update.square().mean(dim=-1, keepdim=True, dtype=dtype)
    except TypeError:
        row_power = update.square().mean(dim=-1, keepdim=True).to(dtype)
    second_momentum.lerp_(row_power, 1.0 - beta2)
    out = update * torch.rsqrt(second_momentum + eps_t)
    return out * (old_norm / (out.norm(dim=(-2, -1), keepdim=True) + eps_t))


def _is_matrix_like_parameter(param: torch.Tensor, *, min_matrix_dim: int) -> bool:
    """Return whether ``param`` is large enough for the spectral matrix path.

    Singleton dimensions do not make a tensor matrix-like. For example, ViT
    ``cls_token`` with shape ``[1, 1, width]`` is effectively a vector, while
    ``pos_embed`` with shape ``[1, patches, width]`` is effectively a matrix.
    """

    if not param.is_floating_point():
        return False
    effective_shape = tuple(int(dim) for dim in param.shape if int(dim) > 1)
    if len(effective_shape) < 2:
        return False
    # The optimizer treats leading dimensions as independent matrix banks.
    # Eligibility must therefore use the final two matrix dimensions; using
    # the first bank dimension would incorrectly route [8, 1024, 1536] by
    # ``8`` and would route skinny [20, 6144, 4] tensors through GramNS.
    rows, cols = effective_shape[-2:]
    return min(rows, cols) >= int(min_matrix_dim)


class GramNewtonSchulz:
    """Batched Gram Newton-Schulz polar approximation used by AnchorMuon."""

    def __init__(
        self,
        *,
        coefficients: Iterable[Iterable[float]] | None = None,
        epsilon: float = 1e-7,
        reset_iterations: Iterable[int] = (2,),
        compute_dtype: torch.dtype | None = None,
    ) -> None:
        self.coefficients = tuple(tuple(float(v) for v in row) for row in (coefficients or POLAR_EXPRESS_COEFFICIENTS))
        self.epsilon = float(epsilon)
        self.reset_iterations = set(int(i) for i in reset_iterations)
        self.compute_dtype = compute_dtype

    def _compute_dtype_for(self, x: torch.Tensor) -> torch.dtype:
        if self.compute_dtype is not None:
            return self.compute_dtype
        return torch.float16 if x.is_cuda else torch.float32

    @torch.no_grad()
    def __call__(self, matrix: torch.Tensor) -> torch.Tensor:
        if matrix.ndim < 2:
            raise ValueError("GramNewtonSchulz expects a tensor with ndim >= 2")

        original_shape = matrix.shape
        x = matrix
        if x.ndim == 2:
            x = x.unsqueeze(0)
        elif x.ndim > 3:
            x = x.reshape(-1, *x.shape[-2:])

        original_dtype = x.dtype
        x = x.to(torch.float32)
        transposed = x.size(-2) > x.size(-1)
        if transposed:
            x = x.mT

        x = x / (x.norm(dim=(-2, -1), keepdim=True) + self.epsilon)
        x = x.to(self._compute_dtype_for(x))

        if max(x.shape[-2:]) > min(x.shape[-2:]):
            x = self._gram_recurrence(x)
        else:
            x = self._standard_recurrence(x)

        if transposed:
            x = x.mT
        return x.to(original_dtype).reshape(original_shape)

    def _standard_recurrence(self, x: torch.Tensor) -> torch.Tensor:
        for a, b, c in self.coefficients:
            gram = x @ x.mT
            # Avoid cuBLASLt's failing nonzero-beta BF16/FP16 addmm path on
            # B300.  These are algebraically identical to baddbmm calls.
            poly = c * (gram @ gram) + b * gram
            x = a * x + poly @ x
        return x

    def _gram_recurrence(self, x: torch.Tensor) -> torch.Tensor:
        """Run the rectangular Gram-NS inverse-estimate recurrence.

        The Gram path keeps a small inverse estimate ``q`` in the reduced
        dimension and refreshes it at the configured reset iterations. This is
        materially different from the square polynomial recurrence and avoids
        paying a second full-size matrix product for rectangular expert and
        projection banks.
        """
        gram = x @ x.mT
        eye = torch.eye(
            gram.size(-1), device=x.device, dtype=x.dtype
        ).expand(gram.size(0), -1, -1).contiguous()
        q = None
        for i, (a, b, c) in enumerate(self.coefficients):
            if i in self.reset_iterations and i != 0:
                if q is None:
                    raise RuntimeError("GramNS reset reached without inverse estimate")
                x = q @ x
                gram = x @ x.mT
                q = None

            # Avoid cuBLASLt's failing nonzero-beta BF16/FP16 baddbmm path on
            # B300.  Keep every operation as an explicit product plus scale;
            # this is algebraically identical and matches the square path.
            z = b * gram + c * (gram @ gram)
            if i == 0 or i in self.reset_iterations:
                q = z + a * eye
            else:
                if q is None:
                    raise RuntimeError("GramNS inverse estimate was not initialized")
                q = a * q + q @ z

            if i < len(self.coefficients) - 1 and i + 1 not in self.reset_iterations:
                rz = a * gram + gram @ z
                gram = a * rz + rz @ z

        if q is None:
            raise RuntimeError("GramNS finished without an inverse estimate")
        return q @ x


class AnchorMuon(torch.optim.Optimizer):
    """PyTorch optimizer implementing the focused AnchorMuon path.

    Preferred use:

        optimizer = AnchorMuon(model)

    Passing a ``torch.nn.Module``, ``model.parameters()``, or
    ``model.named_parameters()`` lets the optimizer route tensors by effective
    shape: matrix-like tensors use PMuonEq/GramNS/NorMuon, while scalar/vector
    tensors use the fallback path. Parameter names are accepted only for
    reporting in ``group_summary()``.

    Call ``optimizer.group_summary()`` before long runs to audit this routing.

    The defaults are intentionally usable. For a new workload, tune ``lr`` and
    ``row_gamma`` first; tune ``fallback_lr`` and ``normuon_beta2`` second;
    leave the remaining arguments fixed unless a targeted diagnostic gives a
    reason to move them. AnchorMuon consumes the current param-group ``lr`` and
    does not own warmup or decay schedules; use normal trainer-side schedulers.
    """

    def __init__(
        self,
        params: ParamsT | Iterable[tuple[str, torch.nn.Parameter]] | torch.nn.Module,
        *,
        lr: float = 8e-3,
        fallback_lr: float | None = None,
        momentum: float = 0.95,
        fallback_betas: tuple[float, float] = (0.9, 0.95),
        fallback_weight_decay: float = 0.0,
        eps: float = 1e-8,
        soda_lambda_scale: float = 1.0,
        soda_lambda_power: float = 1.0,
        pmuoneq_beta: float = 0.90,
        row_gamma: float = 0.35,
        pmuoneq_eps: float = 1e-6,
        normuon_beta2: float = 0.93,
        normuon_eps: float = 1e-10,
        ns_epsilon: float = 1e-7,
        ns_compute_dtype: torch.dtype | None = None,
        min_matrix_dim: int = 32,
        matrix_weight_decay: float = 0.0,
    ) -> None:
        if isinstance(params, torch.nn.Module):
            params = params.parameters()
        fallback_lr = lr if fallback_lr is None else fallback_lr
        if lr < 0.0 or fallback_lr < 0.0:
            raise ValueError("learning rates must be non-negative")
        if fallback_weight_decay < 0.0:
            raise ValueError("fallback_weight_decay must be non-negative")
        if matrix_weight_decay < 0.0:
            raise ValueError("matrix_weight_decay must be non-negative")
        if not 0.0 <= momentum < 1.0:
            raise ValueError("momentum must be in [0, 1)")
        if not 0.0 <= pmuoneq_beta < 1.0:
            raise ValueError("pmuoneq_beta must be in [0, 1)")
        if row_gamma < 0.0:
            raise ValueError("row_gamma must be non-negative")
        if not 0.0 <= normuon_beta2 < 1.0:
            raise ValueError("normuon_beta2 must be in [0, 1)")
        if len(fallback_betas) != 2 or not all(0.0 <= beta < 1.0 for beta in fallback_betas):
            raise ValueError("fallback_betas must contain two values in [0, 1)")
        if eps <= 0.0 or pmuoneq_eps <= 0.0 or normuon_eps <= 0.0 or ns_epsilon <= 0.0:
            raise ValueError("epsilon values must be positive")
        if min_matrix_dim < 1:
            raise ValueError("min_matrix_dim must be >= 1")
        if soda_lambda_scale <= 0.0:
            raise ValueError("soda_lambda_scale must be positive")
        if soda_lambda_power <= 0.0:
            raise ValueError("soda_lambda_power must be positive")
        prepared = self._prepare_param_groups(
            params,
            lr=lr,
            fallback_lr=fallback_lr,
            momentum=momentum,
            fallback_betas=fallback_betas,
            fallback_weight_decay=fallback_weight_decay,
            eps=eps,
            pmuoneq_beta=pmuoneq_beta,
            row_gamma=row_gamma,
            pmuoneq_eps=pmuoneq_eps,
            normuon_beta2=normuon_beta2,
            normuon_eps=normuon_eps,
            min_matrix_dim=min_matrix_dim,
        )
        super().__init__(prepared, defaults={})

        self.soda_lambda_scale = float(soda_lambda_scale)
        self.soda_lambda_power = float(soda_lambda_power)
        self.matrix_weight_decay = float(matrix_weight_decay)
        self._orthogonalizer = GramNewtonSchulz(epsilon=ns_epsilon, compute_dtype=ns_compute_dtype)
        self.last_stats: dict[str, float] = {}

        for group in self.param_groups:
            group.setdefault("k", 0)

    @classmethod
    def from_model(cls, model: torch.nn.Module, **kwargs: Any) -> "AnchorMuon":
        """Construct the optimizer from ``model.parameters()``.

        This is the least error-prone public API for normal training code.
        """

        return cls(model.parameters(), **kwargs)

    @staticmethod
    def _prepare_param_groups(
        params: ParamsT | Iterable[tuple[str, torch.nn.Parameter]],
        *,
        lr: float,
        fallback_lr: float,
        momentum: float,
        fallback_betas: tuple[float, float],
        fallback_weight_decay: float,
        eps: float,
        pmuoneq_beta: float,
        row_gamma: float,
        pmuoneq_eps: float,
        normuon_beta2: float,
        normuon_eps: float,
        min_matrix_dim: int,
    ) -> list[dict[str, Any]]:
        items = list(params)
        if not items:
            raise ValueError("optimizer got an empty parameter list")
        if (
            isinstance(items[0], tuple)
            and len(items[0]) == 2
            and isinstance(items[0][0], str)
            and isinstance(items[0][1], torch.Tensor)
        ):
            return build_param_groups(
                items,  # type: ignore[arg-type]
                lr=lr,
                fallback_lr=fallback_lr,
                momentum=momentum,
                fallback_betas=fallback_betas,
                fallback_weight_decay=fallback_weight_decay,
                eps=eps,
                pmuoneq_beta=pmuoneq_beta,
                row_gamma=row_gamma,
                pmuoneq_eps=pmuoneq_eps,
                normuon_beta2=normuon_beta2,
                normuon_eps=normuon_eps,
                min_matrix_dim=min_matrix_dim,
            )
        if isinstance(items[0], dict):
            groups: list[dict[str, Any]] = []
            seen_params: set[int] = set()

            def apply_defaults(group: dict[str, Any]) -> None:
                is_matrix = bool(group["use_matrix_update"])
                group.setdefault("lr", lr if is_matrix else fallback_lr)
                group.pop("weight_decay", None)
                if is_matrix:
                    group.setdefault("momentum", momentum)
                    group.setdefault("pmuoneq_beta", pmuoneq_beta)
                    group.setdefault("row_gamma", row_gamma)
                    group.setdefault("pmuoneq_eps", pmuoneq_eps)
                    group.setdefault("normuon_beta2", normuon_beta2)
                    group.setdefault("normuon_eps", normuon_eps)
                    group.setdefault("min_matrix_dim", min_matrix_dim)
                    # Forced AdamATan2 parameters (for example MSAR bias)
                    # retain the fallback schedule even though they share a
                    # matrix group for compact optimizer state.
                    group.setdefault("fallback_lr", fallback_lr)
                    group.setdefault("betas", fallback_betas)
                    group.setdefault("fallback_weight_decay", fallback_weight_decay)
                    group.setdefault("eps", eps)
                else:
                    group.setdefault("betas", fallback_betas)
                    group.setdefault("fallback_weight_decay", fallback_weight_decay)
                    group.setdefault("eps", eps)
                if is_matrix:
                    group.setdefault("betas", fallback_betas)
                    group.setdefault("fallback_weight_decay", fallback_weight_decay)
                    group.setdefault("eps", eps)

            def normalized_params(group: dict[str, Any]) -> tuple[list[torch.Tensor], list[str], bool]:
                params_value = group.get("params", [])
                params_in = [params_value] if isinstance(params_value, torch.Tensor) else list(params_value)
                names_in = list(group.get("param_names", []))
                keep_names = len(names_in) == len(params_in)
                params_out: list[torch.Tensor] = []
                names_out: list[str] = []
                for idx, p in enumerate(params_in):
                    ident = id(p)
                    if ident in seen_params:
                        continue
                    seen_params.add(ident)
                    params_out.append(p)
                    if keep_names:
                        names_out.append(names_in[idx])
                return params_out, names_out, keep_names

            for original in items:  # type: ignore[assignment]
                group = dict(original)
                params_in, names_in, keep_names = normalized_params(group)
                if not params_in:
                    continue
                group["params"] = params_in
                if keep_names:
                    group["param_names"] = names_in
                else:
                    group.pop("param_names", None)
                route_is_explicit = "use_matrix_update" in group or "use_muon" in group
                if route_is_explicit:
                    if "use_matrix_update" not in group:
                        group["use_matrix_update"] = group["use_muon"]
                    apply_defaults(group)
                    groups.append(group)
                    continue

                matrix_params: list[torch.Tensor] = []
                matrix_names: list[str] = []
                fallback_params: list[torch.Tensor] = []
                fallback_names: list[str] = []
                for idx, p in enumerate(params_in):
                    use_matrix = (
                        isinstance(p, torch.Tensor)
                        and p.requires_grad
                        and _is_matrix_like_parameter(p, min_matrix_dim=min_matrix_dim)
                    )
                    if use_matrix:
                        matrix_params.append(p)
                        if keep_names:
                            matrix_names.append(names_in[idx])
                    else:
                        fallback_params.append(p)
                        if keep_names:
                            fallback_names.append(names_in[idx])

                base_group = {key: value for key, value in group.items() if key not in {"params", "param_names"}}
                if matrix_params:
                    matrix_group = dict(base_group)
                    matrix_group["params"] = matrix_params
                    if keep_names:
                        matrix_group["param_names"] = matrix_names
                    matrix_group["use_matrix_update"] = True
                    apply_defaults(matrix_group)
                    groups.append(matrix_group)
                if fallback_params:
                    fallback_group = dict(base_group)
                    fallback_group["params"] = fallback_params
                    if keep_names:
                        fallback_group["param_names"] = fallback_names
                    fallback_group["use_matrix_update"] = False
                    apply_defaults(fallback_group)
                    groups.append(fallback_group)
            return groups

        matrix_params: list[torch.Tensor] = []
        fallback_params: list[torch.Tensor] = []
        seen_params: set[int] = set()
        for p in items:  # type: ignore[assignment]
            if not isinstance(p, torch.Tensor):
                raise TypeError("params must be tensors or optimizer param-group dictionaries")
            ident = id(p)
            if ident in seen_params:
                continue
            seen_params.add(ident)
            if p.requires_grad and _is_matrix_like_parameter(p, min_matrix_dim=min_matrix_dim):
                matrix_params.append(p)
            else:
                fallback_params.append(p)

        groups: list[dict[str, Any]] = []
        if matrix_params:
            groups.append(
                {
                    "params": matrix_params,
                    "use_matrix_update": True,
                    "lr": lr,
                    "momentum": momentum,
                    "pmuoneq_beta": pmuoneq_beta,
                    "row_gamma": row_gamma,
                    "pmuoneq_eps": pmuoneq_eps,
                    "normuon_beta2": normuon_beta2,
                    "normuon_eps": normuon_eps,
                    "min_matrix_dim": min_matrix_dim,
                    "fallback_lr": fallback_lr,
                    "betas": fallback_betas,
                    "fallback_weight_decay": fallback_weight_decay,
                    "eps": eps,
                }
            )
        if fallback_params:
            groups.append(
                {
                    "params": fallback_params,
                    "use_matrix_update": False,
                    "lr": fallback_lr,
                    "betas": fallback_betas,
                    "fallback_weight_decay": fallback_weight_decay,
                    "eps": eps,
                }
            )
        return groups

    def train(self) -> "AnchorMuon":
        return self

    def eval(self) -> "AnchorMuon":
        return self

    def load_state_dict(self, state_dict: dict[str, Any]) -> None:
        """Restore optimizer state without narrowing its FP32 accumulators.

        ``torch.optim.Optimizer.load_state_dict`` casts floating state tensors
        to the destination parameter dtype.  The model parameters are BF16,
        while AnchorMuon deliberately keeps its master parameters, SODA
        anchors, momentum, row statistics, and AdamATan2 moments in FP32.
        Re-cast those tensors after the base loader runs so a checkpoint
        resume has the same numerical state as a continuous run.
        """
        # PyTorch's generic loader casts every floating state tensor to the
        # destination parameter dtype.  Capture the serialized state first,
        # then restore it by parameter order after the loader has rebuilt its
        # parameter-id map; otherwise FP32 masters would be irreversibly
        # narrowed to BF16 during resume.
        saved_state = state_dict.get("state", {})
        saved_groups = state_dict.get("param_groups", [])
        super().load_state_dict(state_dict)
        for saved_group, current_group in zip(saved_groups, self.param_groups):
            for saved_id, param in zip(saved_group.get("params", []), current_group["params"]):
                source = saved_state.get(saved_id)
                if not isinstance(source, dict):
                    continue
                destination = self.state[param]
                for key, value in source.items():
                    if isinstance(value, torch.Tensor) and value.is_floating_point():
                        # ``Optimizer.state_dict`` is intentionally a shallow
                        # copy.  A caller may therefore pass the returned
                        # dictionary directly to a second optimizer (as
                        # happens in parity checks) while the first optimizer
                        # is still live.  Reusing a same-device tensor here
                        # would alias the two optimizers' moments and make the
                        # second step depend on whether the first optimizer
                        # happened to run first.  Clone every restored tensor
                        # so checkpoint restore has value semantics even when
                        # no serialization round-trip intervenes.
                        destination[key] = value.detach().to(
                            device=param.device, dtype=torch.float32
                        ).clone(memory_format=torch.preserve_format)
        fp32_keys = {
            "master_param",
            "soda_z0",
            "momentum_buffer",
            "pmuoneq_row_ema",
            "normuon_second_momentum",
            "exp_avg",
            "exp_avg_sq",
        }
        for param, state in self.state.items():
            for key in fp32_keys:
                value = state.get(key)
                if isinstance(value, torch.Tensor):
                    state[key] = value.to(device=param.device, dtype=torch.float32)

    def group_summary(self) -> list[dict[str, Any]]:
        """Return a small serializable summary of optimizer parameter routing."""

        summary: list[dict[str, Any]] = []
        for index, group in enumerate(self.param_groups):
            params = list(group["params"])
            names = list(group.get("param_names", []))
            summary.append(
                {
                    "index": index,
                    "use_matrix_update": bool(group.get("use_matrix_update", False)),
                    "lr": float(group.get("lr", 0.0)),
                    "param_count": len(params),
                    "numel": int(sum(p.numel() for p in params)),
                    "named": bool(names),
                    "param_names": names,
                    "fallback_mode": "atan2",
                }
            )
        return summary

    def _soda_anchor(self, p: torch.Tensor) -> torch.Tensor:
        state = self.state[p]
        master = self._master_param(p)
        anchor = state.get("soda_z0")
        if not isinstance(anchor, torch.Tensor) or anchor.shape != p.shape:
            anchor = state["soda_z0"] = torch.clone(master, memory_format=torch.preserve_format)
        elif anchor.device != p.device or anchor.dtype != torch.float32:
            anchor = state["soda_z0"] = anchor.to(device=p.device, dtype=torch.float32)
        return anchor

    def _master_param(self, p: torch.Tensor) -> torch.Tensor:
        """Return the FP32 update copy for a low-precision model parameter.

        Forward and backward stay on the fused BF16 model weights, while all
        optimizer arithmetic is applied to this persistent FP32 copy.  The
        rounded BF16 value is copied back only after a complete update, so
        small Muon/AdamATan2 and decay updates are not lost to BF16 spacing.
        """
        if p.dtype == torch.float32:
            return p
        state = self.state[p]
        master = state.get("master_param")
        if not isinstance(master, torch.Tensor) or master.shape != p.shape:
            master = state["master_param"] = p.detach().to(torch.float32).clone(
                memory_format=torch.preserve_format
            )
        elif master.device != p.device or master.dtype != torch.float32:
            master = state["master_param"] = master.to(device=p.device, dtype=torch.float32)
        return master

    def _soda_weight(self, t: int) -> float:
        return min(1.0, self.soda_lambda_scale / float(t + 1) ** self.soda_lambda_power)

    def _pre_matrix_source(self, p: torch.Tensor, grad: torch.Tensor, momentum_beta: float) -> tuple[torch.Tensor, torch.Tensor]:
        state = self.state[p]
        momentum = state.get("momentum_buffer")
        if isinstance(momentum, torch.Tensor) and momentum.shape == grad.shape:
            if momentum.device != grad.device or momentum.dtype != torch.float32:
                momentum = state["momentum_buffer"] = momentum.to(device=grad.device, dtype=torch.float32)
        else:
            momentum = state["momentum_buffer"] = torch.zeros_like(grad, dtype=torch.float32)
        g = grad.detach().to(torch.float32)
        momentum.lerp_(g, 1.0 - momentum_beta)
        source = torch.lerp(g, momentum, momentum_beta)
        return _matrix_view(source), _matrix_view(g)

    def _ensure_matrix_state(
        self,
        p: torch.Tensor,
        row_shape: tuple[int, ...],
        device: torch.device,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        state = self.state[p]
        row_ema = state.get("pmuoneq_row_ema")
        if isinstance(row_ema, torch.Tensor) and tuple(row_ema.shape) == row_shape:
            if row_ema.device != device or row_ema.dtype != torch.float32:
                row_ema = state["pmuoneq_row_ema"] = row_ema.to(device=device, dtype=torch.float32)
        else:
            # Start with no observed gradient energy.  The row factors are
            # normalized after the ridge, so a zero EMA gives the first real
            # gradient an immediate relative preconditioner; initializing to
            # one would suppress PMuonEq for hundreds of updates on this
            # workload's small gradients.
            row_ema = state["pmuoneq_row_ema"] = torch.zeros(*row_shape, device=device, dtype=torch.float32)
        second = state.get("normuon_second_momentum")
        second_shape = (*row_shape, 1)
        if isinstance(second, torch.Tensor) and second.shape == second_shape:
            if second.device != device or second.dtype != torch.float32:
                second = state["normuon_second_momentum"] = second.to(device=device, dtype=torch.float32)
        else:
            second = state["normuon_second_momentum"] = torch.zeros(*second_shape, device=device, dtype=torch.float32)
        return row_ema, second

    def _transform_matrix_bucket(
        self,
        entries: list[tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]],
        group: dict[str, Any],
    ) -> torch.Tensor:
        sources = torch.stack([entry[2].to(torch.float32) for entry in entries], dim=0)
        grads = torch.stack([entry[3].to(torch.float32) for entry in entries], dim=0)

        row_buffers: list[torch.Tensor] = []
        second_buffers: list[torch.Tensor] = []
        for p, _anchor, source, _grad in entries:
            # State follows every independent matrix in a stacked bank. For a
            # normal 2-D weight this is simply (rows,), while an expert bank
            # has shape (experts, rows).
            row_shape = tuple(int(v) for v in source.shape[:-1])
            row_ema, second = self._ensure_matrix_state(p, row_shape, grads.device)
            row_buffers.append(row_ema)
            second_buffers.append(second)

        row_stack = torch.stack(row_buffers, dim=0)
        beta_p = float(group["pmuoneq_beta"])
        row_stack.mul_(beta_p).add_(grads.square().mean(dim=-1), alpha=1.0 - beta_p)
        row_factor = _diag_inverse_power(
            row_stack, gamma=float(group["row_gamma"]), eps=float(group["pmuoneq_eps"])
        )

        preconditioned = sources * row_factor.unsqueeze(-1)
        update = self._orthogonalizer(preconditioned)
        update = update * (0.2 * math.sqrt(max(update.size(-2), update.size(-1))))

        second_stack = torch.stack(second_buffers, dim=0)
        update = _normuon_row_normalize(
            update,
            second_stack,
            beta2=float(group["normuon_beta2"]),
            eps=float(group["normuon_eps"]),
        )

        for idx, (p, _anchor, _source, _grad) in enumerate(entries):
            self.state[p]["pmuoneq_row_ema"].copy_(row_stack[idx])
            self.state[p]["normuon_second_momentum"].copy_(second_stack[idx])
        return update

    def _apply_matrix_residual_updates(
        self,
        entries: list[tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]],
        updates: torch.Tensor,
        *,
        lr: float,
        soda_weight: float,
    ) -> int:
        """Apply independent matrix residual updates with foreach kernels.

        The residual equation for each matrix is unchanged: SODA anchor pull,
        optional decoupled decay, then the GramNS update.  Foreach only batches
        these elementwise operations across the already same-shape chunk; the
        FP32 master-to-parameter copies remain per parameter so dtype and
        storage semantics stay identical to the scalar path.
        """
        records = []
        for (p, anchor, _source, _grad), update in zip(
            entries, updates.unbind(0), strict=True
        ):
            master = self._master_param(p)
            records.append(
                (
                    p,
                    master,
                    anchor,
                    update.reshape_as(master),
                    bool(getattr(p, "_no_weight_decay", False)),
                )
            )
        if not records:
            return 0

        masters = [record[1] for record in records]
        anchors = [record[2] for record in records]
        torch._foreach_lerp_(masters, anchors, soda_weight)

        if self.matrix_weight_decay:
            decay_records = [record for record in records if not record[4]]
            if decay_records:
                torch._foreach_mul_(
                    [record[1] for record in decay_records],
                    1.0 - lr * self.matrix_weight_decay,
                )

        torch._foreach_add_(
            masters,
            [record[3] for record in records],
            alpha=-lr,
        )
        for p, master, _anchor, _update, _no_weight_decay in records:
            if master is not p:
                p.copy_(master.to(dtype=p.dtype))
        return len(records)

    def _step_matrix_group(self, group: dict[str, Any], *, lr: float, t: int) -> dict[str, float]:
        beta_m = float(group["momentum"])
        entries: list[tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]] = []
        fallback_count = 0
        for p in group["params"]:
            grad = p.grad
            if grad is None:
                continue
            if grad.is_sparse:
                raise RuntimeError(
                    "AnchorMuon does not support sparse gradients. "
                    "Use dense gradients for this parameter or a different optimizer."
                )
            if getattr(p, "_force_fallback", False) or not _is_matrix_like_parameter(
                grad, min_matrix_dim=int(group.get("min_matrix_dim", 32))
            ):
                fallback_count += self._step_fallback_param(
                    p, group, lr=float(group.get("fallback_lr", lr)), t=t
                )
                continue
            anchor = self._soda_anchor(p)
            source, grad_matrix = self._pre_matrix_source(p, grad, beta_m)
            entries.append((p, anchor, source, grad_matrix))

        if not entries:
            return {
                "matrix_count": 0.0,
                "fallback_count": float(fallback_count),
                "soda_weight": float(self._soda_weight(t)),
            }

        buckets: dict[tuple[torch.device, torch.Size], list[tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]]] = defaultdict(list)
        for entry in entries:
            buckets[(entry[2].device, entry[2].shape)].append(entry)

        soda_weight = self._soda_weight(t)
        matrix_count = 0
        for bucket_entries in buckets.values():
            # GramNS forms several square intermediates whose size is the
            # reduced matrix dimension. Process a bounded number of same-shape
            # banks at a time so optimizer work cannot consume the last few
            # GiB after a large CUDA-graph private pool has been captured.
            for start in range(0, len(bucket_entries), MATRIX_BUCKET_CHUNK_SIZE):
                chunk_entries = bucket_entries[
                    start : start + MATRIX_BUCKET_CHUNK_SIZE
                ]
                updates = self._transform_matrix_bucket(chunk_entries, group)
                matrix_count += self._apply_matrix_residual_updates(
                    chunk_entries,
                    updates,
                    lr=lr,
                    soda_weight=soda_weight,
                )

        return {
            "matrix_count": float(matrix_count),
            "fallback_count": float(fallback_count),
            "soda_weight": float(soda_weight),
        }

    def _step_fallback_param(
        self,
        p: torch.Tensor,
        group: dict[str, Any],
        *,
        lr: float,
        t: int,
    ) -> int:
        grad = p.grad
        if grad is None:
            return 0
        if grad.is_sparse:
            raise RuntimeError(
                "AnchorMuon does not support sparse gradients. "
                    "Use dense gradients for this parameter or a different optimizer."
            )
        state = self.state[p]
        exp_avg_sq = state.get("exp_avg_sq")
        if isinstance(exp_avg_sq, torch.Tensor) and exp_avg_sq.shape == p.shape:
            if exp_avg_sq.device != p.device or exp_avg_sq.dtype != torch.float32:
                exp_avg_sq = state["exp_avg_sq"] = exp_avg_sq.to(device=p.device, dtype=torch.float32)
        else:
            exp_avg_sq = state["exp_avg_sq"] = torch.zeros_like(p, dtype=torch.float32)
            state["step"] = 0
        exp_avg = state.get("exp_avg")
        if not isinstance(exp_avg, torch.Tensor) or exp_avg.shape != p.shape:
            exp_avg = state["exp_avg"] = torch.zeros_like(p, dtype=torch.float32)
        elif isinstance(exp_avg, torch.Tensor) and (exp_avg.device != p.device or exp_avg.dtype != torch.float32):
            exp_avg = state["exp_avg"] = exp_avg.to(device=p.device, dtype=torch.float32)
        state["step"] = int(state.get("step", 0)) + 1
        beta1, beta2 = group.get("betas", (0.9, 0.999))
        eps = float(group.get("eps", 1e-8))
        g = grad.detach().to(torch.float32)
        master = self._master_param(p)
        master.lerp_(end=self._soda_anchor(p), weight=self._soda_weight(t))
        fallback_weight_decay = 0.0 if getattr(p, "_no_weight_decay", False) else float(group.get("fallback_weight_decay", 0.0))
        if fallback_weight_decay != 0.0:
            master.mul_(1.0 - lr * fallback_weight_decay)

        exp_avg_sq.mul_(float(beta2)).addcmul_(g, g, value=1.0 - float(beta2))
        step = int(state["step"])
        bias2 = max(1.0 - float(beta2) ** step, 1e-16)
        denom = (exp_avg_sq / bias2).sqrt().add_(eps)
        if exp_avg is None:
            raise RuntimeError("AdamATan2 fallback state was not initialized")
        exp_avg.mul_(float(beta1)).add_(g, alpha=1.0 - float(beta1))
        bias1 = max(1.0 - float(beta1) ** step, 1e-16)
        update = torch.atan2(exp_avg / bias1, denom)
        master.add_(update, alpha=-lr)
        if master is not p:
            p.copy_(master.to(dtype=p.dtype))
        return 1

    def _step_fallback_group(self, group: dict[str, Any], *, lr: float, t: int) -> dict[str, float]:
        fallback_count = 0
        for p in group["params"]:
            fallback_count += self._step_fallback_param(p, group, lr=lr, t=t)
        return {
            "matrix_count": 0.0,
            "fallback_count": float(fallback_count),
            "soda_weight": float(self._soda_weight(t)),
        }

    @torch.no_grad()
    def step(self, closure: Callable[[], torch.Tensor] | None = None) -> torch.Tensor | None:
        loss = None
        if closure is not None:
            with torch.enable_grad():
                loss = closure()

        stats = {
            "matrix_count": 0.0,
            "fallback_count": 0.0,
            "soda_weight_sum": 0.0,
            "group_count": 0.0,
        }
        for group in self.param_groups:
            k = int(group.get("k", 0))
            t = k + 1
            lr = float(group["lr"])

            if group.get("use_matrix_update", False):
                group_stats = self._step_matrix_group(group, lr=lr, t=t)
            else:
                group_stats = self._step_fallback_group(group, lr=lr, t=t)
            stats["matrix_count"] += group_stats["matrix_count"]
            stats["fallback_count"] += group_stats["fallback_count"]
            stats["soda_weight_sum"] += group_stats["soda_weight"]
            stats["group_count"] += 1.0
            group["k"] = t

        stats["soda_weight"] = stats["soda_weight_sum"] / max(stats["group_count"], 1.0)
        del stats["soda_weight_sum"]
        self.last_stats = stats
        return loss


def build_param_groups(
    named_parameters: Iterable[tuple[str, torch.nn.Parameter]],
    *,
    lr: float = 8e-3,
    fallback_lr: float | None = None,
    momentum: float = 0.95,
    fallback_betas: tuple[float, float] = (0.9, 0.95),
    fallback_weight_decay: float = 0.0,
    eps: float = 1e-8,
    pmuoneq_beta: float = 0.90,
    row_gamma: float = 0.35,
    pmuoneq_eps: float = 1e-6,
    normuon_beta2: float = 0.93,
    normuon_eps: float = 1e-10,
    min_matrix_dim: int = 32,
) -> list[dict[str, Any]]:
    """Split named parameters into matrix and fallback groups.

    Routing uses effective tensor shape only. Singleton dimensions are ignored,
    so ``[1, 1, width]`` is a vector and ``[1, tokens, width]`` is a matrix.
    Parameter names are retained only for group summaries and diagnostics.

    Tied parameters are deduplicated by object identity. The effective shape of
    the shared tensor decides the route regardless of aliases.
    """

    if min_matrix_dim < 1:
        raise ValueError("min_matrix_dim must be >= 1")
    fallback_lr = lr if fallback_lr is None else fallback_lr
    if fallback_weight_decay < 0.0:
        raise ValueError("fallback_weight_decay must be non-negative")

    matrix_params: list[torch.nn.Parameter] = []
    matrix_names: list[str] = []
    fallback_params: list[torch.nn.Parameter] = []
    fallback_names: list[str] = []

    unique: dict[int, tuple[torch.nn.Parameter, list[str]]] = {}
    for name, p in named_parameters:
        if not p.requires_grad:
            continue
        key = id(p)
        if key not in unique:
            unique[key] = (p, [name])
        else:
            unique[key][1].append(name)

    for _key, (p, names) in unique.items():
        use_matrix = _is_matrix_like_parameter(p, min_matrix_dim=min_matrix_dim)
        display_name = "|".join(names)
        if use_matrix:
            matrix_params.append(p)
            matrix_names.append(display_name)
        else:
            fallback_params.append(p)
            fallback_names.append(display_name)

    groups: list[dict[str, Any]] = []
    if matrix_params:
        groups.append(
            {
                "params": matrix_params,
                "param_names": matrix_names,
                "use_matrix_update": True,
                "lr": lr,
                "momentum": momentum,
                "pmuoneq_beta": pmuoneq_beta,
                "row_gamma": row_gamma,
                "pmuoneq_eps": pmuoneq_eps,
                "normuon_beta2": normuon_beta2,
                "normuon_eps": normuon_eps,
                "min_matrix_dim": min_matrix_dim,
                "fallback_lr": fallback_lr,
                "betas": fallback_betas,
                "fallback_weight_decay": fallback_weight_decay,
                "eps": eps,
            }
        )
    if fallback_params:
        groups.append(
            {
                "params": fallback_params,
                "param_names": fallback_names,
                "use_matrix_update": False,
                "lr": fallback_lr,
                "betas": fallback_betas,
                "fallback_weight_decay": fallback_weight_decay,
                "eps": eps,
            }
        )
    return groups


__all__ = [
    "__version__",
    "AnchorMuon",
    "build_param_groups",
]
