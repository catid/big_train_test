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

It does expose a small fallback-only ``fallback_mode`` knob for scalar/vector
tensors. The default is now the AdamATan2 fallback because it gave the best
validation loss/accuracy in the latest fallback sweep while remaining tied with
RMS on official test accuracy. ``"rms"`` and ``"adamc"`` remain available for
fallback-path experiments and do not alter the matrix direction.

The current best-supported recipe comes from a CIFAR-10 train/validation/test
split with the official test split evaluated only once at the end:

    model:    vit5_micro, 458,858 trainable parameters
    data:     CIFAR-10 45k train / 5k validation / official 10k test
    training: 50 epochs, 4,350 steps, batch size 512
    seeds:    123, 456, 789, 101112, 131415
    schedule: trainer-side WSD with 80-step warmup

    lr=0.014, fallback_lr=0.007, fallback_mode="atan2", row_gamma=0.35,
    pmuoneq_beta=0.90, normuon_beta2=0.93

    official test loss: 0.4147 +/- 0.0152
    official test acc:  87.57% +/- 0.39
    step time:          17.51 +/- 0.16 ms

A 5-seed CIFAR-10 follow-up found that ``fallback_mode="atan2"``,
``lr=0.014``, and ``fallback_lr=0.007`` slightly improved final validation
accuracy/loss and was essentially tied with RMS on official test accuracy
within seed variance. AdamATan2 is now the constructor default for fallback
parameters so new users get that stronger validation-loss recipe by default.

Language-model transfer note: bounded byte-level 49.4M GPT checks now have two
useful starting points. WikiText-103 selected ``lr=0.0015``,
``fallback_lr=0.00075``, ``fallback_mode="atan2"``, ``row_gamma=0.55``, and
``soda_lambda_scale=0.01`` after a 213-candidate 1200-step HPO and 1600-step
family replay. FineWeb-Edu needed its own tuning: a 255-candidate 1600-step HPO
with 10k-step family replay selected ``lr=0.0012``, ``fallback_lr=0.0012``,
``fallback_mode="rms"``, ``row_gamma=0.45``, ``soda_lambda_scale=0.003``,
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
use the same SODA anchor plus AdamATan2 by default. Optional fallback modes
exist for narrow experiments on scalar/vector parameters only, but they do not
affect the matrix direction generator. Automatic matrix eligibility is based
only on effective shape after ignoring singleton dimensions: for example
``[1, 1, width]`` is a vector, while ``[1, tokens, width]`` is a matrix. Names
are never used for routing in this standalone file.

The default ``fallback_mode="atan2"`` keeps Adam first/second moments but uses
an AdamATan2-style bounded angular update:

       m_t = beta_1 m_{t-1} + (1 - beta_1) g_t
       v_t = beta_2 v_{t-1} + (1 - beta_2) g_t^2
       update = atan2(m_hat_t, sqrt(v_hat_t) + eps)

The optional ``fallback_mode="rms"`` uses the earlier RMS-style second-moment
fallback:

       v_t = beta_2 v_{t-1} + (1 - beta_2) g_t^2
       update = g_t / (sqrt(v_t / (1 - beta_2^t)) + eps)

The optional ``fallback_mode="adamc"`` uses the AdamC adaptive direction on
fallback tensors. If ``fallback_weight_decay`` is nonzero, it applies AdamC's
``lr^2 * weight_decay`` decoupled decay to the fallback tensor before the
learned update. The default keeps ``fallback_weight_decay=0`` so the shippable
recipe still uses SODA as its only regularizer.

There is no ordinary weight decay path in this standalone optimizer. SODA is the
only regularization mechanism applied by AnchorMuon.

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

        fallback_mode = "atan2"
            Scalar/vector fallback update. Leave at "atan2" unless
            specifically testing "rms" or "adamc" for fallback-only ablations.

        fallback_betas = (0.9, 0.95)
            Fallback first/second moment defaults. The first value is ignored
            by "rms" mode.

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
  (arXiv:2605.11172). Here SODA is always enabled and replaces ordinary weight
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
FALLBACK_MODES = {"rms", "atan2", "adamc"}

# Portable ANVIL-II ablation (Keller Jordan, modded-nanogpt PR #360). The
# production PR uses fused Triton bank kernels; this reference keeps the twin
# rails and six polynomial maps auditable on the H-Net parameter shapes.
ANVIL_II_MAPS = (
    (3.923798038567, -6.095026865488, 3.905234618423),
    (3.278126713798, -3.328923386476, 0.989127286973),
    (3.505298394150, -5.137358782410, 1.968325560615),
    (2.815058591845, -3.685181239622, 1.417196497642),
    (2.245503932403, -2.443826979899, 0.963091710461),
    (2.256537145403, -2.166840097229, 0.929501253245),
)


def _anvil_ii_polar(x: torch.Tensor) -> torch.Tensor:
    x = x.float()
    scale = x.square().sum(dim=(-2, -1), keepdim=True).sqrt().clamp_min(1e-6) * 1.05
    x = x / scale
    for a, b, c in ANVIL_II_MAPS:
        gram = x @ x.mT
        x = a * x + (b * gram + c * (gram @ gram)) @ x
    return x


class AnvilII(torch.optim.Optimizer):
    """Reference ANVIL-II matrix optimizer with AdamATan2 fallback."""

    def __init__(self, params, lr=1.2e-3, fallback_lr=None, weight_decay=0.01,
                 momentum_fast=0.85, momentum_slow=0.98, fast_weight=0.4385,
                 engage_step=514, fallback_betas=(0.9, 0.95), eps=1e-8):
        fallback_lr = lr if fallback_lr is None else fallback_lr
        # Keep immutable schedule bases in the group.  The trainer updates the
        # live rates for warmup/decay each step; without these fields a resume
        # or a custom fallback rate silently resets matrix updates to the
        # matrix default and loses the requested fallback/matrix ratio.
        super().__init__(params, dict(lr=lr, base_lr=lr,
                                      fallback_lr=fallback_lr,
                                      base_fallback_lr=fallback_lr,
                                      weight_decay=weight_decay,
                                      momentum_fast=momentum_fast,
                                      momentum_slow=momentum_slow,
                                      fast_weight=fast_weight,
                                      engage_step=engage_step,
                                      betas=fallback_betas, eps=eps))

    @torch.no_grad()
    def step(self, closure=None):
        loss = None
        if closure is not None:
            with torch.enable_grad():
                loss = closure()
        for group in self.param_groups:
            group["step"] = int(group.get("step", 0)) + 1
            step = group["step"]
            for p in group["params"]:
                if p.grad is None:
                    continue
                g = p.grad.detach().float()
                state = self.state[p]
                if p.ndim >= 2 and min(p.shape[-2:]) > 1:
                    fast = state.setdefault("fast", torch.zeros_like(p, dtype=torch.float32))
                    slow = state.setdefault("slow", torch.zeros_like(p, dtype=torch.float32))
                    fast.lerp_(g, 1.0 - group["momentum_fast"])
                    slow.lerp_(g, 1.0 - group["momentum_slow"])
                    w = group["fast_weight"] if step >= group["engage_step"] else 1.0
                    velocity = w * fast + (1.0 - w) * slow
                    velocity = g.lerp(velocity, group["momentum_fast"])
                    update = _anvil_ii_polar(_matrix_view(velocity)).reshape_as(p)
                    agree = (g * slow).mean().sign().item()
                    decay = 0.0 if getattr(p, "_no_weight_decay", False) else group["weight_decay"]
                    p.mul_(1.0 - group["lr"] * decay * max(0.0, agree))
                    p.add_(update.to(p.dtype), alpha=-group["lr"])
                else:
                    avg = state.setdefault("avg", torch.zeros_like(p, dtype=torch.float32))
                    sq = state.setdefault("sq", torch.zeros_like(p, dtype=torch.float32))
                    beta1, beta2 = group["betas"]
                    avg.mul_(beta1).add_(g, alpha=1.0 - beta1)
                    sq.mul_(beta2).addcmul_(g, g, value=1.0 - beta2)
                    update = torch.atan2(avg, sq.sqrt().add_(group["eps"]))
                    decay = 0.0 if getattr(p, "_no_weight_decay", False) else group["weight_decay"]
                    p.mul_(1.0 - group["fallback_lr"] * decay)
                    p.add_(update.to(p.dtype), alpha=-group["fallback_lr"])
        return loss


@torch.no_grad()
def _matrix_view(x: torch.Tensor) -> torch.Tensor:
    effective_shape = tuple(int(dim) for dim in x.shape if int(dim) > 1)
    if len(effective_shape) < 2:
        raise ValueError("matrix update requires at least two non-singleton dimensions")
    if len(effective_shape) == 2:
        return x.reshape(effective_shape)
    rows = effective_shape[0]
    cols = math.prod(effective_shape[1:])
    return x.reshape(rows, cols)


@torch.no_grad()
def _diag_inverse_power(values: torch.Tensor, *, gamma: float, eps: float) -> torch.Tensor:
    x = values.to(torch.float32)
    scale = x.mean(dim=-1, keepdim=True).clamp_min(1.0) if x.ndim > 1 else x.mean().clamp_min(1.0)
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
    old_norm = update.norm(dim=(-2, -1), keepdim=True)
    # Keep the running row statistic in the update dtype.  ``dtype`` was an
    # undefined local here, so every GramNS/AnchorMuon step crashed before the
    # first parameter update on the production model.
    row_power = update.square().mean(dim=-1, keepdim=True, dtype=update.dtype)
    second_momentum.lerp_(row_power, 1.0 - beta2)
    # Keep ``eps`` as a Python scalar.  Creating a device tensor here is a
    # host-to-device allocation during CUDA graph capture and makes the first
    # captured optimizer step fail with ``cudaErrorStreamCaptureUnsupported``.
    out = update * torch.rsqrt(second_momentum + eps)
    return out * (old_norm / (out.norm(dim=(-2, -1), keepdim=True) + eps))


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
    rows = effective_shape[0]
    cols = math.prod(effective_shape[1:])
    return min(rows, cols) >= int(min_matrix_dim)


def _muon_name_excluded(name: str, param: torch.Tensor) -> bool:
    """Exclude matrices whose semantics are not compatible with Muon.

    Muon's polar update is appropriate for dense projection matrices.  A
    depthwise/convolutional kernel can look matrix-like after singleton
    dimensions are removed, and the vocabulary/output head benefits from the
    AdamATan2 fallback because its rows are directly supervised logits.  Keep
    those tensors on the numerically stable fallback path even when their
    effective shape passes the generic matrix test.
    """

    lowered = str(name).lower()
    if any(token in lowered for token in ("lm_head", "output_head", "conv1d", "depthwise")):
        return True
    if param.ndim >= 3 and any(token in lowered for token in (".conv", "conv.")):
        return True
    return False


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
        # The inverse-estimate recurrence used by the upstream research
        # implementation diverges for this B300 workload with rectangular
        # BF16/FP16 matrices.  The direct Gram polynomial is algebraically
        # equivalent to the standard Newton-Schulz update and remains finite;
        # matmul plus scalar operations also avoids cuBLASLt's nonzero-beta
        # addmm failure.  The smaller Gram dimension is used after the caller's
        # transpose, so this remains practical for Transformer matrices.
        for a, b, c in self.coefficients:
            gram = x @ x.mT
            poly = c * (gram @ gram) + b * gram
            x = a * x + poly @ x
        return x


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
        fallback_mode: str = "atan2",
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
        min_matrix_dim: int = 2,
        use_soda: bool = True,
        use_pmuoneq: bool = True,
        use_normuon: bool = True,
        matrix_weight_decay: float = 0.0,
    ) -> None:
        if isinstance(params, torch.nn.Module):
            params = params.named_parameters()
        fallback_lr = lr if fallback_lr is None else fallback_lr
        if lr < 0.0 or fallback_lr < 0.0:
            raise ValueError("learning rates must be non-negative")
        fallback_mode = str(fallback_mode).lower()
        if fallback_mode not in FALLBACK_MODES:
            raise ValueError(f"fallback_mode must be one of {sorted(FALLBACK_MODES)}")
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
        if soda_lambda_scale <= 0.0 and use_soda:
            raise ValueError("soda_lambda_scale must be positive; SODA is always enabled")
        if soda_lambda_power <= 0.0:
            raise ValueError("soda_lambda_power must be positive")
        prepared = self._prepare_param_groups(
            params,
            lr=lr,
            fallback_lr=fallback_lr,
            fallback_mode=fallback_mode,
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
        self.use_soda = bool(use_soda)
        self.use_pmuoneq = bool(use_pmuoneq)
        self.use_normuon = bool(use_normuon)
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

        # Preserve names so semantic exclusions (depthwise kernels and the
        # output head) are routed to AdamATan2 instead of being flattened into
        # the GramNS matrix path.
        return cls(model.named_parameters(), **kwargs)

    @staticmethod
    def _prepare_param_groups(
        params: ParamsT | Iterable[tuple[str, torch.nn.Parameter]],
        *,
        lr: float,
        fallback_lr: float,
        fallback_mode: str,
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
                fallback_mode=fallback_mode,
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
                group.setdefault("base_lr", float(group["lr"]))
                if "fallback_lr" in group:
                    group.setdefault("base_fallback_lr", float(group["fallback_lr"]))
                group.pop("weight_decay", None)
                if is_matrix:
                    group.setdefault("momentum", momentum)
                    group.setdefault("pmuoneq_beta", pmuoneq_beta)
                    group.setdefault("row_gamma", row_gamma)
                    group.setdefault("pmuoneq_eps", pmuoneq_eps)
                    group.setdefault("normuon_beta2", normuon_beta2)
                    group.setdefault("normuon_eps", normuon_eps)
                    group.setdefault("min_matrix_dim", min_matrix_dim)
                else:
                    group.setdefault("fallback_mode", fallback_mode)
                    group.setdefault("betas", fallback_betas)
                    group.setdefault("fallback_weight_decay", fallback_weight_decay)
                    group.setdefault("eps", eps)
                if is_matrix:
                    group.setdefault("fallback_mode", fallback_mode)
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
                    "base_lr": lr,
                    "momentum": momentum,
                    "pmuoneq_beta": pmuoneq_beta,
                    "row_gamma": row_gamma,
                    "pmuoneq_eps": pmuoneq_eps,
                    "normuon_beta2": normuon_beta2,
                    "normuon_eps": normuon_eps,
                    "min_matrix_dim": min_matrix_dim,
                }
            )
        if fallback_params:
            groups.append(
                {
                    "params": fallback_params,
                    "use_matrix_update": False,
                    "lr": fallback_lr,
                    "base_lr": fallback_lr,
                    "fallback_mode": fallback_mode,
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
                    "fallback_mode": str(group.get("fallback_mode", "atan2")),
                }
            )
        return summary

    def _soda_anchor(self, p: torch.Tensor) -> torch.Tensor:
        state = self.state[p]
        anchor = state.get("soda_z0")
        if anchor is None:
            anchor = state["soda_z0"] = torch.clone(p, memory_format=torch.preserve_format)
        return anchor

    def _soda_weight(self, t: int) -> float:
        if not self.use_soda:
            return 0.0
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
        rows: int,
        device: torch.device,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        state = self.state[p]
        row_ema = state.get("pmuoneq_row_ema")
        if isinstance(row_ema, torch.Tensor) and row_ema.shape == (rows,):
            if row_ema.device != device or row_ema.dtype != torch.float32:
                row_ema = state["pmuoneq_row_ema"] = row_ema.to(device=device, dtype=torch.float32)
        else:
            row_ema = state["pmuoneq_row_ema"] = torch.ones(rows, device=device, dtype=torch.float32)
        second = state.get("normuon_second_momentum")
        second_shape = (rows, 1)
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
        _batch, rows, _cols = grads.shape

        row_buffers: list[torch.Tensor] = []
        second_buffers: list[torch.Tensor] = []
        for p, _anchor, _source, _grad in entries:
            row_ema, second = self._ensure_matrix_state(p, rows, grads.device)
            row_buffers.append(row_ema)
            second_buffers.append(second)

        row_stack = torch.stack(row_buffers, dim=0)
        if self.use_pmuoneq:
            beta_p = float(group["pmuoneq_beta"])
            row_stack.mul_(beta_p).add_(grads.square().mean(dim=2), alpha=1.0 - beta_p)
            row_factor = _diag_inverse_power(
                row_stack, gamma=float(group["row_gamma"]), eps=float(group["pmuoneq_eps"])
            )
        else:
            row_factor = torch.ones_like(row_stack)

        preconditioned = sources * row_factor[:, :, None]
        update = self._orthogonalizer(preconditioned)
        update = update * (0.2 * math.sqrt(max(update.size(-2), update.size(-1))))

        second_stack = torch.stack(second_buffers, dim=0)
        if self.use_normuon:
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
                    "Use dense gradients for this parameter or a different optimizer for sparse embeddings."
                )
            if not _is_matrix_like_parameter(grad, min_matrix_dim=int(group.get("min_matrix_dim", 2))):
                fallback_count += self._step_fallback_param(p, group, lr=lr, t=t)
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
            updates = self._transform_matrix_bucket(bucket_entries, group)
            for (p, anchor, _source, _grad), update in zip(bucket_entries, updates.unbind(0), strict=True):
                p.lerp_(end=anchor, weight=soda_weight)
                matrix_decay = 0.0 if getattr(p, "_no_weight_decay", False) else self.matrix_weight_decay
                if matrix_decay:
                    p.mul_(1.0 - lr * matrix_decay)
                p.add_(update.reshape_as(p).to(p.dtype), alpha=-lr)
                matrix_count += 1

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
                "Use dense gradients for this parameter or a different optimizer for sparse embeddings."
            )
        state = self.state[p]
        mode = str(group.get("fallback_mode", "atan2")).lower()
        if mode not in FALLBACK_MODES:
            raise ValueError(f"fallback_mode must be one of {sorted(FALLBACK_MODES)}")
        exp_avg_sq = state.get("exp_avg_sq")
        if isinstance(exp_avg_sq, torch.Tensor) and exp_avg_sq.shape == p.shape:
            if exp_avg_sq.device != p.device or exp_avg_sq.dtype != torch.float32:
                exp_avg_sq = state["exp_avg_sq"] = exp_avg_sq.to(device=p.device, dtype=torch.float32)
        else:
            exp_avg_sq = state["exp_avg_sq"] = torch.zeros_like(p, dtype=torch.float32)
            state["step"] = 0
        exp_avg = state.get("exp_avg")
        if mode in {"atan2", "adamc"} and (
            not isinstance(exp_avg, torch.Tensor) or exp_avg.shape != p.shape
        ):
            exp_avg = state["exp_avg"] = torch.zeros_like(p, dtype=torch.float32)
        elif isinstance(exp_avg, torch.Tensor) and (exp_avg.device != p.device or exp_avg.dtype != torch.float32):
            exp_avg = state["exp_avg"] = exp_avg.to(device=p.device, dtype=torch.float32)
        state["step"] = int(state.get("step", 0)) + 1
        beta1, beta2 = group.get("betas", (0.9, 0.999))
        eps = float(group.get("eps", 1e-8))
        g = grad.detach().to(torch.float32)
        p.lerp_(end=self._soda_anchor(p), weight=self._soda_weight(t))
        fallback_weight_decay = 0.0 if getattr(p, "_no_weight_decay", False) else float(group.get("fallback_weight_decay", 0.0))
        if fallback_weight_decay != 0.0:
            p.mul_(1.0 - lr * fallback_weight_decay)

        exp_avg_sq.mul_(float(beta2)).addcmul_(g, g, value=1.0 - float(beta2))
        step = int(state["step"])
        bias2 = max(1.0 - float(beta2) ** step, 1e-16)
        denom = (exp_avg_sq / bias2).sqrt().add_(eps)
        if mode == "rms":
            update = g / denom
        else:
            if exp_avg is None:
                raise RuntimeError("fallback first-moment state was not initialized")
            exp_avg.mul_(float(beta1)).add_(g, alpha=1.0 - float(beta1))
            bias1 = max(1.0 - float(beta1) ** step, 1e-16)
            if mode == "atan2":
                update = torch.atan2(exp_avg / bias1, denom)
            else:
                update = (exp_avg / bias1) / denom
        p.add_(update.to(p.dtype), alpha=-lr)
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
    fallback_mode: str = "atan2",
    momentum: float = 0.95,
    fallback_betas: tuple[float, float] = (0.9, 0.95),
    fallback_weight_decay: float = 0.0,
    eps: float = 1e-8,
    pmuoneq_beta: float = 0.90,
    row_gamma: float = 0.35,
    pmuoneq_eps: float = 1e-6,
    normuon_beta2: float = 0.93,
    normuon_eps: float = 1e-10,
    min_matrix_dim: int = 2,
) -> list[dict[str, Any]]:
    """Split named parameters into matrix and fallback groups.

    Routing uses effective tensor shape plus semantic exclusions. Singleton
    dimensions are ignored, so ``[1, 1, width]`` is a vector and
    ``[1, tokens, width]`` is a matrix; depthwise convolution kernels and the
    output head remain on AdamATan2 even when their shape looks matrix-like.
    Parameter names are retained for both routing and diagnostics.

    Tied parameters are deduplicated by object identity. The effective shape of
    the shared tensor decides the route regardless of aliases.
    """

    if min_matrix_dim < 1:
        raise ValueError("min_matrix_dim must be >= 1")
    fallback_lr = lr if fallback_lr is None else fallback_lr
    fallback_mode = str(fallback_mode).lower()
    if fallback_mode not in FALLBACK_MODES:
        raise ValueError(f"fallback_mode must be one of {sorted(FALLBACK_MODES)}")
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
        display_name = "|".join(names)
        use_matrix = (
            not _muon_name_excluded(display_name, p)
            and _is_matrix_like_parameter(p, min_matrix_dim=min_matrix_dim)
        )
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
            }
        )
    if fallback_params:
        groups.append(
            {
                "params": fallback_params,
                "param_names": fallback_names,
                "use_matrix_update": False,
                "lr": fallback_lr,
                "fallback_mode": fallback_mode,
                "betas": fallback_betas,
                "fallback_weight_decay": fallback_weight_decay,
                "eps": eps,
            }
        )
    return groups


__all__ = [
    "__version__",
    "AnchorMuon",
    "AnvilII",
    "build_param_groups",
]
