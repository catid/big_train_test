"""Pinned Transformer Engine integration for the canonical H-Net.

The production model has no precision fallback.  Importing :mod:`hnet` therefore
requires the reviewed Transformer Engine checkout to be installed and loadable
against the active PyTorch/CUDA runtime.  Keeping this check at package import
time prevents a BF16 run from being misreported as FP8 training.
"""

from __future__ import annotations

import os
import ctypes

# The environment contains CUDA 13 PyTorch plus a CUDA 12 FlashAttention
# wheel.  cuDNN frontend otherwise chooses libcudart.so.12 by load order,
# which can select the wrong ABI for the fused FP8 DPA kernels.
os.environ.setdefault("CUDNN_FRONTEND_CUDART_LIB_NAME", "libcudart.so.13")


def _preload_pinned_cuda_blas() -> None:
    """Load the cuBLAS ABI used to build the pinned TE checkout.

    PyTorch's cu13 wheel may load an older ``libcublasLt`` before TE imports;
    that wheel does not export the grouped-layout symbol required by the
    patched TE binary.  Resolve the exact system CUDA 13.4 libraries used at
    build time before importing TE, and fail with a useful message if they are
    absent rather than allowing a later unresolved-symbol error.
    """
    candidates = (
        "/usr/local/cuda-13.4/targets/x86_64-linux/lib",
        "/usr/local/cuda/lib64",
    )
    root = next((p for p in candidates if os.path.exists(os.path.join(p, "libcublasLt.so.13"))), None)
    if root is None:
        raise RuntimeError(
            "The pinned Transformer Engine requires CUDA 13.4 cuBLASLt "
            "(libcublasLt.so.13) before importing PyTorch."
        )
    try:
        ctypes.CDLL(os.path.join(root, "libcublasLt.so.13"), mode=ctypes.RTLD_GLOBAL)
        ctypes.CDLL(os.path.join(root, "libcublas.so.13"), mode=ctypes.RTLD_GLOBAL)
    except OSError as exc:
        raise RuntimeError("Unable to load the pinned CUDA 13.4 cuBLAS libraries") from exc


_preload_pinned_cuda_blas()

try:
    import transformer_engine.pytorch as te
    from transformer_engine.common.recipe import DelayedScaling, Format
except Exception as exc:  # pragma: no cover - the error is environment-specific
    raise RuntimeError(
        "The canonical H-Net requires the pinned patched Transformer Engine from "
        "/home/catid/fp8/TransformerEngine. Install/rebuild that checkout against "
        "the active PyTorch/CUDA/cuDNN runtime before importing hnet."
    ) from exc

import torch

if not te.is_fp8_available():
    raise RuntimeError(
        "Transformer Engine imported, but its FP8 kernels are unavailable on this "
        "GPU/runtime; refusing a silent BF16 training run."
    )

TELinear = te.Linear
TEGroupedLinear = te.GroupedLinear
TEDotProductAttention = te.DotProductAttention
te_autocast = te.autocast


class AlignedTELinear(torch.nn.Module):
    """TE Linear with static alignment padding for arbitrary token counts.

    Transformer Engine's FP8 GEMM requires the flattened token dimension to be
    divisible by eight and the feature dimensions to be divisible by sixteen.
    H-Net's routing/chunking can produce a valid packed stream with any token
    count (including 127 or 4095), and the boundary/router heads have narrow
    output widths such as one or eight.  Pad those dimensions inside the TE
    call, then slice the result back to the public shape.  The padding is
    tensor-only and remains replay-safe for a fixed graph bound.
    """

    def __init__(self, in_features, out_features, *, bias=True, device=None, dtype=None):
        super().__init__()
        self.in_features = int(in_features)
        self.out_features = int(out_features)
        self.padded_out_features = (self.out_features + 15) // 16 * 16
        self.linear = TELinear(
            self.in_features,
            self.padded_out_features,
            bias=bias,
            device=device,
            params_dtype=dtype,
        )

    @property
    def weight(self):
        return self.linear.weight

    @property
    def bias(self):
        return self.linear.bias

    def forward(self, x):
        if x.shape[-1] != self.in_features:
            raise ValueError(
                f"expected final dimension {self.in_features}, got {x.shape[-1]}"
            )
        original_shape = x.shape[:-1]
        flat = x.reshape(-1, self.in_features)
        rows = flat.shape[0]
        padded_rows = (rows + 7) // 8 * 8
        if padded_rows != rows:
            flat = torch.cat((flat, flat.new_zeros((padded_rows - rows, self.in_features))), dim=0)
        out = self.linear(flat)
        out = out[:rows, :self.out_features]
        return out.reshape(*original_shape, self.out_features)


def make_linear(in_features: int, out_features: int, *, bias: bool = True,
                device=None, dtype=None):
    """Construct a Transformer Engine Linear with the model's parameter dtype."""

    if dtype is None:
        dtype = torch.get_default_dtype()
    return AlignedTELinear(
        in_features, out_features, bias=bias, device=device, dtype=dtype
    )


def fp8_recipe():
    """Return the pinned delayed-scaling HYBRID FP8 recipe."""

    # The patched TE checkout selects its corrected CroissAttn backward path for
    # delayed-scaling FP8 DPA during training.  The recipe is created once per
    # process so its amax history is shared by all TE modules in the step.
    return DelayedScaling(
        fp8_format=Format.HYBRID,
        amax_history_len=16,
        amax_compute_algo="max",
        margin=0,
        # DPA has its own FP8 gate.  Without this flag the outer autocast
        # quantizes projections but silently leaves attention in BF16, which
        # would make the CroissAttn correction inactive.
        fp8_dpa=True,
    )


_FP8_RECIPE = fp8_recipe()


def assert_fp8_attention_backend() -> None:
    """Reject TE's quadratic unfused attention selection.

    The canonical model requires the pinned CroissAttn-capable fused backend;
    running TE's correctness fallback would silently violate the FP8 contract
    and makes full length training infeasible.  The backend is probed by the
    DPA module on the first real forward, so this guard checks the runtime
    prerequisite (cuDNN) up front and the DPA call remains fail-loud.
    """
    try:
        from transformer_engine.pytorch.utils import get_cudnn_version
        version = tuple(get_cudnn_version())
    except Exception as exc:
        raise RuntimeError("Unable to query cuDNN for FP8 attention") from exc
    # Canonical H-Net routes the chunked middle stack through packed ``thd``
    # DPA, whose fused FP8 backend requires cuDNN 9.23 or newer on B300.
    if version < (9, 23, 0):
        raise RuntimeError(
            "Pinned packed FP8 CroissAttn requires cuDNN >= 9.23; found "
            f"{'.'.join(map(str, version))}. Install the reviewed cuDNN runtime."
        )


def assert_fused_attention_selected() -> None:
    """Fail if TE selected its quadratic unfused attention implementation."""
    try:
        from transformer_engine.pytorch.attention.dot_product_attention import (
            dot_product_attention as dpa_impl,
        )
        backends = dpa_impl._attention_backends
        if backends.get("use_unfused_attention") and not backends.get("use_fused_attention"):
            raise RuntimeError(
                "Transformer Engine selected UnfusedDotProductAttention; "
                "the canonical H-Net requires the fused FP8 CroissAttn backend."
            )
    except RuntimeError:
        raise
    except Exception as exc:
        raise RuntimeError("Unable to verify Transformer Engine attention backend") from exc


def fp8_context():
    """Create the mandatory FP8 autocast context for a training step."""
    reduction_group = None
    if torch.distributed.is_available() and torch.distributed.is_initialized():
        reduction_group = torch.distributed.group.WORLD
    return te_autocast(
        enabled=True,
        recipe=_FP8_RECIPE,
        amax_reduction_group=reduction_group,
    )


def assert_fp8_runtime() -> None:
    """Fail loudly if the runtime has been switched away from real FP8 kernels."""

    if os.environ.get("NVTE_FRAMEWORK") not in (None, "", "pytorch", "all"):
        raise RuntimeError("NVTE_FRAMEWORK must include pytorch for H-Net training")
    if not te.is_fp8_available():
        raise RuntimeError("Transformer Engine FP8 support is unavailable")
    assert_fp8_attention_backend()
