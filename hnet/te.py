"""Pinned Transformer Engine integration for the canonical H-Net.

The production model has no precision fallback.  Importing :mod:`hnet` therefore
requires the reviewed Transformer Engine checkout to be installed and loadable
against the active PyTorch/CUDA runtime.  The trainer performs the capability
and provenance check after selecting each DDP rank's CUDA device, preventing a
BF16 run from being misreported as FP8 training without touching the wrong GPU.
"""

from __future__ import annotations

import os
import ctypes
import hashlib
import importlib.util
import json
import site
import subprocess
from pathlib import Path

# The environment contains CUDA 13 PyTorch plus a CUDA 12 FlashAttention
# wheel.  cuDNN frontend otherwise chooses libcudart.so.12 by load order,
# which can select the wrong ABI for the fused FP8 DPA kernels.
os.environ.setdefault("CUDNN_FRONTEND_CUDART_LIB_NAME", "libcudart.so.13")
# The canonical run requires the patched delayed-scaling DPA backward.  Set
# the required switch before importing Transformer Engine so its backend reads
# the same value during module initialization, then validate it below rather
# than silently falling back to an FP8-forward/BF16-backward path.
os.environ.setdefault("NVTE_FP8_DPA_BWD", "1")
# The reviewed Transformer Engine build gates CroissAttn's unrounded-output
# backward correction with this process-wide switch.  The canonical config
# enables CroissAttn on its FA/T layers (SWA uses FlashAttention directly), so
# silently leaving the switch at its default would train a different backward.
os.environ.setdefault("NVTE_DPA_CROISSATTN_AUX", "1")

_GRAPH_CAPTURE = False
_GROUPED_LINEAR_ORIGINAL = None
_MODULE_GROUPED_LINEAR_ORIGINAL = None
_GROUPED_LINEAR_EAGER_ORIGINAL = None
_EXPECTED_TE_COMMIT = "9646c3ef22455b6948cbe870f1aaf20fc8238d69"
_CUBLASLT_PATH = None
_CUBLASLT_VERSION = None
_RUNTIME_MANIFEST = Path(__file__).resolve().parents[1] / "runtime_manifest.json"


def _preload_pinned_cuda_blas() -> None:
    """Load the cuBLAS ABI used to build the pinned TE checkout.

    PyTorch's cu13 wheel may load an older ``libcublasLt`` before TE imports;
    that wheel does not export the grouped-layout symbol required by the
    patched TE binary.  Resolve the exact system CUDA 13.4 libraries used at
    build time before importing TE, and fail with a useful message if they are
    absent rather than allowing a later unresolved-symbol error.
    """
    global _CUBLASLT_PATH, _CUBLASLT_VERSION
    venv_candidates = tuple(
        os.path.join(prefix, "nvidia", "cu13", "lib")
        for prefix in site.getsitepackages()
    )
    candidates = venv_candidates + (
        "/usr/local/cuda-13.4/targets/x86_64-linux/lib",
        "/usr/local/cuda/lib64",
    )
    root = next((p for p in candidates if os.path.exists(os.path.join(p, "libcublasLt.so.13"))), None)
    if root is None:
        raise RuntimeError(
            "The pinned Transformer Engine requires CUDA 13.4 cuBLASLt "
            "(libcublasLt.so.13) before importing PyTorch."
        )
    # The pinned TE grouped-GEMM path is corrupted by the CUDA 13.4
    # libcublasLt 13.8.0.4 build.  Do not run sparse MoE with an unreviewed
    # library: fail before importing TE rather than silently training bad
    # expert weights.  A later 13.8.x patch is required for this contract.
    resolved_lt = os.path.realpath(os.path.join(root, "libcublasLt.so.13"))
    try:
        lt_probe = ctypes.CDLL(resolved_lt, mode=ctypes.RTLD_LOCAL)
        lt_probe.cublasLtGetVersion.restype = ctypes.c_size_t
        lt_version = int(lt_probe.cublasLtGetVersion())
    except (AttributeError, OSError, TypeError, ValueError) as exc:
        raise RuntimeError(
            "Unable to query the libcublasLt version required by the canonical "
            f"FP8 MoE path: {resolved_lt}"
        ) from exc
    if lt_version < 130801:
        raise RuntimeError(
            "The canonical FP8 MoE path requires libcublasLt >= 13.8.1; "
            f"found version code {lt_version} at {resolved_lt}. "
            "Install the reviewed CUDA patch before training."
        )
    _CUBLASLT_PATH = resolved_lt
    _CUBLASLT_VERSION = lt_version
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

    def __init__(
        self,
        in_features,
        out_features,
        *,
        bias=True,
        device=None,
        dtype=None,
        fp8=True,
    ):
        super().__init__()
        self.in_features = int(in_features)
        self.out_features = int(out_features)
        self.fp8 = bool(fp8)
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
        # Some fixed numerical boundaries (the FP32 residual path and LM head)
        # intentionally remain outside FP8 even when an enclosing H-Net stack
        # has an active autocast region.  Keep this explicit at the module
        # boundary so a caller cannot accidentally quantize those projections.
        if self.fp8:
            out = self.linear(flat)
        else:
            with te_autocast(enabled=False):
                out = self.linear(flat)
        out = out[:rows, :self.out_features]
        return out.reshape(*original_shape, self.out_features)


def make_linear(
    in_features: int,
    out_features: int,
    *,
    bias: bool = True,
    device=None,
    dtype=None,
    fp8: bool = True,
):
    """Construct a Transformer Engine Linear with the model's parameter dtype."""

    if dtype is None:
        dtype = torch.get_default_dtype()
    return AlignedTELinear(
        in_features,
        out_features,
        bias=bias,
        device=device,
        dtype=dtype,
        fp8=fp8,
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


def set_graph_capture(enabled: bool) -> None:
    """Select graph-safe fixed-split dispatch for CUDA graph capture.

    Delayed scaling has no native grouped quantizer in the pinned TE build, so
    graph mode keeps its split-quantize GEMM but supplies a fixed Python split
    tuple from the router. Eager delayed-scaling dispatch is unchanged.
    """
    global _GRAPH_CAPTURE, _GROUPED_LINEAR_ORIGINAL, _MODULE_GROUPED_LINEAR_ORIGINAL
    global _GROUPED_LINEAR_EAGER_ORIGINAL
    _GRAPH_CAPTURE = bool(enabled)
    import transformer_engine.pytorch.ops.basic.grouped_linear as grouped_linear
    import transformer_engine.pytorch.module.grouped_linear as grouped_linear_module

    if _GROUPED_LINEAR_ORIGINAL is None:
        _GROUPED_LINEAR_ORIGINAL = grouped_linear.is_op_fuser_grouped_tensor_path_supported
    if _MODULE_GROUPED_LINEAR_ORIGINAL is None:
        _MODULE_GROUPED_LINEAR_ORIGINAL = grouped_linear_module.is_module_grouped_tensor_path_supported
    if _GROUPED_LINEAR_EAGER_ORIGINAL is None:
        _GROUPED_LINEAR_EAGER_ORIGINAL = grouped_linear_module._grouped_linear_eager

    if _GRAPH_CAPTURE:
        original = _GROUPED_LINEAR_ORIGINAL
        def graph_safe(recipe, dtype, split_alignment=128):
            if recipe is not None and recipe.delayed():
                return True
            return original(recipe, dtype, split_alignment)

        grouped_linear.is_op_fuser_grouped_tensor_path_supported = graph_safe

        # Delayed scaling is intentionally left on TE's split-quantize path:
        # this pinned build has no delayed native grouped quantizer.  Replace
        # only its host ``m_splits.tolist()`` conversion with a tuple supplied
        # by H-Net's fixed-capacity router (see SparseMoESwiGLU).  The GEMM
        # itself remains TE's fused grouped operation and no device-to-host
        # read occurs during capture.
        def graph_safe_eager(inp, m_splits, fwd_args, weights_and_biases, is_grad_enabled):
            if fwd_args.m_splits is None and not fwd_args.use_grouped_tensor_path:
                static_splits = getattr(m_splits, "_hnet_static_splits", None)
                if static_splits is None:
                    raise RuntimeError(
                        "graph-safe delayed GroupedLinear requires H-Net static m_splits"
                    )
                fwd_args.m_splits = static_splits
            return _GROUPED_LINEAR_EAGER_ORIGINAL(
                inp, m_splits, fwd_args, weights_and_biases, is_grad_enabled
            )

        grouped_linear_module._grouped_linear_eager = graph_safe_eager
    else:
        grouped_linear.is_op_fuser_grouped_tensor_path_supported = _GROUPED_LINEAR_ORIGINAL
        grouped_linear_module.is_module_grouped_tensor_path_supported = _MODULE_GROUPED_LINEAR_ORIGINAL
        grouped_linear_module._grouped_linear_eager = _GROUPED_LINEAR_EAGER_ORIGINAL


def graph_capture_enabled() -> bool:
    """Return whether model code should avoid host reads during capture."""

    return bool(_GRAPH_CAPTURE)


def update_fp8_scaling(*, forward: bool) -> None:
    """Commit delayed-scaling amax history after a captured replay.

    TE intentionally skips these updates while ``_graph=True``.  Captured
    training therefore performs the same reductions on the host side between
    replays, using the static amax buffers populated by the preceding graph.
    """

    from transformer_engine.pytorch.quantization import FP8GlobalStateManager

    FP8GlobalStateManager.reduce_and_update_fp8_tensors(forward=bool(forward))


def reset_fp8_scaling_state(model: torch.nn.Module) -> None:
    """Reset module-local delayed-scaling tensors between calibration trials.

    Transformer Engine's global amax buffers contain references to the module
    ``amax_history`` and ``scale`` tensors.  Calling
    ``FP8GlobalStateManager.reset()`` alone discards those buffers but leaves
    each module's ``buffer_index_and_autocast_key`` registration marker in
    place; the next autocast therefore refuses to register the module again
    and its scales stop updating.  Reset the referenced tensors in place and
    keep the registration intact.  This gives each calibration candidate the
    same unit-scale/zero-history starting state without invalidating TE's
    process-global bookkeeping.
    """

    for module in model.modules():
        reset = getattr(module, "reset_fp8_meta_tensors", None)
        if reset is not None:
            reset()


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
    """Require TE's fused FP8 backend for the CroissAttn-enabled stages.

    TE probes FlashAttention before fused attention.  Merely rejecting the
    unfused fallback is therefore insufficient: a future backend probe could
    select FlashAttention, silently bypassing the patched CroissAttn backward
    correction while still passing the old assertion.
    """
    try:
        from transformer_engine.pytorch.attention.dot_product_attention import (
            dot_product_attention as dpa_impl,
        )
        backends = dpa_impl._attention_backends
        if (
            not backends.get("use_fused_attention")
            or backends.get("use_flash_attention")
            or backends.get("use_unfused_attention")
        ):
            raise RuntimeError(
                "Transformer Engine did not select the fused FP8 CroissAttn backend; "
                f"backends={backends}"
            )
    except RuntimeError:
        raise
    except Exception as exc:
        raise RuntimeError("Unable to verify Transformer Engine attention backend") from exc


def fp8_context(*, graph: bool | None = None):
    """Create the mandatory FP8 autocast context for a training step."""
    reduction_group = None
    if torch.distributed.is_available() and torch.distributed.is_initialized():
        reduction_group = torch.distributed.group.WORLD
    if graph is None:
        graph = _GRAPH_CAPTURE
    return te_autocast(
        enabled=True,
        recipe=_FP8_RECIPE,
        amax_reduction_group=reduction_group,
        _graph=bool(graph),
    )


def assert_fp8_runtime() -> None:
    """Fail loudly if the runtime has been switched away from real FP8 kernels."""

    if os.environ.get("NVTE_FRAMEWORK") not in (None, "", "pytorch", "all"):
        raise RuntimeError("NVTE_FRAMEWORK must include pytorch for H-Net training")
    if not te.is_fp8_available():
        reason = None
        try:
            available = te.is_fp8_available(return_reason=True)
            if isinstance(available, tuple):
                reason = available[1]
        except TypeError:
            pass
        suffix = f": {reason}" if reason else ""
        raise RuntimeError(f"Transformer Engine FP8 support is unavailable{suffix}")
    expected_root = Path("/home/catid/fp8/TransformerEngine").resolve()
    te_path = Path(te.__file__).resolve()
    if expected_root not in te_path.parents:
        raise RuntimeError(
            "Transformer Engine was imported from an unreviewed location: "
            f"{te_path}; expected a build under {expected_root}"
        )
    te_commit = transformer_engine_git_commit()
    if te_commit != _EXPECTED_TE_COMMIT:
        raise RuntimeError(
            "Transformer Engine checkout is not the reviewed CroissAttn commit: "
            f"expected {_EXPECTED_TE_COMMIT}, found {te_commit}"
        )
    try:
        tracked_diff = subprocess.check_output(
            ["git", "diff", "--quiet"], cwd=expected_root,
            stderr=subprocess.DEVNULL,
        )
    except subprocess.CalledProcessError as exc:
        raise RuntimeError(
            "Transformer Engine checkout has tracked source modifications; "
            "refusing an unreviewed FP8 kernel"
        ) from exc
    except (OSError, subprocess.SubprocessError) as exc:
        raise RuntimeError("Unable to verify Transformer Engine source cleanliness") from exc
    if os.environ.get("NVTE_FP8_DPA_BWD") != "1":
        raise RuntimeError(
            "The canonical H-Net requires NVTE_FP8_DPA_BWD=1 for FP8 attention backward"
        )
    if os.environ.get("NVTE_DPA_CROISSATTN_AUX") != "1":
        raise RuntimeError(
            "The canonical H-Net requires NVTE_DPA_CROISSATTN_AUX=1 for the "
            "patched CroissAttn FP8 backward correction"
        )
    _verify_runtime_binaries()
    assert_fp8_attention_backend()


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(16 * 1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _runtime_binary_paths() -> dict[str, Path]:
    """Resolve the exact shared objects used by the canonical runtime."""
    root = Path("/home/catid/fp8/TransformerEngine").resolve()
    torch_libs = sorted(root.glob("transformer_engine_torch*.so"))
    if len(torch_libs) != 1:
        raise RuntimeError(
            "Expected exactly one Transformer Engine torch extension under "
            f"{root}, found {torch_libs}"
        )
    paths = {
        "transformer_engine_core": root / "libtransformer_engine.so",
        "transformer_engine_torch": torch_libs[0],
        "transformer_engine_nccl_ep": root / "libnccl_ep.so",
    }
    for module_name in (
        "flash_attn_2_cuda",
        "selective_scan_cuda",
        "causal_conv1d_cuda",
    ):
        spec = importlib.util.find_spec(module_name)
        if spec is None or not spec.origin or spec.origin == "built-in":
            raise RuntimeError(f"Unable to resolve required extension {module_name}")
        paths[module_name] = Path(spec.origin).resolve()
    if _CUBLASLT_PATH is None:
        raise RuntimeError("cuBLASLt must be resolved before runtime verification")
    paths["libcublasLt"] = Path(_CUBLASLT_PATH).resolve()
    paths["libcublas"] = paths["libcublasLt"].with_name("libcublas.so.13")
    return paths


def runtime_binary_provenance() -> dict[str, dict[str, str]]:
    """Return paths and hashes for the reviewed runtime binaries."""
    result = {}
    for name, path in _runtime_binary_paths().items():
        if not path.is_file():
            raise RuntimeError(f"Required runtime binary is missing: {path}")
        result[name] = {"path": str(path), "sha256": _sha256_file(path)}
    return result


def _verify_runtime_binaries() -> None:
    try:
        manifest = json.loads(_RUNTIME_MANIFEST.read_text())
    except (OSError, ValueError) as exc:
        raise RuntimeError(f"Unable to read runtime allowlist {_RUNTIME_MANIFEST}") from exc
    if manifest.get("transformer_engine_commit") != _EXPECTED_TE_COMMIT:
        raise RuntimeError("runtime allowlist has an unexpected Transformer Engine commit")
    expected = manifest.get("binaries")
    if not isinstance(expected, dict):
        raise RuntimeError("runtime allowlist is missing its binary hashes")
    observed = runtime_binary_provenance()
    for name, record in observed.items():
        expected_hash = expected.get(name)
        if expected_hash != record["sha256"]:
            raise RuntimeError(
                f"unreviewed runtime binary {name}: expected {expected_hash!r}, "
                f"found {record['sha256']!r} at {record['path']}"
            )


def transformer_engine_git_commit() -> str | None:
    """Return the reviewed TE checkout commit for checkpoint provenance."""

    root = Path("/home/catid/fp8/TransformerEngine")
    try:
        return subprocess.check_output(
            ["git", "rev-parse", "HEAD"], cwd=root, text=True,
            stderr=subprocess.DEVNULL,
        ).strip()
    except (OSError, subprocess.SubprocessError):
        return None


def cublaslt_provenance() -> dict[str, str | int | None]:
    """Return the exact cuBLASLt library accepted by the FP8 MoE guard."""

    return {"path": _CUBLASLT_PATH, "version_code": _CUBLASLT_VERSION}
