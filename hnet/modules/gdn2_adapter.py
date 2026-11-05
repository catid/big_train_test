"""Adapter for the inspected external GDN-2 implementation.

The production H-Net keeps the NVIDIA/FLa source outside this repository.  A
named GDN2 block therefore imports the dependency lazily and never substitutes
another mixer when the dependency is missing.  This module is safe to
import in tooling that only reads a config; constructing ``GDN2Adapter`` is
the point at which the external package is required.
"""

from __future__ import annotations

import importlib
import importlib.util
import os
import sys
import types
from pathlib import Path
from typing import Any

import torch
from torch import nn


class GDN2DependencyError(ImportError):
    """Raised when the configured external GDN-2 symbol cannot be loaded."""


def _prepare_external_namespace(module_name: str) -> Path | None:
    """Expose the pinned GDN2 checkout without running ``lit_gpt.__init__``.

    The upstream checkout's package initializer imports the full Lit-GPT
    application (and therefore Lightning).  H-Net only needs ``gdn2.py``;
    creating a namespace package for the configured checkout lets Python load
    that module while keeping the dependency boundary explicit.  We do this
    only for the known external package prefix and only when ``GDN2_ROOT`` is
    set, so an unrelated installed package cannot be selected silently.
    """
    if not module_name.startswith("lit_gpt."):
        return None
    root_value = os.environ.get("GDN2_ROOT")
    if not root_value:
        raise GDN2DependencyError(
            "GDN2_ROOT must point to the inspected external checkout when "
            f"loading {module_name!r}"
        )
    root = Path(root_value).expanduser().resolve()
    package_dir = root / "lit_gpt"
    if not package_dir.is_dir():
        raise GDN2DependencyError(
            f"GDN2_ROOT={str(root)!r} has no lit_gpt package directory"
        )
    existing = sys.modules.get("lit_gpt")
    if existing is None:
        package = types.ModuleType("lit_gpt")
        package.__path__ = [str(package_dir)]
        package.__package__ = "lit_gpt"
        package.__spec__ = importlib.util.spec_from_loader(
            "lit_gpt", loader=None, is_package=True
        )
        sys.modules["lit_gpt"] = package
    else:
        paths = getattr(existing, "__path__", ())
        if not any(Path(path).resolve() == package_dir for path in paths):
            raise GDN2DependencyError(
                "an unrelated lit_gpt package is already loaded; refusing to "
                f"mix it with GDN2_ROOT={str(root)!r}"
            )
    return root


def load_gdn2_symbol(path: str):
    """Resolve a configured ``module:attribute`` GDN-2 class lazily."""
    try:
        module_name, attr_name = path.split(":", 1)
    except ValueError as exc:
        raise GDN2DependencyError(
            f"gdn2_module must use 'module:attribute' syntax, got {path!r}"
        ) from exc
    root = _prepare_external_namespace(module_name)
    try:
        module = importlib.import_module(module_name)
    except Exception as exc:
        raise GDN2DependencyError(
            "GDN2 blocks require the inspected external dependency "
            f"{path!r}; no in-tree fallback is provided"
        ) from exc
    if root is not None:
        module_file = getattr(module, "__file__", None)
        if module_file is None:
            raise GDN2DependencyError(
                f"configured GDN2 module {path!r} has no source file"
            )
        try:
            Path(module_file).resolve().relative_to(root)
        except ValueError as exc:
            raise GDN2DependencyError(
                f"configured GDN2 module {path!r} resolved outside "
                f"GDN2_ROOT={str(root)!r}"
            ) from exc
    try:
        return getattr(module, attr_name)
    except AttributeError as exc:
        raise GDN2DependencyError(
            f"configured GDN2 dependency {path!r} has no {attr_name!r}"
        ) from exc


class GDN2Adapter(nn.Module):
    """Expose external ``GatedDeltaNet2`` through H-Net's mixer contract.

    The reference implementation returns ``(output, attentions, cache)`` and
    uses FLA cache objects.  H-Net's training path only needs the first item;
    inference is rejected until a deliberate cache bridge is supplied rather
    than silently allocating an incompatible cache.
    """

    def __init__(self, d_model: int, config, *, device=None, dtype=None):
        super().__init__()
        if d_model != int(config.gdn2_hidden_size):
            raise ValueError(
                "GDN2Adapter width mismatch: block d_model must equal "
                f"gdn2_hidden_size ({d_model} != {config.gdn2_hidden_size})"
            )
        if int(config.gdn2_num_heads) * int(config.gdn2_head_dim) != d_model:
            raise ValueError(
                "GDN2Adapter requires gdn2_num_heads * gdn2_head_dim == d_model"
            )
        cls = load_gdn2_symbol(str(config.gdn2_module))
        kwargs: dict[str, Any] = {
            "hidden_size": int(config.gdn2_hidden_size),
            "expand_v": float(config.gdn2_expand_v),
            "head_dim": int(config.gdn2_head_dim),
            "num_heads": int(config.gdn2_num_heads),
            "mode": "chunk",
            "use_short_conv": bool(config.gdn2_use_short_conv),
            "fused_projections": bool(config.gdn2_fused_projections),
            "backend": str(config.gdn2_backend),
            "disable_recompute": bool(config.gdn2_disable_recompute),
            "device": device,
            "dtype": dtype,
        }
        self.mixer = cls(**kwargs)
        # Enforce placement so a CUDA H-Net cannot retain a CPU GDN2 submodule.
        if device is not None or dtype is not None:
            self.mixer.to(device=device, dtype=dtype)

    def forward(self, hidden_states: torch.Tensor, *, mask=None, inference_params=None, **kwargs):
        if inference_params is not None:
            raise RuntimeError(
                "GDN2Adapter inference needs an explicit FLA/H-Net cache bridge"
            )
        packed = hidden_states.dim() == 2
        if packed:
            # Packed callers supply [tokens, width] tensors together with
            # cu_seqlens.  The external API is batch-major, but accepts the
            # same cu_seqlens metadata once a singleton batch axis is added.
            hidden_states = hidden_states.unsqueeze(0)
        result = self.mixer(hidden_states, attention_mask=mask, **kwargs)
        output = result[0] if isinstance(result, tuple) else result
        return output.squeeze(0) if packed else output

    def allocate_inference_cache(self, *args, **kwargs):
        raise RuntimeError(
            "GDN2Adapter has no implicit H-Net inference cache; configure the "
            "external FLA cache bridge before decoding"
        )

    def step(self, *args, **kwargs):
        raise RuntimeError(
            "GDN2Adapter.step is unavailable until FLA cache state is adapted"
        )
