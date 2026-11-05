"""Adapter for the inspected external GDN-2 implementation.

The production H-Net keeps the NVIDIA/FLa source outside this repository.  A
named GDN2 block therefore imports the dependency lazily and never substitutes
another mixer when the dependency is missing.  This module is safe to
import in tooling that only reads a config; constructing ``GDN2Adapter`` is
the point at which the external package is required.
"""

from __future__ import annotations

import importlib
from typing import Any

import torch
from torch import nn


class GDN2DependencyError(ImportError):
    """Raised when the configured external GDN-2 symbol cannot be loaded."""


def load_gdn2_symbol(path: str):
    """Resolve a configured ``module:attribute`` GDN-2 class lazily."""
    try:
        module_name, attr_name = path.split(":", 1)
    except ValueError as exc:
        raise GDN2DependencyError(
            f"gdn2_module must use 'module:attribute' syntax, got {path!r}"
        ) from exc
    try:
        module = importlib.import_module(module_name)
    except Exception as exc:
        raise GDN2DependencyError(
            "GDN2 blocks require the inspected external dependency "
            f"{path!r}; no in-tree fallback is provided"
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
