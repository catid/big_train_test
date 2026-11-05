"""Building blocks used by the canonical GDN2/LoL/FA H-Net.

Exports are lazy because the production blocks load CUDA-only extensions.
"""

__all__ = [
    "GDN2Adapter",
    "GDN2DependencyError",
    "HybridStack",
    "LatentMoE",
    "load_gdn2_symbol",
]


def __getattr__(name):
    if name in {"GDN2Adapter", "GDN2DependencyError", "load_gdn2_symbol"}:
        from . import gdn2_adapter
        return getattr(gdn2_adapter, name)
    if name == "HybridStack":
        from .hybrid import HybridStack
        return HybridStack
    if name == "LatentMoE":
        from .latent_moe import LatentMoE
        return LatentMoE
    raise AttributeError(name)
