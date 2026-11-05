"""Canonical raw-byte H-Net model package.

Model classes are resolved lazily because their production CUDA extensions are
optional during config/dependency inspection.
"""

from .models import HNetConfig

__all__ = ["HNet", "HNetConfig", "HNetForByteCausalLM"]


def __getattr__(name):
    if name in {"HNet", "HNetForByteCausalLM"}:
        from . import models
        return getattr(models, name)
    raise AttributeError(name)
