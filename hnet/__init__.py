"""Canonical raw-byte H-Net package."""

# Load Transformer Engine before any model module imports torch.  The pinned
# checkout owns the CUDA/cuDNN library selection; importing torch first can load
# a conflicting wheel copy and make the TE extension fail to resolve symbols.
from .te import TEGroupedLinear, TELinear, assert_fp8_runtime, fp8_context, make_linear

from .models.config_hnet import HNetConfig
from .models.mixer_seq import HNetForByteCausalLM

__all__ = ["HNetConfig", "HNetForByteCausalLM", "TEGroupedLinear", "TELinear"]
