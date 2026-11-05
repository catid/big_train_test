"""The production raw-byte H-Net model and its configuration.

Keep model imports lazy so config validation and dependency inspection work on
hosts that do not have the CUDA attention/FP8 extensions installed.
"""

from .config_hnet import HNetConfig

__all__ = ["HNetConfig", "HNet", "HNetForByteCausalLM"]


def __getattr__(name):
    if name == "HNet":
        from .hnet import HNet
        return HNet
    if name == "HNetForByteCausalLM":
        from .mixer_seq import HNetForByteCausalLM
        return HNetForByteCausalLM
    raise AttributeError(name)
