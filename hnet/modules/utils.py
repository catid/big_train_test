from dataclasses import asdict

import torch


def get_seq_idx(cu_seqlens, device=None, total_length=None):
    """Build per-token document ids without reading a CUDA scalar on the host.

    Packed callers already know the flattened activation length from the
    source tensor.  Requiring that length explicitly avoids
    ``torch.zeros(cu_seqlens[-1])`` converting the final device value to a
    Python integer, which synchronizes the host and breaks CUDA graph capture.
    The legacy inferred form remains available for non-captured callers.
    """
    if total_length is None:
        total_length = int(cu_seqlens[-1].item())
    total_length = int(total_length)
    if total_length < 0:
        raise ValueError("total_length must be non-negative")
    if device is None:
        device = cu_seqlens.device
    # Positive-length packed documents are represented by the interior end
    # offsets. ``searchsorted(..., right=True)`` maps each token position to
    # the document whose end lies strictly after it. This replaces the old
    # zero-fill + indexed write + cumsum sequence (four full-token kernels)
    # with an arange and one search kernel, while retaining the [1, T] int32
    # contract expected by the Mamba and MoE paths.
    positions = torch.arange(total_length, device=device, dtype=cu_seqlens.dtype)
    boundaries = cu_seqlens[1:-1]
    seq_idx = torch.searchsorted(boundaries, positions, right=True)
    seq_idx = seq_idx.unsqueeze(0).to(dtype=torch.int32)

    return seq_idx


def get_stage_cfg(cfg, stage_idx):
    return {
        k: v[stage_idx] if isinstance(v, list) else v for k, v in asdict(cfg).items()
    }


def apply_optimization_params(
    param: torch.Tensor,
    **kwargs,
) -> None:
    """
    Annotates a parameter with optimization parameters.

    Specifically, updates the parameter's `_optim` attribute with the given kwargs.
    """

    if hasattr(param, "_optim"):
        param._optim.update(kwargs)
    else:
        param._optim = kwargs
