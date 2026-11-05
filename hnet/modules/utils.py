import torch


def get_seq_idx(cu_seqlens, device=None):
    """Build Mamba's per-token document IDs without a full-size fill/scatter.

    The old implementation allocated a zero-filled int64 vector, scattered
    document starts into it, scanned it, then cast the result to int32.  That
    sequence launches several bandwidth-bound kernels for every packed stage.
    The cumulative lengths are already sorted, so a single bucketize over an
    int32 token range computes the same IDs directly in the dtype consumed by
    the Mamba kernels.  ``right=True`` is required because a token at a
    document's first offset belongs to the document on its right.
    """
    device = cu_seqlens.device if device is None else device
    total = cu_seqlens[-1]
    positions = torch.arange(total, device=device, dtype=torch.int32)
    # ``searchsorted``/``bucketize`` return int64 on current PyTorch releases;
    # cast once to the int32 contract expected by mamba_chunk_scan_combined.
    seq_idx = torch.searchsorted(cu_seqlens[1:], positions, right=True)
    return seq_idx.to(dtype=torch.int32).unsqueeze(0)


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
