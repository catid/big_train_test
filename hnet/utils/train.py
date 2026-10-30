"""
This file contains utility functions for training.

NOTE: This file is not used inside the HNet package, but contains useful utilities for training the model itself.
"""

import torch

from hnet.modules.dc import RoutingModuleOutput

def load_balancing_loss(
    router_output: RoutingModuleOutput,
    N: float,
    valid_mask: torch.Tensor | None = None,
) -> torch.Tensor:
    """
    Compute the load balancing loss.
    
    NOTE: This is the loss we used for all experiments. It computes the loss on each device/minibatch, and then averages the loss over all devices/minibatches.
    It is possible that computing the loss on each example is better, or that computing the statistics over the entire (global) batch would have been better.

    Args:
        router_output: The output of the routing module.
        N: The number of "experts", i.e. the downsampling factor. Can be a float (e.g. 2.5) or an integer (e.g. 3), but must be greater than 1.

    Returns:
        A single tensor, the load balancing loss.
    """
    boundary_prob = router_output.boundary_prob
    tokenized_prob = boundary_prob[..., -1]
    # Balance the router's raw hard decisions. Capacity clipping is a causal
    # execution limit, not a learning signal: using the clipped mask makes
    # true_ratio saturate at 1/N and gives the probability term zero gradient
    # precisely when the router over-emits boundaries. That leaves no force
    # pushing the router back below capacity and causes the tail of each
    # sequence to be decoded from the final retained latent. The effective
    # mask remains available on the output for execution/overflow diagnostics.
    if valid_mask is None:
        weights = torch.ones_like(tokenized_prob, dtype=torch.float32)
    else:
        if valid_mask.shape != tokenized_prob.shape or valid_mask.dtype is not torch.bool:
            raise ValueError(
                "router valid_mask must be bool with the same shape as boundary_prob"
            )
        weights = valid_mask.to(dtype=torch.float32)
    denominator = weights.sum().clamp_min(1.0)
    true_ratio = (router_output.boundary_mask.float() * weights).sum() / denominator
    average_prob = (tokenized_prob.float() * weights).sum() / denominator

    return (
        (1 - true_ratio) * (1 - average_prob) +
        (true_ratio) * (average_prob) * (N-1)
    ) * N / (N-1)

