"""Loss terms used by the canonical raw-byte H-Net trainer."""

from __future__ import annotations

import torch

from hnet.modules.dc import RoutingModuleOutput


def load_balancing_loss(
    router_output: RoutingModuleOutput,
    target_ratio: float,
) -> torch.Tensor:
    """Penalize deviation from the requested learned-chunk boundary rate.

    The router probabilities and hard boundary mask are accumulated in FP32;
    this keeps the regularizer well behaved when the model activations run in
    BF16/FP8.
    """
    if target_ratio <= 1:
        raise ValueError("target_ratio must be greater than one")
    boundary_prob = router_output.boundary_prob
    tokenized_prob = boundary_prob[..., -1]
    boundary_mask = router_output.boundary_mask
    true_ratio = boundary_mask.float().mean()
    average_prob = tokenized_prob.float().mean()
    return (
        (1 - true_ratio) * (1 - average_prob)
        + true_ratio * average_prob * (target_ratio - 1)
    ) * target_ratio / (target_ratio - 1)


__all__ = ["load_balancing_loss"]
