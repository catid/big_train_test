"""Triton kernels for the padded MSAR depthwise convolution.

The forward MSAR expansion stays on PyTorch's depthwise convolution kernel.
Both backward correlations are short causal reductions and can be computed
without PyTorch's generic grouped-convolution backward launches. The kernels
preserve the public convolution equation while keeping optimizer-visible
parameter shapes and dtypes unchanged.
"""

from __future__ import annotations

import torch
import triton
import triton.language as tl


@triton.jit
def _causal_depthwise_dx(
    grad_y_ptr,
    weight_ptr,
    grad_x_ptr,
    batch,
    channels,
    length,
    kernel,
    BLOCK_T: tl.constexpr,
):
    """Backward input for y[t] = sum_lag x[t-lag] w[K-1-lag]."""

    program = tl.program_id(0)
    batch_index = program // channels
    channel = program % channels
    for tile_start in range(0, length, BLOCK_T):
        token = tile_start + tl.arange(0, BLOCK_T)
        token_mask = token < length
        grad_x = tl.zeros((BLOCK_T,), dtype=tl.float32)
        for lag in range(0, kernel):
            output_token = token + lag
            output_mask = token_mask & (output_token < length)
            grad_y = tl.load(
                grad_y_ptr
                + batch_index * channels * length
                + channel * length
                + output_token,
                mask=output_mask,
                other=0,
            ).to(tl.float32)
            weight = tl.load(weight_ptr + channel * kernel + (kernel - 1 - lag)).to(
                tl.float32
            )
            grad_x += grad_y * weight
        tl.store(
            grad_x_ptr
            + batch_index * channels * length
            + channel * length
            + token,
            grad_x,
            mask=token_mask,
        )


def _depthwise_dx(grad_y: torch.Tensor, weight: torch.Tensor) -> torch.Tensor:
    if not grad_y.is_cuda or not weight.is_cuda:
        raise RuntimeError("MSAR Triton depthwise backward requires CUDA tensors")
    if grad_y.ndim != 3 or weight.ndim != 3 or weight.shape[1] != 1:
        raise ValueError(
            "expected grad_y=(B,C,L) and depthwise weight=(C,1,K), got "
            f"{tuple(grad_y.shape)} and {tuple(weight.shape)}"
        )
    batch, channels, length = grad_y.shape
    if weight.shape[0] != channels:
        raise ValueError("depthwise weight channel count must match grad_y")
    kernel = weight.shape[-1]
    grad_x = torch.empty_like(grad_y)
    block_t = 256
    _causal_depthwise_dx[(batch * channels,)](
        grad_y,
        weight,
        grad_x,
        batch,
        channels,
        length,
        kernel,
        BLOCK_T=block_t,
        num_warps=4,
    )
    return grad_x


@triton.jit
def _causal_depthwise_dw(
    grad_y_ptr,
    x_ptr,
    grad_w_ptr,
    batch,
    channels,
    length,
    kernel: tl.constexpr,
    BLOCK_T: tl.constexpr,
):
    """Backward weight for ``y[t] = sum_lag x[t-lag] w[K-1-lag]``."""

    channel = tl.program_id(0)
    # Compute one coefficient at a time. Keeping the reduction vector scalar in
    # the compile-time coefficient loop avoids materializing a [K,BLOCK_T]
    # tensor and works across Triton 3.x's constexpr indexing rules.
    for coefficient in range(kernel):
        lag = kernel - 1 - coefficient
        accum = tl.zeros((BLOCK_T,), dtype=tl.float32)
        for batch_index in range(batch):
            for tile_start in range(0, length, BLOCK_T):
                token = tile_start + tl.arange(0, BLOCK_T)
                token_mask = token < length
                x_value = tl.load(
                    x_ptr
                    + batch_index * channels * length
                    + channel * length
                    + token,
                    mask=token_mask,
                    other=0,
                ).to(tl.float32)
                output_token = token + lag
                output_mask = token_mask & (output_token < length)
                grad_value = tl.load(
                    grad_y_ptr
                    + batch_index * channels * length
                    + channel * length
                    + output_token,
                    mask=output_mask,
                    other=0,
                ).to(tl.float32)
                accum += x_value * grad_value
        tl.store(
            grad_w_ptr + channel * kernel + coefficient,
            tl.sum(accum, axis=0),
        )


def _depthwise_dw(grad_y: torch.Tensor, x: torch.Tensor, kernel: int) -> torch.Tensor:
    if not grad_y.is_cuda or not x.is_cuda:
        raise RuntimeError("MSAR Triton depthwise backward requires CUDA tensors")
    if grad_y.ndim != 3 or x.ndim != 3 or grad_y.shape != x.shape:
        raise ValueError(
            "expected x and grad_y with matching (B,C,L) shapes, got "
            f"{tuple(x.shape)} and {tuple(grad_y.shape)}"
        )
    batch, channels, length = x.shape
    if kernel <= 0:
        raise ValueError(f"depthwise kernel must be positive, got {kernel}")
    grad_w = torch.empty((channels * kernel,), device=x.device, dtype=x.dtype)
    _causal_depthwise_dw[(channels,)](
        grad_y,
        x,
        grad_w,
        batch,
        channels,
        length,
        kernel=kernel,
        BLOCK_T=256,
        num_warps=4,
    )
    return grad_w.view(channels, 1, kernel)


class _CausalDepthwiseFunction(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x: torch.Tensor, weight: torch.Tensor) -> torch.Tensor:
        if x.ndim != 3 or weight.ndim != 3 or weight.shape[1] != 1:
            raise ValueError(
                "expected x=(B,C,L) and depthwise weight=(C,1,K), got "
                f"{tuple(x.shape)} and {tuple(weight.shape)}"
            )
        if x.shape[1] != weight.shape[0]:
            raise ValueError("depthwise weight channel count must match x")
        # The Triton backward uses flat BCHL addressing.  Preserve the public
        # shape while making that layout explicit for any packed caller.
        x = x.contiguous()
        weight = weight.contiguous()
        length = x.shape[-1]
        kernel = weight.shape[-1]
        # Symmetric padding followed by the causal prefix is algebraically
        # equal to explicit left padding and avoids materializing that pad.
        output = torch.nn.functional.conv1d(
            x,
            weight,
            padding=kernel - 1,
            groups=x.shape[1],
        )[..., :length]
        ctx.save_for_backward(x, weight)
        return output

    @staticmethod
    def backward(ctx, grad_output: torch.Tensor):
        x, weight = ctx.saved_tensors
        grad_output = grad_output.contiguous()
        grad_x = _depthwise_dx(grad_output, weight)
        # The generic grouped-convolution weight backward is a major launch in
        # the padded MSAR graph. The causal correlation has a fixed short
        # kernel, so compute it directly without materializing left padding.
        grad_weight = _depthwise_dw(grad_output, x, weight.shape[-1])
        return grad_x, grad_weight


def causal_depthwise_conv1d(x: torch.Tensor, weight: torch.Tensor) -> torch.Tensor:
    """Depthwise causal convolution with a Triton input-gradient path."""

    return _CausalDepthwiseFunction.apply(x, weight)

