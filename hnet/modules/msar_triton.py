"""Triton kernels for the padded MSAR depthwise convolution.

The forward MSAR expansion stays on PyTorch's depthwise convolution kernel.
Its input-gradient kernel, however, is a short causal correlation and can be
computed without the generic convolution backward launch.  The weight
gradient deliberately remains PyTorch's grouped-convolution implementation so
the custom path changes only the dominant input-gradient work.
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
        # Keep the existing grouped-convolution weight-gradient kernel.  This
        # avoids changing optimizer-visible parameter math while replacing
        # only the expensive input-gradient launch.
        padded_x = torch.nn.functional.pad(x, (weight.shape[-1] - 1, 0))
        grad_weight = torch.nn.grad.conv1d_weight(
            padded_x,
            weight.shape,
            grad_output,
            groups=x.shape[1],
        )
        return grad_x, grad_weight


def causal_depthwise_conv1d(x: torch.Tensor, weight: torch.Tensor) -> torch.Tensor:
    """Depthwise causal convolution with a Triton input-gradient path."""

    return _CausalDepthwiseFunction.apply(x, weight)

