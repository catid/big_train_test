"""Triton kernels for the padded MSAR depthwise convolution.

The forward MSAR expansion stays on PyTorch's depthwise convolution kernel.
Its input and weight-gradient kernels are short causal correlations and can be
computed without the generic grouped-convolution backward launches.
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


@triton.jit
def _causal_depthwise_dw(
    grad_y_ptr,
    x_ptr,
    grad_weight_ptr,
    batch,
    channels,
    length,
    BLOCK_T: tl.constexpr,
    KERNEL: tl.constexpr,
    K_PAD: tl.constexpr,
):
    """Weight gradient for the causal depthwise correlation.

    The forward uses ``w[K - 1 - lag]`` at output position ``t``.  One
    program owns a channel and accumulates all K coefficients in FP32 while
    streaming the input/gradient tiles over the batch.  Keeping the
    coefficient loop inside the program avoids the generic grouped-conv
    weight-gradient setup and its per-call workspace for this tiny fixed
    depthwise kernel.
    """

    channel = tl.program_id(0)
    acc = tl.zeros((K_PAD,), dtype=tl.float32)
    coefficient = tl.arange(0, K_PAD)
    coefficient_mask = coefficient < KERNEL
    for batch_index in range(batch):
        base = (batch_index * channels + channel) * length
        for tile_start in range(0, length, BLOCK_T):
            token = tile_start + tl.arange(0, BLOCK_T)
            token_mask = token < length
            grad_y = tl.load(grad_y_ptr + base + token, mask=token_mask, other=0).to(
                tl.float32
            )
            # Keep the coefficient axis explicit so each coefficient gets a
            # distinct reduction.  K_PAD is a power of two (required by
            # Triton block shapes); lanes beyond KERNEL are masked.
            source_token = token[:, None] - (KERNEL - 1 - coefficient[None, :])
            source_mask = token_mask[:, None] & (source_token >= 0) & coefficient_mask[None, :]
            source = tl.load(
                x_ptr + base + source_token,
                mask=source_mask,
                other=0,
            ).to(tl.float32)
            acc += tl.sum(grad_y[:, None] * source, axis=0)
    # The parameter is BF16 in production; the cast is performed by the
    # store rather than narrowing each partial accumulation.
    tl.store(
        grad_weight_ptr + channel * KERNEL + coefficient,
        acc,
        mask=coefficient_mask,
    )


def _depthwise_dw(
    grad_y: torch.Tensor,
    x: torch.Tensor,
    kernel: int,
    output_dtype: torch.dtype,
) -> torch.Tensor:
    if not grad_y.is_cuda or not x.is_cuda:
        raise RuntimeError("MSAR Triton depthwise backward requires CUDA tensors")
    batch, channels, length = grad_y.shape
    grad_weight = torch.empty(
        (channels, 1, kernel), device=grad_y.device, dtype=torch.float32
    )
    _causal_depthwise_dw[(channels,)](
        grad_y,
        x,
        grad_weight,
        batch,
        channels,
        length,
        BLOCK_T=256,
        KERNEL=kernel,
        K_PAD=16,
        num_warps=4,
    )
    return grad_weight.to(dtype=output_dtype)


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
        # The depthwise weight gradient is the same causal correlation as the
        # forward, with one independent reduction per channel.  Keep its
        # accumulation in FP32 inside the fused Triton kernel and narrow only
        # the final result to the parameter dtype.
        grad_weight = _depthwise_dw(
            grad_output, x, weight.shape[-1], weight.dtype
        )
        return grad_x, grad_weight


def causal_depthwise_conv1d(x: torch.Tensor, weight: torch.Tensor) -> torch.Tensor:
    """Depthwise causal convolution with a Triton input-gradient path."""

    return _CausalDepthwiseFunction.apply(x, weight)

