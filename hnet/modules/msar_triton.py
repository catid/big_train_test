"""Triton kernels for the padded MSAR depthwise convolution.

The MSAR expansion and both gradient correlations use compact causal Triton
kernels. This avoids the generic grouped-convolution launches selected by
``torch.conv1d`` at the production batch and channel sizes while preserving
the zero-left-padding contract.
"""

from __future__ import annotations

import torch
import triton
import triton.language as tl


@triton.jit
def _causal_depthwise_dy(
    x_ptr,
    weight_ptr,
    output_ptr,
    batch,
    channels,
    length,
    BLOCK_T: tl.constexpr,
    KERNEL: tl.constexpr,
):
    """Forward causal depthwise correlation for the MSAR stream bank.

    ``torch.conv1d`` was previously used here.  At the production B16/S4096
    shape it selected the generic depthwise path and accounted for roughly
    29 ms of every replay.  The kernel keeps the same zero-left-padding and
    filter orientation, accumulates in FP32, and narrows once at the output.
    """

    program = tl.program_id(0)
    batch_index = program // channels
    channel = program % channels
    base = (batch_index * channels + channel) * length
    for tile_start in range(0, length, BLOCK_T):
        token = tile_start + tl.arange(0, BLOCK_T)
        token_mask = token < length
        acc = tl.zeros((BLOCK_T,), dtype=tl.float32)
        for coefficient in range(KERNEL):
            source_token = token - (KERNEL - 1 - coefficient)
            source_mask = token_mask & (source_token >= 0)
            source = tl.load(
                x_ptr + base + source_token,
                mask=source_mask,
                other=0,
            ).to(tl.float32)
            weight = tl.load(weight_ptr + channel * KERNEL + coefficient).to(tl.float32)
            acc += source * weight
        tl.store(output_ptr + base + token, acc, mask=token_mask)


@triton.jit
def _weighted_reduce_forward(
    weights_ptr,
    bank_ptr,
    output_ptr,
    batch,
    length,
    width,
    weights_stride_n,
    weights_stride_b,
    weights_stride_l,
    bank_stride_n,
    bank_stride_b,
    bank_stride_l,
    bank_stride_d,
    BLOCK_D: tl.constexpr,
    N: tl.constexpr,
):
    """Reduce a fixed MSAR source bank without a source-sized temporary."""

    pid = tl.program_id(0)
    blocks_d = tl.cdiv(width, BLOCK_D)
    token_pid = pid // blocks_d
    d_pid = pid % blocks_d
    batch_index = token_pid // length
    token = token_pid % length
    d = d_pid * BLOCK_D + tl.arange(0, BLOCK_D)
    d_mask = d < width
    acc = tl.zeros((BLOCK_D,), dtype=tl.float32)
    for source in range(N):
        weight = tl.load(
            weights_ptr
            + source * weights_stride_n
            + batch_index * weights_stride_b
            + token * weights_stride_l
        ).to(tl.float32)
        value = tl.load(
            bank_ptr
            + source * bank_stride_n
            + batch_index * bank_stride_b
            + token * bank_stride_l
            + d * bank_stride_d,
            mask=d_mask,
            other=0,
        ).to(tl.float32)
        acc += weight * value
    tl.store(
        output_ptr + (batch_index * length + token) * width + d,
        acc,
        mask=d_mask,
    )


@triton.jit
def _weighted_reduce_grad_bank(
    grad_output_ptr,
    weights_ptr,
    grad_bank_ptr,
    batch,
    length,
    width,
    weights_stride_n,
    weights_stride_b,
    weights_stride_l,
    bank_stride_n,
    bank_stride_b,
    bank_stride_l,
    bank_stride_d,
    grad_bank_stride_n,
    grad_bank_stride_b,
    grad_bank_stride_l,
    grad_bank_stride_d,
    BLOCK_D: tl.constexpr,
    N: tl.constexpr,
):
    """Gradient of the weighted source reduction with one launch."""

    pid = tl.program_id(0)
    blocks_d = tl.cdiv(width, BLOCK_D)
    token_pid = pid // blocks_d
    d_pid = pid % blocks_d
    batch_index = token_pid // length
    token = token_pid % length
    d = d_pid * BLOCK_D + tl.arange(0, BLOCK_D)
    d_mask = d < width
    grad_output = tl.load(
        grad_output_ptr + (batch_index * length + token) * width + d,
        mask=d_mask,
        other=0,
    ).to(tl.float32)
    for source in range(N):
        weight = tl.load(
            weights_ptr
            + source * weights_stride_n
            + batch_index * weights_stride_b
            + token * weights_stride_l
        ).to(tl.float32)
        tl.store(
            grad_bank_ptr
            + (source * grad_bank_stride_n + batch_index * grad_bank_stride_b
               + token * grad_bank_stride_l + d * grad_bank_stride_d),
            grad_output * weight,
            mask=d_mask,
        )


@triton.jit
def _weighted_reduce_grad_weights(
    grad_output_ptr,
    bank_ptr,
    grad_weights_ptr,
    batch,
    length,
    width,
    bank_stride_n,
    bank_stride_b,
    bank_stride_l,
    bank_stride_d,
    BLOCK_D: tl.constexpr,
    N: tl.constexpr,
):
    """Gradient of the source weights, accumulated in FP32."""

    pid = tl.program_id(0)
    tokens = batch * length
    source = pid // tokens
    token_pid = pid % tokens
    batch_index = token_pid // length
    token = token_pid % length
    acc = tl.zeros((BLOCK_D,), dtype=tl.float32)
    # BLOCK_D is also the reduction tile.  Keep a scalar accumulator per
    # program so the output ordering follows the source-major PyTorch sum.
    total = tl.zeros((), dtype=tl.float32)
    for d_start in range(0, width, BLOCK_D):
        d = d_start + tl.arange(0, BLOCK_D)
        d_mask = d < width
        grad_output = tl.load(
            grad_output_ptr + (batch_index * length + token) * width + d,
            mask=d_mask,
            other=0,
        ).to(tl.float32)
        value = tl.load(
            bank_ptr
            + source * bank_stride_n
            + batch_index * bank_stride_b
            + token * bank_stride_l
            + d * bank_stride_d,
            mask=d_mask,
            other=0,
        ).to(tl.float32)
        total += tl.sum(grad_output * value, axis=0)
    tl.store(
        grad_weights_ptr + source * batch * length + batch_index * length + token,
        total,
    )


class _WeightedSourceReduce(torch.autograd.Function):
    """FP32 MSAR weighted reduction with fused forward and backward maps."""

    @staticmethod
    def forward(ctx, weights: torch.Tensor, bank: torch.Tensor) -> torch.Tensor:
        if not weights.is_cuda or not bank.is_cuda:
            raise RuntimeError("MSAR weighted reduction requires CUDA tensors")
        if weights.ndim != 3 or bank.ndim != 4:
            raise ValueError("expected weights=(N,B,L) and bank=(N,B,L,D)")
        if weights.shape[:3] != bank.shape[:3]:
            raise ValueError("MSAR source and weight shapes do not match")
        if weights.dtype != torch.float32 or bank.dtype not in (
            torch.bfloat16,
            torch.float32,
        ):
            raise TypeError("MSAR weighted reduction expects FP32 weights and BF16/FP32 bank")
        n, batch, length, width = bank.shape
        output = torch.empty((batch, length, width), device=bank.device, dtype=torch.float32)
        # The production MSAR width is 1536.  A 1024-wide tile uses two
        # feature tiles instead of three and is faster on the B300 shape.
        # This dimension only partitions independent output features; the
        # source accumulation order is unchanged. Keep the weight-gradient
        # tile at 256 below so its FP32 reduction order remains canonical.
        block_d = 1024
        grid = (batch * length * triton.cdiv(width, block_d),)
        _weighted_reduce_forward[grid](
            weights,
            bank,
            output,
            batch,
            length,
            width,
            *weights.stride(),
            *bank.stride(),
            BLOCK_D=block_d,
            N=n,
            num_warps=4,
        )
        ctx.save_for_backward(weights, bank)
        return output

    @staticmethod
    def backward(ctx, grad_output: torch.Tensor):
        weights, bank = ctx.saved_tensors
        n, batch, length, width = bank.shape
        grad_output = grad_output.contiguous().float()
        grad_bank = torch.empty_like(bank)
        # Match the forward feature tiling. The gradient writes are
        # independent across feature tiles, so this changes launch count only
        # and preserves the per-element/source arithmetic.
        block_d = 1024
        _weighted_reduce_grad_bank[(batch * length * triton.cdiv(width, block_d),)](
            grad_output,
            weights,
            grad_bank,
            batch,
            length,
            width,
            *weights.stride(),
            *bank.stride(),
            *grad_bank.stride(),
            BLOCK_D=block_d,
            N=n,
            num_warps=4,
        )
        grad_weights = torch.empty_like(weights)
        _weighted_reduce_grad_weights[(n * batch * length,)](
            grad_output,
            bank,
            grad_weights,
            batch,
            length,
            width,
            *bank.stride(),
            BLOCK_D=256,
            N=n,
            num_warps=4,
        )
        return grad_weights, grad_bank


def weighted_source_reduce(weights: torch.Tensor, bank: torch.Tensor) -> torch.Tensor:
    """Reduce ``weights[:,B,L] * bank[:,B,L,D]`` in FP32 without a temporary."""

    if not weights.is_cuda or not bank.is_cuda:
        return (weights.unsqueeze(-1) * bank).sum(dim=0)
    return _WeightedSourceReduce.apply(weights, bank)


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


def _depthwise_dy(x: torch.Tensor, weight: torch.Tensor) -> torch.Tensor:
    """Fused forward path matching the causal grouped-convolution contract."""

    if not x.is_cuda or not weight.is_cuda:
        raise RuntimeError("MSAR Triton depthwise forward requires CUDA tensors")
    if x.ndim != 3 or weight.ndim != 3 or weight.shape[1] != 1:
        raise ValueError(
            "expected x=(B,C,L) and depthwise weight=(C,1,K), got "
            f"{tuple(x.shape)} and {tuple(weight.shape)}"
        )
    if x.shape[1] != weight.shape[0]:
        raise ValueError("depthwise weight channel count must match x")
    batch, channels, length = x.shape
    kernel = weight.shape[-1]
    output = torch.empty_like(x)
    _causal_depthwise_dy[(batch * channels,)](
        x,
        weight,
        output,
        batch,
        channels,
        length,
        BLOCK_T=256,
        KERNEL=kernel,
        num_warps=4,
    )
    return output


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
        # Use the same fused correlation in forward and backward.  This
        # avoids the generic depthwise-convolution launch selected by
        # ``torch.conv1d`` for the large grouped MSAR bank.
        output = _depthwise_dy(x, weight)
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

