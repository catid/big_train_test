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
def _packed_depthwise_fwd(
    x_ptr,
    seq_ptr,
    weight_ptr,
    out_ptr,
    batch,
    length,
    width,
    x_stride_b,
    x_stride_l,
    x_stride_d,
    seq_stride_b,
    seq_stride_l,
    weight_stride_d,
    weight_stride_k,
    out_stride_b,
    out_stride_l,
    out_stride_d,
    BLOCK_T: tl.constexpr,
    KERNEL: tl.constexpr,
):
    """Causal depthwise MSAR expansion with packed-document resets."""
    pid = tl.program_id(0)
    b = pid // width
    d = pid % width
    for tile_start in range(0, length, BLOCK_T):
        token = tile_start + tl.arange(0, BLOCK_T)
        token_mask = token < length
        acc = tl.zeros((BLOCK_T,), dtype=tl.float32)
        current_seq = tl.load(
            seq_ptr + b * seq_stride_b + token * seq_stride_l,
            mask=token_mask,
            other=0,
        )
        for coefficient in range(KERNEL):
            lag = KERNEL - 1 - coefficient
            source_token = token - lag
            source_mask = token_mask & (source_token >= 0)
            source_seq = tl.load(
                seq_ptr + b * seq_stride_b + source_token * seq_stride_l,
                mask=source_mask,
                other=-1,
            )
            same_document = source_mask & (source_seq == current_seq)
            value = tl.load(
                x_ptr
                + b * x_stride_b
                + source_token * x_stride_l
                + d * x_stride_d,
                mask=same_document,
                other=0,
            ).to(tl.float32)
            weight = tl.load(
                weight_ptr + d * weight_stride_d + coefficient * weight_stride_k
            ).to(tl.float32)
            acc += value * weight
        tl.store(
            out_ptr + b * out_stride_b + token * out_stride_l + d * out_stride_d,
            acc,
            mask=token_mask,
        )


@triton.jit
def _packed_depthwise_dx(
    grad_out_ptr,
    seq_ptr,
    weight_ptr,
    grad_x_ptr,
    batch,
    length,
    width,
    grad_out_stride_b,
    grad_out_stride_l,
    grad_out_stride_d,
    seq_stride_b,
    seq_stride_l,
    weight_stride_d,
    weight_stride_k,
    grad_x_stride_b,
    grad_x_stride_l,
    grad_x_stride_d,
    BLOCK_T: tl.constexpr,
    KERNEL: tl.constexpr,
):
    pid = tl.program_id(0)
    b = pid // width
    d = pid % width
    for tile_start in range(0, length, BLOCK_T):
        token = tile_start + tl.arange(0, BLOCK_T)
        token_mask = token < length
        current_seq = tl.load(
            seq_ptr + b * seq_stride_b + token * seq_stride_l,
            mask=token_mask,
            other=0,
        )
        acc = tl.zeros((BLOCK_T,), dtype=tl.float32)
        for lag in range(KERNEL):
            future_token = token + lag
            future_mask = token_mask & (future_token < length)
            future_seq = tl.load(
                seq_ptr + b * seq_stride_b + future_token * seq_stride_l,
                mask=future_mask,
                other=-1,
            )
            same_document = future_mask & (future_seq == current_seq)
            grad = tl.load(
                grad_out_ptr
                + b * grad_out_stride_b
                + future_token * grad_out_stride_l
                + d * grad_out_stride_d,
                mask=same_document,
                other=0,
            ).to(tl.float32)
            coefficient = KERNEL - 1 - lag
            weight = tl.load(
                weight_ptr + d * weight_stride_d + coefficient * weight_stride_k
            ).to(tl.float32)
            acc += grad * weight
        tl.store(
            grad_x_ptr + b * grad_x_stride_b + token * grad_x_stride_l + d * grad_x_stride_d,
            acc,
            mask=token_mask,
        )


@triton.jit
def _packed_depthwise_dw(
    x_ptr,
    grad_out_ptr,
    seq_ptr,
    grad_weight_ptr,
    batch,
    length,
    width,
    x_stride_b,
    x_stride_l,
    x_stride_d,
    grad_out_stride_b,
    grad_out_stride_l,
    grad_out_stride_d,
    seq_stride_b,
    seq_stride_l,
    grad_weight_stride_d,
    grad_weight_stride_k,
    BLOCK_T: tl.constexpr,
    KERNEL: tl.constexpr,
):
    pid = tl.program_id(0)
    d = pid // KERNEL
    coefficient = pid % KERNEL
    lag = KERNEL - 1 - coefficient
    total = tl.zeros((), dtype=tl.float32)
    for batch_index in range(batch):
        for tile_start in range(0, length, BLOCK_T):
            token = tile_start + tl.arange(0, BLOCK_T)
            token_mask = token < length
            source_token = token - lag
            source_mask = token_mask & (source_token >= 0)
            output_seq = tl.load(
                seq_ptr + batch_index * seq_stride_b + token * seq_stride_l,
                mask=token_mask,
                other=0,
            )
            source_seq = tl.load(
                seq_ptr + batch_index * seq_stride_b + source_token * seq_stride_l,
                mask=source_mask,
                other=-1,
            )
            same_document = source_mask & (source_seq == output_seq)
            value = tl.load(
                x_ptr
                + batch_index * x_stride_b
                + source_token * x_stride_l
                + d * x_stride_d,
                mask=same_document,
                other=0,
            ).to(tl.float32)
            grad = tl.load(
                grad_out_ptr
                + batch_index * grad_out_stride_b
                + token * grad_out_stride_l
                + d * grad_out_stride_d,
                mask=same_document,
                other=0,
            ).to(tl.float32)
            total += tl.sum(value * grad, axis=0)
    tl.store(
        grad_weight_ptr + d * grad_weight_stride_d + coefficient * grad_weight_stride_k,
        total,
    )


class _PackedCausalDepthwise(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x, weight, seq_idx):
        if x.ndim != 3 or weight.ndim != 2 or seq_idx.ndim != 2:
            raise ValueError("packed MSAR expects x=(B,L,D), weight=(D,K), seq_idx=(B,L)")
        if not (x.is_cuda and weight.is_cuda and seq_idx.is_cuda):
            raise RuntimeError("packed MSAR expansion requires CUDA tensors")
        if x.shape[:2] != seq_idx.shape or x.shape[-1] != weight.shape[0]:
            raise ValueError("packed MSAR tensors have incompatible shapes")
        x = x.contiguous()
        weight = weight.contiguous()
        seq_idx = seq_idx.contiguous()
        batch, length, width = x.shape
        kernel = weight.shape[-1]
        output = torch.empty_like(x)
        _packed_depthwise_fwd[(batch * width,)](
            x,
            seq_idx,
            weight,
            output,
            batch,
            length,
            width,
            *x.stride(),
            *seq_idx.stride(),
            *weight.stride(),
            *output.stride(),
            BLOCK_T=256,
            KERNEL=kernel,
            num_warps=4,
        )
        ctx.save_for_backward(x, weight, seq_idx)
        return output

    @staticmethod
    def backward(ctx, grad_output):
        x, weight, seq_idx = ctx.saved_tensors
        grad_output = grad_output.contiguous()
        batch, length, width = x.shape
        kernel = weight.shape[-1]
        grad_x = torch.empty_like(x)
        grad_weight = torch.empty_like(weight)
        _packed_depthwise_dx[(batch * width,)](
            grad_output,
            seq_idx,
            weight,
            grad_x,
            batch,
            length,
            width,
            *grad_output.stride(),
            *seq_idx.stride(),
            *weight.stride(),
            *grad_x.stride(),
            BLOCK_T=256,
            KERNEL=kernel,
            num_warps=4,
        )
        _packed_depthwise_dw[(width * kernel,)](
            x,
            grad_output,
            seq_idx,
            grad_weight,
            batch,
            length,
            width,
            *x.stride(),
            *grad_output.stride(),
            *seq_idx.stride(),
            *grad_weight.stride(),
            BLOCK_T=256,
            KERNEL=kernel,
            num_warps=4,
        )
        return grad_x, grad_weight, None


def packed_causal_depthwise_conv1d(x, weight, seq_idx):
    """Depthwise causal convolution that resets at packed document boundaries."""
    return _PackedCausalDepthwise.apply(x, weight, seq_idx)


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
        # source accumulation order is unchanged. The weight-gradient path
        # uses a 1024-wide reduction tile below, which cuts its six feature
        # passes to two at the production width while retaining FP32 math.
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
            # Weight-gradient programs are scalar reductions over D; two
            # warps improve occupancy on the B300 shape versus the generic
            # four-warp launch without changing arithmetic.
            num_warps=2,
        )
        return grad_weights, grad_bank


def weighted_source_reduce(weights: torch.Tensor, bank: torch.Tensor) -> torch.Tensor:
    """Reduce ``weights[:,B,L] * bank[:,B,L,D]`` in FP32 without a temporary."""

    if not weights.is_cuda or not bank.is_cuda:
        return (weights.unsqueeze(-1) * bank).sum(dim=0)
    return _WeightedSourceReduce.apply(weights, bank)


# The cached padded MSAR path keeps each source in its own allocation.  The
# old reduction first concatenated those singleton source banks into a new
# [N, B, L, D] tensor solely to feed ``weighted_source_reduce``.  Keep the
# source pointers as ordinary kernel arguments instead.  Eight pointers cover
# the canonical five-boundary stack and leave room for deeper configurations;
# the constexpr source count keeps unused arguments out of the loop.
@triton.jit
def _weighted_reduce_ptr_forward(
    weights_ptr,
    output_ptr,
    batch,
    length,
    width,
    weights_stride_n,
    weights_stride_b,
    weights_stride_l,
    bank0_ptr,
    bank1_ptr,
    bank2_ptr,
    bank3_ptr,
    bank4_ptr,
    bank5_ptr,
    bank6_ptr,
    bank7_ptr,
    bank_stride_b,
    bank_stride_l,
    bank_stride_d,
    BLOCK_D: tl.constexpr,
    N: tl.constexpr,
):
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
        if source == 0:
            bank_ptr = bank0_ptr
        elif source == 1:
            bank_ptr = bank1_ptr
        elif source == 2:
            bank_ptr = bank2_ptr
        elif source == 3:
            bank_ptr = bank3_ptr
        elif source == 4:
            bank_ptr = bank4_ptr
        elif source == 5:
            bank_ptr = bank5_ptr
        elif source == 6:
            bank_ptr = bank6_ptr
        else:
            bank_ptr = bank7_ptr
        weight = tl.load(
            weights_ptr + source * weights_stride_n
            + batch_index * weights_stride_b + token * weights_stride_l
        ).to(tl.float32)
        value = tl.load(
            bank_ptr + batch_index * bank_stride_b + token * bank_stride_l
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
def _weighted_reduce_ptr_grad_bank(
    grad_output_ptr,
    weights_ptr,
    grad_bank0_ptr,
    grad_bank1_ptr,
    grad_bank2_ptr,
    grad_bank3_ptr,
    grad_bank4_ptr,
    grad_bank5_ptr,
    grad_bank6_ptr,
    grad_bank7_ptr,
    batch,
    length,
    width,
    weights_stride_n,
    weights_stride_b,
    weights_stride_l,
    bank_stride_b,
    bank_stride_l,
    bank_stride_d,
    BLOCK_D: tl.constexpr,
    N: tl.constexpr,
):
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
        if source == 0:
            grad_bank_ptr = grad_bank0_ptr
        elif source == 1:
            grad_bank_ptr = grad_bank1_ptr
        elif source == 2:
            grad_bank_ptr = grad_bank2_ptr
        elif source == 3:
            grad_bank_ptr = grad_bank3_ptr
        elif source == 4:
            grad_bank_ptr = grad_bank4_ptr
        elif source == 5:
            grad_bank_ptr = grad_bank5_ptr
        elif source == 6:
            grad_bank_ptr = grad_bank6_ptr
        else:
            grad_bank_ptr = grad_bank7_ptr
        weight = tl.load(
            weights_ptr + source * weights_stride_n
            + batch_index * weights_stride_b + token * weights_stride_l
        ).to(tl.float32)
        tl.store(
            grad_bank_ptr + batch_index * bank_stride_b + token * bank_stride_l
            + d * bank_stride_d,
            grad_output * weight,
            mask=d_mask,
        )


@triton.jit
def _weighted_reduce_ptr_grad_weights(
    grad_output_ptr,
    bank0_ptr,
    bank1_ptr,
    bank2_ptr,
    bank3_ptr,
    bank4_ptr,
    bank5_ptr,
    bank6_ptr,
    bank7_ptr,
    grad_weights_ptr,
    batch,
    length,
    width,
    bank_stride_b,
    bank_stride_l,
    bank_stride_d,
    BLOCK_D: tl.constexpr,
    N: tl.constexpr,
):
    pid = tl.program_id(0)
    tokens = batch * length
    source = pid // tokens
    token_pid = pid % tokens
    batch_index = token_pid // length
    token = token_pid % length
    if source == 0:
        bank_ptr = bank0_ptr
    elif source == 1:
        bank_ptr = bank1_ptr
    elif source == 2:
        bank_ptr = bank2_ptr
    elif source == 3:
        bank_ptr = bank3_ptr
    elif source == 4:
        bank_ptr = bank4_ptr
    elif source == 5:
        bank_ptr = bank5_ptr
    elif source == 6:
        bank_ptr = bank6_ptr
    else:
        bank_ptr = bank7_ptr
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
            bank_ptr + batch_index * bank_stride_b + token * bank_stride_l
            + d * bank_stride_d,
            mask=d_mask,
            other=0,
        ).to(tl.float32)
        total += tl.sum(grad_output * value, axis=0)
    tl.store(grad_weights_ptr + source * batch * length + token_pid, total)


class _WeightedSourceReducePointers(torch.autograd.Function):
    """Reduce singleton source banks without materializing their concatenation."""

    @staticmethod
    def forward(ctx, weights: torch.Tensor, *banks: torch.Tensor) -> torch.Tensor:
        if not banks or len(banks) > 8:
            raise ValueError("pointer MSAR reduction supports one through eight banks")
        if weights.ndim != 3 or any(bank.ndim != 3 for bank in banks):
            raise ValueError("expected weights=(N,B,L) and banks=(B,L,D)")
        n, batch, length = weights.shape
        if n != len(banks) or any(bank.shape[:2] != (batch, length) for bank in banks):
            raise ValueError("pointer MSAR source and weight shapes do not match")
        if weights.dtype != torch.float32 or any(
            bank.dtype not in (torch.bfloat16, torch.float32) for bank in banks
        ):
            raise TypeError("pointer MSAR reduction expects FP32 weights and BF16/FP32 banks")
        width = banks[0].shape[-1]
        if any(bank.shape[-1] != width or not bank.is_contiguous() for bank in banks):
            raise ValueError("pointer MSAR banks must be contiguous and equally wide")
        output = torch.empty((batch, length, width), device=weights.device, dtype=torch.float32)
        padded = list(banks) + [banks[0]] * (8 - n)
        block_d = 1024
        _weighted_reduce_ptr_forward[(batch * length * triton.cdiv(width, block_d),)](
            weights,
            output,
            batch,
            length,
            width,
            weights.stride(0),
            weights.stride(1),
            weights.stride(2),
            *padded,
            banks[0].stride(0),
            banks[0].stride(1),
            banks[0].stride(2),
            BLOCK_D=block_d,
            N=n,
            num_warps=4,
        )
        ctx.save_for_backward(weights, *banks)
        return output

    @staticmethod
    def backward(ctx, grad_output: torch.Tensor):
        saved = ctx.saved_tensors
        weights, *banks = saved
        n, batch, length = weights.shape
        width = banks[0].shape[-1]
        grad_output = grad_output.contiguous().float()
        grad_weights = torch.empty_like(weights)
        grad_banks = [torch.empty_like(bank) for bank in banks]
        padded_banks = banks + [banks[0]] * (8 - n)
        padded_grads = grad_banks + [grad_banks[0]] * (8 - n)
        block_d = 1024
        _weighted_reduce_ptr_grad_bank[(batch * length * triton.cdiv(width, block_d),)](
            grad_output,
            weights,
            *padded_grads,
            batch,
            length,
            width,
            weights.stride(0),
            weights.stride(1),
            weights.stride(2),
            banks[0].stride(0),
            banks[0].stride(1),
            banks[0].stride(2),
            BLOCK_D=block_d,
            N=n,
            num_warps=4,
        )
        _weighted_reduce_ptr_grad_weights[(n * batch * length,)](
            grad_output,
            *padded_banks,
            grad_weights,
            batch,
            length,
            width,
            banks[0].stride(0),
            banks[0].stride(1),
            banks[0].stride(2),
            BLOCK_D=256,
            N=n,
            num_warps=2,
        )
        return (grad_weights, *grad_banks)


def weighted_source_reduce_pointers(weights: torch.Tensor, banks) -> torch.Tensor:
    """Pointer reduction for a list of contiguous [B,L,D] source banks."""

    if not weights.is_cuda:
        return sum(weight.unsqueeze(-1) * bank for weight, bank in zip(weights, banks))
    return _WeightedSourceReducePointers.apply(weights, *tuple(banks))


@triton.jit
def _msar_mix_forward(
    mix_ptr, s0_ptr, s1_ptr, s2_ptr, s3_ptr, out_ptr,
    tokens, width, BLOCK_D: tl.constexpr,
):
    """Fuse the four-stream MSAR weighted sum into one launch."""
    pid = tl.program_id(0)
    offsets = pid * BLOCK_D + tl.arange(0, BLOCK_D)
    token = pid // tl.cdiv(width, BLOCK_D)
    d = offsets % width
    mask = (token < tokens) & (offsets < (token + 1) * width)
    base = token * width
    m0 = tl.load(mix_ptr + token * 4 + 0).to(tl.float32)
    m1 = tl.load(mix_ptr + token * 4 + 1).to(tl.float32)
    m2 = tl.load(mix_ptr + token * 4 + 2).to(tl.float32)
    m3 = tl.load(mix_ptr + token * 4 + 3).to(tl.float32)
    x0 = tl.load(s0_ptr + base + d, mask=mask, other=0).to(tl.float32)
    x1 = tl.load(s1_ptr + base + d, mask=mask, other=0).to(tl.float32)
    x2 = tl.load(s2_ptr + base + d, mask=mask, other=0).to(tl.float32)
    x3 = tl.load(s3_ptr + base + d, mask=mask, other=0).to(tl.float32)
    out = x0 * m0 + x1 * m1
    out += x2 * m2
    out += x3 * m3
    tl.store(out_ptr + base + d, out, mask=mask)


@triton.jit
def _msar_mix_grad_mix(
    grad_ptr, s0_ptr, s1_ptr, s2_ptr, s3_ptr, grad_mix_ptr,
    tokens, width, BLOCK_D: tl.constexpr,
):
    """Gradient of the fused MSAR stream sum with respect to its gates."""
    token = tl.program_id(0)
    d = tl.arange(0, BLOCK_D)
    acc0 = tl.zeros((), dtype=tl.float32)
    acc1 = tl.zeros((), dtype=tl.float32)
    acc2 = tl.zeros((), dtype=tl.float32)
    acc3 = tl.zeros((), dtype=tl.float32)
    base = token * width
    for start in range(0, width, BLOCK_D):
        idx = start + d
        mask = (token < tokens) & (idx < width)
        grad = tl.load(grad_ptr + base + idx, mask=mask, other=0).to(tl.float32)
        acc0 += tl.sum(grad * tl.load(s0_ptr + base + idx, mask=mask, other=0).to(tl.float32))
        acc1 += tl.sum(grad * tl.load(s1_ptr + base + idx, mask=mask, other=0).to(tl.float32))
        acc2 += tl.sum(grad * tl.load(s2_ptr + base + idx, mask=mask, other=0).to(tl.float32))
        acc3 += tl.sum(grad * tl.load(s3_ptr + base + idx, mask=mask, other=0).to(tl.float32))
    tl.store(grad_mix_ptr + token * 4 + 0, acc0)
    tl.store(grad_mix_ptr + token * 4 + 1, acc1)
    tl.store(grad_mix_ptr + token * 4 + 2, acc2)
    tl.store(grad_mix_ptr + token * 4 + 3, acc3)


@triton.jit
def _msar_mix_grad_streams(
    grad_ptr, mix_ptr, g0_ptr, g1_ptr, g2_ptr, g3_ptr,
    tokens, width, BLOCK_D: tl.constexpr,
):
    """Gradient of the fused MSAR stream sum with respect to its streams."""
    pid = tl.program_id(0)
    offsets = pid * BLOCK_D + tl.arange(0, BLOCK_D)
    token = pid // tl.cdiv(width, BLOCK_D)
    d = offsets % width
    mask = (token < tokens) & (offsets < (token + 1) * width)
    base = token * width
    grad = tl.load(grad_ptr + base + d, mask=mask, other=0).to(tl.float32)
    m0 = tl.load(mix_ptr + token * 4 + 0).to(tl.float32)
    m1 = tl.load(mix_ptr + token * 4 + 1).to(tl.float32)
    m2 = tl.load(mix_ptr + token * 4 + 2).to(tl.float32)
    m3 = tl.load(mix_ptr + token * 4 + 3).to(tl.float32)
    tl.store(g0_ptr + base + d, grad * m0, mask=mask)
    tl.store(g1_ptr + base + d, grad * m1, mask=mask)
    tl.store(g2_ptr + base + d, grad * m2, mask=mask)
    tl.store(g3_ptr + base + d, grad * m3, mask=mask)


@triton.jit
def _msar_mix_grad_fused(
    grad_ptr, mix_ptr, s0_ptr, s1_ptr, s2_ptr, s3_ptr,
    g0_ptr, g1_ptr, g2_ptr, g3_ptr, grad_mix_ptr,
    tokens, width, BLOCK_D: tl.constexpr,
):
    """Fuse MSAR mix stream and gate gradients in one token program.

    The stream-gradient map and gate-gradient map both read ``grad`` and the
    four source streams.  Keeping one program per token lets the loop write
    each stream gradient while accumulating the four FP32 dot products in the
    same tile order as the previous two kernels.  The output stream dtype is
    preserved by the stores; only the gate derivatives remain FP32.
    """
    token = tl.program_id(0)
    d = tl.arange(0, BLOCK_D)
    base = token * width
    m0 = tl.load(mix_ptr + token * 4 + 0).to(tl.float32)
    m1 = tl.load(mix_ptr + token * 4 + 1).to(tl.float32)
    m2 = tl.load(mix_ptr + token * 4 + 2).to(tl.float32)
    m3 = tl.load(mix_ptr + token * 4 + 3).to(tl.float32)
    acc0 = tl.zeros((), dtype=tl.float32)
    acc1 = tl.zeros((), dtype=tl.float32)
    acc2 = tl.zeros((), dtype=tl.float32)
    acc3 = tl.zeros((), dtype=tl.float32)
    for start in range(0, width, BLOCK_D):
        idx = start + d
        mask = (token < tokens) & (idx < width)
        grad = tl.load(grad_ptr + base + idx, mask=mask, other=0).to(tl.float32)
        s0 = tl.load(s0_ptr + base + idx, mask=mask, other=0).to(tl.float32)
        s1 = tl.load(s1_ptr + base + idx, mask=mask, other=0).to(tl.float32)
        s2 = tl.load(s2_ptr + base + idx, mask=mask, other=0).to(tl.float32)
        s3 = tl.load(s3_ptr + base + idx, mask=mask, other=0).to(tl.float32)
        tl.store(g0_ptr + base + idx, grad * m0, mask=mask)
        tl.store(g1_ptr + base + idx, grad * m1, mask=mask)
        tl.store(g2_ptr + base + idx, grad * m2, mask=mask)
        tl.store(g3_ptr + base + idx, grad * m3, mask=mask)
        acc0 += tl.sum(grad * s0, axis=0)
        acc1 += tl.sum(grad * s1, axis=0)
        acc2 += tl.sum(grad * s2, axis=0)
        acc3 += tl.sum(grad * s3, axis=0)
    tl.store(grad_mix_ptr + token * 4 + 0, acc0)
    tl.store(grad_mix_ptr + token * 4 + 1, acc1)
    tl.store(grad_mix_ptr + token * 4 + 2, acc2)
    tl.store(grad_mix_ptr + token * 4 + 3, acc3)


class _MSARMix(torch.autograd.Function):
    @staticmethod
    def forward(ctx, mix, s0, s1, s2, s3, result_dtype=None):
        if not all(x.is_cuda for x in (mix, s0, s1, s2, s3)):
            raise RuntimeError("MSAR mix requires CUDA tensors")
        mix, s0, s1, s2, s3 = (x.contiguous() for x in (mix, s0, s1, s2, s3))
        shape = s0.shape
        if mix.shape != shape[:-1] + (4,) or any(x.shape != shape for x in (s1, s2, s3)):
            raise ValueError("expected mix=(B,L,4) and four matching streams")
        # Padded training uses [B,L,D], while packed/chunked calls use
        # [tokens,D].  The flattened storage contract is identical.
        width = shape[-1]
        tokens = s0.numel() // width
        # The recurrent MSAR state is consumed at the block's parameter
        # precision.  Allocate that destination directly so the graph does
        # not launch a separate full activation cast after the mix kernel.
        out = torch.empty(
            s0.shape,
            device=s0.device,
            dtype=s0.dtype if result_dtype is None else result_dtype,
        )
        grid = (tokens * triton.cdiv(width, 256),)
        _msar_mix_forward[grid](mix, s0, s1, s2, s3, out, tokens, width, BLOCK_D=256, num_warps=4)
        ctx.save_for_backward(mix, s0, s1, s2, s3)
        return out

    @staticmethod
    def backward(ctx, grad_out):
        mix, s0, s1, s2, s3 = ctx.saved_tensors
        width = s0.shape[-1]
        tokens = s0.numel() // width
        grad_out = grad_out.contiguous().float()
        grad_mix = torch.empty_like(mix)
        grads = [torch.empty_like(s0) for _ in range(4)]
        _msar_mix_grad_fused[(tokens,)](
            grad_out,
            mix,
            s0,
            s1,
            s2,
            s3,
            *grads,
            grad_mix,
            tokens,
            width,
            BLOCK_D=256,
            num_warps=4,
        )
        return (grad_mix, *grads, None)


def fused_msar_mix(mix: torch.Tensor, streams, result_dtype=None):
    """Fuse the canonical four-stream MSAR weighted sum."""
    if len(streams) != 4:
        raise ValueError(f"production MSAR expects four streams, got {len(streams)}")
    return _MSARMix.apply(mix, *streams, result_dtype)


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
        # Weight-gradient reduction is bandwidth-bound.  A 512-token tile halves the
        # launch grid at the canonical B16/S4096 shape and benchmarks faster
        # across the source-bank sizes while retaining the same FP32
        # accumulation and causal indexing.
        BLOCK_T=512,
        KERNEL=kernel,
        K_PAD=16,
        num_warps=2,
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
    # Match the larger backward tile used by the input-gradient kernel.  The
    # weight-gradient reduction is especially launch-bound because each
    # program owns one channel and streams the full batch/sequence.
    block_t = 512
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


@triton.jit
def _msar_gate_forward(
    start_ptr, output_ptr, residual_ptr, result_ptr, scale_ptr, elements,
    BLOCK: tl.constexpr,
):
    pid = tl.program_id(0)
    offsets = pid * BLOCK + tl.arange(0, BLOCK)
    mask = offsets < elements
    start = tl.load(start_ptr + offsets, mask=mask, other=0).to(tl.float32)
    output = tl.load(output_ptr + offsets, mask=mask, other=0).to(tl.float32)
    residual = tl.load(residual_ptr + offsets, mask=mask, other=0).to(tl.float32)
    scale = tl.load(scale_ptr).to(tl.float32)
    tl.store(result_ptr + offsets, start + (output + residual - start) * scale, mask=mask)


@triton.jit
def _msar_gate_backward(
    grad_result_ptr, start_ptr, output_ptr, residual_ptr, scale_ptr,
    grad_start_ptr, grad_output_ptr, grad_residual_ptr, grad_scale_partials_ptr,
    elements,
    BLOCK: tl.constexpr,
):
    pid = tl.program_id(0)
    offsets = pid * BLOCK + tl.arange(0, BLOCK)
    mask = offsets < elements
    grad_result = tl.load(grad_result_ptr + offsets, mask=mask, other=0).to(tl.float32)
    start = tl.load(start_ptr + offsets, mask=mask, other=0).to(tl.float32)
    output = tl.load(output_ptr + offsets, mask=mask, other=0).to(tl.float32)
    residual = tl.load(residual_ptr + offsets, mask=mask, other=0).to(tl.float32)
    scale = tl.load(scale_ptr).to(tl.float32)
    grad_output = grad_result * scale
    tl.store(grad_start_ptr + offsets, grad_result * (1.0 - scale), mask=mask)
    tl.store(grad_output_ptr + offsets, grad_output, mask=mask)
    tl.store(grad_residual_ptr + offsets, grad_output, mask=mask)
    # Keep the scale derivative in the same pass as the activation gradients.
    # The old spelling widened three complete activations and launched a
    # second reduction after this kernel.  One FP32 partial per program keeps
    # the reduction deterministic while avoiding those full-size temporaries.
    partial = tl.sum(grad_result * (output + residual - start), axis=0)
    tl.store(grad_scale_partials_ptr + pid, partial)


class _MSARGate(torch.autograd.Function):
    @staticmethod
    def forward(ctx, start, output, residual, scale, result_dtype=None):
        if not (start.is_cuda and output.is_cuda and residual.is_cuda and scale.is_cuda):
            raise RuntimeError("MSAR gate requires CUDA tensors")
        if start.shape != output.shape or start.shape != residual.shape:
            raise ValueError("MSAR gate tensors must have matching shapes")
        start, output, residual, scale = (
            start.contiguous(), output.contiguous(), residual.contiguous(), scale.contiguous()
        )
        # The gate accumulates in FP32, but the recurrent MSAR state is stored
        # at the block parameter precision.  Let the kernel narrow directly
        # into its final destination when requested, avoiding a second full
        # activation cast immediately after the fused gate.
        result = torch.empty_like(
            start, dtype=torch.float32 if result_dtype is None else result_dtype
        )
        elements = start.numel()
        _msar_gate_forward[(triton.cdiv(elements, 256),)](
            start, output, residual, result, scale, elements, BLOCK=256, num_warps=4
        )
        ctx.save_for_backward(start, output, residual, scale)
        return result

    @staticmethod
    def backward(ctx, grad_result):
        start, output, residual, scale = ctx.saved_tensors
        elements = start.numel()
        grad_result = grad_result.contiguous()
        grad_start, grad_output, grad_residual = (
            torch.empty_like(start), torch.empty_like(output), torch.empty_like(residual)
        )
        blocks = triton.cdiv(elements, 256)
        grad_scale_partials = torch.empty(
            (blocks,), device=grad_result.device, dtype=torch.float32
        )
        _msar_gate_backward[(blocks,)](
            grad_result, start, output, residual, scale,
            grad_start, grad_output, grad_residual, grad_scale_partials,
            elements, BLOCK=256, num_warps=4
        )
        grad_scale = grad_scale_partials.sum()
        return grad_start, grad_output, grad_residual, grad_scale, None


def fused_msar_gate(start, output, residual, scale, result_dtype=None):
    """Fuse FP32 MSAR residual addition and gated interpolation."""
    return _MSARGate.apply(start, output, residual, scale, result_dtype)
