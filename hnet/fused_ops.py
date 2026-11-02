"""Small graph-safe fused kernels used by the canonical optimizer.

The pointer tables are materialized before CUDA graph capture.  Replay only
launches the Triton kernel and performs no allocator or host/device transfer.
"""
from __future__ import annotations

import torch
import triton
import triton.language as tl


@triton.jit
def _fp32_to_bf16_ptrs(
    source_ptrs,
    destination_ptrs,
    block_tensor,
    block_local,
    lengths,
    block_size: tl.constexpr,
):
    block_id = tl.program_id(0)
    tensor_id = tl.load(block_tensor + block_id)
    local_block = tl.load(block_local + block_id)
    length = tl.load(lengths + tensor_id)
    source_address = tl.load(source_ptrs + tensor_id)
    destination_address = tl.load(destination_ptrs + tensor_id)
    offsets = local_block * block_size + tl.arange(0, block_size)
    mask = offsets < length
    source = tl.cast(source_address, tl.pointer_type(tl.float32))
    destination = tl.cast(destination_address, tl.pointer_type(tl.bfloat16))
    values = tl.load(source + offsets, mask=mask, other=0.0)
    tl.store(destination + offsets, values, mask=mask)


class Fp32ToBf16CastCache:
    """Batch FP32-master to BF16-parameter casts with stable pointer tables."""

    def __init__(self, block_size: int = 256):
        if block_size <= 0 or block_size & (block_size - 1):
            raise ValueError("block_size must be a positive power of two")
        self.block_size = int(block_size)
        self._tables = {}

    @staticmethod
    def _key(sources, destinations):
        return tuple(
            (int(source.data_ptr()), int(destination.data_ptr()), int(source.numel()))
            for source, destination in zip(sources, destinations, strict=True)
        )

    def _get_tables(self, sources, destinations):
        key = self._key(sources, destinations)
        tables = self._tables.get(key)
        if tables is not None:
            return tables
        device = sources[0].device
        if any(
            source.device != device
            or destination.device != device
            or source.dtype != torch.float32
            or destination.dtype != torch.bfloat16
            or source.numel() != destination.numel()
            for source, destination in zip(sources, destinations, strict=True)
        ):
            raise ValueError("fused cast requires matching CUDA FP32 to BF16 tensors")
        if device.type != "cuda":
            raise ValueError("fused cast requires CUDA tensors")
        lengths = [int(source.numel()) for source in sources]
        block_counts = [
            (length + self.block_size - 1) // self.block_size for length in lengths
        ]
        block_tensor_cpu = torch.cat(
            [
                torch.full((count,), index, dtype=torch.int32)
                for index, count in enumerate(block_counts)
            ]
        )
        block_local_cpu = torch.cat(
            [torch.arange(count, dtype=torch.int32) for count in block_counts]
        )
        source_ptrs = torch.tensor(
            [int(source.data_ptr()) for source in sources],
            dtype=torch.int64,
            device=device,
        )
        destination_ptrs = torch.tensor(
            [int(destination.data_ptr()) for destination in destinations],
            dtype=torch.int64,
            device=device,
        )
        block_tensor = block_tensor_cpu.to(device=device, non_blocking=True)
        block_local = block_local_cpu.to(device=device, non_blocking=True)
        length_tensor = torch.tensor(lengths, dtype=torch.int64, device=device)
        tables = (
            source_ptrs,
            destination_ptrs,
            block_tensor,
            block_local,
            length_tensor,
            int(block_tensor.numel()),
        )
        self._tables[key] = tables
        return tables

    def copy_(self, sources, destinations):
        sources = tuple(sources)
        destinations = tuple(destinations)
        if len(sources) != len(destinations):
            raise ValueError("source and destination lists must have equal length")
        if not sources:
            return
        tables = self._get_tables(sources, destinations)
        source_ptrs, destination_ptrs, block_tensor, block_local, lengths, blocks = tables
        _fp32_to_bf16_ptrs[(blocks,)](
            source_ptrs,
            destination_ptrs,
            block_tensor,
            block_local,
            lengths,
            block_size=self.block_size,
        )


@triton.jit
def _fp32_update_bf16_ptrs(
    master_ptrs,
    anchor_ptrs,
    update_ptrs,
    parameter_ptrs,
    block_tensor,
    block_local,
    lengths,
    lr_ptr,
    soda_ptr,
    weight_decay: tl.constexpr,
    block_size: tl.constexpr,
):
    block_id = tl.program_id(0)
    tensor_id = tl.load(block_tensor + block_id)
    local_block = tl.load(block_local + block_id)
    length = tl.load(lengths + tensor_id)
    master_address = tl.load(master_ptrs + tensor_id)
    anchor_address = tl.load(anchor_ptrs + tensor_id)
    update_address = tl.load(update_ptrs + tensor_id)
    parameter_address = tl.load(parameter_ptrs + tensor_id)
    offsets = local_block * block_size + tl.arange(0, block_size)
    mask = offsets < length
    master = tl.cast(master_address, tl.pointer_type(tl.float32))
    anchor = tl.cast(anchor_address, tl.pointer_type(tl.float32))
    update = tl.cast(update_address, tl.pointer_type(tl.float32))
    parameter = tl.cast(parameter_address, tl.pointer_type(tl.bfloat16))
    lr = tl.load(lr_ptr)
    soda = tl.load(soda_ptr)
    old = tl.load(master + offsets, mask=mask, other=0.0)
    z0 = tl.load(anchor + offsets, mask=mask, other=0.0)
    direction = tl.load(update + offsets, mask=mask, other=0.0)
    value = ((1.0 - soda) * old + soda * z0) * (1.0 - lr * weight_decay) - lr * direction
    tl.store(master + offsets, value, mask=mask)
    tl.store(parameter + offsets, value, mask=mask)


class Fp32UpdateBf16Cache:
    """Fused SODA/decay/update and FP32-master to BF16-parameter write."""

    def __init__(self, block_size: int = 256):
        self.block_size = int(block_size)
        self._tables = {}

    @staticmethod
    def _key(masters, anchors, updates, parameters, weight_decay):
        return (
            float(weight_decay),
            tuple(
                (int(master.data_ptr()), int(anchor.data_ptr()), int(update.data_ptr()), int(parameter.data_ptr()), int(master.numel()))
                for master, anchor, update, parameter in zip(masters, anchors, updates, parameters, strict=True)
            ),
        )

    def prepare(self, masters, anchors, updates, parameters, weight_decay):
        key = self._key(masters, anchors, updates, parameters, weight_decay)
        if key in self._tables:
            return
        device = masters[0].device
        if any(
            not (master.is_cuda and anchor.is_cuda and update.is_cuda and parameter.is_cuda)
            or master.dtype != torch.float32 or anchor.dtype != torch.float32
            or update.dtype != torch.float32 or parameter.dtype != torch.bfloat16
            or master.numel() != anchor.numel() or master.numel() != update.numel()
            or master.numel() != parameter.numel()
            for master, anchor, update, parameter in zip(masters, anchors, updates, parameters, strict=True)
        ):
            raise ValueError("fused update requires matching CUDA FP32/BF16 tensors")
        lengths = [int(master.numel()) for master in masters]
        counts = [(length + self.block_size - 1) // self.block_size for length in lengths]
        block_tensor = torch.cat([torch.full((count,), i, dtype=torch.int32) for i, count in enumerate(counts)]).to(device)
        block_local = torch.cat([torch.arange(count, dtype=torch.int32) for count in counts]).to(device)
        tables = (
            torch.tensor([int(x.data_ptr()) for x in masters], dtype=torch.int64, device=device),
            torch.tensor([int(x.data_ptr()) for x in anchors], dtype=torch.int64, device=device),
            torch.tensor([int(x.data_ptr()) for x in updates], dtype=torch.int64, device=device),
            torch.tensor([int(x.data_ptr()) for x in parameters], dtype=torch.int64, device=device),
            block_tensor,
            block_local,
            torch.tensor(lengths, dtype=torch.int64, device=device),
            int(block_tensor.numel()),
            float(weight_decay),
        )
        self._tables[key] = tables

    def apply(self, masters, anchors, updates, parameters, *, lr, soda, weight_decay):
        key = self._key(masters, anchors, updates, parameters, weight_decay)
        tables = self._tables.get(key)
        if tables is None:
            raise RuntimeError("fused update tables must be prepared before graph capture")
        mp, ap, up, pp, bt, bl, lengths, blocks, wd = tables
        if not isinstance(lr, torch.Tensor) or not isinstance(soda, torch.Tensor):
            raise ValueError("fused graph update requires CUDA scalar tensors")
        _fp32_update_bf16_ptrs[(blocks,)](
            mp, ap, up, pp, bt, bl, lengths, lr, soda,
            weight_decay=wd, block_size=self.block_size,
            # The reference PyTorch foreach path evaluates each FP32
            # operation with a separate round.  Triton's default FMA
            # contraction changes the result by an ulp at BF16 boundaries;
            # disable contraction so the fused path remains bitwise parity
            # with the exact optimizer recurrence.
            enable_fp_fusion=False,
        )
