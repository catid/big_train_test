#!/usr/bin/env python3
"""Benchmark the document-local byte stream used by the canonical trainer.

This measures only input staging and DMA. It does not instantiate a model or
claim a model training throughput result. For the production CUDA-graph path,
run it with ``torchrun`` so the measured value includes the slowest rank::

    torchrun --standalone --nproc_per_node=8 tools/benchmark_byte_stream.py \
      --manifest /data/corpus/train.manifest.json --batch 18 --steps 100
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import time
from pathlib import Path

import torch
import torch.distributed as dist

# Allow ``python tools/benchmark_byte_stream.py`` from the repository root.
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from train_hnet_600m import ByteStream, _load_manifest, _load_manifest_distributed


def _distributed_context() -> tuple[int, int, torch.device]:
    world = int(os.environ.get("WORLD_SIZE", "1"))
    rank = int(os.environ.get("RANK", "0"))
    if world > 1:
        if not torch.cuda.is_available():
            raise RuntimeError("distributed byte-stream benchmark requires CUDA")
        local_rank = int(os.environ.get("LOCAL_RANK", rank))
        torch.cuda.set_device(local_rank)
        dist.init_process_group("nccl", device_id=torch.device("cuda", local_rank))
        return rank, world, torch.device("cuda", local_rank)
    device = torch.device("cuda", torch.cuda.current_device()) if torch.cuda.is_available() else torch.device("cpu")
    return rank, world, device


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", required=True)
    parser.add_argument("--batch", type=int, default=16)
    parser.add_argument("--seq", type=int, default=8192)
    parser.add_argument("--steps", type=int, default=100)
    parser.add_argument("--warmup", type=int, default=10)
    parser.add_argument("--seed", type=int, default=1729)
    parser.add_argument(
        "--skip-manifest-hash",
        action="store_true",
        help="skip startup content hashing after the manifest was independently verified",
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    if args.batch < 1 or args.seq < 2 or args.steps < 1 or args.warmup < 0:
        raise ValueError("batch, steps, and warmup must be positive; seq must be at least 2")
    rank, world, device = _distributed_context()
    # Match the production trainer's startup contract: rank zero verifies the
    # large immutable payload/index/metadata digests once, while other ranks
    # perform structural checks and compare the resulting identity.  Calling
    # ``_load_manifest`` directly here made an eight-rank benchmark hash the
    # entire corpus eight times before measuring the loader.
    if args.skip_manifest_hash:
        manifest = _load_manifest(args.manifest, verify_hashes=False)
    else:
        manifest = _load_manifest_distributed(
            args.manifest,
            None,
            rank,
            world,
            hash_mode="rank0" if world > 1 else "strict",
        )
    stream = ByteStream(manifest, args.seq, args.batch, rank, world, seed=args.seed)
    direct = device.type == "cuda"
    buffers = None
    consumer_stream = None
    if direct:
        buffers = (
            torch.empty((args.batch, args.seq), dtype=torch.uint8, device=device),
            torch.empty((args.batch, args.seq), dtype=torch.uint8, device=device),
            torch.empty((args.batch, args.seq), dtype=torch.bool, device=device),
            torch.empty((args.batch, args.seq), dtype=torch.bool, device=device),
        )
        consumer_stream = torch.cuda.Stream(device=device)
    if world > 1:
        dist.barrier()
    for step in range(args.warmup):
        stream.get(
            step,
            device,
            consumer_stream=consumer_stream,
            device_buffers=buffers,
        )
    if direct:
        torch.cuda.synchronize(device)
    if world > 1:
        dist.barrier()
    start = time.perf_counter()
    for step in range(args.steps):
        stream.get(
            step + args.warmup,
            device,
            consumer_stream=consumer_stream,
            device_buffers=buffers,
        )
    if direct:
        torch.cuda.synchronize(device)
    elapsed = time.perf_counter() - start
    elapsed_tensor = torch.tensor([elapsed], device=device)
    if world > 1:
        dist.all_reduce(elapsed_tensor, op=dist.ReduceOp.MAX)
    if rank == 0:
        result = {
            "manifest": str(manifest.path),
            "manifest_version": manifest.version,
            "documents": stream.document_count,
            "batch": args.batch,
            "seq": args.seq,
            "world": world,
            "device": str(device),
            "direct_dma": direct,
            "full_windows": stream.full_windows,
            "steps": args.steps,
            "max_ms_per_batch": float(elapsed_tensor.item() * 1000.0 / args.steps),
            "max_bytes_per_second": float(
                args.batch * args.seq * 2 * args.steps / elapsed_tensor.item()
            ),
        }
        print(json.dumps(result, sort_keys=True), flush=True)
    if world > 1:
        dist.destroy_process_group()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
