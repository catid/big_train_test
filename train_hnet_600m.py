"""Direct 600M H-Net raw-byte trainer.

The model consumes UTF-8 bytes through a frozen categorical byte code, then
learns boundaries with H-Net routing from the first update. There is no BPE or
trainable token embedding. The script keeps the CE and router losses in FP32,
supports Muon/GramNS, ANVIL-II, AdamATan2 fallback, FBT multi-pass training,
and captures a fixed-shape CUDA graph when requested.
"""

from __future__ import annotations

import argparse
import json
import math
import os
import subprocess
import time
from pathlib import Path

import torch
import torch.distributed as dist
import torch.nn.functional as F
from torch.nn.parallel import DistributedDataParallel as DDP

from hnet.models.config_hnet import AttnConfig, HNetConfig, SSMConfig
from hnet.models.mixer_seq import HNetForByteCausalLM
from hnet.utils.train import load_balancing_loss


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--data", required=True, help="raw UTF-8 byte stream")
    p.add_argument("--config", default="configs/hnet_1stage_600m_loop2.json")
    p.add_argument("--seq", type=int, default=4096)
    p.add_argument("--batch", type=int, default=1)
    p.add_argument("--steps", type=int, default=100000)
    p.add_argument("--warmup", type=int, default=2000)
    p.add_argument("--lr", type=float, default=1.2e-3)
    p.add_argument("--fallback-lr", type=float, default=6e-4)
    p.add_argument("--weight-decay", type=float, default=0.05)
    p.add_argument("--optimizer", choices=["gramns", "anvil", "adamw"], default="gramns")
    p.add_argument("--fbt-passes", type=int, default=2)
    p.add_argument("--fbt-lambda", type=float, default=1.0)
    p.add_argument("--ratio-weight", type=float, default=0.03)
    p.add_argument("--ratio-target", type=float, default=6.0)
    p.add_argument("--log-every", type=int, default=10)
    p.add_argument("--ckpt-every", type=int, default=1000)
    p.add_argument("--out", default="/home/catid/hnet/artifacts/hnet_600m_train.json")
    p.add_argument("--checkpoint", default="/home/catid/hnet/artifacts/hnet_600m_last.pt")
    p.add_argument("--cuda-graphs", action=argparse.BooleanOptionalAction, default=True)
    p.add_argument("--autotune-batch", action="store_true")
    return p.parse_args()


class ByteStream:
    def __init__(self, path, seq, batch, rank, world):
        self.data = torch.from_file(str(path), shared=False, size=-1, dtype=torch.uint8)
        self.seq = seq
        self.batch = batch
        self.rank = rank
        self.world = world
        self.stride = batch * seq
        usable = (self.data.numel() - 1) // (self.stride * world) * self.stride * world
        if usable <= 0:
            raise ValueError("byte stream is shorter than one distributed batch")
        self.n_batches = usable // (self.stride * world)

    def get(self, step, device):
        base = ((step % self.n_batches) * self.world + self.rank) * self.stride
        x = self.data[base : base + self.stride].view(self.batch, self.seq)
        # The final target byte is loaded by taking a contiguous one-byte
        # overlap from the stream. This keeps every rank's sequence fixed for
        # CUDA graph replay.
        y = self.data[base + 1 : base + self.stride + 1].view(self.batch, self.seq)
        return x.to(device=device, non_blocking=True), y.to(device=device, non_blocking=True)


def load_config(path):
    raw = json.loads(Path(path).read_text())
    return HNetConfig(
        **{k: v for k, v in raw.items() if k not in {"attn_cfg", "ssm_cfg"}},
        attn_cfg=AttnConfig(**raw["attn_cfg"]),
        ssm_cfg=SSMConfig(**raw["ssm_cfg"]),
    )


def init_dist():
    if "RANK" not in os.environ:
        return 0, 1, torch.device("cuda" if torch.cuda.is_available() else "cpu")
    dist.init_process_group("nccl")
    rank = dist.get_rank()
    local = int(os.environ.get("LOCAL_RANK", rank))
    torch.cuda.set_device(local)
    return rank, dist.get_world_size(), torch.device("cuda", local)


def annotate_decay(model):
    for name, p in model.named_parameters():
        if name.endswith(".bias") or "norm" in name.lower() or "msar_bias" in name:
            p._no_weight_decay = True


def make_optimizer(model, args):
    try:
        from scripts.optimizer_experiments import AnchorMuon, AnvilII
    except ModuleNotFoundError:
        from optimizer_experiments import AnchorMuon, AnvilII
    annotate_decay(model)
    if args.optimizer == "adamw":
        decay, no_decay = [], []
        for p in model.parameters():
            (no_decay if getattr(p, "_no_weight_decay", False) else decay).append(p)
        return torch.optim.AdamW(
            [{"params": decay, "weight_decay": args.weight_decay}, {"params": no_decay, "weight_decay": 0.0}],
            lr=args.lr, betas=(0.9, 0.95), fused=True,
        )
    if args.optimizer == "anvil":
        return AnvilII(model.parameters(), lr=args.lr, fallback_lr=args.fallback_lr, weight_decay=args.weight_decay)
    return AnchorMuon.from_model(
        model,
        lr=args.lr,
        fallback_lr=args.fallback_lr,
        fallback_mode="atan2",
        matrix_weight_decay=args.weight_decay,
        fallback_weight_decay=args.weight_decay,
        momentum=0.95,
        pmuoneq_beta=0.90,
        row_gamma=0.45,
        normuon_beta2=0.93,
    )


def ce(logits, target):
    # TE's CE fixes apply here too: cast logits and reduction inputs to FP32,
    # then reject non-finite values before any optimizer state is mutated.
    return F.cross_entropy(logits.float().reshape(-1, logits.shape[-1]), target.reshape(-1))


def loss_for_output(out, target, args):
    losses = [ce(out.logits, target)]
    if out.feedback_logits:
        losses = [ce(out.feedback_logits[0], target)]
        if len(out.feedback_logits) > 1:
            losses.append(args.fbt_lambda * torch.stack([ce(x, target) for x in out.feedback_logits[1:]]).mean())
    ratio = out.logits.new_zeros((), dtype=torch.float32)
    for router in out.bpred_output:
        ratio = ratio + load_balancing_loss(router, args.ratio_target).float()
    return losses[0] + (sum(losses[1:]) if len(losses) > 1 else 0.0) + args.ratio_weight * ratio, losses[0], ratio


class GraphStep:
    def __init__(self, model, optimizer, x, y, args):
        self.model, self.optimizer, self.args = model, optimizer, args
        self.x = torch.zeros_like(x)
        self.y = torch.zeros_like(y)
        self.static_loss = torch.zeros((), device=x.device)
        self.graph = torch.cuda.CUDAGraph()
        # Static all-true mask keeps dynamic chunk lengths bounded by config
        # and makes every allocation replay-safe.
        self.mask = torch.ones_like(self.x, dtype=torch.bool)
        torch.cuda.synchronize()
        for _ in range(3):
            out = model(self.x, mask=self.mask, fbt_passes=args.fbt_passes)
            loss, _, _ = loss_for_output(out, self.y, args)
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            optimizer.step()
        torch.cuda.synchronize()
        with torch.cuda.graph(self.graph):
            out = model(self.x, mask=self.mask, fbt_passes=args.fbt_passes)
            self.static_loss, _, _ = loss_for_output(out, self.y, args)
            optimizer.zero_grad(set_to_none=True)
            self.static_loss.backward()
            optimizer.step()

    def __call__(self, x, y):
        self.x.copy_(x)
        self.y.copy_(y)
        self.graph.replay()
        return self.static_loss


def main():
    args = parse_args()
    rank, world, device = init_dist()
    torch.manual_seed(1729 + rank)
    cfg = load_config(args.config)
    cfg.max_seq_len = args.seq
    model = HNetForByteCausalLM(cfg, device=device, dtype=torch.bfloat16).train()
    n_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    if rank == 0:
        print(f"H-Net params={n_params:,} trainable; byte_lift_params=0; world={world}", flush=True)
    if world > 1:
        model = DDP(model, device_ids=[device.index], broadcast_buffers=False, bucket_cap_mb=256, static_graph=True)
    optimizer = make_optimizer(model.module if world > 1 else model, args)
    stream = ByteStream(args.data, args.seq, args.batch, rank, world)
    graph_step = None
    if args.cuda_graphs and device.type == "cuda":
        x0, y0 = stream.get(0, device)
        graph_step = GraphStep(model, optimizer, x0, y0, args)
    log = {"args": vars(args), "params": n_params, "world": world, "loss": [], "ratio": [], "time_ms": [], "nonfinite": []}
    for step in range(args.steps):
        x, y = stream.get(step, device)
        lr_scale = min(1.0, (step + 1) / max(1, args.warmup))
        for group in optimizer.param_groups:
            group["lr"] = args.lr * lr_scale
            if "fallback_lr" in group:
                group["fallback_lr"] = args.fallback_lr * lr_scale
        t0 = time.perf_counter()
        if graph_step is not None:
            loss = graph_step(x, y)
            ratio = torch.tensor(float("nan"), device=device)
        else:
            mask = torch.ones_like(x, dtype=torch.bool)
            out = model(x, mask=mask, fbt_passes=args.fbt_passes)
            loss, _, ratio = loss_for_output(out, y, args)
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            grad_norm = torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            finite = bool(torch.isfinite(loss) and torch.isfinite(grad_norm))
            if finite:
                optimizer.step()
            else:
                log["nonfinite"].append(step)
        if device.type == "cuda":
            torch.cuda.synchronize()
        elapsed = (time.perf_counter() - t0) * 1e3
        reduced = loss.detach().float()
        if world > 1:
            dist.all_reduce(reduced)
            reduced /= world
        log["loss"].append(float(reduced))
        log["ratio"].append(float(ratio.detach().float()))
        log["time_ms"].append(elapsed)
        if rank == 0 and (step < 3 or step % args.log_every == 0):
            print(f"step={step} loss={float(reduced):.6f} time_ms={elapsed:.1f}", flush=True)
        if rank == 0 and args.ckpt_every and (step + 1) % args.ckpt_every == 0:
            path = Path(args.checkpoint)
            path.parent.mkdir(parents=True, exist_ok=True)
            torch.save({"step": step, "config": vars(cfg), "model": (model.module if world > 1 else model).state_dict(), "optimizer": optimizer.state_dict()}, path)
    if rank == 0:
        Path(args.out).parent.mkdir(parents=True, exist_ok=True)
        Path(args.out).write_text(json.dumps(log, indent=2))
    if world > 1:
        dist.destroy_process_group()


if __name__ == "__main__":
    main()
