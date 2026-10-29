"""Direct 600M H-Net raw-byte trainer.

The model consumes UTF-8 bytes through a frozen categorical byte code, then
learns boundaries with H-Net routing from the first update. There is no BPE or
trainable token embedding. The script keeps the CE and router losses in FP32,
supports Muon/GramNS, ANVIL-II, AdamATan2 fallback, FBT multi-pass training,
and captures a fixed-shape CUDA graph when requested.
"""

from __future__ import annotations

import argparse
import dataclasses
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
    p.add_argument(
        "--power-every",
        type=int,
        default=100,
        help="sample nvidia-smi power at this cadence; reuse the latest reading in logs",
    )
    p.add_argument("--ckpt-every", type=int, default=1000)
    p.add_argument("--out", default="/home/catid/hnet/artifacts/hnet_600m_train.json")
    p.add_argument("--checkpoint", default="/home/catid/hnet/artifacts/hnet_600m_last.pt")
    p.add_argument(
        "--resume",
        default=None,
        help="checkpoint to resume (model, optimizer, and step); defaults to a fresh run",
    )
    p.add_argument("--cuda-graphs", action=argparse.BooleanOptionalAction, default=True)
    p.add_argument("--autotune-batch", action="store_true")
    p.add_argument("--target-power", type=float, default=1000.0)
    return p.parse_args()


class ByteStream:
    def __init__(self, path, seq, batch, rank, world):
        self.data = torch.from_file(
            str(path), shared=False, size=os.path.getsize(path), dtype=torch.uint8
        )
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
        # Input and target are adjacent in the byte stream. Copy the shared
        # ``stride + 1`` span once, then form overlapping views on-device;
        # this removes one host-to-device launch and avoids transferring the
        # duplicated overlap byte twice while preserving exact sequence
        # boundaries for every rank and CUDA-graph replay.
        batch = self.data[base : base + self.stride + 1].to(
            device=device, non_blocking=True
        )
        x = batch[:-1].view(self.batch, self.seq)
        y = batch[1:].view(self.batch, self.seq)
        return x, y


def load_config(path):
    raw = json.loads(Path(path).read_text())
    return HNetConfig(
        **{k: v for k, v in raw.items() if k not in {"attn_cfg", "ssm_cfg"}},
        attn_cfg=AttnConfig(**raw["attn_cfg"]),
        ssm_cfg=SSMConfig(**raw["ssm_cfg"]),
    )


def _config_dict(cfg):
    """Return a stable, serializable architecture signature for checkpoints."""
    if dataclasses.is_dataclass(cfg):
        cfg = dataclasses.asdict(cfg)
    if isinstance(cfg, dict):
        return {k: _config_dict(v) if isinstance(v, (dict,)) or dataclasses.is_dataclass(v) else v
                for k, v in cfg.items()}
    return cfg


def _validate_resume(checkpoint, cfg, args, n_params):
    """Reject resumes that silently mix incompatible model/data shapes."""
    saved_params = checkpoint.get("params")
    if saved_params is not None and int(saved_params) != int(n_params):
        raise ValueError(
            f"resume parameter mismatch: checkpoint={saved_params}, current={n_params}"
        )
    saved_seq = checkpoint.get("seq")
    if saved_seq is not None and int(saved_seq) != int(args.seq):
        raise ValueError(
            f"resume sequence mismatch: checkpoint={saved_seq}, current={args.seq}"
        )
    saved_batch = checkpoint.get("batch")
    if saved_batch is not None and int(saved_batch) != int(args.batch):
        raise ValueError(
            f"resume batch mismatch: checkpoint={saved_batch}, current={args.batch}"
        )
    saved_cfg = checkpoint.get("config_signature")
    if saved_cfg is None:
        # Older checkpoints stored ``vars(cfg)``.  It is still useful to
        # validate the fields when they contain plain dictionaries.
        saved_cfg = checkpoint.get("config")
    if saved_cfg is not None:
        expected = _config_dict(cfg)
        saved_cfg = _config_dict(saved_cfg)
        # Ignore max_seq_len from old checkpoints whose config was saved
        # before the command-line sequence length was applied.
        if isinstance(saved_cfg, dict):
            saved_cfg = dict(saved_cfg)
            saved_cfg.pop("max_seq_len", None)
            expected = dict(expected)
            expected.pop("max_seq_len", None)
            if saved_cfg != expected:
                raise ValueError("resume architecture/configuration does not match current config")


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
    # The canonical trainer uses the repository's reviewed optimizer module.
    # Do not silently switch to an alternate scripts/ checkout: a training run
    # must fail loudly if its pinned optimizer implementation is unavailable.
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
    moe_aux_value = getattr(out, "moe_aux_loss", None)
    moe_aux = (moe_aux_value.float() if moe_aux_value is not None
               else out.logits.new_zeros((), dtype=torch.float32))
    return (losses[0] + (sum(losses[1:]) if len(losses) > 1 else 0.0)
            + args.ratio_weight * ratio + moe_aux), losses[0], ratio


def read_power_watts():
    try:
        raw = subprocess.check_output(
            ["nvidia-smi", "--query-gpu=power.draw", "--format=csv,noheader,nounits"],
            text=True,
            timeout=2,
        )
        values = [float(x.strip()) for x in raw.splitlines() if x.strip()]
        return values
    except (OSError, subprocess.SubprocessError, ValueError):
        return []


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
            optimizer.zero_grad(set_to_none=False)
            loss.backward()
        optimizer.zero_grad(set_to_none=False)
        with torch.cuda.graph(self.graph):
            out = model(self.x, mask=self.mask, fbt_passes=args.fbt_passes)
            self.static_loss, _, self.static_ratio = loss_for_output(out, self.y, args)
            self.static_loss.backward()

    def __call__(self, x, y):
        self.x.copy_(x)
        self.y.copy_(y)
        # Keep Python-side optimizer state and learning-rate schedules outside
        # the graph; the graph contains only fixed-shape forward/backward.
        self.optimizer.zero_grad(set_to_none=False)
        self.graph.replay()
        if not bool(torch.isfinite(self.static_loss)):
            return self.static_loss, self.static_ratio
        torch.nn.utils.clip_grad_norm_(self.model.parameters(), 1.0)
        self.optimizer.step()
        return self.static_loss, self.static_ratio


def _cuda_graphs_supported(cfg, args):
    """Return whether this model has replay-stable CUDA graph shapes.

    H-Net routing compacts each sequence according to a learned boundary mask.
    ``static_chunking`` pads the compacted representation to the configured
    bound, so the graph sees fixed tensor shapes while the mask still carries
    the learned boundaries.  The optimizer update remains outside the graph,
    which keeps Muon/ANVIL counters and learning-rate schedules correct.
    """
    if not bool(getattr(cfg, "static_chunking", False)):
        return False, "static_chunking is required for replay-stable routed shapes"
    if args.fbt_passes != 1:
        return False, "FBT multi-pass execution is not CUDA-graph stable"
    if args.warmup:
        return False, "learning-rate warmup is outside the captured graph"
    if args.optimizer != "adamw":
        return False, "Muon/ANVIL Python state counters stay outside the graph"
    # The selected S/SWA/T/FA hierarchy uses only FlashAttention mixers, so
    # graph capture is stream-safe once the fixed-shape gates above pass.
    layout_text = str(getattr(cfg, "arch_layout", ""))
    if "m" in layout_text.lower():
        return False, "Mamba2 encoder/decoder kernels are not CUDA-graph stream safe"
    return True, ""


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
        model = DDP(
            model,
            device_ids=[device.index],
            broadcast_buffers=False,
            bucket_cap_mb=256,
            static_graph=True,
            gradient_as_bucket_view=True,
        )
    optimizer = make_optimizer(model.module if world > 1 else model, args)
    start_step = 0
    resume_state = None
    if args.resume:
        resume_path = Path(args.resume)
        if not resume_path.is_file():
            raise FileNotFoundError(f"resume checkpoint not found: {resume_path}")
        # Checkpoints contain only tensors, optimizer state, and plain
        # configuration metadata.  Explicitly disable ``weights_only`` so
        # older checkpoints with nested dataclass metadata remain loadable.
        resume_state = torch.load(resume_path, map_location="cpu", weights_only=False)
        _validate_resume(resume_state, cfg, args, n_params)
        target_model = model.module if world > 1 else model
        missing, unexpected = target_model.load_state_dict(
            resume_state["model"], strict=True
        )
        if missing or unexpected:
            raise ValueError(f"resume state mismatch: missing={missing}, unexpected={unexpected}")
        optimizer.load_state_dict(resume_state["optimizer"])
        start_step = int(resume_state.get("step", -1)) + 1
        if start_step < 0 or start_step > args.steps:
            raise ValueError(f"resume step {start_step} is outside requested [0, {args.steps}]")
        saved_rng = resume_state.get("rng_state_by_rank")
        if saved_rng is not None and rank < len(saved_rng):
            cpu_rng, cuda_rng = saved_rng[rank]
            if cpu_rng is not None:
                torch.set_rng_state(cpu_rng)
            if device.type == "cuda" and cuda_rng is not None:
                torch.cuda.set_rng_state(cuda_rng, device=device)
        elif rank == 0 and "rng_state" in resume_state:
            # Compatibility with checkpoints written before per-rank RNGs.
            torch.set_rng_state(resume_state["rng_state"])
            if device.type == "cuda" and resume_state.get("cuda_rng_state") is not None:
                torch.cuda.set_rng_state(resume_state["cuda_rng_state"], device=device)
        if rank == 0:
            print(f"Resumed checkpoint {resume_path} at step {start_step}", flush=True)
    stream = ByteStream(args.data, args.seq, args.batch, rank, world)
    graph_step = None
    if args.cuda_graphs and device.type == "cuda":
        graph_ok, reason = _cuda_graphs_supported(cfg, args)
        if graph_ok:
            x0, y0 = stream.get(0, device)
            graph_step = GraphStep(model, optimizer, x0, y0, args)
        elif rank == 0:
            print(f"CUDA graphs disabled: {reason}", flush=True)
    log = {
        "args": vars(args),
        "params": n_params,
        "world": world,
        "loss": [],
        "ratio": [],
        "time_ms": [],
        "power_w": [],
        "nonfinite": [],
    }
    latest_power = []
    for step in range(start_step, args.steps):
        x, y = stream.get(step, device)
        lr_scale = min(1.0, (step + 1) / max(1, args.warmup))
        for group in optimizer.param_groups:
            # AnchorMuon splits matrix and fallback tensors into separate
            # groups, while Anvil keeps both rates in one group.  Preserve the
            # requested fallback rate in either representation.
            if "use_matrix_update" in group:
                base_lr = args.lr if group["use_matrix_update"] else args.fallback_lr
                group["lr"] = base_lr * lr_scale
            else:
                group["lr"] = args.lr * lr_scale
            if "fallback_lr" in group:
                group["fallback_lr"] = args.fallback_lr * lr_scale
        t0 = time.perf_counter()
        if graph_step is not None:
            loss, ratio = graph_step(x, y)
            if not bool(torch.isfinite(loss)):
                log["nonfinite"].append(step)
                raise RuntimeError("non-finite loss encountered during CUDA-graph replay")
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
        reduced_ratio = ratio.detach().float()
        if world > 1:
            dist.all_reduce(reduced)
            reduced /= world
            dist.all_reduce(reduced_ratio)
            reduced_ratio /= world
        log["loss"].append(float(reduced))
        log["ratio"].append(float(reduced_ratio))
        log["time_ms"].append(elapsed)
        if rank == 0 and args.power_every > 0 and (
            step == start_step or step % args.power_every == 0
        ):
            latest_power = read_power_watts()
        if rank == 0:
            log["power_w"].append(list(latest_power))
        if rank == 0 and (step < 3 or step % args.log_every == 0):
            print(
                f"step={step} loss={float(reduced):.6f} time_ms={elapsed:.1f} "
                f"power_w={','.join(f'{x:.0f}' for x in latest_power) or 'n/a'}",
                flush=True,
            )
        if args.ckpt_every and (step + 1) % args.ckpt_every == 0:
            local_rng = (
                torch.get_rng_state(),
                torch.cuda.get_rng_state(device=device) if device.type == "cuda" else None,
            )
            rng_by_rank = [None] * world if rank == 0 else None
            if world > 1:
                dist.gather_object(local_rng, rng_by_rank, dst=0)
            else:
                rng_by_rank = [local_rng]
        if rank == 0 and args.ckpt_every and (step + 1) % args.ckpt_every == 0:
            path = Path(args.checkpoint)
            path.parent.mkdir(parents=True, exist_ok=True)
            torch.save(
                {
                    "step": step,
                    "seq": args.seq,
                    "batch": args.batch,
                    "params": n_params,
                    "config_signature": _config_dict(cfg),
                    "model": (model.module if world > 1 else model).state_dict(),
                    "optimizer": optimizer.state_dict(),
                    "rng_state_by_rank": rng_by_rank,
                    "rng_state": rng_by_rank[0][0],
                    "cuda_rng_state": rng_by_rank[0][1],
                },
                path,
            )
    if rank == 0:
        Path(args.out).parent.mkdir(parents=True, exist_ok=True)
        Path(args.out).write_text(json.dumps(log, indent=2))
    if world > 1:
        dist.destroy_process_group()


if __name__ == "__main__":
    main()
