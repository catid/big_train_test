"""Direct 600M H-Net raw-byte trainer.

The model consumes UTF-8 bytes through a frozen categorical byte code, then
learns boundaries with H-Net routing from the first update. There is no BPE or
trainable token embedding. The script keeps the CE and router losses in FP32,
uses AnchorMuon/GramNS with its AdamATan2 fallback, FBT multi-pass training,
and reports why CUDA graph capture is unavailable for the canonical
DDP+FBT2+AnchorMuon path.
"""

from __future__ import annotations

import argparse
import dataclasses
import json
import os
import subprocess
import time
from pathlib import Path

# Import hnet first: it loads the pinned Transformer Engine before torch can
# select a conflicting CUDA library set.
from hnet.te import assert_fp8_runtime, fp8_context

import torch
import torch.distributed as dist
import torch.nn.functional as F
from torch.nn.parallel import DistributedDataParallel as DDP

from hnet.models.config_hnet import AttnConfig, HNetConfig, SSMConfig
from hnet.models.mixer_seq import HNetForByteCausalLM
from hnet.utils.train import load_balancing_loss

CANONICAL_CONFIG = Path(__file__).resolve().parent / "configs" / "hnet_1stage_600m_loop2.json"
FBT_PASSES = 2


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--data", required=True, help="raw UTF-8 byte stream")
    p.add_argument("--seq", type=int, default=4096)
    p.add_argument(
        "--batch",
        type=int,
        default=1,
        help="per-rank microbatch; use --auto-batch to size from GPU memory",
    )
    p.add_argument(
        "--auto-batch",
        action=argparse.BooleanOptionalAction,
        default=False,
        help="choose the largest conservative full-model batch from GPU memory",
    )
    p.add_argument("--data-seed", type=int, default=1729)
    p.add_argument("--steps", type=int, default=100000)
    p.add_argument("--warmup", type=int, default=2000)
    p.add_argument("--lr", type=float, default=1.2e-3)
    p.add_argument("--fallback-lr", type=float, default=6e-4)
    p.add_argument("--weight-decay", type=float, default=0.05)
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
    p.add_argument(
        "--cuda-graphs",
        action=argparse.BooleanOptionalAction,
        default=False,
        help="request CUDA graph capture; rejected because canonical DDP+FBT2 capture is unsupported",
    )
    return p.parse_args()


class ByteStream:
    def __init__(self, path, seq, batch, rank, world, seed=1729):
        self.data = torch.from_file(
            str(path), shared=False, size=os.path.getsize(path), dtype=torch.uint8
        )
        self.seq = seq
        self.batch = batch
        self.rank = rank
        self.world = world
        self.seed = int(seed)
        self.stride = batch * seq
        usable = (self.data.numel() - 1) // (self.stride * world) * self.stride * world
        if usable <= 0:
            raise ValueError("byte stream is shorter than one distributed batch")
        self.n_batches = usable // (self.stride * world)
        self._epoch = None
        self._permutation = None

    def _epoch_permutation(self, epoch):
        """Return a reproducible window permutation for one data epoch."""
        if self._epoch != epoch:
            generator = torch.Generator(device="cpu")
            generator.manual_seed(self.seed + epoch)
            self._permutation = torch.randperm(self.n_batches, generator=generator)
            self._epoch = epoch
        return self._permutation

    def get(self, step, device):
        epoch, offset = divmod(int(step), self.n_batches)
        window = int(self._epoch_permutation(epoch)[offset])
        base = (window * self.world + self.rank) * self.stride
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


def load_config():
    raw = json.loads(CANONICAL_CONFIG.read_text())
    cfg = HNetConfig(
        **{k: v for k, v in raw.items() if k not in {"attn_cfg", "ssm_cfg"}},
        attn_cfg=AttnConfig(**raw["attn_cfg"]),
        ssm_cfg=SSMConfig(**raw["ssm_cfg"]),
    )
    cfg.validate_canonical()
    return cfg


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
    saved_data_seed = checkpoint.get("data_seed")
    if saved_data_seed is not None and int(saved_data_seed) != int(args.data_seed):
        raise ValueError(
            f"resume data-seed mismatch: checkpoint={saved_data_seed}, current={args.data_seed}"
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
        if "msar_bias" in name:
            p._force_fallback = True


def make_optimizer(model, args):
    # The canonical trainer uses the repository's reviewed optimizer module.
    # Do not silently switch to an alternate scripts/ checkout: a training run
    # must fail loudly if its pinned optimizer implementation is unavailable.
    from optimizer_experiments import AnchorMuon
    annotate_decay(model)
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
        soda_lambda_scale=0.003,
        min_matrix_dim=32,
    )


def ce(logits, target):
    # TE's CE fixes apply here too: cast logits and reduction inputs to FP32,
    # then reject non-finite values before any optimizer state is mutated.
    return F.cross_entropy(logits.float().reshape(-1, logits.shape[-1]), target.reshape(-1))


def loss_for_output(out, target, args):
    """Compute the FBT objective and retain every component for diagnostics."""
    standard_ce = ce(out.logits, target)
    feedback_values = [ce(x, target) for x in (out.feedback_logits or [])]
    if feedback_values:
        feedback_ce = torch.stack(feedback_values).mean()
        # With two FBT passes, the first-pass logits are retained separately
        # and the final pass is ``standard_ce``. This is algebraically the
        # previous two-pass objective while avoiding a duplicate final head.
        objective = feedback_values[0] + args.fbt_lambda * standard_ce
        if len(feedback_values) > 1:
            objective = objective + args.fbt_lambda * torch.stack(feedback_values[1:]).mean()
    else:
        feedback_ce = standard_ce.new_zeros(())
        objective = standard_ce
    ratio = out.logits.new_zeros((), dtype=torch.float32)
    for router in out.bpred_output:
        ratio = ratio + load_balancing_loss(router, args.ratio_target).float()
    moe_aux_value = getattr(out, "moe_aux_loss", None)
    moe_aux = (moe_aux_value.float() if moe_aux_value is not None
               else out.logits.new_zeros((), dtype=torch.float32))
    objective = objective + args.ratio_weight * ratio + moe_aux
    metrics = {
        "objective": objective,
        "standard_ce": standard_ce,
        "feedback_ce": feedback_ce,
        "router_loss": ratio,
        "moe_aux": moe_aux,
        "ratio_target_error": (ratio.detach() - float(args.ratio_target)).abs(),
    }
    return objective, metrics


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


def main():
    args = parse_args()
    assert_fp8_runtime()
    rank, world, device = init_dist()
    if args.auto_batch:
        if device.type != "cuda":
            raise RuntimeError("--auto-batch requires CUDA")
        # The full model keeps a large recurrent activation graph. Reserve
        # 25% of VRAM for NCCL/allocator work and choose a deterministic local
        # batch; power is measured after the first step and recorded for later
        # closed-loop tuning rather than changing shapes mid-graph.
        total_gib = torch.cuda.get_device_properties(device).total_memory / (1024**3)
        args.batch = max(1, min(8, int(total_gib // 64)))
        if rank == 0:
            print(f"auto_batch selected per-rank batch={args.batch} from {total_gib:.1f} GiB", flush=True)
    torch.manual_seed(1729 + rank)
    cfg = load_config()
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
    stream = ByteStream(args.data, args.seq, args.batch, rank, world, seed=args.data_seed)
    if args.cuda_graphs:
        raise RuntimeError(
            "CUDA graphs are not supported for the canonical DDP+FBT2+AnchorMuon "
            "training path: DDP gradient-bucket logging is illegal during capture"
        )
    log = {
        "args": vars(args),
        "params": n_params,
        "world": world,
        "optimizer_groups": optimizer.group_summary(),
        "loss": [],
        "objective": [],
        "standard_ce": [],
        "feedback_ce": [],
        "router_loss": [],
        "moe_aux": [],
        "ratio_target_error": [],
        "time_ms": [],
        "power_w": [],
        "nonfinite": [],
    }
    if rank == 0:
        print(
            "optimizer_groups="
            + json.dumps(optimizer.group_summary(), separators=(",", ":")),
            flush=True,
        )
    latest_power = []
    for step in range(start_step, args.steps):
        x, y = stream.get(step, device)
        lr_scale = min(1.0, (step + 1) / max(1, args.warmup))
        for group in optimizer.param_groups:
            # AnchorMuon splits matrix and AdamATan2 fallback tensors into
            # separate groups; preserve the requested fallback rate.
            if "use_matrix_update" in group:
                base_lr = args.lr if group["use_matrix_update"] else args.fallback_lr
                group["lr"] = base_lr * lr_scale
            else:
                group["lr"] = args.lr * lr_scale
            if "fallback_lr" in group:
                group["fallback_lr"] = args.fallback_lr * lr_scale
        t0 = time.perf_counter()
        mask = torch.ones_like(x, dtype=torch.bool)
        optimizer.zero_grad(set_to_none=True)
        with fp8_context():
            out = model(x, mask=mask, fbt_passes=FBT_PASSES)
            loss, metrics = loss_for_output(out, y, args)
        # Transformer Engine updates delayed-scaling state during the forward
        # autocast scope; its backward kernels consume that state and should be
        # launched after leaving the context.
        loss.backward()
        grad_norm = torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        finite_local = torch.isfinite(loss) & torch.isfinite(grad_norm)
        # A single rank with an invalid batch must not advance its optimizer
        # while the other DDP ranks do.  Reduce the finite flag before any
        # optimizer mutation so every rank takes the same branch.
        finite_flag = finite_local.to(dtype=torch.int32)
        if world > 1:
            dist.all_reduce(finite_flag, op=dist.ReduceOp.MIN)
        finite = bool(finite_flag.item())
        if finite:
            optimizer.step()
        else:
            log["nonfinite"].append(step)
        if device.type == "cuda":
            torch.cuda.synchronize()
        elapsed = (time.perf_counter() - t0) * 1e3
        reduced = loss.detach().float()
        reduced_metrics = {name: value.detach().float() for name, value in metrics.items()}
        if world > 1:
            dist.all_reduce(reduced)
            reduced /= world
            for value in reduced_metrics.values():
                dist.all_reduce(value)
                value /= world
        log["loss"].append(float(reduced))
        for name, value in reduced_metrics.items():
            log[name].append(float(value))
        log["time_ms"].append(elapsed)
        if rank == 0 and args.power_every > 0 and (
            step == start_step or step % args.power_every == 0
        ):
            latest_power = read_power_watts()
        if rank == 0:
            log["power_w"].append(list(latest_power))
        if rank == 0 and (step < 3 or step % args.log_every == 0):
            print(
                f"step={step} objective={float(reduced):.6f} "
                f"ce={float(reduced_metrics['standard_ce']):.6f} "
                f"fb_ce={float(reduced_metrics['feedback_ce']):.6f} "
                f"router={float(reduced_metrics['router_loss']):.6f} "
                f"moe={float(reduced_metrics['moe_aux']):.6f} "
                f"time_ms={elapsed:.1f} "
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
            checkpoint = {
                    "step": step,
                    "seq": args.seq,
                    "batch": args.batch,
                    "data_seed": args.data_seed,
                    "params": n_params,
                    "config_signature": _config_dict(cfg),
                    "model": (model.module if world > 1 else model).state_dict(),
                    "optimizer": optimizer.state_dict(),
                    "rng_state_by_rank": rng_by_rank,
                    "rng_state": rng_by_rank[0][0],
                    "cuda_rng_state": rng_by_rank[0][1],
                }
            temp_path = path.with_name(f".{path.name}.tmp.{os.getpid()}")
            torch.save(
                checkpoint, temp_path,
            )
            os.replace(temp_path, path)
    if rank == 0:
        Path(args.out).parent.mkdir(parents=True, exist_ok=True)
        Path(args.out).write_text(json.dumps(log, indent=2))
    if world > 1:
        dist.destroy_process_group()


if __name__ == "__main__":
    main()
