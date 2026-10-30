"""Direct 600M H-Net raw-byte trainer.

The model consumes UTF-8 bytes through a frozen categorical byte code, then
learns boundaries with H-Net routing from the first update. There is no BPE or
trainable token embedding. The script keeps the CE and router losses in FP32,
uses AnchorMuon/GramNS with its AdamATan2 fallback, FBT multi-pass training,
and supports CUDA-graph capture for the fixed-shape padded training contract.
"""

from __future__ import annotations

import argparse
import contextlib
import dataclasses
import hashlib
import json
import math
import os
import subprocess
import time
from pathlib import Path

# Import hnet first: it loads the pinned Transformer Engine before torch can
# select a conflicting CUDA library set.
import hnet.te as hnet_te
from hnet.te import assert_fp8_runtime

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
        default=True,
        help="calibrate a DDP-safe batch against the measured per-GPU power target",
    )
    p.add_argument(
        "--auto-batch-max",
        type=int,
        default=32,
        help="upper bound for power-calibration candidates (powers of two)",
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
    p.add_argument(
        "--target-ce",
        type=float,
        default=4.0,
        help="plain byte cross-entropy target recorded in the run log",
    )
    p.add_argument("--log-every", type=int, default=10)
    p.add_argument(
        "--power-every",
        type=int,
        default=100,
        help="sample nvidia-smi power at this cadence; reuse the latest reading in logs",
    )
    p.add_argument(
        "--target-power-w",
        type=float,
        default=1000.0,
        help="per-GPU power target recorded for batch tuning diagnostics",
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
        default=True,
        help="capture the fixed-shape canonical forward/backward with CUDA graphs",
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


def _runtime_provenance():
    """Capture the runtime identity that produced a checkpoint.

    FP8 state is tied to the CUDA/cuDNN/Transformer Engine build.  Recording
    these values makes a resume auditable and lets us reject a checkpoint
    produced by a different kernel/runtime combination instead of silently
    changing the optimization path.
    """
    repo = Path(__file__).resolve().parent
    try:
        git_sha = subprocess.check_output(
            ["git", "rev-parse", "HEAD"], cwd=repo, text=True, stderr=subprocess.DEVNULL
        ).strip()
    except (OSError, subprocess.SubprocessError):
        git_sha = None
    try:
        git_diff = subprocess.check_output(
            ["git", "diff", "HEAD", "--binary"],
            cwd=repo,
            stderr=subprocess.DEVNULL,
        )
        git_diff_sha256 = hashlib.sha256(git_diff).hexdigest()
        git_dirty = bool(git_diff)
    except (OSError, subprocess.SubprocessError):
        git_diff_sha256 = None
        git_dirty = None
    try:
        te_path = str(Path(hnet_te.te.__file__).resolve())
    except (AttributeError, OSError):
        te_path = None
    recipe = getattr(hnet_te, "_FP8_RECIPE", None)
    return {
        "git_sha": git_sha,
        "git_dirty": git_dirty,
        "git_diff_sha256": git_diff_sha256,
        "transformer_engine": getattr(hnet_te.te, "__version__", None),
        "transformer_engine_path": te_path,
        "transformer_engine_git_commit": hnet_te.transformer_engine_git_commit(),
        "cublaslt": hnet_te.cublaslt_provenance(),
        "torch": torch.__version__,
        "cuda_runtime": torch.version.cuda,
        "cudnn": torch.backends.cudnn.version(),
        "fp8_recipe": {
            "type": type(recipe).__name__ if recipe is not None else None,
            "fp8_format": str(getattr(recipe, "fp8_format", None)),
            "amax_history_len": getattr(recipe, "amax_history_len", None),
            "amax_compute_algo": getattr(recipe, "amax_compute_algo", None),
            "fp8_dpa": bool(getattr(recipe, "fp8_dpa", False)),
        },
        "nvte_fp8_dpa_bwd": os.environ.get("NVTE_FP8_DPA_BWD"),
        "nvte_croissattn_aux": os.environ.get("NVTE_DPA_CROISSATTN_AUX"),
    }


def _validate_resume(checkpoint, cfg, args, n_params, world, runtime):
    """Reject resumes that silently mix incompatible model/data shapes."""
    saved_world = checkpoint.get("world_size")
    if saved_world is None:
        raise ValueError("resume checkpoint is missing world_size provenance")
    if int(saved_world) != int(world):
        raise ValueError(
            f"resume world-size mismatch: checkpoint={saved_world}, current={world}"
        )
    saved_runtime = checkpoint.get("runtime")
    if not isinstance(saved_runtime, dict):
        raise ValueError("resume checkpoint is missing runtime provenance")
    for key in (
        "git_sha",
        "git_dirty",
        "git_diff_sha256",
        "transformer_engine",
        "transformer_engine_path",
        "transformer_engine_git_commit",
        "cublaslt",
        "torch",
        "cuda_runtime",
        "cudnn",
        "fp8_recipe",
        "nvte_fp8_dpa_bwd",
        "nvte_croissattn_aux",
    ):
        if saved_runtime.get(key) != runtime.get(key):
            raise ValueError(
                f"resume runtime mismatch for {key}: "
                f"checkpoint={saved_runtime.get(key)!r}, current={runtime.get(key)!r}"
            )
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
    return F.cross_entropy(
        logits.float().reshape(-1, logits.shape[-1]), target.long().reshape(-1)
    )


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
    boundary_ratios = []
    for router in out.bpred_output:
        ratio = ratio + load_balancing_loss(router, args.ratio_target).float()
        boundary_ratios.append(router.boundary_mask.float().mean())
    moe_aux_value = getattr(out, "moe_aux_loss", None)
    moe_aux = (moe_aux_value.float() if moe_aux_value is not None
               else out.logits.new_zeros((), dtype=torch.float32))
    objective = objective + args.ratio_weight * ratio + moe_aux
    routing_ratio = (
        torch.stack(boundary_ratios).mean()
        if boundary_ratios
        else standard_ce.new_zeros((), dtype=torch.float32)
    )
    target_ratio = getattr(args, "_target_ratio_tensor", None)
    if target_ratio is None or target_ratio.device != standard_ce.device:
        target_ratio = standard_ce.new_tensor(1.0 / float(args.ratio_target))
    metrics = {
        "objective": objective,
        "standard_ce": standard_ce,
        "feedback_ce": feedback_ce,
        "router_loss": ratio,
        "moe_aux": moe_aux,
        "routing_ratio": routing_ratio,
        "ratio_target_error": (routing_ratio - target_ratio).abs(),
    }
    return objective, metrics


def read_power_watts(device_index=None):
    try:
        query = ["nvidia-smi"]
        if device_index is not None:
            query.extend(["-i", str(int(device_index))])
        query.extend(["--query-gpu=power.draw", "--format=csv,noheader,nounits"])
        raw = subprocess.check_output(
            query,
            text=True,
            timeout=2,
        )
        values = [float(x.strip()) for x in raw.splitlines() if x.strip()]
        return values
    except (OSError, subprocess.SubprocessError, ValueError):
        return []


def calibrate_batch_to_power(model, optimizer, args, device, rank, world):
    """Choose a per-rank batch from complete DDP training steps.

    Power is affected materially by gradient synchronization and the optimizer,
    especially for AnchorMuon's GramNS update.  Measuring only a forward and
    backward before DDP therefore underestimates the draw used by the real
    trainer.  Each candidate below executes the same complete update as the
    training loop, then restores a CPU snapshot of the initial model and clears
    optimizer state before the next candidate.  The selected batch is thus
    apples-to-apples with the first real step and never depends on a warmed
    candidate's parameters or moments.
    """
    if device.type != "cuda":
        raise RuntimeError("--auto-batch requires CUDA")
    total_gib = torch.cuda.get_device_properties(device).total_memory / (1024**3)
    # Keep a generous memory reserve for the real optimizer, DDP buckets, and
    # checkpointing. B300's 275 GiB permits candidates through eight for this
    # model; smaller devices receive a proportionally lower bound.
    if int(args.auto_batch_max) < 1:
        raise ValueError("--auto-batch-max must be positive")
    # Leave room for the optimizer master/state tensors, DDP buckets, and
    # checkpoint allocations that are absent during this pre-optimizer sweep.
    max_batch = max(1, min(int(args.auto_batch_max), int(total_gib // 16)))
    available_batches = (os.path.getsize(args.data) - 1) // (args.seq * max(1, world))
    if available_batches < 1:
        raise ValueError("byte stream is shorter than one distributed batch")
    # Do not select a candidate the configured stream cannot supply. This is
    # especially useful for deterministic calibration fixtures and keeps the
    # selected DDP shape valid before ByteStream is constructed.
    max_batch = min(max_batch, int(available_batches))
    candidates = []
    candidate = 1
    while candidate <= max_batch:
        candidates.append(candidate)
        candidate *= 2

    target_model = model.module if isinstance(model, DDP) else model
    # The snapshot is intentionally CPU-resident so candidate tensors and
    # optimizer state can be released before the next measurement.  It also
    # keeps every rank's reset exact after DDP has broadcast the initial model.
    initial_state = {
        name: value.detach().cpu().clone()
        for name, value in target_model.state_dict().items()
    }
    cpu_rng = torch.get_rng_state()
    cuda_rng = torch.cuda.get_rng_state(device=device)
    measurements = []
    try:
        for candidate in candidates:
            if world > 1:
                dist.barrier()
            powers = []
            oom = False
            x = y = mask = out = loss = metrics = None
            try:
                # Deterministic inputs keep candidate comparisons reproducible;
                # model randomness still exercises the real FBT path.
                generator = torch.Generator(device=device)
                generator.manual_seed(1729 + candidate)
                x = torch.randint(
                    0, 256, (candidate, args.seq), device=device,
                    dtype=torch.uint8, generator=generator,
                )
                y = torch.randint(
                    0, 256, (candidate, args.seq), device=device,
                    dtype=torch.uint8, generator=generator,
                )
                mask = torch.ones_like(x, dtype=torch.bool)
                for iteration in range(4):
                    # The first two updates warm kernels and allocator state;
                    # the latter two are the steady-state power samples.
                    model.zero_grad(set_to_none=True)
                    out = model(x, mask=mask, fbt_passes=FBT_PASSES)
                    loss, metrics = loss_for_output(out, y, args)
                    loss.backward()
                    grad_norm = torch.nn.utils.clip_grad_norm_(
                        model.parameters(), 1.0
                    )
                    finite_flag = (torch.isfinite(loss) & torch.isfinite(grad_norm)).to(
                        dtype=torch.int32
                    )
                    if world > 1:
                        dist.all_reduce(finite_flag, op=dist.ReduceOp.MIN)
                    if not bool(finite_flag.item()):
                        raise FloatingPointError(
                            f"non-finite calibration update at candidate batch {candidate}"
                        )
                    optimizer.step()
                    torch.cuda.synchronize(device)
                    if iteration >= 2:
                        power_device = (
                            device.index
                            if device.index is not None
                            else torch.cuda.current_device()
                        )
                        measured = read_power_watts(power_device)
                        if measured:
                            powers.append(float(measured[0]))
            except RuntimeError as exc:
                if "out of memory" in str(exc).lower():
                    oom = True
                else:
                    raise
            finally:
                model.zero_grad(set_to_none=True)
                # Restore both parameters and optimizer moments before the next
                # candidate.  This is required because GramNS/AdamATan2 state
                # and master copies otherwise make later measurements depend on
                # the preceding batch size.
                target_model.load_state_dict(initial_state, strict=True)
                optimizer.state.clear()
                for group in optimizer.param_groups:
                    group["k"] = 0
                del x, y, mask, out, loss, metrics
                if oom:
                    torch.cuda.empty_cache()

            oom_flag = torch.tensor([int(oom)], device=device, dtype=torch.int32)
            if world > 1:
                dist.all_reduce(oom_flag, op=dist.ReduceOp.MAX)
            if bool(oom_flag.item()):
                if rank == 0:
                    print(
                        f"auto_batch candidate={candidate} stopped at OOM; "
                        f"selected from prior candidates",
                        flush=True,
                    )
                break

            local_power = sum(powers) / len(powers) if powers else float("nan")
            sample_flag = torch.tensor(
                [int(bool(powers))], device=device, dtype=torch.int32
            )
            if world > 1:
                # Every rank must have an actual local reading before the
                # aggregate is considered a power calibration point.
                dist.all_reduce(sample_flag, op=dist.ReduceOp.MIN)
            if not bool(sample_flag.item()):
                raise RuntimeError(
                    "--auto-batch requires nvidia-smi power samples on every rank"
                )
            power_tensor = torch.tensor(
                [local_power if math.isfinite(local_power) else 0.0],
                device=device, dtype=torch.float32,
            )
            if world > 1:
                dist.all_reduce(power_tensor, op=dist.ReduceOp.SUM)
                power_tensor /= world
            mean_power = float(power_tensor.item())
            measurements.append({"batch": candidate, "power_w": mean_power})
            if rank == 0:
                print(
                    f"auto_batch candidate={candidate} mean_power_w={mean_power:.1f}",
                    flush=True,
                )
    finally:
        target_model.load_state_dict(initial_state, strict=True)
        optimizer.state.clear()
        for group in optimizer.param_groups:
            group["k"] = 0
        torch.set_rng_state(cpu_rng)
        torch.cuda.set_rng_state(cuda_rng, device=device)

    if not measurements:
        raise RuntimeError("--auto-batch could not complete a full FP8 calibration step")
    target = float(args.target_power_w)
    max_power = max(item["power_w"] for item in measurements)
    min_power = min(item["power_w"] for item in measurements)
    if max_power < target:
        # If the whole sweep is below the requested draw, the closest point is
        # the wrong policy: a small noisy difference can select batch 1 and
        # leave available GPU capacity unused.  Use the largest fully
        # validated candidate and record the remaining gap explicitly.
        selected = max(measurements, key=lambda item: item["batch"])
        selection_policy = "largest_below_target"
    elif min_power > target:
        # If even batch 1 exceeds the target, preserve the smallest validated
        # shape so the trainer does not knowingly overdraw the GPU.
        selected = min(measurements, key=lambda item: item["batch"])
        selection_policy = "smallest_above_target"
    else:
        selected = min(
            measurements,
            key=lambda item: (abs(item["power_w"] - target), -item["batch"]),
        )
        selection_policy = "closest_target"
    selected_tensor = torch.tensor([selected["batch"]], device=device, dtype=torch.int64)
    if world > 1:
        dist.broadcast(selected_tensor, src=0)
    selected_batch = int(selected_tensor.item())
    if rank == 0:
        print(
            f"auto_batch selected per-rank batch={selected_batch} "
            f"target_power_w={target:.1f} policy={selection_policy} "
            f"measurements={json.dumps(measurements)}",
            flush=True,
        )
    args._auto_batch_calibration = measurements
    args._auto_batch_selection_policy = selection_policy
    return selected_batch


class CUDAGraphStep:
    """Captured canonical forward/backward with optimizer updates outside.

    The H-Net has data-dependent routing, but its static chunking contract makes
    every intermediate tensor bounded by the configured ``(batch, seq)`` shape.
    We therefore capture only tensor work: inputs are copied into persistent
    device buffers before replay, while clipping, the finite flag reduction, and
    AnchorMuon/AdamATan2 updates remain outside the graph.  Keeping the update
    outside avoids capturing optimizer state initialization and lets DDP perform
    its normal synchronization between replays. For DDP, capture runs with
    reducer synchronization disabled and a flat gradient all-reduce is issued
    after replay, because DDP's runtime logger is not graph-safe.
    """

    def __init__(self, model, optimizer, args, batch, seq, device, world, capture_stream=None):
        if device.type != "cuda":
            raise RuntimeError("CUDA graphs require a CUDA device")
        self.model = model
        self.optimizer = optimizer
        self.args = args
        self.device = device
        self.world = int(world)
        self.capture_stream = capture_stream
        self.ddp = world > 1 and isinstance(model, DDP)
        self._ddp_params = list(model.parameters()) if self.ddp else []
        self._ddp_grad_flat = None
        # Materialize scalar diagnostics before capture.  Constructing a new
        # tensor from a Python float inside a CUDA capture is rejected by the
        # runtime even though the arithmetic itself is graph safe.
        args._target_ratio_tensor = torch.tensor(
            1.0 / float(args.ratio_target), dtype=torch.float32, device=device
        )
        self.static_x = torch.empty((batch, seq), dtype=torch.uint8, device=device)
        self.static_y = torch.empty((batch, seq), dtype=torch.uint8, device=device)
        self.static_mask = torch.ones((batch, seq), dtype=torch.bool, device=device)
        self.graph = torch.cuda.CUDAGraph()
        self.loss = None
        self.metrics = None

        # Warm up all lazy TE kernels, routing caches, autograd buffers, and
        # DDP buckets before capture.  Do not mutate parameters in warmup: the
        # first real optimizer update remains step zero.
        # Eager FP8 exits update delayed-scaling history; capture mode must be
        # enabled only after this calibration because TE intentionally skips
        # those reductions while ``_graph=True``.
        hnet_te.set_graph_capture(False)
        # Warm up with a deterministic nondegenerate byte window.  An all-zero
        # prompt can make the learned router emit a single latent boundary;
        # the pinned TE CroissAttn kernel intentionally rejects the resulting
        # length-one BSHD call.  The graph itself remains shape-static and is
        # replayed with arbitrary data after this representative warmup.
        warm_x = (torch.arange(self.static_x.numel(), device=self.device,
                               dtype=torch.int32).reshape_as(self.static_x) % 251).to(torch.uint8)
        warm_y = torch.roll(warm_x, shifts=-1, dims=1)
        warm_stream_context = (
            torch.cuda.stream(self.capture_stream)
            if self.capture_stream is not None
            else contextlib.nullcontext()
        )
        with warm_stream_context:
            for _ in range(3):
                self.optimizer.zero_grad(set_to_none=True)
                warm_out = self.model(warm_x, mask=self.static_mask, fbt_passes=FBT_PASSES)
                warm_loss, _ = loss_for_output(warm_out, warm_y, args)
                warm_loss.backward()
        # Do not retain an eager autograd graph while capturing the next graph;
        # AccumulateGrad nodes remember the warmup stream and CUDA rejects a
        # capture that makes the legacy stream depend on those stale nodes.
        del warm_out, warm_loss, _
        # Gradients must have stable addresses for replay.  The captured zero
        # operation then clears those buffers before each backward pass.
        self.optimizer.zero_grad(set_to_none=False)
        self.static_x.copy_(warm_x)
        self.static_y.copy_(warm_y)
        torch.cuda.synchronize(device)
        if self.ddp:
            # DDP's C++ logger performs a host-side runtime-stat call in
            # ``DistributedDataParallel._pre_forward``. Disable reducer sync
            # for capture and perform a flat gradient all-reduce after replay;
            # this keeps the captured graph free of logger/NCCL setup calls.
            self.model.require_backward_grad_sync = False
        hnet_te.set_graph_capture(True)
        try:
            # Capture on the same non-default stream used for DDP construction
            # and warmup. Single-GPU capture keeps PyTorch's default stream
            # behavior; distributed capture must avoid AccumulateGrad stream
            # dependencies created by DDP initialization.
            capture_context = (
                torch.cuda.stream(self.capture_stream)
                if self.capture_stream is not None
                else contextlib.nullcontext()
            )
            with capture_context:
                if self.capture_stream is None:
                    graph_context = torch.cuda.graph(self.graph)
                else:
                    graph_context = torch.cuda.graph(
                        self.graph, stream=self.capture_stream
                    )
                with graph_context:
                    self.optimizer.zero_grad(set_to_none=False)
                    graph_out = self.model(
                        self.static_x, mask=self.static_mask, fbt_passes=FBT_PASSES
                    )
                    self.loss, self.metrics = loss_for_output(graph_out, self.static_y, args)
                    self.loss.backward()
        except Exception as exc:
            raise RuntimeError(
                "canonical CUDA graph capture failed; inspect the TE/CUDA graph "
                f"trace rather than silently running eager mode: {exc}"
            ) from exc
        torch.cuda.synchronize(device)
        # Seed the first replay with the calibrated amax history generated by
        # capture. Subsequent replays commit both histories below.
        hnet_te.update_fp8_scaling(forward=True)
        hnet_te.update_fp8_scaling(forward=False)

    def replay(self, x, y):
        if self.capture_stream is None:
            self.static_x.copy_(x)
            self.static_y.copy_(y)
            self.graph.replay()
        else:
            current_stream = torch.cuda.current_stream(self.device)
            self.capture_stream.wait_stream(current_stream)
            with torch.cuda.stream(self.capture_stream):
                self.static_x.copy_(x)
                self.static_y.copy_(y)
                self.graph.replay()
            current_stream.wait_stream(self.capture_stream)
        # Delayed scaling updates are deliberately outside the captured region;
        # the next replay consumes the histories produced by this one.
        hnet_te.update_fp8_scaling(forward=True)
        hnet_te.update_fp8_scaling(forward=False)
        if self.ddp:
            self._sync_ddp_grads()
        return self.loss, self.metrics

    def _sync_ddp_grads(self):
        """Average captured local gradients without entering the graph."""

        total = sum(p.numel() for p in self._ddp_params)
        if self._ddp_grad_flat is None or self._ddp_grad_flat.numel() != total:
            self._ddp_grad_flat = torch.zeros(
                (total,), device=self.device, dtype=torch.float32
            )
        flat = self._ddp_grad_flat
        presence = torch.zeros(
            (len(self._ddp_params),), device=self.device, dtype=torch.int32
        )
        offset = 0
        for index, param in enumerate(self._ddp_params):
            count = param.numel()
            view = flat[offset : offset + count]
            if param.grad is None:
                view.zero_()
            else:
                presence[index] = 1
                view.copy_(param.grad.detach().reshape(-1).float())
            offset += count
        dist.all_reduce(flat, op=dist.ReduceOp.SUM)
        dist.all_reduce(presence, op=dist.ReduceOp.MAX)
        flat.div_(float(self.world))
        offset = 0
        for index, param in enumerate(self._ddp_params):
            count = param.numel()
            view = flat[offset : offset + count].view_as(param)
            if not bool(presence[index].item()):
                param.grad = None
            elif param.grad is None:
                param.grad = view.to(dtype=param.dtype)
            else:
                param.grad.copy_(view.to(dtype=param.grad.dtype))
            offset += count


def main():
    args = parse_args()
    # Select the graph-safe TE grouped-tensor dispatch before constructing the
    # model.  The capture helper temporarily disables it for eager warmup and
    # restores it for the graph itself.
    hnet_te.set_graph_capture(args.cuda_graphs)
    rank, world, device = init_dist()
    # Validate FP8 only after selecting the local CUDA device. Import-time
    # capability checks would otherwise initialize GPU 0 in every DDP rank.
    assert_fp8_runtime()
    # DDP broadcasts parameters later, but calibration runs before DDP wraps
    # the model. Use identical initial weights on every rank so power choices
    # compare the same kernels and activations; the data stream has its own
    # rank-aware deterministic partition.
    torch.manual_seed(1729)
    cfg = load_config()
    cfg.max_seq_len = args.seq
    runtime = _runtime_provenance()
    model = HNetForByteCausalLM(cfg, device=device, dtype=torch.bfloat16).train()
    n_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    if rank == 0:
        print(f"H-Net params={n_params:,} trainable; byte_lift_params=0; world={world}", flush=True)
    graph_stream = (
        torch.cuda.Stream(device=device)
        if args.cuda_graphs and world > 1
        else None
    )
    if world > 1:
        ddp_context = (
            torch.cuda.stream(graph_stream)
            if graph_stream is not None
            else contextlib.nullcontext()
        )
        with ddp_context:
            model = DDP(
                model,
                device_ids=[device.index],
                broadcast_buffers=False,
                bucket_cap_mb=256,
                gradient_as_bucket_view=True,
                static_graph=bool(args.cuda_graphs),
            )
    optimizer = make_optimizer(model.module if world > 1 else model, args)
    if args.auto_batch:
        # Calibrate after DDP and optimizer construction so the measured step
        # includes gradient synchronization and the real AnchorMuon update.
        # No graph is captured until this shape has been selected.
        args.batch = calibrate_batch_to_power(
            model, optimizer, args, device, rank, world
        )
    start_step = 0
    resume_state = None
    if args.resume:
        resume_path = Path(args.resume)
        if not resume_path.is_file():
            raise FileNotFoundError(f"resume checkpoint not found: {resume_path}")
        # Checkpoints are loaded through PyTorch's restricted deserializer;
        # never execute arbitrary pickle globals from an artifact. Transformer
        # Engine's delayed-scaling extra state is the one legacy exception, and
        # is enabled only after provenance validation below.
        with torch.serialization.safe_globals([torch.torch_version.TorchVersion]):
            resume_state = torch.load(
                resume_path, map_location="cpu", weights_only=True
            )
        _validate_resume(resume_state, cfg, args, n_params, world, runtime)
        target_model = model.module if world > 1 else model
        unsafe_key = "NVTE_ALLOW_UNSAFE_PICKLE_EXTRA_STATE"
        previous_unsafe = os.environ.get(unsafe_key)
        os.environ[unsafe_key] = "1"
        try:
            missing, unexpected = target_model.load_state_dict(
                resume_state["model"], strict=True
            )
        finally:
            if previous_unsafe is None:
                os.environ.pop(unsafe_key, None)
            else:
                os.environ[unsafe_key] = previous_unsafe
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
    graph_step = None
    if args.cuda_graphs:
        graph_step = CUDAGraphStep(
            model,
            optimizer,
            args,
            args.batch,
            args.seq,
            device,
            world,
            capture_stream=graph_stream,
        )
    log = {
        # Internal capture-only tensors live on the Namespace for the loss
        # closure but must never be serialized into the JSON run record.
        "args": {key: value for key, value in vars(args).items() if not key.startswith("_")},
        "params": n_params,
        "world": world,
        "runtime": runtime,
        "config_signature": _config_dict(cfg),
        "optimizer_groups": optimizer.group_summary(),
        # ``loss`` is deliberately the plain byte CE.  The optimized scalar
        # (CE + FBT/router/MoE terms) is recorded separately as ``objective``.
        "loss": [],
        "objective": [],
        "standard_ce": [],
        "feedback_ce": [],
        "router_loss": [],
        "moe_aux": [],
        "routing_ratio": [],
        "ratio_target_error": [],
        "time_ms": [],
        "power_w": [],
        "power_target_w": float(args.target_power_w),
        "power_gap_w": [],
        "target_ce": float(args.target_ce),
        "target_reached_step": None,
        "auto_batch_calibration": getattr(args, "_auto_batch_calibration", []),
        "auto_batch_selection_policy": getattr(
            args, "_auto_batch_selection_policy", None
        ),
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
        if graph_step is not None:
            loss, metrics = graph_step.replay(x, y)
        else:
            mask = torch.ones_like(x, dtype=torch.bool)
            optimizer.zero_grad(set_to_none=True)
            # FP8 autocast regions are owned by the model at each repeated-layer
            # boundary.  The canonical H-Net reuses the middle layers for two
            # recurrent executions and two FBT passes; one trainer-wide region
            # would make delayed-scaling TE state invalid on the second call.
            out = model(x, mask=mask, fbt_passes=FBT_PASSES)
            loss, metrics = loss_for_output(out, y, args)
            # Transformer Engine updates delayed-scaling state during the forward
            # autocast scope; its backward kernels consume that state and should
            # be launched after leaving the context.
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
        standard_ce_value = float(reduced_metrics["standard_ce"])
        log["loss"].append(standard_ce_value)
        for name, value in reduced_metrics.items():
            log[name].append(float(value))
        if log["target_reached_step"] is None and standard_ce_value <= float(args.target_ce):
            log["target_reached_step"] = int(step)
        log["time_ms"].append(elapsed)
        if rank == 0 and args.power_every > 0 and (
            step == start_step or step % args.power_every == 0
        ):
            latest_power = read_power_watts()
        if rank == 0:
            log["power_w"].append(list(latest_power))
            if latest_power:
                log["power_gap_w"].append(
                    [float(args.target_power_w) - float(value) for value in latest_power]
                )
            else:
                log["power_gap_w"].append([])
        if rank == 0 and (step < 3 or step % args.log_every == 0):
            print(
                f"step={step} objective={float(reduced):.6f} "
                f"ce={standard_ce_value:.6f} "
                f"fb_ce={float(reduced_metrics['feedback_ce']):.6f} "
                f"router={float(reduced_metrics['router_loss']):.6f} "
                f"moe={float(reduced_metrics['moe_aux']):.6f} "
                f"route_ratio={float(reduced_metrics['routing_ratio']):.6f} "
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
                    "world_size": world,
                    "runtime": runtime,
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
