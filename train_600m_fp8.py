#!/usr/bin/env python3
"""600M GPT training with TE delayed-scaling FP8 and CroissAttn correction.

The model keeps BF16 master parameters throughout the Transformer.  The first
and final Transformer blocks run under a disabled TE FP8 autocast context,
while the middle blocks use delayed-scaling FP8 (including FP8 DPA backward).
The final vocabulary projection is deliberately FP32 and receives an FP32
input so the language-model loss has a numerically stable output path.

Run with ``torchrun``.  On this workstation Transformer Engine must be loaded
with the host CUDA 13 cuBLASLt first; the launch examples in the repository
set ``LD_PRELOAD`` and ``LD_LIBRARY_PATH`` accordingly.
"""
from __future__ import annotations

import argparse
import contextlib
import json
import math
import os
import time
from pathlib import Path

import numpy as np
import torch
import torch.distributed as dist
import torch.nn as nn
import torch.nn.functional as F
import transformer_engine
import transformer_engine.pytorch as te
from transformer_engine.common import recipe
from optimizer_experiments import AnchorMuon
from vha_attention_te import VHAAttention
from kronlstm_te import KronLSTMAttention


class TokenStream:
    """Deterministic uint16 token stream with rank and step sharding."""

    def __init__(self, path: str, seq: int, batch: int, world: int, seed: int):
        self.data = np.memmap(path, dtype=np.uint16, mode="r")
        self.seq, self.batch, self.world = seq, batch, world
        self.n_chunks = (len(self.data) - 1) // seq
        self.perm = np.random.default_rng(seed).permutation(self.n_chunks)

    def get(self, step: int, rank: int, device: torch.device) -> torch.Tensor:
        base = (step * self.world + rank) * self.batch
        slots = (base + np.arange(self.batch)) % self.n_chunks
        indices = self.perm[slots] * self.seq
        arr = np.stack([self.data[i : i + self.seq + 1] for i in indices]).astype(np.int64)
        return torch.from_numpy(arr).to(device=device, non_blocking=True)


def _expand_attention_pattern(layers: int, spec: str) -> list[str]:
    pattern = [item.strip().upper() for item in spec.split(",") if item.strip()]
    aliases = {"KRON": "KDR", "KRONLSTM": "KDR"}
    pattern = [aliases.get(item, item) for item in pattern]
    if not pattern or any(item not in {"KDR", "SWA", "FA", "VHA"} for item in pattern):
        raise ValueError("--attention-pattern must contain only KDR, SWA, FA, and VHA entries")
    return [pattern[index % len(pattern)] for index in range(layers)]


class VHALayer(nn.Module):
    """Pre-norm Transformer block with VHA attention and TE FP8 linears."""

    def __init__(self, args: argparse.Namespace, layer_number: int, device: torch.device):
        super().__init__()
        bf16 = torch.bfloat16
        self.ln1 = te.LayerNorm(args.hidden, device=device, params_dtype=bf16)
        self.attn = VHAAttention(
            hidden_size=args.hidden,
            num_attention_heads=args.heads,
            num_key_value_heads=args.vha_kv_heads,
            head_dim=args.hidden // args.heads,
            postmix_rank=args.vha_postmix_rank,
            window_size=None,
            params_dtype=bf16,
            device=device,
            seq_length=args.seq,
            micro_batch_size=args.batch,
            layer_number=layer_number,
        )
        self.ln2 = te.LayerNorm(args.hidden, device=device, params_dtype=bf16)
        self.fc1 = te.Linear(args.hidden, args.ffn, bias=False, params_dtype=bf16, device=device)
        self.fc2 = te.Linear(args.ffn, args.hidden, bias=False, params_dtype=bf16, device=device)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = x + self.attn(self.ln1(x))
        x = x + self.fc2(F.gelu(self.fc1(self.ln2(x)), approximate="tanh"))
        return x


class KDRLayer(nn.Module):
    """Pre-norm Transformer block using the paper's KronLSTM mixer."""

    def __init__(self, args: argparse.Namespace, device: torch.device):
        super().__init__()
        bf16 = torch.bfloat16
        self.ln1 = te.LayerNorm(args.hidden, device=device, params_dtype=bf16)
        self.attn = KronLSTMAttention(
            hidden_size=args.hidden,
            num_heads=args.heads,
            params_dtype=bf16,
            device=device,
            chunk_size=args.kdr_chunk_size,
            checkpoint_chunks=args.kdr_checkpoint,
        )
        self.ln2 = te.LayerNorm(args.hidden, device=device, params_dtype=bf16)
        self.fc1 = te.Linear(args.hidden, args.ffn, bias=False, params_dtype=bf16, device=device)
        self.fc2 = te.Linear(args.ffn, args.hidden, bias=False, params_dtype=bf16, device=device)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = x + self.attn(self.ln1(x))
        x = x + self.fc2(F.gelu(self.fc1(self.ln2(x)), approximate="tanh"))
        return x


class GPT600M(nn.Module):
    def __init__(self, args: argparse.Namespace, device: torch.device):
        super().__init__()
        self.fp8_enabled = args.mode == "fp8"
        self.fp8_recipe = None
        self.external_fp8_autocast = False
        self.attention_pattern = _expand_attention_pattern(args.layers, args.attention_pattern)
        bf16 = torch.bfloat16

        # Embeddings, Transformer weights, and normalization stay BF16.
        self.emb = nn.Embedding(args.vocab, args.hidden, device=device, dtype=bf16)
        self.layers = nn.ModuleList(
            [
                (
                    VHALayer(args, i + 1, device)
                    if self.attention_pattern[i] == "VHA"
                    else KDRLayer(args, device)
                    if self.attention_pattern[i] == "KDR"
                    else te.TransformerLayer(
                    hidden_size=args.hidden,
                    ffn_hidden_size=args.ffn,
                    num_attention_heads=args.heads,
                    layer_number=i + 1,
                    self_attn_mask_type="causal",
                    window_size=(args.swa_window, 0)
                    if self.attention_pattern[i] == "SWA"
                    else None,
                    hidden_dropout=0.0,
                    attention_dropout=0.0,
                    params_dtype=bf16,
                    device=device,
                    seq_length=args.seq,
                    micro_batch_size=args.batch,
                    attn_input_format="sbhd",
                    bias=False,
                    fuse_qkv_params=True,
                    )
                )
                for i in range(args.layers)
            ]
        )
        self.norm = nn.LayerNorm(args.hidden, device=device, dtype=bf16)
        # Keep both the projection weights and its input/output computation FP32.
        self.head = nn.Linear(
            args.hidden, args.vocab, bias=False, device=device, dtype=torch.float32
        )
        self.apply(self._init)

    @staticmethod
    def _init(module: nn.Module) -> None:
        if isinstance(module, (nn.Linear, nn.Embedding)):
            nn.init.normal_(module.weight, mean=0.0, std=0.02)

    def forward(self, tokens: torch.Tensor) -> torch.Tensor:
        x = self.emb(tokens).transpose(0, 1).contiguous()
        context = (
            contextlib.nullcontext()
            if self.external_fp8_autocast
            else te.fp8_autocast(enabled=self.fp8_enabled, fp8_recipe=self.fp8_recipe)
        )
        with context:
            # These explicit disabled scopes are the precision boundaries.
            with te.fp8_autocast(enabled=False):
                x = self.layers[0](x)
            for layer in self.layers[1:-1]:
                x = layer(x)
            if len(self.layers) > 1:
                with te.fp8_autocast(enabled=False):
                    x = self.layers[-1](x)
            x = self.norm(x)
            logits = self.head(x.float())
        return logits.transpose(0, 1).contiguous()


def _dtype_counts(module: nn.Module) -> dict[str, int]:
    counts: dict[str, int] = {}
    for parameter in module.parameters():
        key = str(parameter.dtype).replace("torch.", "")
        counts[key] = counts.get(key, 0) + parameter.numel()
    return counts


def _layer_diagnostics(model: GPT600M, mode: str) -> dict[str, object]:
    def layer_info(layer: nn.Module) -> dict[str, object]:
        return {"dtypes": _dtype_counts(layer), "training": layer.training}

    return {
        "precision_plan": {
            "first_transformer": "bf16 (fp8 autocast disabled)",
            "middle_transformers": (
                "delayed-scaling fp8 (fp8_dpa backward)" if mode == "fp8" else "bf16"
            ),
            "last_transformer": "bf16 (fp8 autocast disabled)",
            "output_head": "fp32 weights and fp32 matmul",
        },
        "embedding_dtypes": _dtype_counts(model.emb),
        "first_layer": layer_info(model.layers[0]),
        "middle_layers": [layer_info(layer) for layer in model.layers[1:-1]],
        "last_layer": layer_info(model.layers[-1]),
        "norm_dtypes": _dtype_counts(model.norm),
        "head_dtypes": _dtype_counts(model.head),
    }


def _make_recipe(mode: str):
    if mode == "fp8":
        # Delayed scaling is the tested production path.  fp8_dpa enables the
        # patched TE attention backward and its CroissAttn auxiliary correction.
        return recipe.DelayedScaling(
            margin=0,
            fp8_format=recipe.Format.HYBRID,
            amax_history_len=1,
            amax_compute_algo="most_recent",
            fp8_dpa=True,
            fp8_mha=False,
        )
    return None


def _muon_adjust_lr(lr: float, adjust_lr_fn: str, shape: torch.Size) -> float:
    rows, cols = shape[:2]
    if adjust_lr_fn in (None, "original"):
        return lr * math.sqrt(max(1.0, rows / cols))
    if adjust_lr_fn == "match_rms_adamw":
        return lr * 0.2 * math.sqrt(max(rows, cols))
    if adjust_lr_fn == "spectral_unclamped":
        return lr * math.sqrt(rows / cols)
    raise ValueError(f"unknown Muon LR adjustment: {adjust_lr_fn}")


class StableMuon(torch.optim.Optimizer):
    """Muon with matmul-based Newton-Schulz for B300 cuBLASLt compatibility.

    PyTorch's stock Muon uses ``torch.addmm`` with a nonzero beta for the
    Newton-Schulz polynomial.  On this CUDA/cuBLASLt combination that call
    fails for BF16/FP16 with ``CUBLAS_STATUS_NOT_INITIALIZED``.  Matmul plus
    scalar multiply/add computes the same polynomial without that backend
    path and keeps the BF16 momentum/update behavior of stock Muon.
    """

    def __init__(
        self,
        params,
        *,
        lr: float,
        weight_decay: float,
        momentum: float,
        nesterov: bool = True,
        ns_steps: int = 5,
        adjust_lr_fn: str = "match_rms_adamw",
        eps: float = 1e-7,
    ):
        if ns_steps >= 100:
            raise ValueError("ns_steps must be less than 100")
        defaults = dict(
            lr=lr,
            weight_decay=weight_decay,
            momentum=momentum,
            nesterov=nesterov,
            ns_steps=ns_steps,
            adjust_lr_fn=adjust_lr_fn,
            eps=eps,
        )
        super().__init__(params, defaults)
        for group in self.param_groups:
            for parameter in group["params"]:
                if parameter.ndim != 2:
                    raise ValueError("StableMuon only supports 2-D parameters")

    @staticmethod
    def _zeropower(gradient, ns_steps, eps):
        update = gradient.bfloat16()
        transposed = update.shape[0] > update.shape[1]
        if transposed:
            update = update.T
        update = update / update.norm().clamp(min=eps)
        a, b, c = 3.4445, -4.7750, 2.0315
        for _ in range(ns_steps):
            gram = update @ update.T
            # Equivalent to addmm(gram, gram, gram, beta=b, alpha=c),
            # without cuBLASLt's failing nonzero-beta BF16 path.
            gram_update = c * (gram @ gram) + b * gram
            update = a * update + gram_update @ update
        return update.T if transposed else update

    @torch.no_grad()
    def step(self, closure=None):
        loss = None
        if closure is not None:
            with torch.enable_grad():
                loss = closure()
        for group in self.param_groups:
            lr = group["lr"]
            momentum = group["momentum"]
            for parameter in group["params"]:
                if parameter.grad is None:
                    continue
                if parameter.grad.is_sparse:
                    raise RuntimeError("StableMuon does not support sparse gradients")
                state = self.state[parameter]
                if "momentum_buffer" not in state:
                    state["momentum_buffer"] = torch.zeros_like(parameter.grad)
                buffer = state["momentum_buffer"]
                gradient = parameter.grad
                buffer.lerp_(gradient, 1.0 - momentum)
                update = gradient.lerp(buffer, momentum) if group["nesterov"] else buffer
                update = self._zeropower(update, group["ns_steps"], group["eps"])
                parameter.mul_(1.0 - lr * group["weight_decay"])
                parameter.add_(
                    update,
                    alpha=-_muon_adjust_lr(lr, group["adjust_lr_fn"], parameter.shape),
                )
        return loss


def _build_optimizers(model: nn.Module, args: argparse.Namespace):
    """Build staged GramNS/AnchorMuon with AdamATan2 fallback routing."""
    named = [(name, parameter) for name, parameter in model.named_parameters() if parameter.requires_grad]
    if args.optimizer == "adamw":
        optimizer = torch.optim.AdamW(
            [parameter for _, parameter in named],
            lr=args.adamw_lr,
            betas=(0.9, 0.95),
            weight_decay=args.weight_decay,
            fused=True,
        )
        return {"adamw": optimizer}, {"adamw": sum(parameter.numel() for _, parameter in named)}

    matrix_named = [
        (name, parameter)
        for name, parameter in named
        if parameter.ndim == 2 and not name.startswith(("emb.", "head."))
    ]
    fallback_named = [(name, parameter) for name, parameter in named if (name, parameter) not in matrix_named]
    if not matrix_named:
        raise RuntimeError("no 2-D hidden matrices were found for GramNS/AnchorMuon")
    variant = args.optimizer
    use_soda = variant in {"anchor_soda", "anchor_pmuoneq", "anchor_normuon", "anchor"}
    use_pmuoneq = variant in {"anchor_pmuoneq", "anchor_normuon", "anchor"}
    use_normuon = variant in {"anchor_normuon", "anchor"}
    muon_groups = [
        {
            "params": [parameter for _, parameter in matrix_named],
            "param_names": [name for name, _ in matrix_named],
            "use_matrix_update": True,
        },
        {
            "params": [parameter for _, parameter in fallback_named],
            "param_names": [name for name, _ in fallback_named],
            "use_matrix_update": False,
        },
    ]
    optimizer = AnchorMuon(
        muon_groups,
        lr=args.muon_lr,
        fallback_lr=args.muon_fallback_lr,
        fallback_mode="atan2",
        fallback_betas=(0.9, 0.95),
        fallback_weight_decay=args.weight_decay,
        momentum=args.muon_momentum,
        soda_lambda_scale=args.soda_lambda_scale,
        pmuoneq_beta=args.pmuoneq_beta,
        row_gamma=args.anchor_row_gamma,
        normuon_beta2=args.normuon_beta2,
        ns_compute_dtype=torch.float16,
        min_matrix_dim=2,
        use_soda=use_soda,
        use_pmuoneq=use_pmuoneq,
        use_normuon=use_normuon,
        matrix_weight_decay=args.weight_decay,
    )
    optimizers = {"muon": optimizer}
    counts = {
        "muon": sum(parameter.numel() for _, parameter in matrix_named),
        "fallback": sum(parameter.numel() for _, parameter in fallback_named),
        "muon_matrices": len(matrix_named),
        "fallback_tensors": len(fallback_named),
    }
    return optimizers, counts


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--data", default="/home/catid/fp4/data/fineweb_tok")
    parser.add_argument("--mode", choices=("fp8", "bf16"), default="fp8")
    parser.add_argument(
        "--optimizer",
        choices=("gramns", "anchor_soda", "anchor_pmuoneq", "anchor_normuon", "anchor", "adamw"),
        default="gramns",
    )
    parser.add_argument("--out", default="")
    parser.add_argument("--seq", type=int, default=512)
    parser.add_argument("--batch", type=int, default=1)
    parser.add_argument("--hidden", type=int, default=1536)
    parser.add_argument("--ffn", type=int, default=6144)
    parser.add_argument("--heads", type=int, default=24)
    parser.add_argument("--layers", type=int, default=16)
    parser.add_argument(
        "--attention-pattern",
        default="KDR,KDR,KDR,VHA",
        help="repeated per-layer pattern; KDR is KronLSTM, VHA is compressed full attention",
    )
    parser.add_argument(
        "--swa-window",
        type=int,
        default=2048,
        help="number of causal keys to retain to the left in SWA layers",
    )
    parser.add_argument("--vha-kv-heads", type=int, default=2)
    parser.add_argument("--vha-postmix-rank", type=int, default=6)
    parser.add_argument("--kdr-chunk-size", type=int, default=64)
    parser.add_argument(
        "--kdr-checkpoint",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="checkpoint KronLSTM chunks to reduce activation memory",
    )
    parser.add_argument("--vocab", type=int, default=50304)
    parser.add_argument("--tokens", type=int, default=20_000_000_000)
    parser.add_argument("--steps", type=int, default=0)
    parser.add_argument("--warmup", type=int, default=200)
    parser.add_argument("--lr", type=float, default=None, help="override the selected optimizer LR")
    parser.add_argument("--adamw-lr", type=float, default=6e-4)
    parser.add_argument("--muon-lr", type=float, default=1e-3)
    parser.add_argument("--muon-fallback-lr", type=float, default=1e-3)
    parser.add_argument("--muon-momentum", type=float, default=0.95)
    parser.add_argument(
        "--muon-adjust-lr",
        choices=("original", "match_rms_adamw", "spectral_unclamped"),
        default="match_rms_adamw",
    )
    parser.add_argument("--muon-ns-steps", type=int, default=5)
    parser.add_argument("--weight-decay", type=float, default=0.05)
    parser.add_argument("--soda-lambda-scale", type=float, default=0.003)
    parser.add_argument("--pmuoneq-beta", type=float, default=0.90)
    parser.add_argument("--anchor-row-gamma", type=float, default=0.45)
    parser.add_argument("--normuon-beta2", type=float, default=0.93)
    parser.add_argument("--min-lr", type=float, default=6e-5)
    parser.add_argument("--seed", type=int, default=1234)
    parser.add_argument("--log-every", type=int, default=10)
    parser.add_argument("--ckpt-every", type=int, default=0)
    parser.add_argument("--resume", default="")
    parser.add_argument("--inspect-only", action="store_true")
    args = parser.parse_args()

    if args.lr is not None:
        if args.optimizer != "adamw":
            args.muon_lr = args.lr
        else:
            args.adamw_lr = args.lr
    if args.swa_window <= 0:
        raise ValueError("--swa-window must be positive")

    # The patched TE backend defaults this switch to enabled.  Set it
    # explicitly for reproducibility while still allowing a caller to turn it
    # off for an ablation with NVTE_FP8_DPA_BWD=0.
    if args.mode == "fp8":
        os.environ.setdefault("NVTE_FP8_DPA_BWD", "1")

    local = int(os.environ.get("LOCAL_RANK", "0"))
    torch.cuda.set_device(local)
    dist.init_process_group("nccl")
    rank, world = dist.get_rank(), dist.get_world_size()
    device = torch.device("cuda", local)
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)

    model = GPT600M(args, device)
    n_params = sum(parameter.numel() for parameter in model.parameters())
    steps = args.steps or math.ceil(args.tokens / (world * args.batch * args.seq))
    fp8_recipe = _make_recipe(args.mode)
    model.fp8_recipe = fp8_recipe
    diagnostics = _layer_diagnostics(model, args.mode)
    if rank == 0:
        print(
            json.dumps(
                {
                    "mode": args.mode,
                    "optimizer": args.optimizer,
                    "params": n_params,
                    "params_m": n_params / 1e6,
                    "transformer_engine": str(Path(transformer_engine.__file__).resolve()),
                    "recipe": {
                        "name": type(fp8_recipe).__name__ if fp8_recipe else None,
                        "fp8_dpa": bool(getattr(fp8_recipe, "fp8_dpa", False)),
                        "fp8_mha": bool(getattr(fp8_recipe, "fp8_mha", False)),
                        "amax_history_len": getattr(fp8_recipe, "amax_history_len", None),
                    },
                    "croissattn_env": {
                        "NVTE_FP8_DPA_BWD": os.environ.get("NVTE_FP8_DPA_BWD", "<default>"),
                        "aux_correction_expected": bool(
                            args.mode == "fp8" and fp8_recipe is not None and fp8_recipe.delayed()
                        ),
                    },
                    "dtypes": diagnostics,
                    "attention_pattern": model.attention_pattern,
                    "swa_window": args.swa_window,
                },
                indent=2,
            ),
            flush=True,
        )
    if args.inspect_only:
        dist.barrier(device_ids=[local])
        dist.destroy_process_group()
        return

    if not os.path.exists(os.path.join(args.data, "train.bin")):
        raise FileNotFoundError(f"missing token stream: {os.path.join(args.data, 'train.bin')}")
    train = TokenStream(os.path.join(args.data, "train.bin"), args.seq, args.batch, world, args.seed)
    if train.n_chunks == 0:
        raise RuntimeError("training stream is empty")

    ddp = nn.parallel.DistributedDataParallel(model, device_ids=[local])
    optimizers, optimizer_counts = _build_optimizers(ddp.module, args)
    start = 0
    if args.resume:
        state = torch.load(args.resume, map_location=device, weights_only=False)
        ddp.module.load_state_dict(state["model"])
        if "optimizers" in state:
            for name, optimizer in optimizers.items():
                optimizer.load_state_dict(state["optimizers"][name])
        elif "optimizer" in state and "adamw" in optimizers:
            optimizers["adamw"].load_state_dict(state["optimizer"])
        start = int(state["step"]) + 1

    def lr_at(step: int, base_lr: float) -> float:
        if step < args.warmup:
            return base_lr * (step + 1) / max(1, args.warmup)
        t = (step - args.warmup) / max(1, steps - args.warmup)
        min_lr = args.min_lr * (base_lr / 6e-4)
        return min_lr + 0.5 * (base_lr - min_lr) * (1 + math.cos(math.pi * min(1.0, t)))

    log: dict[str, object] = {
        "args": vars(args),
        "world": world,
        "params": n_params,
        "attention_pattern": model.attention_pattern,
        "swa_window": args.swa_window,
        "transformer_engine": str(Path(transformer_engine.__file__).resolve()),
        "optimizer": args.optimizer,
        "optimizer_param_counts": optimizer_counts,
        "weight_decay": args.weight_decay,
        "muon": {
            "lr": args.muon_lr,
            "momentum": args.muon_momentum,
            "ns_steps": args.muon_ns_steps,
            "adjust_lr": args.muon_adjust_lr,
        },
        "adamw": {"lr": args.adamw_lr},
        "diagnostics": diagnostics,
        "recipe": {
            "name": type(fp8_recipe).__name__ if fp8_recipe else None,
            "fp8_dpa": bool(getattr(fp8_recipe, "fp8_dpa", False)),
            "fp8_mha": bool(getattr(fp8_recipe, "fp8_mha", False)),
            "amax_history_len": getattr(fp8_recipe, "amax_history_len", None),
        },
        "croissattn_env": {
            "NVTE_FP8_DPA_BWD": os.environ.get("NVTE_FP8_DPA_BWD", "<default>"),
            "aux_correction_expected": bool(
                args.mode == "fp8" and fp8_recipe is not None and fp8_recipe.delayed()
            ),
        },
        "steps": [],
        "loss": [],
        "grad_norm": [],
        "time_ms": [],
        "nonfinite": [],
    }
    if rank == 0:
        print(
            f"mode={args.mode} params={n_params/1e6:.2f}M world={world} "
            f"tokens={steps*world*args.batch*args.seq/1e9:.3f}B steps={steps}",
            flush=True,
        )

    times: list[float] = []
    for step in range(start, steps):
        for name, optimizer in optimizers.items():
            for group in optimizer.param_groups:
                # AnchorMuon carries separate matrix and AdamATan2 fallback
                # groups. Preserve the requested fallback LR instead of
                # overwriting every group with the matrix rate.
                if name == "muon":
                    base_lr = args.muon_lr if group.get("use_matrix_update", False) else args.muon_fallback_lr
                else:
                    base_lr = args.adamw_lr
                group["lr"] = lr_at(step, base_lr)
        tok = train.get(step, rank, device)
        x, y = tok[:, :-1], tok[:, 1:]
        torch.cuda.synchronize()
        t0 = time.perf_counter()
        logits = ddp(x)
        loss = F.cross_entropy(logits.float().flatten(0, 1), y.flatten())
        for optimizer in optimizers.values():
            optimizer.zero_grad(set_to_none=True)
        loss.backward()
        gnorm = torch.nn.utils.clip_grad_norm_(ddp.parameters(), 1.0)
        finite = bool(torch.isfinite(loss) and torch.isfinite(gnorm))
        if finite:
            for optimizer in optimizers.values():
                optimizer.step()
        torch.cuda.synchronize()
        elapsed = time.perf_counter() - t0
        reduced_loss = loss.detach().clone()
        dist.all_reduce(reduced_loss)
        reduced_loss = reduced_loss.item() / world
        if not finite:
            log["nonfinite"].append(step)
        log["steps"].append(step)
        log["loss"].append(reduced_loss)
        log["grad_norm"].append(float(gnorm))
        log["time_ms"].append(elapsed * 1e3)
        if step >= start + 5:
            times.append(elapsed)
        if rank == 0 and (step < start + 3 or step % args.log_every == 0 or step == steps - 1):
            print(
                f"step={step} loss={reduced_loss:.6f} grad={float(gnorm):.4f} "
                f"time_ms={elapsed*1e3:.1f}",
                flush=True,
            )
        if args.ckpt_every and (step + 1) % args.ckpt_every == 0 and rank == 0:
            path = args.out or "train_600m_fp8"
            torch.save(
                {
                    "step": step,
                    "model": ddp.module.state_dict(),
                    "optimizers": {
                        name: optimizer.state_dict() for name, optimizer in optimizers.items()
                    },
                },
                path + f".step{step+1}.pt",
            )
        if len(log["nonfinite"]) >= 5:
            raise RuntimeError("too many non-finite steps")

    if rank == 0:
        mean = sum(times) / len(times) if times else float("nan")
        log["mean_ms"] = mean * 1e3
        log["tokens_per_s"] = world * args.batch * args.seq / mean if times else float("nan")
        print(
            f"RESULT mode={args.mode} params={n_params/1e6:.2f}M "
            f"mean_ms={mean*1e3:.2f} Mtok/s={log['tokens_per_s']/1e6:.3f} "
            f"final_loss={log['loss'][-1]:.6f} nonfinite={len(log['nonfinite'])}",
            flush=True,
        )
        if args.out:
            Path(args.out).parent.mkdir(parents=True, exist_ok=True)
            Path(args.out).write_text(json.dumps(log, indent=2))
    dist.destroy_process_group()


if __name__ == "__main__":
    main()
