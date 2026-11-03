"""Direct 600M H-Net raw-byte trainer.

The model consumes document-local UTF-8 bytes from a validated manifest through
a frozen categorical byte code, then learns boundaries with H-Net routing from
the first update. There is no BPE or trainable token embedding. The script
keeps the CE and router losses in FP32, uses AnchorMuon/GramNS with its
AdamATan2 fallback, FBT multi-pass training, and supports CUDA-graph capture
for the fixed-shape padded training contract.
"""

from __future__ import annotations

import argparse
import contextlib
import dataclasses
import datetime
import hashlib
import importlib.metadata
import json
import math
import os
import subprocess
import sys
import time
from pathlib import Path, PurePosixPath

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
CANONICAL_CUDA_GRAPHS = True
ADAM_ATAN2_LR_RATIO = 0.5
# With activation recomputation disabled, B=18 at S=4096 is the largest
# validated complete full-model step with optimizer state on a B300. B=20
# and larger exhaust the CUDA graph pool during the first complete update.
# Power calibration still chooses a smaller shape when the per-GPU target is
# reached before this bound.
CANONICAL_MAX_BATCH = 18


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument(
        "--manifest",
        required=True,
        help="versioned document manifest; its data_path identifies the raw UTF-8 byte file",
    )
    p.add_argument(
        "--data",
        default=None,
        help="optional explicit data path; it must match the manifest data_path",
    )
    p.add_argument("--seq", type=int, default=4096)
    p.add_argument(
        "--auto-batch-max",
        type=int,
        default=CANONICAL_MAX_BATCH,
        help="upper bound for power-calibration candidates (validated complete-step cap is 18)",
    )
    p.add_argument("--data-seed", type=int, default=1729)
    p.add_argument("--steps", type=int, default=100000)
    p.add_argument("--warmup", type=int, default=2000)
    p.add_argument("--lr", type=float, default=1.2e-3)
    p.add_argument(
        "--lr-decay-fraction",
        type=float,
        default=0.20,
        help="final linear WSqD cooldown fraction of the post-warmup horizon",
    )
    p.add_argument(
        "--lr-shift",
        type=float,
        default=5000.0,
        help="T0 shift used by the horizon-free inverse-square-root WSqD base",
    )
    p.add_argument("--weight-decay", type=float, default=0.05)
    p.add_argument("--fbt-lambda", type=float, default=1.0)
    p.add_argument("--ratio-weight", type=float, default=0.03)
    p.add_argument(
        "--ratio-target",
        type=float,
        default=8.0,
        help="target byte-to-latent ratio; 8 matches 4096 tokens and 512 latent capacity",
    )
    p.add_argument(
        "--target-ce",
        type=float,
        default=4.0,
        help="plain byte cross-entropy target recorded in the run log",
    )
    p.add_argument("--log-every", type=int, default=10)
    p.add_argument(
        "--eval-every",
        type=int,
        default=40,
        help="run the complete-model held-out evaluation every N optimizer steps (0 disables)",
    )
    p.add_argument(
        "--eval-batches",
        type=int,
        default=1,
        help="number of fixed-shape held-out byte batches per evaluation",
    )
    p.add_argument(
        "--eval-manifest",
        default=None,
        help="optional held-out document manifest; training manifest is used when omitted",
    )
    p.add_argument(
        "--mmlu-data",
        default=None,
        help="optional JSONL multiple-choice file with question/choices/answer fields",
    )
    p.add_argument(
        "--arc-challenge-data",
        default=None,
        help="optional JSONL ARC-Challenge file (question/choices/answerKey fields)",
    )
    p.add_argument(
        "--hellaswag-data",
        default=None,
        help="optional JSONL HellaSwag file (ctx/endings/label fields)",
    )
    p.add_argument(
        "--eval-max-items",
        type=int,
        default=0,
        help="maximum multiple-choice items per benchmark (0 means all records)",
    )
    p.add_argument(
        "--peak-tflops",
        type=float,
        default=1000.0,
        help="per-GPU FP8 peak used for the reported model-FLOP utilization estimate",
    )
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
    return p.parse_args()


def _validate_schedule_args(args):
    """Validate the canonical warmup + WSqD schedule parameters."""
    if int(args.seq) != 4096:
        raise ValueError(
            "the canonical full-system trainer requires --seq=4096; "
            "shorter sequence shapes are unsupported"
        )
    if int(args.warmup) < 0:
        raise ValueError("warmup must be non-negative")
    if not (0.0 < float(args.lr_decay_fraction) < 1.0):
        raise ValueError("lr-decay-fraction must be strictly between zero and one")
    if float(args.lr_shift) < 0.0:
        raise ValueError("lr-shift must be non-negative")
    if float(args.lr) <= 0.0:
        raise ValueError("lr must be positive")
    if int(args.eval_every) < 0:
        raise ValueError("eval-every must be non-negative")
    if int(args.log_every) < 1:
        raise ValueError("log-every must be positive")
    if int(args.eval_batches) < 1:
        raise ValueError("eval-batches must be positive")
    if int(args.eval_max_items) < 0:
        raise ValueError("eval-max-items must be non-negative")
    if float(args.peak_tflops) <= 0.0:
        raise ValueError("peak-tflops must be positive")


def _wsqd_scale(step: int, args) -> float:
    """Return the scalar for WSqD after linear warmup.

    The post-warmup base follows the paper's shifted inverse-square-root
    schedule, ``c0 / sqrt(t + T0)``, and its final ``alpha`` fraction is a
    linear cooldown to zero.  ``c0`` is chosen so the first post-warmup step
    equals the requested peak ``args.lr``.  The horizon only selects the
    cooldown start; changing ``args.steps`` leaves the base trajectory intact.
    """
    warmup = int(args.warmup)
    if step < warmup:
        return (step + 1) / max(1, warmup)
    t = max(1, int(step) - warmup + 1)
    # A resumed run may request more total steps than the checkpoint's
    # original horizon. Keep the original cooldown boundary so extending a
    # run cannot jump its learning rate back into the high-LR stable region.
    schedule_steps = int(getattr(args, "_lr_horizon_steps", args.steps))
    horizon = max(1, schedule_steps - warmup)
    alpha = float(args.lr_decay_fraction)
    shift = float(args.lr_shift)
    stable_horizon = max(1, int(math.floor((1.0 - alpha) * horizon)))
    base = math.sqrt((1.0 + shift) / (t + shift))
    if t <= stable_horizon:
        return base
    cooldown = max(1, horizon - stable_horizon)
    # WSqD freezes the base value at the stable/cooldown boundary, then
    # linearly decays that value. Do not continue the inverse-square-root
    # decrease during cooldown.
    stable_base = math.sqrt((1.0 + shift) / (stable_horizon + shift))
    return stable_base * max(0.0, (horizon - t) / cooldown)


@dataclasses.dataclass(frozen=True)
class DocumentManifest:
    """Validated index for document-local byte windows.

    The raw byte file is immutable and the manifest is the sampling contract.
    Every document is sampled independently, so no input or target pair can
    cross a document boundary.  Keeping offsets instead of materializing text
    preserves mmap-style access for multi-hundred-billion-byte corpora.
    """

    path: Path
    data_path: Path
    data_size: int
    data_sha256: str
    documents: tuple[dict, ...]
    total_token_equivalents: int
    manifest_sha256: str


def _verify_document_hashes(data_path: Path, documents: list[dict]) -> None:
    """Verify every manifest range against the payload before training.

    The payload digest protects the file as a whole; per-document digests also
    protect offsets and deduplication boundaries. This is a startup check for
    the corpus build, not a per-step operation.
    """
    with data_path.open("rb") as handle:
        for document in sorted(documents, key=lambda item: item["offset"]):
            handle.seek(document["offset"])
            digest = hashlib.sha256()
            remaining = int(document["length"])
            while remaining:
                chunk = handle.read(min(16 * 1024 * 1024, remaining))
                if not chunk:
                    raise ValueError(
                        f"manifest document {document['id']!r} extends past payload"
                    )
                digest.update(chunk)
                remaining -= len(chunk)
            if digest.hexdigest() != document["sha256"].lower():
                raise ValueError(
                    f"manifest document {document['id']!r} sha256 does not match payload"
                )


def _load_manifest(path: str | os.PathLike[str], data_override: str | None = None) -> DocumentManifest:
    manifest_path = Path(path).expanduser().resolve()
    if not manifest_path.is_file():
        raise FileNotFoundError(f"manifest not found: {manifest_path}")
    try:
        raw = json.loads(manifest_path.read_text())
    except json.JSONDecodeError as exc:
        raise ValueError(f"invalid JSON manifest: {manifest_path}") from exc
    if not isinstance(raw, dict) or int(raw.get("version", -1)) != 1:
        raise ValueError("document manifest requires integer version=1")
    declared_data = raw.get("data_path")
    if not isinstance(declared_data, str) or not declared_data:
        raise ValueError("document manifest requires a non-empty data_path")
    if Path(declared_data).is_absolute():
        raise ValueError("manifest data_path must be relative to the manifest")
    data_path = (manifest_path.parent / declared_data).resolve()
    try:
        data_path.relative_to(manifest_path.parent)
    except ValueError as exc:
        raise ValueError("manifest data_path escapes its manifest directory") from exc
    if data_override is not None:
        override_path = Path(data_override).expanduser().resolve()
        if override_path != data_path:
            raise ValueError(
                "--data must resolve to the same path as manifest data_path; "
                f"manifest={data_path}, override={override_path}"
            )
    if not data_path.is_file():
        raise FileNotFoundError(f"manifest data_path is not a regular file: {data_path}")
    data_size = int(data_path.stat().st_size)
    declared_size = raw.get("data_size")
    if declared_size is None or int(declared_size) != data_size:
        raise ValueError(
            f"manifest data_size mismatch: manifest={declared_size}, actual={data_size}"
        )
    declared_hash = raw.get("data_sha256")
    if not isinstance(declared_hash, str) or len(declared_hash) != 64:
        raise ValueError("document manifest requires a 64-character data_sha256")
    data_sha256 = _sha256_file(data_path)
    if data_sha256 != declared_hash.lower():
        raise ValueError(
            f"manifest data_sha256 mismatch: manifest={declared_hash}, actual={data_sha256}"
        )
    documents = raw.get("documents")
    if not isinstance(documents, list) or not documents:
        raise ValueError("document manifest requires a non-empty documents list")
    normalized = []
    intervals = []
    seen_ids = set()
    for index, document in enumerate(documents):
        if not isinstance(document, dict):
            raise ValueError(f"manifest document {index} must be an object")
        required = ("id", "offset", "length", "token_count", "source", "license", "language", "sha256")
        missing = [key for key in required if key not in document]
        if missing:
            raise ValueError(f"manifest document {index} missing fields: {missing}")
        doc_id = document["id"]
        if not isinstance(doc_id, str) or not doc_id or doc_id in seen_ids:
            raise ValueError(f"manifest document {index} has a missing or duplicate id")
        seen_ids.add(doc_id)
        offset = int(document["offset"])
        length = int(document["length"])
        token_count = int(document["token_count"])
        if offset < 0 or length < 2 or offset + length > data_size:
            raise ValueError(
                f"manifest document {doc_id!r} has invalid byte range "
                f"offset={offset}, length={length}, data_size={data_size}"
            )
        if token_count < 0:
            raise ValueError(f"manifest document {doc_id!r} has negative token_count")
        for key in ("source", "license", "language"):
            if not isinstance(document[key], str) or not document[key]:
                raise ValueError(f"manifest document {doc_id!r} requires non-empty {key}")
        doc_hash = document["sha256"]
        if not isinstance(doc_hash, str) or len(doc_hash) != 64:
            raise ValueError(f"manifest document {doc_id!r} requires a 64-character sha256")
        normalized_doc = dict(document)
        normalized_doc.update(offset=offset, length=length, token_count=token_count)
        normalized.append(normalized_doc)
        intervals.append((offset, offset + length, doc_id))
    intervals.sort()
    for previous, current in zip(intervals, intervals[1:]):
        if current[0] < previous[1]:
            raise ValueError(
                f"manifest document ranges overlap: {previous[2]!r} and {current[2]!r}"
            )
    _verify_document_hashes(data_path, normalized)
    total_tokens = sum(document["token_count"] for document in normalized)
    declared_tokens = raw.get("total_token_equivalents", total_tokens)
    if int(declared_tokens) != total_tokens:
        raise ValueError(
            "manifest total_token_equivalents does not equal the sum of document token_count"
        )
    return DocumentManifest(
        path=manifest_path,
        data_path=data_path,
        data_size=data_size,
        data_sha256=data_sha256,
        documents=tuple(normalized),
        total_token_equivalents=total_tokens,
        manifest_sha256=_sha256_file(manifest_path),
    )


class ByteStream:
    def __init__(self, manifest: DocumentManifest, seq, batch, rank, world, seed=1729):
        self.data = torch.from_file(
            str(manifest.data_path), shared=False, size=manifest.data_size, dtype=torch.uint8
        )
        self.manifest = manifest
        self.documents = manifest.documents
        self.seq = seq
        self.batch = batch
        self.rank = rank
        self.world = world
        self.seed = int(seed)
        self.stride = batch * seq
        self.document_count = len(self.documents)
        # A document-local window has no padding when every document contains
        # at least one complete input/target pair of the configured length.
        # The canonical FineWeb shards commonly satisfy this invariant.  In
        # that case the graph's all-ones masks are static and need not be
        # rebuilt, copied to the device, or copied into the graph-owned input
        # tensors on every replay.  Keep the general masked path below for
        # manifests containing short documents.
        self.full_windows = all(
            int(document["length"]) >= int(seq) + 1
            for document in self.documents
        )
        global_rows = batch * world
        # Round the epoch up to a complete global batch. Repeating only the
        # first few document indices is preferable to silently dropping the
        # tail of a FineWeb shard, and keeps every rank's static shape valid.
        self.n_documents = (
            (self.document_count + global_rows - 1) // global_rows
        ) * global_rows
        self.n_batches = self.n_documents // global_rows
        # Reuse a ring of pinned host staging buffers so document-local
        # sampling does not allocate pageable tensors or force a synchronous
        # host-to-device copy on every graph replay. Transfers are queued on a
        # dedicated stream and the consumer stream waits only on the selected
        # slot's event; this overlaps the next H2D transfer with the previous
        # model/optimizer replay.
        self._pinned = bool(torch.cuda.is_available())
        self._stage_count = 4 if self._pinned else 1
        self._copy_stream = (
            torch.cuda.Stream(device=torch.cuda.current_device())
            if self._pinned
            else None
        )
        self._staged = [
            (
                torch.empty((self.batch, self.seq), dtype=torch.uint8, pin_memory=self._pinned),
                torch.empty((self.batch, self.seq), dtype=torch.uint8, pin_memory=self._pinned),
                torch.empty((self.batch, self.seq), dtype=torch.bool, pin_memory=self._pinned),
                torch.empty((self.batch, self.seq), dtype=torch.bool, pin_memory=self._pinned),
            )
            for _ in range(self._stage_count)
        ]
        self._stage_events = (
            [torch.cuda.Event() for _ in range(self._stage_count)]
            if self._pinned
            else []
        )
        self._stage_used = [False] * self._stage_count
        self._stage_index = 0
        self._epoch = None
        self._permutation = None

    def _epoch_permutation(self, epoch):
        """Return a reproducible window permutation for one data epoch."""
        if self._epoch != epoch:
            generator = torch.Generator(device="cpu")
            generator.manual_seed(self.seed + epoch)
            # Permute document rows, not batches. ``get`` slices this tensor
            # by ``batch * world``; permuting only n_batches would silently
            # omit almost the entire manifest once there is more than one
            # batch per epoch.
            permutation = torch.randperm(self.document_count, generator=generator)
            pad = self.n_documents - self.document_count
            if pad:
                repeats = (pad + self.document_count - 1) // self.document_count
                permutation = torch.cat(
                    (permutation, permutation.repeat(repeats)[:pad])
                )
            self._permutation = permutation
            self._epoch = epoch
        return self._permutation

    @staticmethod
    def _window_start(seed: int, epoch: int, document_index: int, length: int, seq: int) -> int:
        max_start = max(0, length - (seq + 1))
        if max_start == 0:
            return 0
        value = (int(seed) ^ (int(epoch) * 0x9E3779B1) ^ (int(document_index) * 0x85EBCA77))
        value &= 0xFFFFFFFFFFFFFFFF
        value ^= value >> 30
        value = (value * 0xBF58476D1CE4E5B9) & 0xFFFFFFFFFFFFFFFF
        value ^= value >> 27
        return int(value % (max_start + 1))

    def get(
        self,
        step,
        device,
        *,
        omit_masks=False,
        consumer_stream=None,
        device_buffers=None,
    ):
        """Load one batch, optionally DMAing directly into graph inputs.

        ``CUDAGraphStep`` owns persistent input tensors.  Passing those four
        tensors here lets the copy stream write them directly from the pinned
        staging rows, avoiding a temporary GPU batch followed by a second
        device-to-device copy immediately before replay.  The returned
        objects are the supplied buffers, so the graph replay can recognize
        that its inputs are already populated.  The event is still recorded
        after every DMA and waited on by ``consumer_stream`` before replay.
        """
        if omit_masks and not self.full_windows:
            raise ValueError("omit_masks requires a full-window document stream")
        if device_buffers is not None:
            if device.type != "cuda" or len(device_buffers) != 4:
                raise ValueError(
                    "device_buffers requires four CUDA tensors: x, y, active_mask, loss_mask"
                )
            expected = ((self.batch, self.seq), (self.batch, self.seq),
                        (self.batch, self.seq), (self.batch, self.seq))
            for index, (buffer, shape) in enumerate(zip(device_buffers, expected)):
                if not isinstance(buffer, torch.Tensor) or buffer.device != device or tuple(buffer.shape) != shape:
                    raise ValueError(f"device_buffers[{index}] has incompatible shape/device")
        epoch, offset = divmod(int(step), self.n_batches)
        permutation = self._epoch_permutation(epoch)
        first = (offset * self.world + self.rank) * self.batch
        document_indices = permutation[first : first + self.batch]
        slot = self._stage_index
        self._stage_index = (slot + 1) % self._stage_count
        if self._pinned and self._stage_used[slot]:
            # The event is recorded after the prior H2D reads from this slot;
            # waiting here protects the pinned buffer before it is rewritten.
            self._stage_events[slot].synchronize()
        x_cpu, y_cpu, active_cpu, loss_cpu = self._staged[slot]
        if not omit_masks:
            x_cpu.zero_()
            y_cpu.zero_()
            active_cpu.zero_()
            loss_cpu.zero_()
        for row, document_index in enumerate(document_indices.tolist()):
            document = self.documents[int(document_index)]
            start = self._window_start(self.seed, epoch, int(document_index), document["length"], self.seq)
            available = min(self.seq, document["length"] - 1)
            span = self.data[document["offset"] + start : document["offset"] + start + available + 1]
            x_cpu[row, :available].copy_(span[:-1])
            y_cpu[row, :available].copy_(span[1:])
            if not omit_masks:
                active_cpu[row, :available] = True
                loss_cpu[row, :available] = True
        if not self._pinned or device.type != "cuda":
            if device_buffers is not None:
                raise ValueError("device_buffers require pinned CUDA staging")
            outputs = [
                x_cpu.to(device=device, non_blocking=True),
                y_cpu.to(device=device, non_blocking=True),
            ]
            if not omit_masks:
                outputs.extend(
                    [
                        active_cpu.to(device=device, non_blocking=True),
                        loss_cpu.to(device=device, non_blocking=True),
                    ]
                )
            else:
                outputs.extend((None, None))
            return tuple(outputs)
        if device_buffers is not None and consumer_stream is not None:
            # The graph may still be reading its persistent input tensors from
            # the previous replay.  Make the DMA producer wait for that graph
            # stream before overwriting the same addresses; the consumer-side
            # event below then establishes the opposite dependency for the
            # new replay.
            self._copy_stream.wait_stream(consumer_stream)
        with torch.cuda.stream(self._copy_stream):
            if device_buffers is None:
                outputs = [
                    x_cpu.to(device=device, non_blocking=True),
                    y_cpu.to(device=device, non_blocking=True),
                ]
                if not omit_masks:
                    outputs.extend(
                        [
                            active_cpu.to(device=device, non_blocking=True),
                            loss_cpu.to(device=device, non_blocking=True),
                        ]
                    )
            else:
                # Direct DMA into the persistent graph inputs.  This removes
                # the temporary GPU allocations and the replay-time D2D copy.
                x_dst, y_dst, active_dst, loss_dst = device_buffers
                x_dst.copy_(x_cpu, non_blocking=True)
                y_dst.copy_(y_cpu, non_blocking=True)
                if not omit_masks:
                    active_dst.copy_(active_cpu, non_blocking=True)
                    loss_dst.copy_(loss_cpu, non_blocking=True)
                outputs = [x_dst, y_dst]
                if not omit_masks:
                    outputs.extend([active_dst, loss_dst])
            self._stage_events[slot].record(self._copy_stream)
        # Asynchronous dependency insertion; the host remains free while the
        # copy stream performs DMA.  CUDA-graph replay can wait on the DMA
        # event directly from its capture stream.  Waiting on the default
        # stream here would force replay to add a second cross-stream
        # dependency (capture_stream.wait_stream(default)) before launching
        # the graph, even though no default-stream work produces these inputs.
        wait_stream = consumer_stream
        if wait_stream is None:
            wait_stream = torch.cuda.current_stream(device)
        wait_stream.wait_event(self._stage_events[slot])
        self._stage_used[slot] = True
        if omit_masks:
            outputs.extend((None, None))
        return tuple(outputs)


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


def _sha256_file(path: Path) -> str:
    """Hash a provenance file in bounded chunks without loading it in RAM."""
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(16 * 1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _hash_installed_distribution(distribution) -> dict:
    """Hash the immutable files recorded by an installed distribution.

    Source-built CUDA packages often do not publish a source commit through
    ``importlib.metadata``.  Their installed ``RECORD`` is still an
    auditable source/build artifact, however: hash every file listed by that
    record and include the path, size, and content digest in a stable manifest.
    Python bytecode caches are not part of a wheel's immutable install and are
    deliberately excluded so importing a module cannot change resume
    provenance.  Missing manifest entries fail closed instead of allowing an
    incomplete package install to masquerade as the reviewed dependency.
    """
    files = distribution.files
    if files is None:
        raise RuntimeError(
            f"installed distribution {distribution.metadata.get('Name')!r} "
            "does not expose a file manifest; cannot record source provenance"
        )
    # A wheel may legitimately install a small launcher outside site-packages
    # (apache-tvm-ffi records ``../../../bin/tvm-ffi-config``).  Keep the
    # traversal check, but scope it to the active virtualenv/prefix rather than
    # rejecting that reviewed launcher or allowing an arbitrary filesystem
    # escape.
    install_prefix = Path(sys.prefix).resolve()
    records = []
    for package_path in files:
        relative = str(package_path).replace("\\", "/")
        pure = PurePosixPath(relative)
        # Absolute RECORD entries are malformed.  Traversal is accepted only
        # when the resolved file remains inside this interpreter's prefix;
        # this covers the TVM FFI launcher while failing closed on arbitrary
        # paths outside the reviewed environment.
        if pure.is_absolute():
            raise RuntimeError(
                f"invalid installed-file path for {distribution.metadata.get('Name')!r}: "
                f"{relative!r}"
            )
        if any(part == "__pycache__" for part in pure.parts) or pure.suffix == ".pyc":
            continue
        path = Path(distribution.locate_file(package_path))
        if path.is_symlink():
            raise RuntimeError(
                f"installed manifest entry is a symlink for "
                f"{distribution.metadata.get('Name')!r}: {relative}"
            )
        resolved_path = path.resolve()
        try:
            resolved_path.relative_to(install_prefix)
        except ValueError as exc:
            raise RuntimeError(
                f"installed manifest entry escapes its environment prefix for "
                f"{distribution.metadata.get('Name')!r}: {relative}"
            ) from exc
        try:
            stat = resolved_path.stat()
        except OSError as exc:
            raise RuntimeError(
                f"installed file is missing for {distribution.metadata.get('Name')!r}: "
                f"{relative}"
            ) from exc
        if not resolved_path.is_file():
            raise RuntimeError(
                f"installed manifest entry is not a regular file for "
                f"{distribution.metadata.get('Name')!r}: {relative}"
            )
        records.append(
            {
                "path": relative,
                "size": int(stat.st_size),
                "sha256": _sha256_file(resolved_path),
            }
        )
    if not records:
        raise RuntimeError(
            f"installed distribution {distribution.metadata.get('Name')!r} "
            "has no immutable files to hash"
        )
    records.sort(key=lambda item: item["path"])
    manifest = json.dumps(records, sort_keys=True, separators=(",", ":")).encode(
        "utf-8"
    )
    return {
        "file_count": len(records),
        "tree_sha256": hashlib.sha256(manifest).hexdigest(),
        "files": records,
    }


def _dependency_provenance() -> dict:
    """Record the installed Python environment used by a checkpoint.

    Binary hashes below identify the loaded CUDA extensions, but the Python
    distribution metadata also matters: a different FlashAttention, Mamba-3, or
    causal-conv wrapper can select a different ABI or call convention while
    leaving the extension filename unchanged.  Keep the complete normalized
    ``pip freeze`` text and a digest of each required distribution's METADATA.
    The source-built packages do not expose authoritative source commits in
    this environment, so this records an immutable installed-tree manifest in
    addition to metadata that is actually present.
    """
    try:
        freeze = subprocess.check_output(
            [sys.executable, "-m", "pip", "freeze", "--all", "--disable-pip-version-check"],
            text=True,
            stderr=subprocess.DEVNULL,
        ).replace("\r\n", "\n")
    except (OSError, subprocess.SubprocessError) as exc:
        raise RuntimeError("unable to capture complete pip freeze provenance") from exc
    if not freeze.strip():
        raise RuntimeError("pip freeze returned an empty dependency manifest")

    distributions = {}
    # Mamba-3's CuteDSL decode wrapper calls the TVM FFI kwargs adapter with
    # ``map_dataclass_to_tuple``.  Include the adapter in checkpoint
    # provenance so a resume cannot silently pick an ABI-incompatible 0.1.9
    # installation (which fails only when the first cached decode compiles).
    for requested_name in (
        "flash-attn",
        "mamba-ssm",
        "causal-conv1d",
        "apache-tvm-ffi",
    ):
        try:
            distribution = importlib.metadata.distribution(requested_name)
        except importlib.metadata.PackageNotFoundError as exc:
            raise RuntimeError(
                f"required installed distribution is missing: {requested_name}"
            ) from exc
        metadata_text = distribution.read_text("METADATA") or ""
        direct_url_text = distribution.read_text("direct_url.json") or ""
        distributions[requested_name] = {
            "name": distribution.metadata.get("Name"),
            "version": distribution.version,
            "metadata_sha256": hashlib.sha256(
                metadata_text.encode("utf-8")
            ).hexdigest(),
            "direct_url_sha256": (
                hashlib.sha256(direct_url_text.encode("utf-8")).hexdigest()
                if direct_url_text
                else None
            ),
            "installed_tree": _hash_installed_distribution(distribution),
        }
    return {
        "pip_freeze": freeze,
        "pip_freeze_sha256": hashlib.sha256(freeze.encode("utf-8")).hexdigest(),
        "distributions": distributions,
    }


def _runtime_provenance(
    data_path: str | os.PathLike[str], manifest: DocumentManifest
):
    """Capture the runtime identity that produced a checkpoint.

    FP8 state is tied to the CUDA/cuDNN/Transformer Engine build.  Recording
    these values makes a resume auditable and lets us reject a checkpoint
    produced by a different kernel/runtime combination instead of silently
    changing the optimization path.
    """
    repo = Path(__file__).resolve().parent
    data_file = Path(data_path).expanduser().resolve()
    if not data_file.is_file():
        raise FileNotFoundError(f"training data is not a regular file: {data_file}")
    data_size = int(data_file.stat().st_size)
    if data_file != manifest.data_path:
        raise ValueError(
            "runtime provenance data path does not match the loaded manifest"
        )
    # _load_manifest already verified the payload size and digest. Reuse that
    # result instead of hashing a 300–500 GB corpus a second time on every DDP
    # rank before training starts.
    if data_size != manifest.data_size:
        raise ValueError("manifest payload size changed after validation")
    data_sha256 = manifest.data_sha256
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
    dependencies = _dependency_provenance()
    return {
        "git_sha": git_sha,
        "git_dirty": git_dirty,
        "git_diff_sha256": git_diff_sha256,
        "transformer_engine": getattr(hnet_te.te, "__version__", None),
        "transformer_engine_path": te_path,
        "transformer_engine_git_commit": hnet_te.transformer_engine_git_commit(),
        "runtime_binaries": hnet_te.runtime_binary_provenance(),
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
        # The byte stream is part of the training function. Record its content
        # identity so a resume cannot silently continue against a replaced or
        # truncated corpus while retaining the same model/optimizer state.
        "data_size": data_size,
        "data_sha256": data_sha256,
        "manifest_path": str(manifest.path),
        "manifest_size": int(manifest.path.stat().st_size),
        "manifest_sha256": manifest.manifest_sha256,
        "manifest_version": 1,
        "manifest_doc_count": len(manifest.documents),
        "manifest_total_token_equivalents": manifest.total_token_equivalents,
        # Record the concrete MoE routing operators used by the canonical
        # trainer.  This prevents a run log from being mistaken for a fused
        # route when only the downstream graph-safe permutation was fused.
        "moe_router": "hnet.modules.latent_moe.FusedTokenRouter",
        "moe_router_kernel": (
            "transformer_engine.pytorch.router.fused_compute_score_for_moe_aux_loss"
        ),
        "moe_dispatch_kernel": "te_moe::permute_mask_map_fwd",
        "dependencies": dependencies,
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
        "runtime_binaries",
        "cublaslt",
        "torch",
        "cuda_runtime",
        "cudnn",
        "fp8_recipe",
        "nvte_fp8_dpa_bwd",
        "nvte_croissattn_aux",
        "moe_router",
        "moe_router_kernel",
        "moe_dispatch_kernel",
        "data_size",
        "data_sha256",
        "manifest_path",
        "manifest_size",
        "manifest_sha256",
        "manifest_version",
        "manifest_doc_count",
        "manifest_total_token_equivalents",
        "dependencies",
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
    saved_schedule = checkpoint.get("lr_schedule")
    if saved_schedule is not None:
        expected_schedule = {
            "name": "WSqD",
            "warmup_steps": int(args.warmup),
            "decay_fraction": float(args.lr_decay_fraction),
            "shift": float(args.lr_shift),
            "horizon_steps": int(getattr(args, "_lr_horizon_steps", args.steps)),
        }
        saved_base = dict(saved_schedule)
        saved_horizon = int(saved_base.pop("horizon_steps", expected_schedule["horizon_steps"]))
        expected_base = dict(expected_schedule)
        expected_base.pop("horizon_steps")
        if saved_base != expected_base or int(args.steps) < saved_horizon:
            raise ValueError(
                "resume learning-rate schedule mismatch: "
                f"checkpoint={saved_schedule!r}, current={expected_schedule!r}"
            )
        # Keep the original horizon for an ordinary resume, but extend it to
        # the requested endpoint when a run is deliberately continued.  Using
        # the old horizon after it has elapsed would make every extension step
        # receive the cooldown's exact zero scale.
        args._lr_horizon_steps = max(saved_horizon, int(args.steps))
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
        if torch.cuda.is_available():
            # Use an explicit logical ordinal even for single-process runs.
            # ``torch.device("cuda")`` has ``index=None``; passing that to
            # power logging would query every physical GPU and defeat the
            # CUDA_VISIBLE_DEVICES mapping in ``read_power_watts``.
            torch.cuda.set_device(0)
            return 0, 1, torch.device("cuda", torch.cuda.current_device())
        return 0, 1, torch.device("cpu")
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
        if getattr(p, "_lerope_frequency", False) or name.endswith(".log_scale"):
            # LeRoPE frequencies are a coordinate system, not model weights:
            # keep them out of decay and in the AdamATan2 scalar path.
            p._no_weight_decay = True
            p._force_fallback = True


@torch.no_grad()
def _clip_lerope_gradients(model, max_norm: float = 1.0):
    """Clip shared LeRoPE frequency gradients independently of model norm."""
    # Parameter identity is stable for the lifetime of a trainer. Cache the
    # filtered list so the per-step clipping path does not rebuild Python
    # lists or walk the full 600M-parameter module tree after every replay.
    parameters = getattr(model, "_hnet_lerope_parameters", None)
    if parameters is None:
        parameters = tuple(
            p for p in model.parameters() if getattr(p, "_lerope_frequency", False)
        )
        setattr(model, "_hnet_lerope_parameters", parameters)
    grads = [p.grad for p in parameters if p.grad is not None]
    if not grads:
        return None
    norm = torch.linalg.vector_norm(torch.stack([g.float().norm() for g in grads]))
    coef = torch.clamp(
        torch.as_tensor(max_norm, device=norm.device, dtype=norm.dtype)
        / (norm + 1e-6),
        max=1.0,
    )
    for grad in grads:
        grad.mul_(coef.to(dtype=grad.dtype))
    return norm


@torch.no_grad()
def _clip_non_lerope_gradients(model, max_norm: float = 1.0):
    """Clip the ordinary model gradients without re-clipping LeRoPE.

    LeRoPE's frequency gradient has a separate norm in the method's reference
    recipe.  Passing all parameters to ``clip_grad_norm_`` after that step
    silently couples the frequency channels back into the global norm and can
    suppress every model update at long context.  Keep this helper shared by
    eager and CUDA-graph training paths.
    """

    parameters = getattr(model, "_hnet_non_lerope_parameters", None)
    if parameters is None:
        parameters = tuple(
            parameter
            for parameter in model.parameters()
            if parameter.requires_grad
            and not getattr(parameter, "_lerope_frequency", False)
        )
        setattr(model, "_hnet_non_lerope_parameters", parameters)
    parameters = [parameter for parameter in parameters if parameter.grad is not None]
    if not parameters:
        return torch.zeros((), device=next(model.parameters()).device)
    # The canonical graph path leaves dense gradients for every trainable
    # parameter. Explicit foreach dispatch batches the norm and scaling work
    # instead of selecting the conservative per-tensor implementation.
    return torch.nn.utils.clip_grad_norm_(parameters, max_norm, foreach=True)


def make_optimizer(model, args):
    # The canonical trainer uses the repository's reviewed optimizer module.
    # Do not silently switch to an alternate scripts/ checkout: a training run
    # must fail loudly if its pinned optimizer implementation is unavailable.
    from optimizer_experiments import AnchorMuon
    annotate_decay(model)
    return AnchorMuon.from_model(
        model,
        lr=args.lr,
        fallback_lr=args.lr * ADAM_ATAN2_LR_RATIO,
        matrix_weight_decay=args.weight_decay,
        fallback_weight_decay=args.weight_decay,
        momentum=0.95,
        pmuoneq_beta=0.90,
        row_gamma=0.45,
        normuon_beta2=0.93,
        soda_lambda_scale=0.003,
        min_matrix_dim=32,
    )


def ce(logits, target, loss_mask=None):
    # TE's CE fixes apply here too: cast logits and reduction inputs to FP32,
    # then reject non-finite values before any optimizer state is mutated.
    flat_logits = logits.float().reshape(-1, logits.shape[-1])
    flat_target = target.long().reshape(-1)
    if loss_mask is None:
        return F.cross_entropy(flat_logits, flat_target)
    if loss_mask.shape != target.shape or loss_mask.dtype is not torch.bool:
        raise ValueError("loss_mask must be bool with the same shape as target")
    per_token = F.cross_entropy(flat_logits, flat_target, reduction="none")
    weights = loss_mask.reshape(-1).to(dtype=per_token.dtype)
    return (per_token * weights).sum() / weights.sum().clamp_min(1.0)


def loss_for_output(out, target, args, loss_mask=None, active_mask=None):
    """Compute the FBT objective and retain every component for diagnostics."""
    final_ce = ce(out.logits, target, loss_mask)
    feedback_values = [ce(x, target, loss_mask) for x in (out.feedback_logits or [])]
    if feedback_values:
        # ``feedback_logits`` contains the earlier plain pass; ``out.logits``
        # is the final feedback-conditioned pass. Keep the public metric names
        # aligned with that meaning so ``loss``/target-ce always refer to the
        # ordinary (plain) byte CE documented for the trainer.
        standard_ce = feedback_values[0]
        feedback_ce = final_ce
        # With two FBT passes, the first-pass logits are retained separately
        # and the final pass is already available as ``out.logits``. This
        # avoids a duplicate final-head evaluation.
        objective = standard_ce + args.fbt_lambda * feedback_ce
        if len(feedback_values) > 1:
            objective = objective + args.fbt_lambda * torch.stack(feedback_values[1:]).mean()
    else:
        standard_ce = final_ce
        feedback_ce = final_ce.new_zeros(())
        objective = final_ce
    ratio = out.logits.new_zeros((), dtype=torch.float32)
    boundary_ratios = []
    soft_boundary_ratios = []
    for router_index, router in enumerate(out.bpred_output):
        router_mask = active_mask if router_index == 0 else None
        ratio = ratio + load_balancing_loss(
            router, args.ratio_target, valid_mask=router_mask
        ).float()
        if router_mask is None:
            boundary_ratios.append(router.boundary_mask.float().mean())
            soft_boundary_ratios.append(router.boundary_prob[..., -1].float().mean())
        else:
            weights = router_mask.to(dtype=torch.float32)
            denominator = weights.sum().clamp_min(1.0)
            boundary_ratios.append(
                (router.boundary_mask.float() * weights).sum() / denominator
            )
            soft_boundary_ratios.append(
                (router.boundary_prob[..., -1].float() * weights).sum() / denominator
            )
    moe_aux_value = getattr(out, "moe_aux_loss", None)
    moe_aux = (moe_aux_value.float() if moe_aux_value is not None
               else out.logits.new_zeros((), dtype=torch.float32))
    moe_balance_value = getattr(out, "moe_balance_loss", None)
    moe_balance = (moe_balance_value.float() if moe_balance_value is not None
                   else out.logits.new_zeros((), dtype=torch.float32))
    objective = objective + args.ratio_weight * ratio + moe_aux
    routing_ratio = (
        torch.stack(boundary_ratios).mean()
        if boundary_ratios
        else standard_ce.new_zeros((), dtype=torch.float32)
    )
    soft_routing_ratio = (
        torch.stack(soft_boundary_ratios).mean()
        if soft_boundary_ratios
        else standard_ce.new_zeros((), dtype=torch.float32)
    )
    target_ratio = getattr(args, "_target_ratio_tensor", None)
    if target_ratio is None or target_ratio.device != standard_ce.device:
        target_ratio = standard_ce.new_tensor(1.0 / float(args.ratio_target))
    expert_load_value = getattr(out, "moe_expert_load", None)
    if expert_load_value is None:
        expert_load_value = out.logits.new_zeros((8,), dtype=torch.float32)
    metrics = {
        "objective": objective,
        "standard_ce": standard_ce,
        "feedback_ce": feedback_ce,
        "router_loss": ratio,
        "moe_aux": moe_aux,
        "moe_balance_loss": moe_balance,
        "moe_expert_load": expert_load_value.float(),
        "routing_ratio": routing_ratio,
        "soft_routing_ratio": soft_routing_ratio,
        "ratio_target_error": (routing_ratio - target_ratio).abs(),
    }
    return objective, metrics


def _lr_phase(step: int, args) -> str:
    """Name the active WSqD phase for human-readable training telemetry."""
    warmup = int(args.warmup)
    if int(step) < warmup:
        return "warmup"
    horizon = max(1, int(getattr(args, "_lr_horizon_steps", args.steps)) - warmup)
    t = max(1, int(step) - warmup + 1)
    stable_horizon = max(
        1, int(math.floor((1.0 - float(args.lr_decay_fraction)) * horizon))
    )
    return "stable" if t <= stable_horizon else "decay"


@torch.no_grad()
def _gini_from_load(load: torch.Tensor) -> torch.Tensor:
    """Compute the expert-load Gini coefficient from an aggregated vector."""
    values = load.detach().float().reshape(-1)
    total = values.sum()
    if values.numel() < 2 or not bool(torch.isfinite(total).item()) or float(total.item()) <= 0.0:
        return values.new_zeros(())
    ordered = torch.sort(values).values
    n = ordered.numel()
    rank = torch.arange(1, n + 1, device=ordered.device, dtype=ordered.dtype)
    return ((2.0 * rank - n - 1.0) * ordered).sum() / (float(n) * total).clamp_min(1.0)


@torch.no_grad()
def _expert_gini(model) -> torch.Tensor:
    """Return mean accepted-route Gini coefficient across LatentMoE layers.

    LatentMoE records the static-shape accepted route counts during its normal
    forward.  Reading those tensors after a graph replay is capture-safe; the
    sort and reduction happen only on the logging stream, outside the graph.
    A perfectly balanced eight-expert layer is zero and a single active expert
    approaches one.  Missing statistics are reported as a zero scalar so a
    fresh graph warmup cannot make logging fail.
    """
    values = []
    for module in model.modules():
        counts = getattr(module, "_expert_counts", None)
        if not isinstance(counts, torch.Tensor) or counts.numel() < 2:
            continue
        counts = counts.detach().float().reshape(-1)
        total = counts.sum()
        if not bool(torch.isfinite(total).item()) or float(total.item()) <= 0.0:
            continue
        ordered = torch.sort(counts).values
        n = ordered.numel()
        rank = torch.arange(1, n + 1, device=ordered.device, dtype=ordered.dtype)
        numerator = ((2.0 * rank - n - 1.0) * ordered).sum()
        values.append(numerator / (float(n) * total).clamp_min(1.0))
    if not values:
        return torch.zeros((), device=next(model.parameters()).device, dtype=torch.float32)
    return torch.stack(values).mean()


def _read_jsonl_records(path: str | os.PathLike[str], limit: int = 0) -> list[dict]:
    """Read a bounded benchmark JSONL file without importing benchmark code."""
    records = []
    with Path(path).expanduser().open("r", encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, 1):
            if limit and len(records) >= int(limit):
                break
            if not line.strip():
                continue
            try:
                record = json.loads(line)
            except json.JSONDecodeError as exc:
                raise ValueError(f"invalid benchmark JSONL at {path}:{line_number}") from exc
            if not isinstance(record, dict):
                raise ValueError(f"benchmark record at {path}:{line_number} is not an object")
            records.append(record)
    return records


def _normalize_mcq_record(record: dict, benchmark: str) -> tuple[str, list[str], int] | None:
    """Normalize MMLU, ARC, and HellaSwag records to prompt/choices/answer."""
    if benchmark == "hellaswag":
        prompt_value = record.get("ctx", record.get("context"))
        if prompt_value is None:
            prompt_value = " ".join(
                str(record.get(key, "")) for key in ("ctx_a", "ctx_b") if record.get(key)
            )
        prompt = str(prompt_value or "")
        choices = record.get("endings", record.get("choices"))
        answer = record.get("label", record.get("answer"))
    else:
        prompt = str(record.get("question", record.get("prompt", "")))
        choices = record.get("choices")
        answer = record.get("answer", record.get("answerKey", record.get("label")))
        if isinstance(choices, dict):
            choices = choices.get("text", choices.get("texts"))
    if not prompt or not isinstance(choices, list) or len(choices) < 2:
        return None
    choices = [str(choice.get("text", "")) if isinstance(choice, dict) else str(choice) for choice in choices]
    if isinstance(answer, str):
        value = answer.strip()
        if len(value) == 1 and value.upper() in "ABCDEFGHIJKLMNOPQRSTUVWXYZ":
            answer = ord(value.upper()) - ord("A")
        else:
            try:
                answer = int(value)
            except ValueError:
                return None
    try:
        answer = int(answer)
    except (TypeError, ValueError):
        return None
    if not 0 <= answer < len(choices):
        return None
    return prompt, choices, answer


@torch.no_grad()
def _score_mcq_records(model, records: list[dict], benchmark: str, device, seq: int) -> tuple[float | None, int]:
    """Score byte-level multiple-choice records with the complete H-Net model."""
    if not records:
        return None, 0
    correct = 0
    count = 0
    for raw in records:
        normalized = _normalize_mcq_record(raw, benchmark)
        if normalized is None:
            continue
        prompt, choices, answer = normalized
        prompt_bytes = prompt.encode("utf-8")
        candidate_bytes = [choice.encode("utf-8") for choice in choices]
        if not prompt_bytes or any(not choice for choice in candidate_bytes):
            continue
        if any(len(prompt_bytes) + len(choice) > int(seq) for choice in candidate_bytes):
            continue
        batch = len(candidate_bytes)
        inputs = torch.zeros((batch, seq), dtype=torch.uint8, device=device)
        masks = torch.zeros((batch, seq), dtype=torch.bool, device=device)
        positions = []
        for row, choice in enumerate(candidate_bytes):
            payload = torch.tensor(list(prompt_bytes + choice), dtype=torch.uint8, device=device)
            inputs[row, : payload.numel()].copy_(payload)
            masks[row, : payload.numel()].fill_(True)
            positions.append((len(prompt_bytes) - 1, len(choice)))
        output = model(inputs, mask=masks, fbt_passes=FBT_PASSES)
        scores = []
        for row, (start, length) in enumerate(positions):
            logits = output.logits[row, start : start + length].float()
            target = inputs[row, start + 1 : start + length + 1].long()
            scores.append(F.cross_entropy(logits, target, reduction="sum").neg())
        if int(torch.stack(scores).argmax().item()) == answer:
            correct += 1
        count += 1
    return (float(correct) / count if count else None), count


@torch.no_grad()
def evaluate_model(model, eval_stream, args, device, benchmark_paths: dict[str, str | None]):
    """Run held-out byte loss and configured MCQ suites on the full model."""
    was_training = model.training
    model.eval()
    previous_graph = hnet_te.graph_capture_enabled()
    # Keep the graph-mode TE flag during evaluation when training uses CUDA
    # graphs.  Besides selecting the same fixed-capacity grouped route, this
    # prevents an eager eval autocast from mutating delayed-scaling histories
    # that the captured training graph consumes on its next replay.
    started = time.perf_counter()
    byte_losses = []
    try:
        for batch_index in range(int(args.eval_batches)):
            x, y, active, loss_mask = eval_stream.get(batch_index, device)
            out = model(x, mask=active, fbt_passes=FBT_PASSES)
            byte_losses.append(float(ce(out.logits, y, loss_mask).item()))
        benchmark_scores = {}
        for name, path in benchmark_paths.items():
            if path is None:
                benchmark_scores[name] = (None, 0)
                continue
            records = _read_jsonl_records(path, int(args.eval_max_items))
            benchmark_scores[name] = _score_mcq_records(
                model, records, name, device, int(args.seq)
            )
    finally:
        hnet_te.set_graph_capture(previous_graph)
        if was_training:
            model.train()
    return {
        "loss": sum(byte_losses) / max(1, len(byte_losses)),
        "mmlu": benchmark_scores.get("mmlu", (None, 0)),
        "arc_challenge": benchmark_scores.get("arc_challenge", (None, 0)),
        "hellaswag": benchmark_scores.get("hellaswag", (None, 0)),
        "eval_took": time.perf_counter() - started,
    }


def _physical_gpu_index(logical_index):
    """Map a CUDA logical ordinal to the nvidia-smi device selector.

    ``torch.cuda`` ordinals are relative to ``CUDA_VISIBLE_DEVICES`` while
    nvidia-smi's numeric selectors are host ordinals.  Passing the logical
    value directly silently samples the wrong GPU whenever a launcher masks or
    reorders devices.  UUID entries are valid nvidia-smi selectors too, so
    preserve them rather than attempting to parse them as integers.
    """
    if logical_index is None:
        return None
    logical_index = int(logical_index)
    visible = os.environ.get("CUDA_VISIBLE_DEVICES")
    if not visible:
        return str(logical_index)
    entries = [item.strip() for item in visible.split(",") if item.strip()]
    if 0 <= logical_index < len(entries):
        return entries[logical_index]
    # A malformed mask should not make logging crash after a successful step;
    # let nvidia-smi return its normal error and report an empty sample.
    return str(logical_index)


def read_power_watts(device_index=None):
    try:
        query = ["nvidia-smi"]
        if device_index is not None:
            query.extend(["-i", _physical_gpu_index(device_index)])
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


@torch.no_grad()
def _clip_direct_ddp_storage(
    buckets, max_norm: float = 1.0, frequency_parameters=()
):
    """Clip contiguous DDP storage while preserving a separate LeRoPE norm.

    Reducer views put the tiny FP32 LeRoPE tensor in a shared storage bucket.
    Save and restore those views around the bucket-wide scale so the ordinary
    norm never changes their independently clipped gradient. Return
    ``(ordinary_norm, frequency_norm_or_None)`` so the caller can perform the
    same finite check even when no frequency gradient is present.
    """
    if not buckets:
        raise ValueError("direct DDP clipping requires at least one storage bucket")
    frequency_parameters = tuple(
        parameter
        for parameter in frequency_parameters
        if parameter.grad is not None
    )
    # The caller historically clipped LeRoPE first, then this helper clipped
    # the ordinary bucket while saving/restoring the already-scaled frequency
    # views. Do that first step here while retaining the same arithmetic and
    # storage ordering.
    frequency_norm = None
    if frequency_parameters:
        frequency_grads = tuple(parameter.grad for parameter in frequency_parameters)
        frequency_norm = torch.linalg.vector_norm(
            torch.stack([gradient.float().norm() for gradient in frequency_grads])
        )
        frequency_coef = torch.clamp(
            torch.as_tensor(max_norm, device=frequency_norm.device, dtype=frequency_norm.dtype)
            / (frequency_norm + 1e-6),
            max=1.0,
        )
        for gradient in frequency_grads:
            gradient.mul_(frequency_coef.to(dtype=gradient.dtype))
    saved_frequency = tuple(
        (parameter, parameter.grad.detach().clone())
        for parameter in frequency_parameters
    )
    # The direct storage buckets do not retain parameter boundaries.  Compute
    # one norm per bucket, subtracting the tiny frequency-vector contribution;
    # this preserves the independent LeRoPE norm without launching a norm
    # kernel for every parameter tensor.
    norms = torch.stack(torch._foreach_norm(list(buckets), 2.0)).float()
    total_sq = norms.square().sum()
    if saved_frequency:
        frequency_sq = torch.stack(
            [gradient.float().norm().square() for _, gradient in saved_frequency]
        ).sum()
        total_sq = total_sq.sub(frequency_sq).clamp_min(0.0)
    total_norm = total_sq.sqrt()
    clip_coef = torch.clamp(
        torch.as_tensor(max_norm, dtype=total_norm.dtype, device=total_norm.device)
        / (total_norm + 1e-6),
        max=1.0,
    )
    for bucket in buckets:
        bucket.mul_(clip_coef.to(dtype=bucket.dtype))
    for parameter, gradient in saved_frequency:
        parameter.grad.copy_(gradient)
    # Keep one return shape even if a future graph has no frequency gradient;
    # callers can still apply the finite check without a conditional unpack.
    return total_norm, frequency_norm


def _pack_gradients_by_dtype(parameters, device, buckets=None):
    """Pack gradients into persistent flat dtype buckets without collectives.

    Keeping packing separate from the NCCL calls lets calibration put a
    synchronized OOM gate around every allocation.  A rank that runs out of
    memory while building a bucket must not leave healthy ranks entering an
    all-reduce that it can never reach.
    """
    if buckets is None:
        buckets = {
            torch.bfloat16: {"params": [], "buffer": None},
            torch.float32: {"params": [], "buffer": None},
        }
        for parameter in parameters:
            if not parameter.requires_grad:
                continue
            if parameter.dtype not in buckets:
                raise RuntimeError(
                    f"unsupported DDP gradient dtype {parameter.dtype}; "
                    "the canonical graph expects BF16 or FP32 parameters"
                )
            buckets[parameter.dtype]["params"].append(parameter)

    for dtype, bucket in buckets.items():
        params = bucket["params"]
        if not params:
            continue
        total = sum(parameter.numel() for parameter in params)
        flat = bucket["buffer"]
        if flat is None or flat.numel() != total:
            flat = torch.empty((total,), device=device, dtype=dtype)
            bucket["buffer"] = flat
        offset = 0
        for parameter in params:
            count = parameter.numel()
            view = flat[offset : offset + count]
            if parameter.grad is None:
                view.zero_()
            else:
                view.copy_(parameter.grad.detach().reshape(-1))
            offset += count
    return buckets


def _reduce_packed_gradients_by_dtype(buckets, world):
    """Average already-packed gradients with one flat NCCL reduction per dtype."""
    for bucket in buckets.values():
        params = bucket["params"]
        if not params:
            continue
        flat = bucket["buffer"]
        # NCCL performs the world-size normalization in its AVG reduction;
        # this avoids a full-buffer divide kernel after communication.  Keep
        # the same averaging contract as the direct DDP graph path.
        dist.all_reduce(flat, op=dist.ReduceOp.AVG)
        offset = 0
        for parameter in params:
            count = parameter.numel()
            view = flat[offset : offset + count].view_as(parameter)
            if parameter.grad is None:
                parameter.grad = view
            else:
                parameter.grad.copy_(view)
            offset += count


def _average_gradients_by_dtype(parameters, world, device, buckets=None):
    """Average gradients with one flat NCCL reduction per parameter dtype.

    Calibration runs outside the captured graph but must measure the same
    communication shape as production DDP replay. Reducing each parameter
    separately creates hundreds of tiny NCCL launches and overstates the
    steady-state step time. The returned bucket state keeps the flat buffers
    allocated across calibration iterations.
    """
    buckets = _pack_gradients_by_dtype(parameters, device, buckets)
    _reduce_packed_gradients_by_dtype(buckets, world)
    return buckets


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
    # Keep a generous memory reserve for the real optimizer and DDP buckets.
    # The canonical no-checkpoint B=16 graph is validated on B300; smaller
    # devices receive a proportionally lower bound through calibration.
    if int(args.auto_batch_max) < 1:
        raise ValueError("--auto-batch-max must be positive")
    # The full model keeps its activation graph for the canonical B=16 shape;
    # calibration catches a synchronized OOM before any collective that would
    # deadlock a DDP rank. Use the requested bound directly; a conservative
    # memory heuristic would reject the validated graph batch on B300.
    max_batch = min(max(1, int(args.auto_batch_max)), CANONICAL_MAX_BATCH)
    manifest_doc_count = int(getattr(args, "_manifest_doc_count", 0))
    if manifest_doc_count <= 0:
        raise RuntimeError("manifest document count must be set before batch calibration")
    # One document-local window occupies each batch row. Bound the per-rank
    # candidate by the number of complete global rows, rather than payload
    # bytes, so calibration cannot select a shape the manifest cannot supply.
    available_batches = (
        manifest_doc_count + max(1, world) - 1
    ) // max(1, world)
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
    if candidates[-1] != max_batch:
        candidates.append(max_batch)

    target_model = model.module if isinstance(model, DDP) else model
    # Calibration must not invoke DDP's reducer at all.  ``no_sync`` still
    # installs reducer autograd hooks, and repeated static-graph probes can
    # trip PyTorch's reducer state assertion before the explicit OOM gate.
    # Probe the underlying module directly, then average its gradients with
    # the explicit collectives below after every rank passes the gate.
    probe_model = target_model
    calibration_grad_buckets = None
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
    # Calibration performs real eager updates so delayed-scaling histories and
    # power readings match the non-captured kernel path. CUDA graph capture is
    # enabled again before the selected shape is handed to CUDAGraphStep.
    graph_was_enabled = hnet_te.graph_capture_enabled()
    hnet_te.set_graph_capture(False)
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
                    probe_model.zero_grad(set_to_none=True)
                    iteration_oom = False
                    try:
                        # Probe the complete local forward/backward without
                        # entering DDP reducer collectives.  If one rank
                        # OOMs, healthy ranks can still reach the OOM gate
                        # below and report it collectively.
                        out = probe_model(x, mask=mask, fbt_passes=FBT_PASSES)
                        loss, metrics = loss_for_output(out, y, args)
                        loss.backward()
                    except RuntimeError as exc:
                        if "out of memory" not in str(exc).lower():
                            raise
                        iteration_oom = True
                        # Release the failed candidate's allocator blocks
                        # before entering the shared OOM gate.  Other ranks
                        # remain in ``no_sync`` and can therefore reach the
                        # same collective without waiting in DDP's reducer.
                        torch.cuda.empty_cache()
                    # Every rank reports allocation failure before any rank
                    # enters the finite-gradient collective. This prevents a
                    # single-rank OOM from leaving the other DDP ranks parked
                    # in NCCL until the launcher timeout.
                    oom_tensor = torch.tensor(
                        [int(iteration_oom)], device=device, dtype=torch.int32
                    )
                    if world > 1:
                        dist.all_reduce(oom_tensor, op=dist.ReduceOp.MAX)
                    if bool(oom_tensor.item()):
                        oom = True
                        break
                    # Reject a non-finite local objective before reducing its
                    # gradients. This keeps the collective safe while making
                    # the subsequent calibration update match production's
                    # averaged-gradient ordering.
                    finite_flag = torch.isfinite(loss).to(dtype=torch.int32)
                    if world > 1:
                        dist.all_reduce(finite_flag, op=dist.ReduceOp.MIN)
                    if not bool(finite_flag.item()):
                        raise FloatingPointError(
                            f"non-finite calibration update at candidate batch {candidate}"
                        )
                    if world > 1:
                        # Packing is separate from the NCCL calls so an OOM
                        # while allocating a flat bucket is reported by every
                        # rank before any healthy rank enters all_reduce.
                        pack_oom = False
                        try:
                            calibration_grad_buckets = _pack_gradients_by_dtype(
                                probe_model.parameters(), device, calibration_grad_buckets
                            )
                        except RuntimeError as exc:
                            if "out of memory" not in str(exc).lower():
                                raise
                            pack_oom = True
                            torch.cuda.empty_cache()
                        pack_flag = torch.tensor(
                            [int(pack_oom)], device=device, dtype=torch.int32
                        )
                        dist.all_reduce(pack_flag, op=dist.ReduceOp.MAX)
                        if bool(pack_flag.item()):
                            oom = True
                            break
                        # All ranks have now proved that their packed buffers
                        # exist, so no rank can strand the others before this
                        # collective because of a local allocation failure.
                        _reduce_packed_gradients_by_dtype(
                            calibration_grad_buckets, world
                        )
                    clip_oom = False
                    try:
                        lerope_norm = _clip_lerope_gradients(probe_model, 1.0)
                        grad_norm = _clip_non_lerope_gradients(probe_model, 1.0)
                    except RuntimeError as exc:
                        if "out of memory" not in str(exc).lower():
                            raise
                        clip_oom = True
                        grad_norm = torch.zeros((), device=device)
                        torch.cuda.empty_cache()
                    clip_flag = torch.tensor(
                        [int(clip_oom)], device=device, dtype=torch.int32
                    )
                    if world > 1:
                        dist.all_reduce(clip_flag, op=dist.ReduceOp.MAX)
                    if bool(clip_flag.item()):
                        oom = True
                        break
                    finite_grad = torch.isfinite(grad_norm)
                    if lerope_norm is not None:
                        finite_grad = finite_grad & torch.isfinite(lerope_norm)
                    finite_grad_flag = finite_grad.to(dtype=torch.int32)
                    if world > 1:
                        dist.all_reduce(finite_grad_flag, op=dist.ReduceOp.MIN)
                    if not bool(finite_grad_flag.item()):
                        raise FloatingPointError(
                            f"non-finite calibration gradient at candidate batch {candidate}"
                        )
                    step_oom = False
                    try:
                        optimizer.step()
                        # Surface asynchronous allocation failures before the
                        # rank reaches the shared stage gate.
                        torch.cuda.synchronize(device)
                    except RuntimeError as exc:
                        if "out of memory" not in str(exc).lower():
                            raise
                        step_oom = True
                        torch.cuda.empty_cache()
                    step_flag = torch.tensor(
                        [int(step_oom)], device=device, dtype=torch.int32
                    )
                    if world > 1:
                        dist.all_reduce(step_flag, op=dist.ReduceOp.MAX)
                    if bool(step_flag.item()):
                        oom = True
                        break
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
                probe_model.zero_grad(set_to_none=True)
                # Restore both parameters and optimizer moments before the next
                # candidate.  This is required because GramNS/AdamATan2 state
                # and master copies otherwise make later measurements depend on
                # the preceding batch size.
                target_model.load_state_dict(initial_state, strict=True)
                optimizer.state.clear()
                for group in optimizer.param_groups:
                    group["k"] = 0
                hnet_te.reset_fp8_scaling_state(target_model)
                del x, y, mask, out, loss, metrics
                if oom:
                    torch.cuda.empty_cache()
                else:
                    # Release allocator slabs between shapes so a larger
                    # candidate is tested against actual free memory rather
                    # than fragmentation from the previous candidate.
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
                # A per-GPU power limit is a worst-rank constraint. Selecting
                # from the mean could push one rank past the target while the
                # aggregate still appears safe.
                dist.all_reduce(power_tensor, op=dist.ReduceOp.MAX)
            peak_power = float(power_tensor.item())
            measurements.append({"batch": candidate, "power_w": peak_power})
            if rank == 0:
                print(
                    f"auto_batch candidate={candidate} peak_power_w={peak_power:.1f}",
                    flush=True,
                )
            if peak_power >= float(args.target_power_w):
                # Power is monotonic enough across the geometric batch sweep;
                # do not spend more full updates on candidates that are
                # already above the per-GPU safety target.
                break
    finally:
        target_model.load_state_dict(initial_state, strict=True)
        optimizer.state.clear()
        for group in optimizer.param_groups:
            group["k"] = 0
        hnet_te.reset_fp8_scaling_state(target_model)
        torch.set_rng_state(cpu_rng)
        torch.cuda.set_rng_state(cuda_rng, device=device)
        hnet_te.set_graph_capture(graph_was_enabled)

    if not measurements:
        raise RuntimeError("--auto-batch could not complete a full FP8 calibration step")
    target = float(args.target_power_w)
    below_target = [item for item in measurements if item["power_w"] <= target]
    if below_target:
        # If the whole sweep is below the requested draw, the closest point is
        # the wrong policy: use the largest safe candidate and record any
        # remaining gap explicitly. This is also safe when the sweep brackets
        # the target because no known rank exceeds it.
        selected = max(below_target, key=lambda item: item["batch"])
        selection_policy = "largest_below_target"
    else:
        # If even batch 1 exceeds the target, preserve the smallest validated
        # shape so the trainer does not knowingly overdraw the GPU.
        selected = min(measurements, key=lambda item: item["batch"])
        selection_policy = "smallest_above_target"
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
    outside avoids capturing optimizer state initialization. For DDP, capture
    runs with reducer synchronization disabled; replay reduces retained DDP
    bucket views directly when possible, and otherwise uses one flat all-reduce
    per parameter dtype. This keeps the runtime logger out of the graph without
    creating per-parameter collectives.
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
        # DDP is used for construction and eager warmup so its reducer hooks
        # are initialized against the complete parameter set.  Captured
        # replay uses the underlying module after that setup: the wrapper's
        # per-forward logger/DDPSink path is not graph-safe and is redundant
        # because replay performs the explicit post-capture gradient reduction.
        self.forward_model = model.module if self.ddp else model
        # DDP's reducer only tracks trainable parameters.  Filtering here keeps
        # the flat post-capture collective aligned with the reducer and avoids
        # presence bookkeeping for the frozen byte-code buffer.
        self._ddp_params = (
            [param for param in model.parameters() if param.requires_grad]
            if self.ddp
            else []
        )
        # Keep reduction buffers in the parameter precision.  The prior
        # implementation converted every BF16 gradient to one giant FP32
        # buffer (about 2.4 GiB for this model) before NCCL all-reduce, then
        # converted it back.  BF16 gradients are the normal DDP contract and
        # reducing them directly removes the conversion bandwidth and makes
        # the captured step's synchronization considerably cheaper.  The few
        # FP32 parameters retain an FP32 reduction bucket.
        self._ddp_grad_buckets = {
            torch.bfloat16: {"params": [], "buffer": None},
            torch.float32: {"params": [], "buffer": None},
        }
        self._ddp_storage_buckets = None
        self._ddp_direct_storage = False
        if self.ddp:
            for param in self._ddp_params:
                if param.dtype not in self._ddp_grad_buckets:
                    raise RuntimeError(
                        f"unsupported DDP gradient dtype {param.dtype}; "
                        "the canonical graph expects BF16 or FP32 parameters"
                    )
                self._ddp_grad_buckets[param.dtype]["params"].append(param)
        # Materialize scalar diagnostics before capture.  Constructing a new
        # tensor from a Python float inside a CUDA capture is rejected by the
        # runtime even though the arithmetic itself is graph safe.
        args._target_ratio_tensor = torch.tensor(
            1.0 / float(args.ratio_target), dtype=torch.float32, device=device
        )
        self.static_x = torch.empty((batch, seq), dtype=torch.uint8, device=device)
        self.static_y = torch.empty((batch, seq), dtype=torch.uint8, device=device)
        self.static_mask = torch.ones((batch, seq), dtype=torch.bool, device=device)
        self.static_loss_mask = torch.ones((batch, seq), dtype=torch.bool, device=device)
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
                # Keep the DDP reducer's gradient-as-bucket views alive.  The
                # graph replays the underlying module directly (the DDP
                # wrapper performs host-side logger work that is not capture
                # safe), so these views are the only way the captured
                # AccumulateGrad nodes can write directly into coalesced
                # reducer storage.  Dropping them here makes every parameter
                # allocate an independent gradient and turns the post-replay
                # reduction into hundreds of NCCL launches.
                self.optimizer.zero_grad(set_to_none=False)
                warm_out = self.model(warm_x, mask=self.static_mask, fbt_passes=FBT_PASSES)
                warm_loss, _ = loss_for_output(
                    warm_out,
                    warm_y,
                    args,
                    loss_mask=self.static_loss_mask,
                    active_mask=self.static_mask,
                )
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
            # for capture and perform the explicit gradient reduction after
            # replay; this keeps the captured graph free of logger/NCCL setup.
            self.model.require_backward_grad_sync = False
        hnet_te.set_graph_capture(True)
        # TE's DPA backend key includes its ``cuda_graph`` flag.  A normal
        # eager warmup therefore leaves the first graph replay trying to run
        # the cuDNN backend selector while the stream is already capturing;
        # cuDNN handle creation is not capture safe.  Mark one graph-mode
        # forward outside the raw CUDA capture to populate TE's backend cache
        # with the exact graph key, then capture only the already-selected
        # kernels below.
        from transformer_engine.pytorch import graph as te_graph
        backend_warmup_context = (
            torch.cuda.stream(self.capture_stream)
            if self.capture_stream is not None
            else contextlib.nullcontext()
        )
        te_graph.set_capture_start()
        try:
            with backend_warmup_context:
                graph_backend_warmup = self.forward_model(
                    self.static_x, mask=self.static_mask, fbt_passes=FBT_PASSES
                )
                graph_backend_loss, _graph_backend_metrics = loss_for_output(
                    graph_backend_warmup,
                    self.static_y,
                    self.args,
                    loss_mask=self.static_loss_mask,
                    active_mask=self.static_mask,
                )
                # The backward fused-attention selector has the same lazy
                # cuDNN handle as forward. Exercise it under TE's graph-mode
                # key before entering raw CUDA capture, then discard the
                # warmup graph.
                graph_backend_loss.backward()
                # Preserve the reducer-owned gradient views through the
                # graph-backend warmup so capture reuses the same storage.
                self.optimizer.zero_grad(set_to_none=False)
                del graph_backend_warmup
                del graph_backend_loss, _graph_backend_metrics
                torch.cuda.synchronize(self.device)
            # A model may be recaptured after a restore or allocator reset.
            # Clear the per-module first-microbatch markers so this capture
            # refreshes every FP8 weight workspace once before replaying the
            # remaining recurrent/FBT executions with cached weights.
            hnet_te.reset_graph_weight_cache_flags(self.forward_model)
            # Capture on the same non-default stream used for DDP construction
            # and warmup. Keep TE's graph flag active for this raw capture:
            # ending it after backend warmup would let outermost FP8 contexts
            # record backward amax reductions and scale updates inside the
            # graph, then update them a second time on replay.
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
                    graph_out = self.forward_model(
                        self.static_x, mask=self.static_mask, fbt_passes=FBT_PASSES
                    )
                    self.loss, self.metrics = loss_for_output(
                        graph_out,
                        self.static_y,
                        args,
                        loss_mask=self.static_loss_mask,
                        active_mask=self.static_mask,
                    )
                    self.loss.backward()
        except Exception as exc:
            raise RuntimeError(
                "canonical CUDA graph capture failed; inspect the TE/CUDA graph "
                f"trace rather than silently running eager mode: {exc}"
            ) from exc
        finally:
            te_graph.set_capture_end()
        torch.cuda.synchronize(device)
        # Seed the first replay with the calibrated amax history generated by
        # capture. Subsequent replays commit both histories below.
        hnet_te.update_fp8_scaling(forward=True)
        hnet_te.update_fp8_scaling(forward=False)
        if self.ddp:
            self._prepare_ddp_storage_buckets()

    def _prepare_ddp_storage_buckets(self):
        """Validate and retain direct views of DDP's gradient bucket storage.

        ``gradient_as_bucket_view=True`` makes each captured parameter gradient
        a view into one of a few contiguous reducer buckets.  When every rank
        exposes the same dtype/size sequence, reducing those storage views in
        place is algebraically identical to the copy/reduce/copy-back path and
        avoids moving the full model through temporary buffers.  The one-time
        metadata check is deliberately conservative; any mismatch keeps the
        existing flat-buffer implementation.
        """
        groups = []
        seen = set()
        local_ok = True
        signature = []
        dtype_codes = {torch.bfloat16: 1, torch.float32: 2}
        for param in self._ddp_params:
            grad = param.grad
            if (
                grad is None
                or not grad.is_contiguous()
                or grad.dtype not in dtype_codes
            ):
                local_ok = False
                break
            storage = grad.untyped_storage()
            key = (storage.data_ptr(), grad.dtype)
            if key in seen:
                continue
            element_size = grad.element_size()
            nbytes = storage.nbytes()
            if nbytes <= 0 or nbytes % element_size:
                local_ok = False
                break
            flat = torch.empty(0, dtype=grad.dtype, device=self.device).set_(
                storage,
                0,
                (nbytes // element_size,),
                (1,),
            )
            seen.add(key)
            groups.append(flat)
            signature.append((dtype_codes[grad.dtype], int(flat.numel())))

        # A compact device-side signature avoids an object collective and lets
        # all ranks choose the same branch before any gradient collective.
        sig_value = 1469598103934665603
        for dtype_code, numel in signature:
            sig_value = (sig_value ^ (dtype_code * 1000003 + numel))
            sig_value = (sig_value * 1099511628211) % (2**63 - 1)
        meta = torch.tensor(
            [int(local_ok), len(signature), sig_value],
            device=self.device,
            dtype=torch.int64,
        )
        meta_min = meta.clone()
        meta_max = meta.clone()
        dist.all_reduce(meta_min, op=dist.ReduceOp.MIN)
        dist.all_reduce(meta_max, op=dist.ReduceOp.MAX)
        compatible = bool(local_ok) and bool(torch.equal(meta_min, meta_max))
        # A graph capture that did not preserve reducer views can expose one
        # storage per parameter.  Treat that layout as incompatible rather
        # than issuing hundreds of tiny collectives; the flat dtype buckets in
        # _sync_ddp_grads are slower than direct reducer views but remain one
        # collective per dtype and preserve the update exactly.
        if compatible and len(groups) > 16:
            compatible = False
        self._ddp_direct_storage = compatible
        self._ddp_storage_buckets = groups if compatible else None

    def replay(self, x, y, active_mask=None, loss_mask=None):
        # The loader can DMA directly into these persistent graph inputs.  In
        # that mode all four objects are already at their captured addresses;
        # avoid launching four redundant device-to-device copies.  The
        # identity checks intentionally fall back to the old copies for
        # callers that supply ordinary tensors (calibration/tests).
        direct_inputs = x is self.static_x and y is self.static_y
        direct_masks = (
            active_mask is None or active_mask is self.static_mask
        ) and (loss_mask is None or loss_mask is self.static_loss_mask)
        if self.capture_stream is None:
            if not direct_inputs:
                self.static_x.copy_(x)
                self.static_y.copy_(y)
            if active_mask is not None and not direct_masks:
                self.static_mask.copy_(active_mask)
            if loss_mask is not None and not direct_masks:
                self.static_loss_mask.copy_(loss_mask)
            self.graph.replay()
        else:
            current_stream = torch.cuda.current_stream(self.device)
            self.capture_stream.wait_stream(current_stream)
            with torch.cuda.stream(self.capture_stream):
                if not direct_inputs:
                    self.static_x.copy_(x)
                    self.static_y.copy_(y)
                if active_mask is not None and not direct_masks:
                    self.static_mask.copy_(active_mask)
                if loss_mask is not None and not direct_masks:
                    self.static_loss_mask.copy_(loss_mask)
                self.graph.replay()
            current_stream.wait_stream(self.capture_stream)
        # Delayed scaling updates are deliberately outside the captured region;
        # the next replay consumes the histories produced by this one.  Direct
        # DDP storage uses a fixed, small set of contiguous buckets.  Launch
        # those NCCL reductions asynchronously before the TE amax reductions:
        # FP8 scaling only reads/writes TE's amax and scale buffers, so it is
        # independent of the parameter-gradient buckets and can overlap the
        # communication.  Wait before returning so clipping and the optimizer
        # always see averaged gradients.  The conservative flat-buffer path
        # keeps the original ordering because its host copies are not safely
        # overlappable with the scaling updates.
        if self.ddp and self._ddp_direct_storage:
            # The reducer exposes six large contiguous views (five BF16 and
            # one FP32 in the canonical model). NCCL coalesced all-reduce
            # requires one dtype per group, so coalesce each dtype separately
            # rather than passing mixed BF16/FP32 tensors to one group.
            pending = []
            by_dtype = {}
            for flat in self._ddp_storage_buckets:
                by_dtype.setdefault(flat.dtype, []).append(flat)
            for flats in by_dtype.values():
                if len(flats) == 1:
                    pending.append(
                        dist.all_reduce(
                            flats[0], op=dist.ReduceOp.AVG, async_op=True
                        )
                    )
                else:
                    # Use the explicit homogeneous collective.  The private
                    # context manager can retain operations across nested
                    # dtype groups on some NCCL versions, causing the next
                    # FP32 group to be checked together with BF16 tensors.
                    pending.append(
                        dist.all_reduce_coalesced(
                            flats, op=dist.ReduceOp.AVG, async_op=True
                        )
                    )
            hnet_te.update_fp8_scaling(forward=True)
            hnet_te.update_fp8_scaling(forward=False)
            for work in pending:
                work.wait()
            # NCCL's AVG reduction includes the world-size normalization, so
            # avoid a full-buffer divide kernel after communication.
        else:
            hnet_te.update_fp8_scaling(forward=True)
            hnet_te.update_fp8_scaling(forward=False)
            if self.ddp:
                self._sync_ddp_grads()
        return self.loss, self.metrics

    def _sync_ddp_grads(self):
        """Average captured local gradients without entering the graph.

        Reduce one flat buffer per parameter precision.  This preserves the
        usual BF16 DDP communication contract while avoiding a full-model
        BF16-to-FP32 staging copy and its matching copy back to every
        parameter.  All ranks use the same parameter ordering and bucket sizes
        because they construct the model from the same canonical config.
        """

        if self._ddp_direct_storage:
            for flat in self._ddp_storage_buckets:
                dist.all_reduce(flat, op=dist.ReduceOp.AVG)
            return

        for dtype, bucket in self._ddp_grad_buckets.items():
            params = bucket["params"]
            if not params:
                continue
            total = sum(param.numel() for param in params)
            flat = bucket["buffer"]
            if flat is None or flat.numel() != total:
                flat = torch.empty((total,), device=self.device, dtype=dtype)
                bucket["buffer"] = flat
            offset = 0
            for param in params:
                count = param.numel()
                view = flat[offset : offset + count]
                if param.grad is None:
                    view.zero_()
                else:
                    view.copy_(param.grad.detach().reshape(-1))
                offset += count
            dist.all_reduce(flat, op=dist.ReduceOp.AVG)
            offset = 0
            for param in params:
                count = param.numel()
                view = flat[offset : offset + count].view_as(param)
                if param.grad is None:
                    param.grad = view
                else:
                    param.grad.copy_(view)
                offset += count


class CUDAGraphOptimizerStep:
    """Replay-safe AnchorMuon/AdamATan2 update.

    The forward/backward graph leaves gradients in stable storage.  After one
    ordinary update initializes the optimizer state, this helper captures only
    the optimizer update into the same CUDA graph pool.  Learning rate and
    update count are device scalars, so schedule decay and Adam bias
    correction continue to change on every replay without Python launches.
    """

    def __init__(self, optimizer, model_graph: CUDAGraphStep, device, lr: float, step: int):
        if device.type != "cuda":
            raise RuntimeError("graph optimizer requires CUDA")
        self.optimizer = optimizer
        self.device = device
        self.lr = torch.tensor(float(lr), device=device, dtype=torch.float32)
        # ``optimizer.step`` increments this inside the captured graph.  The
        # caller has already completed ``step`` eager updates before capture.
        self.step = torch.tensor(float(step), device=device, dtype=torch.float32)
        optimizer.set_graph_scalars(self.lr, self.step)
        self.graph = torch.cuda.CUDAGraph()
        self.pool = model_graph.graph.pool()

    def capture(self) -> None:
        with torch.cuda.graph(self.graph, pool=self.pool):
            self.optimizer.step()
        torch.cuda.synchronize(self.device)

    def set_lr(self, value: float) -> None:
        self.lr.fill_(float(value))

    def replay(self) -> None:
        self.graph.replay()

    def sync_python_state(self) -> None:
        """Materialize graph counters before checkpointing or process exit."""
        current_step = int(self.step.item())
        for group in self.optimizer.param_groups:
            group["k"] = current_step
            for parameter in group["params"]:
                state = self.optimizer.state.get(parameter)
                if state is not None and "exp_avg_sq" in state:
                    state["step"] = current_step


def main():
    args = parse_args()
    _validate_schedule_args(args)
    args._lr_horizon_steps = int(args.steps)
    # Select the graph-safe TE grouped-tensor dispatch before constructing the
    # model.  The capture helper temporarily disables it for eager warmup and
    # restores it for the graph itself.
    hnet_te.set_graph_capture(CANONICAL_CUDA_GRAPHS)
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
    # The canonical configuration already fixes max_seq_len=4096 and the
    # corresponding 512-latent MoE capacity.  Keep those values immutable so
    # a production run cannot quietly become a shorter partial model.
    cfg.validate_canonical()
    minimum_ratio_target = float(args.seq) / float(cfg.chunk_capacity_tokens)
    if float(args.ratio_target) < minimum_ratio_target:
        raise ValueError(
            "ratio-target is incompatible with the canonical latent capacity: "
            f"need at least {minimum_ratio_target:g} for seq={args.seq} and "
            f"capacity={cfg.chunk_capacity_tokens}, got {args.ratio_target:g}"
        )
    manifest = _load_manifest(args.manifest, args.data)
    args.data = str(manifest.data_path)
    args._manifest_doc_count = len(manifest.documents)
    runtime = _runtime_provenance(args.data, manifest)
    model = HNetForByteCausalLM(cfg, device=device, dtype=torch.bfloat16).train()
    n_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    if rank == 0:
        print(f"H-Net params={n_params:,} trainable; byte_lift_params=0; world={world}", flush=True)
    graph_stream = (
        torch.cuda.Stream(device=device)
        if CANONICAL_CUDA_GRAPHS and world > 1
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
                static_graph=CANONICAL_CUDA_GRAPHS,
            )
    optimizer = make_optimizer(model.module if world > 1 else model, args)
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
        saved_batch = resume_state.get("batch")
        if saved_batch is None or int(saved_batch) < 1:
            raise ValueError("resume checkpoint is missing a valid calibrated batch")
        # A resume must replay the exact static shape selected by its original
        # power calibration.  Re-running the sweep here is both expensive and
        # unsafe: power noise could choose a different batch, causing the
        # subsequent provenance check (and the data order) to disagree with
        # the checkpoint.
        args.batch = int(saved_batch)
        args._auto_batch_calibration = []
        args._auto_batch_selection_policy = "checkpoint_batch"
        _validate_resume(resume_state, cfg, args, n_params, world, runtime)
    else:
        # Calibrate after DDP and optimizer construction so the measured step
        # includes gradient synchronization and the real AnchorMuon update.
        # The canonical trainer has no partial/manual-batch execution path.
        calibration_context = (
            torch.cuda.stream(graph_stream)
            if graph_stream is not None
            else contextlib.nullcontext()
        )
        # DDP construction and its AccumulateGrad hooks run on graph_stream.
        # Keep the direct-module calibration on that same stream; switching to
        # the default stream here creates avoidable stream-mismatch waits before
        # the later graph capture even though calibration bypasses the wrapper.
        with calibration_context:
            args.batch = calibrate_batch_to_power(
                model, optimizer, args, device, rank, world
            )

    if resume_state is not None:
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
    stream = ByteStream(manifest, args.seq, args.batch, rank, world, seed=args.data_seed)
    eval_manifest = manifest if args.eval_manifest is None else _load_manifest(args.eval_manifest)
    eval_stream = ByteStream(
        eval_manifest, args.seq, args.batch, rank, world, seed=args.data_seed + 1_000_003
    )
    graph_step = None
    if CANONICAL_CUDA_GRAPHS:
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
    graph_optimizer_step = None
    # Fresh calibration clears optimizer state. The first finite update is
    # therefore eager and initializes AnchorMuon/AdamATan2. A resumed
    # optimizer also needs one eager update before capture: AnchorMuon lazily
    # creates GramNS identities, fused pointer tables, and update buffers from
    # the first observed gradient. Capturing that allocation would make the
    # resume path graph-unsafe and can fail only after a long capture.
    optimizer_initialized = False
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
        "moe_balance_loss": [],
        "routing_ratio": [],
        "soft_routing_ratio": [],
        "ratio_target_error": [],
        "learning_rate": [],
        "lr_phase": [],
        "grad_norm": [],
        "tokens_per_second": [],
        "mfu": [],
        "expert_gini": [],
        "expert_load": [],
        "eval": [],
        "time_ms": [],
        "power_w": [],
        "power_target_w": float(args.target_power_w),
        "power_gap_w": [],
        "target_ce": float(args.target_ce),
        "lr_schedule": {
            "name": "WSqD",
            "warmup_steps": int(args.warmup),
            "decay_fraction": float(args.lr_decay_fraction),
            "shift": float(args.lr_shift),
            "horizon_steps": int(args._lr_horizon_steps),
        },
        "learning_rate_scale": [],
        "target_reached_step": None,
        "auto_batch_calibration": getattr(args, "_auto_batch_calibration", []),
        "auto_batch_selection_policy": getattr(
            args, "_auto_batch_selection_policy", None
        ),
        "ddp_direct_storage_reduce": bool(
            graph_step is not None and graph_step._ddp_direct_storage
        ),
        "nonfinite": [],
        "eval_manifest_sha256": eval_manifest.manifest_sha256,
        "eval_every": int(args.eval_every),
    }
    if rank == 0:
        print(
            "optimizer_groups="
            + json.dumps(optimizer.group_summary(), separators=(",", ":")),
            flush=True,
        )
    latest_power = []
    metric_names = (
        "objective",
        "standard_ce",
        "feedback_ce",
        "router_loss",
        "moe_aux",
        "moe_balance_loss",
        "routing_ratio",
        "soft_routing_ratio",
        "ratio_target_error",
    )
    # Timing events are recorded every update but synchronized only when the
    # step is reported or evaluated.  The old loop synchronized the entire
    # device after every optimizer replay, preventing the next input DMA and
    # graph launch from overlapping the tail of the current update.
    step_start_event = (
        torch.cuda.Event(enable_timing=True) if device.type == "cuda" else None
    )
    step_end_event = (
        torch.cuda.Event(enable_timing=True) if device.type == "cuda" else None
    )
    for step in range(start_step, args.steps):
        omit_graph_masks = (
            graph_step is not None and bool(getattr(stream, "full_windows", False))
        )
        x, y, active_mask, loss_mask = stream.get(
            step,
            device,
            omit_masks=omit_graph_masks,
            consumer_stream=(
                (
                    graph_step.capture_stream
                    if graph_step.capture_stream is not None
                    else torch.cuda.current_stream(device)
                )
                if graph_step is not None
                else None
            ),
            device_buffers=(
                (graph_step.static_x, graph_step.static_y,
                 graph_step.static_mask, graph_step.static_loss_mask)
                if graph_step is not None else None
            ),
        )
        lr_scale = _wsqd_scale(step, args)
        log["learning_rate_scale"].append(float(lr_scale))
        step_number = int(step) + 1
        report_requested = (
            step < 3
            or step % int(args.log_every) == 0
            or (int(args.eval_every) > 0 and step_number % int(args.eval_every) == 0)
        )
        for group in optimizer.param_groups:
            # AnchorMuon splits matrix and AdamATan2 fallback tensors into
            # separate groups; preserve the requested fallback rate.
            if "use_matrix_update" in group:
                base_lr = (
                    args.lr
                    if group["use_matrix_update"]
                    else args.lr * ADAM_ATAN2_LR_RATIO
                )
                group["lr"] = base_lr * lr_scale
            else:
                group["lr"] = args.lr * lr_scale
            if "fallback_lr" in group:
                group["fallback_lr"] = args.lr * ADAM_ATAN2_LR_RATIO * lr_scale
        t0 = time.perf_counter() if step_start_event is None else None
        if step_start_event is not None:
            step_start_event.record()
        if graph_step is not None:
            loss, metrics = graph_step.replay(x, y, active_mask, loss_mask)
        else:
            mask = active_mask
            optimizer.zero_grad(set_to_none=True)
            # FP8 autocast regions are owned by the model at each repeated-layer
            # boundary.  The canonical H-Net reuses the middle layers for two
            # recurrent executions and two FBT passes; one trainer-wide region
            # would make delayed-scaling TE state invalid on the second call.
            out = model(x, mask=mask, fbt_passes=FBT_PASSES)
            loss, metrics = loss_for_output(
                out, y, args, loss_mask=loss_mask, active_mask=active_mask
            )
            # Transformer Engine updates delayed-scaling state during the forward
            # autocast scope; its backward kernels consume that state and should
            # be launched after leaving the context.
            loss.backward()
        if graph_step is not None and graph_step._ddp_direct_storage:
            trainable_model = model.module if world > 1 else model
            frequency_parameters = getattr(trainable_model, "_hnet_lerope_parameters", None)
            if frequency_parameters is None:
                frequency_parameters = tuple(
                    parameter
                    for parameter in trainable_model.parameters()
                    if getattr(parameter, "_lerope_frequency", False)
                )
                setattr(trainable_model, "_hnet_lerope_parameters", frequency_parameters)
            grad_norm, lerope_norm = _clip_direct_ddp_storage(
                graph_step._ddp_storage_buckets,
                1.0,
                frequency_parameters=frequency_parameters,
            )
        else:
            lerope_norm = _clip_lerope_gradients(
                model.module if world > 1 else model, 1.0
            )
            grad_norm = _clip_non_lerope_gradients(model, 1.0)
        finite_local = torch.isfinite(loss) & torch.isfinite(grad_norm)
        if lerope_norm is not None:
            finite_local = finite_local & torch.isfinite(lerope_norm)
        # A single rank with an invalid batch must not advance its optimizer
        # while the other DDP ranks do.  Reduce the finite flag before any
        # optimizer mutation so every rank takes the same branch.
        finite_flag = finite_local.to(dtype=torch.int32)
        if world > 1:
            dist.all_reduce(finite_flag, op=dist.ReduceOp.MIN)
        finite = bool(finite_flag.item())
        if finite:
            if graph_step is not None:
                if graph_optimizer_step is None and optimizer_initialized:
                    # Capture the update using the current replay's gradients.
                    # CUDA graph capture records kernels without executing them;
                    # the explicit replay immediately below applies this real
                    # training update.
                    current_k = max(
                        int(group.get("k", 0)) for group in optimizer.param_groups
                    )
                    graph_optimizer_step = CUDAGraphOptimizerStep(
                        optimizer,
                        graph_step,
                        device,
                        lr=float(args.lr) * float(lr_scale),
                        step=current_k,
                    )
                    graph_optimizer_step.capture()
                    graph_optimizer_step.replay()
                elif graph_optimizer_step is not None:
                    graph_optimizer_step.set_lr(float(args.lr) * float(lr_scale))
                    graph_optimizer_step.replay()
                else:
                    optimizer.step()
                    optimizer_initialized = True
            else:
                optimizer.step()
        else:
            log["nonfinite"].append(step)
        if step_end_event is not None:
            step_end_event.record()
            if report_requested:
                step_end_event.synchronize()
                elapsed = step_start_event.elapsed_time(step_end_event)
            else:
                elapsed = None
        else:
            elapsed = (time.perf_counter() - t0) * 1e3
        # Aggregate all diagnostics in one NCCL operation.  Keep the metric
        # order explicit so the graph replay and JSON schema remain unchanged.
        # Expert-load telemetry belongs in this same vector: reducing it in a
        # second collective added a launch and synchronization on every
        # reported step despite having no dependency on the optimizer.
        expert_load = metrics["moe_expert_load"].detach().float()
        metric_count = len(metric_names)
        if report_requested:
            diagnostic_vector = torch.cat(
                (
                    torch.stack(
                        [loss.detach().float()]
                        + [metrics[name].detach().float() for name in metric_names[1:]]
                    ),
                    expert_load.reshape(-1),
                )
            )
            if world > 1:
                dist.all_reduce(diagnostic_vector)
                diagnostic_vector.div_(float(world))
            # One device-to-host copy replaces a scalar .item()/float
            # conversion per metric and the separate expert-load copy.
            # Non-reporting steps leave None markers in the JSON arrays and
            # never force a host wait on captured tensors.
            diagnostic_values = diagnostic_vector.cpu().tolist()
            metric_values = diagnostic_values[:metric_count]
            expert_load_values = diagnostic_values[metric_count:]
            # Keep the reduced GPU view for the Gini diagnostic below.
            expert_load = diagnostic_vector[metric_count:]
            reduced = metric_values[0]
            reduced_metrics = dict(zip(metric_names, metric_values))
        else:
            reduced = None
            reduced_metrics = None
            expert_load_values = None

        standard_ce_value = (
            reduced_metrics["standard_ce"] if reduced_metrics is not None else None
        )
        current_lr = float(args.lr) * float(lr_scale)
        # A conservative dense-equivalent estimate keeps MFU comparable across
        # runs while the model's actual sparse expert balance is reported
        # separately. The estimate is telemetry only and never affects updates.
        # ``tokens_per_second`` is the aggregate DDP rate, so scale the
        # per-GPU peak by world size when converting it to a cluster MFU.
        if elapsed is None:
            tokens_per_second = None
            mfu = None
        else:
            tokens_per_second = float(args.batch * args.seq * world) / max(
                elapsed / 1000.0, 1e-9
            )
            mfu = (6.0 * float(n_params) * tokens_per_second) / (
                float(args.peak_tflops) * float(world) * 1e12
            )
        report_step = rank == 0 and report_requested
        expert_gini = (
            float(_gini_from_load(expert_load).item())
            if report_step
            else None
        )
        log["loss"].append(standard_ce_value)
        if reduced_metrics is None:
            for name in metric_names:
                log[name].append(None)
        else:
            for name, value in reduced_metrics.items():
                log[name].append(float(value))
        if (
            log["target_reached_step"] is None
            and standard_ce_value is not None
            and standard_ce_value <= float(args.target_ce)
        ):
            log["target_reached_step"] = int(step)
        log["time_ms"].append(elapsed)
        log["learning_rate"].append(current_lr)
        log["lr_phase"].append(_lr_phase(step, args))
        log["grad_norm"].append(
            float(grad_norm.detach().float().item()) if report_requested else None
        )
        log["tokens_per_second"].append(tokens_per_second)
        log["mfu"].append(mfu)
        log["expert_gini"].append(expert_gini)
        log["expert_load"].append(
            None
            if expert_load_values is None
            else [float(value) for value in expert_load_values]
        )
        if rank == 0 and args.power_every > 0 and (
            step == start_step or step % args.power_every == 0
        ):
            latest_power = read_power_watts(device.index)
        if rank == 0:
            log["power_w"].append(list(latest_power))
            if latest_power:
                log["power_gap_w"].append(
                    [float(args.target_power_w) - float(value) for value in latest_power]
                )
            else:
                log["power_gap_w"].append([])
        if rank == 0 and (step < 3 or step % args.log_every == 0):
            timestamp = datetime.datetime.now(datetime.timezone.utc).isoformat(
                timespec="seconds"
            ).replace("+00:00", "Z")
            print(
                f"[{timestamp} step {step_number}] "
                f"loss={standard_ce_value:.4f} lr={current_lr:.8g} "
                f"lr_phase={_lr_phase(step, args)} grad_norm={float(grad_norm.detach().float().item()):.3f} "
                f"tokens/s={tokens_per_second / 1e6:.2f}M mfu={100.0 * mfu:.1f}% "
                f"moe_balance_loss={float(reduced_metrics['moe_balance_loss']):.4f} expert_gini={float(expert_gini):.4f} "
                f"objective={float(reduced):.6f} fb_ce={float(reduced_metrics['feedback_ce']):.6f} "
                f"router={float(reduced_metrics['router_loss']):.6f} route_ratio={float(reduced_metrics['routing_ratio']):.6f} "
                f"time_ms={elapsed:.1f} power_w={','.join(f'{x:.0f}' for x in latest_power) or 'n/a'}",
                flush=True,
            )
        if int(args.eval_every) > 0 and step_number % int(args.eval_every) == 0:
            if world > 1:
                dist.barrier()
            eval_result = evaluate_model(
                model.module if world > 1 else model,
                eval_stream,
                args,
                device,
                ({
                    "mmlu": args.mmlu_data,
                    "arc_challenge": args.arc_challenge_data,
                    "hellaswag": args.hellaswag_data,
                } if rank == 0 else {
                    "mmlu": None,
                    "arc_challenge": None,
                    "hellaswag": None,
                }),
            )
            eval_loss_tensor = torch.tensor(
                [float(eval_result["loss"])], device=device, dtype=torch.float32
            )
            if world > 1:
                dist.all_reduce(eval_loss_tensor)
                eval_loss_tensor.div_(float(world))
            eval_result["loss"] = float(eval_loss_tensor.item())
            eval_record = {
                "step": step_number,
                "loss": float(eval_result["loss"]),
                "mmlu": (None if eval_result["mmlu"][0] is None else float(eval_result["mmlu"][0])),
                "mmlu_n": int(eval_result["mmlu"][1]),
                "arc_challenge": (None if eval_result["arc_challenge"][0] is None else float(eval_result["arc_challenge"][0])),
                "arc_challenge_n": int(eval_result["arc_challenge"][1]),
                "hellaswag": (None if eval_result["hellaswag"][0] is None else float(eval_result["hellaswag"][0])),
                "hellaswag_n": int(eval_result["hellaswag"][1]),
                "eval_took": float(eval_result["eval_took"]),
            }
            if rank == 0:
                log["eval"].append(eval_record)
                timestamp = datetime.datetime.now(datetime.timezone.utc).isoformat(
                    timespec="seconds"
                ).replace("+00:00", "Z")
                benchmark_text = " ".join(
                    f"{name}={eval_record[name] if eval_record[name] is not None else 'null'}(n={eval_record[name + '_n']})"
                    for name in ("mmlu", "arc_challenge", "hellaswag")
                )
                print(
                    f"[{timestamp} step {step_number}] eval(step {step_number}). "
                    f"loss={eval_record['loss']:.4f} {benchmark_text} "
                    f"eval_took={eval_record['eval_took']:.1f}s",
                    flush=True,
                )
            if world > 1:
                dist.barrier()
        if args.ckpt_every and (step + 1) % args.ckpt_every == 0:
            if graph_optimizer_step is not None:
                graph_optimizer_step.sync_python_state()
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
                    "lr_schedule": {
                        "name": "WSqD",
                        "warmup_steps": int(args.warmup),
                        "decay_fraction": float(args.lr_decay_fraction),
                        "shift": float(args.lr_shift),
                        "horizon_steps": int(args._lr_horizon_steps),
                    },
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
    if graph_optimizer_step is not None:
        graph_optimizer_step.sync_python_state()
    if rank == 0:
        Path(args.out).parent.mkdir(parents=True, exist_ok=True)
        Path(args.out).write_text(json.dumps(log, indent=2))
    if world > 1:
        # CUDA-graph replay plus the explicit post-replay NCCL bucket
        # reductions can leave communicator teardown waiting forever in
        # ``destroy_process_group`` after the run record is written.  This is
        # a one-shot trainer; let normal process exit reclaim the communicator
        # instead of hanging a completed training job.
        pass


if __name__ == "__main__":
    main()
