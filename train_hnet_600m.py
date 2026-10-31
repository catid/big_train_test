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
# With activation recomputation disabled, B=12 at S=4096 is the largest
# validated full-model CUDA-graph shape with useful throughput and memory
# headroom on a B300. B=16 exhausts the 275 GiB device during capture. B=32
# also exceeds the graph pool boundary; calibration must not select either.
CANONICAL_MAX_BATCH = 12


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
        help="upper bound for power-calibration candidates (validated maximum is 16)",
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
        global_rows = batch * world
        # Round the epoch up to a complete global batch. Repeating only the
        # first few document indices is preferable to silently dropping the
        # tail of a FineWeb shard, and keeps every rank's static shape valid.
        self.n_documents = (
            (self.document_count + global_rows - 1) // global_rows
        ) * global_rows
        self.n_batches = self.n_documents // global_rows
        # Reuse pinned host staging buffers so document-local sampling does
        # not allocate pageable tensors or force a synchronous host-to-device
        # copy on every graph replay. The buffers are private to this stream
        # and are fully overwritten before each transfer.
        self._pinned = bool(torch.cuda.is_available())
        self._x_cpu = torch.empty(
            (self.batch, self.seq), dtype=torch.uint8, pin_memory=self._pinned
        )
        self._y_cpu = torch.empty(
            (self.batch, self.seq), dtype=torch.uint8, pin_memory=self._pinned
        )
        self._active_cpu = torch.empty(
            (self.batch, self.seq), dtype=torch.bool, pin_memory=self._pinned
        )
        self._loss_cpu = torch.empty(
            (self.batch, self.seq), dtype=torch.bool, pin_memory=self._pinned
        )
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

    def get(self, step, device):
        epoch, offset = divmod(int(step), self.n_batches)
        permutation = self._epoch_permutation(epoch)
        first = (offset * self.world + self.rank) * self.batch
        document_indices = permutation[first : first + self.batch]
        x_cpu = self._x_cpu
        y_cpu = self._y_cpu
        active_cpu = self._active_cpu
        loss_cpu = self._loss_cpu
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
            active_cpu[row, :available] = True
            loss_cpu[row, :available] = True
        return tuple(
            tensor.to(device=device, non_blocking=True)
            for tensor in (x_cpu, y_cpu, active_cpu, loss_cpu)
        )


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
        # Preserve the checkpoint's original WSqD horizon when a caller asks
        # to continue for additional steps. This keeps cooldown monotonic and
        # avoids a learning-rate jump at the resume boundary.
        args._lr_horizon_steps = saved_horizon
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
    metrics = {
        "objective": objective,
        "standard_ce": standard_ce,
        "feedback_ce": feedback_ce,
        "router_loss": ratio,
        "moe_aux": moe_aux,
        "routing_ratio": routing_ratio,
        "soft_routing_ratio": soft_routing_ratio,
        "ratio_target_error": (routing_ratio - target_ratio).abs(),
    }
    return objective, metrics


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


def _average_gradients_by_dtype(parameters, world, device, buckets=None):
    """Average gradients with one flat NCCL reduction per parameter dtype.

    Calibration runs outside the captured graph but must measure the same
    communication shape as production DDP replay. Reducing each parameter
    separately creates hundreds of tiny NCCL launches and overstates the
    steady-state step time. The returned bucket state keeps the flat buffers
    allocated across calibration iterations.
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
        dist.all_reduce(flat, op=dist.ReduceOp.SUM)
        flat.div_(float(world))
        offset = 0
        for parameter in params:
            count = parameter.numel()
            view = flat[offset : offset + count].view_as(parameter)
            if parameter.grad is None:
                parameter.grad = view
            else:
                parameter.grad.copy_(view)
            offset += count
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
    # The canonical no-checkpoint B=12 graph is validated on B300; smaller
    # devices receive a proportionally lower bound through calibration.
    if int(args.auto_batch_max) < 1:
        raise ValueError("--auto-batch-max must be positive")
    # The full model keeps its activation graph for the canonical B=12 shape;
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
                        # After every rank passes the OOM gate, explicitly
                        # average gradients so this remains a complete DDP
                        # optimizer step for power measurement.  Use the same
                        # two flat dtype buckets as graph replay rather than
                        # hundreds of per-parameter NCCL collectives.
                        calibration_grad_buckets = _average_gradients_by_dtype(
                            probe_model.parameters(),
                            world,
                            device,
                            calibration_grad_buckets,
                        )
                    grad_norm = torch.nn.utils.clip_grad_norm_(
                        probe_model.parameters(), 1.0
                    )
                    finite_grad_flag = torch.isfinite(grad_norm).to(dtype=torch.int32)
                    if world > 1:
                        dist.all_reduce(finite_grad_flag, op=dist.ReduceOp.MIN)
                    if not bool(finite_grad_flag.item()):
                        raise FloatingPointError(
                            f"non-finite calibration gradient at candidate batch {candidate}"
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
        # DDP is used for construction and eager warmup so its reducer hooks
        # are initialized against the complete parameter set.  Captured
        # replay uses the underlying module after that setup: the wrapper's
        # per-forward logger/DDPSink path is not graph-safe and is redundant
        # because replay performs the explicit flat gradient all-reduce below.
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
                self.optimizer.zero_grad(set_to_none=True)
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
            # for capture and perform a flat gradient all-reduce after replay;
            # this keeps the captured graph free of logger/NCCL setup calls.
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
                self.optimizer.zero_grad(set_to_none=True)
                del graph_backend_warmup
                del graph_backend_loss, _graph_backend_metrics
                torch.cuda.synchronize(self.device)
        finally:
            te_graph.set_capture_end()
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
        self._ddp_direct_storage = compatible
        self._ddp_storage_buckets = groups if compatible else None

    def replay(self, x, y, active_mask, loss_mask):
        if self.capture_stream is None:
            self.static_x.copy_(x)
            self.static_y.copy_(y)
            self.static_mask.copy_(active_mask)
            self.static_loss_mask.copy_(loss_mask)
            self.graph.replay()
        else:
            current_stream = torch.cuda.current_stream(self.device)
            self.capture_stream.wait_stream(current_stream)
            with torch.cuda.stream(self.capture_stream):
                self.static_x.copy_(x)
                self.static_y.copy_(y)
                self.static_mask.copy_(active_mask)
                self.static_loss_mask.copy_(loss_mask)
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
        """Average captured local gradients without entering the graph.

        Reduce one flat buffer per parameter precision.  This preserves the
        usual BF16 DDP communication contract while avoiding a full-model
        BF16-to-FP32 staging copy and its matching copy back to every
        parameter.  All ranks use the same parameter ordering and bucket sizes
        because they construct the model from the same canonical config.
        """

        if self._ddp_direct_storage:
            for flat in self._ddp_storage_buckets:
                dist.all_reduce(flat, op=dist.ReduceOp.SUM)
                flat.div_(float(self.world))
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
            dist.all_reduce(flat, op=dist.ReduceOp.SUM)
            flat.div_(float(self.world))
            offset = 0
            for param in params:
                count = param.numel()
                view = flat[offset : offset + count].view_as(param)
                if param.grad is None:
                    param.grad = view
                else:
                    param.grad.copy_(view)
                offset += count


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
        "soft_routing_ratio": [],
        "ratio_target_error": [],
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
    }
    if rank == 0:
        print(
            "optimizer_groups="
            + json.dumps(optimizer.group_summary(), separators=(",", ":")),
            flush=True,
        )
    latest_power = []
    for step in range(start_step, args.steps):
        x, y, active_mask, loss_mask = stream.get(step, device)
        lr_scale = _wsqd_scale(step, args)
        log["learning_rate_scale"].append(float(lr_scale))
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
        t0 = time.perf_counter()
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
        # Aggregate all diagnostics in one NCCL operation.  The previous loop
        # launched one all-reduce for the objective and one for every metric,
        # then converted each scalar independently on the host.  Keep the
        # metric order explicit so the graph replay and the JSON schema remain
        # unchanged while reducing the synchronization/launch count to one.
        metric_names = (
            "objective",
            "standard_ce",
            "feedback_ce",
            "router_loss",
            "moe_aux",
            "routing_ratio",
            "soft_routing_ratio",
            "ratio_target_error",
        )
        metric_vector = torch.stack(
            [loss.detach().float()]
            + [metrics[name].detach().float() for name in metric_names[1:]]
        )
        if world > 1:
            dist.all_reduce(metric_vector)
            metric_vector.div_(float(world))
        # One device-to-host copy replaces a scalar .item()/float conversion
        # per metric.  It also makes the logging path independent of whether
        # graph replay returned aliases into captured storage.
        metric_values = metric_vector.cpu().tolist()
        reduced = metric_values[0]
        reduced_metrics = dict(zip(metric_names, metric_values))
        standard_ce_value = reduced_metrics["standard_ce"]
        log["loss"].append(standard_ce_value)
        for name, value in reduced_metrics.items():
            log[name].append(float(value))
        if log["target_reached_step"] is None and standard_ce_value <= float(args.target_ce):
            log["target_reached_step"] = int(step)
        log["time_ms"].append(elapsed)
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
