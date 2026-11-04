#!/usr/bin/env python3
"""Build immutable UTF-8 payloads and manifests for the H-Net byte trainer.

The builder deliberately has no dependency on the Hugging Face ``datasets``
package.  It consumes exported JSONL/JSONL.GZ/JSONL.ZST records directly and
can consume Parquet when the optional, pinned ``pyarrow`` dependency is
installed.  A source specification supplies the provenance and mixture order;
all sources are streamed, deduplicated together, and split by document before
the payloads are written.

Example source specification::

  [
    {"path": "/data/fineweb_edu/*.jsonl.zst", "source": "fineweb-edu",
     "revision": "...", "license": "ODC-By 1.0 / Common Crawl terms",
     "language": "en", "weight": 0.20, "tier": "edu3"},
    {"path": "/data/fineweb/*.jsonl.zst", "source": "fineweb",
     "revision": "...", "license": "ODC-By 1.0 / Common Crawl terms",
     "language": "en", "weight": 0.45, "tier": "base"}
  ]

For a production mixture, pass ``--target-bytes`` and ``--near-dedup``.  The
builder uses a deterministic normalized-text filter and a small SimHash LSH
index for near duplicates.  The LSH filter is intentionally conservative and
only rejects a candidate after an exact Hamming-distance check.
"""

from __future__ import annotations

import argparse
import contextlib
import datetime as _dt
import gzip
import hashlib
import json
import os
import re
import shutil
import sqlite3
import struct
import subprocess
import sys
import tempfile
import unicodedata
import uuid
from dataclasses import dataclass
from pathlib import Path
from typing import Any, BinaryIO, Iterable, Iterator, TextIO
from urllib.parse import urlsplit


BUILDER_VERSION = "hnet-byte-corpus/1"
COMPACT_INDEX_MAGIC = b"HNETIDX2"
COMPACT_INDEX_VERSION = 1
COMPACT_INDEX_RECORD_SIZE = 56
COMPACT_INDEX_HEADER = struct.Struct("<8sIIQ")
_TOKEN_RE = re.compile(r"\w+|[^\w\s]", re.UNICODE)
_SCALAR_TYPES = (str, int, float, bool)
_METADATA_FIELDS = (
    "url", "date", "dump", "file_path", "score", "int_score",
    "language_score", "repository_id", "repo_id", "domain", "tier",
)


class BuildError(RuntimeError):
    """Raised for an invalid source or an unsafe build condition."""


@dataclass(frozen=True)
class Source:
    path: Path
    source: str
    license: str
    language: str
    revision: str | None
    weight: float
    tier: str | None
    text_field: str
    id_field: str
    group: str
    priority: int
    min_score: float | None
    max_score: float | None
    min_int_score: int | None
    max_int_score: int | None
    allow_estimated_token_count: bool


@dataclass
class Record:
    text: str
    doc_id: str
    source: Source
    metadata: dict[str, Any]
    token_count: int


@dataclass
class Stats:
    seen: int = 0
    written: int = 0
    duplicate_exact: int = 0
    duplicate_normalized: int = 0
    duplicate_near: int = 0
    invalid: int = 0
    filtered: int = 0
    train_bytes: int = 0
    validation_bytes: int = 0


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while True:
            block = handle.read(16 * 1024 * 1024)
            if not block:
                break
            digest.update(block)
    return digest.hexdigest()


def _record_hash(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8", errors="strict")).hexdigest()


def _normalize(text: str) -> str:
    # NFKC and whitespace folding catches formatting-only duplicates but keeps
    # punctuation and case distinctions out of the more conservative SimHash.
    return " ".join(unicodedata.normalize("NFKC", text).casefold().split())


def _simhash(text: str) -> int:
    tokens = _TOKEN_RE.findall(_normalize(text))
    if len(tokens) >= 5:
        features = ("\x1f".join(tokens[i:i + 5]) for i in range(len(tokens) - 4))
    else:
        features = ("\x1f".join(tokens),)
    accum = [0] * 64
    used = False
    for feature in features:
        if not feature:
            continue
        value = int.from_bytes(
            hashlib.blake2b(feature.encode("utf-8"), digest_size=8).digest(),
            "little",
        )
        used = True
        for bit in range(64):
            accum[bit] += 1 if value & (1 << bit) else -1
    if not used:
        return 0
    return sum((1 << bit) for bit, score in enumerate(accum) if score >= 0)


def _hamming(a: int, b: int) -> int:
    return (a ^ b).bit_count()


def _open_text(path: Path) -> TextIO:
    suffixes = "".join(path.suffixes).lower()
    if suffixes.endswith(".gz"):
        return gzip.open(path, "rt", encoding="utf-8", errors="strict")
    if suffixes.endswith(".zst"):
        if shutil.which("zstd") is None:
            raise BuildError("zstd is required for .zst inputs but was not found")
        proc = subprocess.Popen(
            ["zstd", "--quiet", "--decompress", "--stdout", "--no-progress", str(path)],
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
        )
        assert proc.stdout is not None

        class _ZstdText:
            def __init__(self):
                self._stream = proc.stdout
                self._reader = __import__("io").TextIOWrapper(
                    self._stream, encoding="utf-8", errors="strict"
                )

            def __iter__(self):
                return iter(self._reader)

            def __enter__(self):
                return self

            def __exit__(self, exc_type, exc, tb):
                try:
                    self._reader.close()
                finally:
                    stderr = proc.stderr.read().decode("utf-8", errors="replace") if proc.stderr else ""
                    code = proc.wait()
                    if exc_type is None and code != 0:
                        raise BuildError(f"zstd failed for {path}: {stderr.strip()}")

        return _ZstdText()  # type: ignore[return-value]
    return path.open("rt", encoding="utf-8", errors="strict")


def _iter_parquet(path: Path, source: Source) -> Iterator[dict[str, Any]]:
    try:
        import pyarrow.parquet as pq  # type: ignore[import-not-found]
    except ImportError as exc:
        raise BuildError(
            "Parquet input requires a reviewed, pinned pyarrow installation; "
            "export the shard to JSONL or install pyarrow in the data environment"
        ) from exc
    parquet = pq.ParquetFile(path)
    for batch in parquet.iter_batches(batch_size=4096, use_threads=True):
        columns = batch.to_pydict()
        if not columns:
            continue
        names = list(columns)
        rows = len(columns[names[0]])
        for row in range(rows):
            yield {name: columns[name][row] for name in names}


def _iter_raw(path: Path, source: Source) -> Iterator[dict[str, Any]]:
    if path.suffix.lower() == ".parquet":
        yield from _iter_parquet(path, source)
        return
    with _open_text(path) as handle:
        for line_number, line in enumerate(handle, 1):
            if not line.strip():
                continue
            try:
                value = json.loads(line)
            except json.JSONDecodeError as exc:
                raise BuildError(f"invalid JSON at {path}:{line_number}: {exc}") from exc
            if not isinstance(value, dict):
                raise BuildError(f"record at {path}:{line_number} is not an object")
            yield value


def _expand_path(pattern: str) -> list[Path]:
    path = Path(pattern).expanduser()
    if path.is_file():
        return [path]
    if path.is_dir():
        return sorted(p for p in path.rglob("*") if p.is_file())
    matches = sorted(Path().glob(pattern) if not os.path.isabs(pattern) else Path("/").glob(pattern.lstrip("/")))
    return matches


def _load_sources(path: Path) -> list[Source]:
    try:
        raw = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise BuildError(f"unable to read source specification {path}: {exc}") from exc
    if not isinstance(raw, list) or not raw:
        raise BuildError("source specification must be a non-empty JSON list")
    sources: list[Source] = []
    for index, item in enumerate(raw):
        if not isinstance(item, dict):
            raise BuildError(f"source specification entry {index} is not an object")
        for key in ("path", "source", "license", "language"):
            if not isinstance(item.get(key), str) or not item[key]:
                raise BuildError(f"source entry {index} requires non-empty {key}")
        weight = float(item.get("weight", 0.0))
        if weight < 0.0 or not weight == weight:
            raise BuildError(f"source entry {index} has invalid weight")
        paths = _expand_path(item["path"])
        if not paths:
            raise BuildError(f"source entry {index} matched no files: {item['path']}")
        # Keep quality tiers as separate mixture buckets even when they are
        # materialized from the same upstream revision (FineWeb-Edu score>=3
        # and score==2 are the canonical example).  Otherwise the first tier
        # would silently consume the combined quota for both entries.
        tier_name = str(item.get("tier", ""))
        group = f"{item['source']}@{item.get('revision', '')}@{tier_name}"
        default_priority = {
            "edu3": 0, "edu2": 1, "educational": 1, "base": 2,
            "dolma": 3, "multilingual": 4, "reference": 5,
        }.get(str(item.get("tier", "")).casefold(), 5)
        try:
            priority = int(item.get("priority", default_priority))
        except (TypeError, ValueError) as exc:
            raise BuildError(f"source entry {index} has invalid priority") from exc
        def _optional_float(name: str) -> float | None:
            value = item.get(name)
            if value is None:
                return None
            try:
                parsed = float(value)
            except (TypeError, ValueError) as exc:
                raise BuildError(f"source entry {index} has invalid {name}") from exc
            if not parsed == parsed:
                raise BuildError(f"source entry {index} has invalid {name}")
            return parsed
        def _optional_int(name: str) -> int | None:
            value = item.get(name)
            if value is None:
                return None
            try:
                return int(value)
            except (TypeError, ValueError) as exc:
                raise BuildError(f"source entry {index} has invalid {name}") from exc
        min_score = _optional_float("min_score")
        max_score = _optional_float("max_score")
        min_int_score = _optional_int("min_int_score")
        max_int_score = _optional_int("max_int_score")
        if min_score is not None and max_score is not None and min_score > max_score:
            raise BuildError(f"source entry {index} has min_score above max_score")
        if min_int_score is not None and max_int_score is not None and min_int_score > max_int_score:
            raise BuildError(f"source entry {index} has min_int_score above max_int_score")
        for expanded in paths:
            sources.append(Source(
                path=expanded.resolve(), source=item["source"], license=item["license"],
                language=item["language"], revision=item.get("revision"), weight=weight,
                tier=item.get("tier"), text_field=item.get("text_field", "text"),
                id_field=item.get("id_field", "id"),
                group=group,
                priority=priority,
                min_score=min_score,
                max_score=max_score,
                min_int_score=min_int_score,
                max_int_score=max_int_score,
                allow_estimated_token_count=bool(item.get("allow_estimated_token_count", False)),
            ))
    if any(source.weight for source in sources):
        total = sum(source.weight for source in sources)
        if total <= 0.0:
            raise BuildError("source weights must sum to a positive value")
    # Retain high-quality derived tiers before their parent pool. This is the
    # documented FineWeb-Edu-before-FineWeb rule and makes duplicate retention
    # independent of the order in a hand-written source list.
    return sorted(sources, key=lambda item: (item.priority, item.source, str(item.path)))


def _token_count(record: dict[str, Any], text: str, allow_estimated: bool) -> tuple[int, bool]:
    for key in ("token_count", "tokens", "num_tokens"):
        if key in record and record[key] is not None:
            try:
                count = int(record[key])
            except (TypeError, ValueError) as exc:
                raise BuildError(f"invalid token count {record[key]!r}") from exc
            if count < 0:
                raise BuildError("token count cannot be negative")
            return count, False
    if not allow_estimated:
        raise BuildError("record has no token_count; pass --allow-estimated-token-count explicitly")
    # A transparent, deterministic estimate used only for non-FineWeb sources.
    return max(1, round(len(text.encode("utf-8")) / 4)), True


def _record_from_json(value: dict[str, Any], source: Source, line_id: int, allow_estimated: bool) -> Record:
    text = value.get(source.text_field)
    if not isinstance(text, str) or not text:
        raise BuildError(f"record has no non-empty string {source.text_field!r}")
    # Explicitly exercise strict UTF-8 encoding here, before any output file is
    # opened, so a malformed surrogate cannot leave a partially valid corpus.
    text.encode("utf-8", errors="strict")
    raw_id = value.get(source.id_field)
    doc_id = str(raw_id) if raw_id not in (None, "") else f"{source.source}:{source.path.name}:{line_id}"
    token_count, estimated = _token_count(value, text, allow_estimated)
    metadata: dict[str, Any] = {}
    for key in _METADATA_FIELDS:
        item = value.get(key)
        if isinstance(item, _SCALAR_TYPES):
            metadata[key] = item
    if source.tier is not None:
        metadata.setdefault("tier", source.tier)
    if source.revision is not None:
        metadata["revision"] = source.revision
    if estimated:
        metadata["token_count_estimated"] = True
    return Record(text=text, doc_id=doc_id, source=source, metadata=metadata, token_count=token_count)


def _source_accepts(value: dict[str, Any], source: Source) -> bool:
    """Apply source-level quality filters before deduplication and budgeting."""
    if source.min_score is not None or source.max_score is not None:
        raw = value.get("score")
        if raw is None:
            return False
        try:
            score = float(raw)
        except (TypeError, ValueError):
            return False
        if source.min_score is not None and score < source.min_score:
            return False
        if source.max_score is not None and score > source.max_score:
            return False
    if source.min_int_score is not None or source.max_int_score is not None:
        raw = value.get("int_score")
        if raw is None:
            return False
        try:
            int_score = int(raw)
        except (TypeError, ValueError):
            return False
        if source.min_int_score is not None and int_score < source.min_int_score:
            return False
        if source.max_int_score is not None and int_score > source.max_int_score:
            return False
    return True


class DedupIndex:
    def __init__(self, path: Path, near: bool, max_hamming: int):
        self.db = sqlite3.connect(path)
        self.db.execute("PRAGMA journal_mode=OFF")
        self.db.execute("PRAGMA synchronous=OFF")
        self.db.execute("CREATE TABLE exact (content TEXT PRIMARY KEY, normalized TEXT UNIQUE, source TEXT, doc_id TEXT)")
        self.near = near
        self.max_hamming = max_hamming
        if near:
            self.db.execute("CREATE TABLE simhash (value INTEGER NOT NULL, content TEXT PRIMARY KEY)")
            self.db.execute("CREATE INDEX simhash_band ON simhash(value)")
            self.db.execute("CREATE TABLE band (bucket INTEGER NOT NULL, content TEXT NOT NULL, PRIMARY KEY(bucket, content))")
            self.db.execute("CREATE INDEX band_index ON band(bucket)")
        self.db.commit()

    @staticmethod
    def _buckets(value: int) -> tuple[int, ...]:
        return tuple(((value >> shift) & 0xFFFF) | (band << 16) for band, shift in enumerate((0, 16, 32, 48)))

    def check_and_add(self, text: str, source: str, doc_id: str) -> str | None:
        content = _record_hash(text)
        if self.db.execute("SELECT 1 FROM exact WHERE content=?", (content,)).fetchone():
            return "exact"
        normalized = hashlib.sha256(_normalize(text).encode("utf-8")).hexdigest()
        if self.db.execute("SELECT 1 FROM exact WHERE normalized=?", (normalized,)).fetchone():
            return "normalized"
        sim = _simhash(text) if self.near and len(text) >= 64 else None
        if sim is not None:
            candidates = set()
            for bucket in self._buckets(sim):
                candidates.update(row[0] for row in self.db.execute("SELECT content FROM band WHERE bucket=?", (bucket,)))
            for candidate in candidates:
                row = self.db.execute("SELECT value FROM simhash WHERE content=?", (candidate,)).fetchone()
                if row is not None and _hamming(sim, int(row[0]) & ((1 << 64) - 1)) <= self.max_hamming:
                    return "near"
        self.db.execute("INSERT INTO exact(content, normalized, source, doc_id) VALUES(?,?,?,?)", (content, normalized, source, doc_id))
        if sim is not None:
            # SQLite INTEGER is signed; store the exact 64-bit bit pattern in
            # two's-complement form and recover it with ``& ((1<<64)-1)``.
            signed = sim if sim < (1 << 63) else sim - (1 << 64)
            self.db.execute("INSERT INTO simhash(value, content) VALUES(?,?)", (signed, content))
            for bucket in self._buckets(sim):
                self.db.execute("INSERT INTO band(bucket, content) VALUES(?,?)", (bucket, content))
        return None

    def commit(self) -> None:
        self.db.commit()

    def close(self) -> None:
        self.db.close()


def _split_is_validation(record: Record, fraction: float, seed: int) -> bool:
    if fraction <= 0.0:
        return False
    if fraction >= 1.0:
        return True
    url = record.metadata.get("url")
    group = record.metadata.get("repository_id") or record.metadata.get("repo_id")
    if not group and isinstance(url, str) and url:
        group = urlsplit(url).netloc
    if not group:
        group = record.doc_id
    key = f"{seed}:{record.source.source}:{group}".encode("utf-8", errors="strict")
    value = int.from_bytes(hashlib.blake2b(key, digest_size=8).digest(), "big")
    return value < int(fraction * (1 << 64))


def _manifest(
    path: Path,
    payload: Path,
    documents: list[dict[str, Any]],
    split: str,
    sources: list[Source],
    *,
    compact_index: Path | None = None,
    document_count: int | None = None,
    total_token_equivalents: int | None = None,
) -> dict[str, Any]:
    if compact_index is None:
        version = 1
        index_fields = {"documents": documents}
        count = len(documents)
        tokens = sum(int(doc["token_count"]) for doc in documents)
    else:
        version = 2
        count = int(document_count or 0)
        tokens = int(total_token_equivalents or 0)
        index_fields = {
            "document_index": {
                "path": compact_index.name,
                "size": compact_index.stat().st_size,
                "sha256": _sha256_file(compact_index),
                "count": count,
                "record_size": COMPACT_INDEX_RECORD_SIZE,
                "total_token_equivalents": tokens,
                "format": "HNETIDX2",
                "version": COMPACT_INDEX_VERSION,
            }
        }
    return {
        "version": version,
        "data_path": payload.name,
        "data_size": payload.stat().st_size,
        "data_sha256": _sha256_file(payload),
        "total_token_equivalents": tokens,
        **index_fields,
        "split": split,
        "builder": BUILDER_VERSION,
        "built_at": _dt.datetime.now(_dt.timezone.utc).isoformat().replace("+00:00", "Z"),
        "source_revisions": sorted({f"{s.source}@{s.revision}" for s in sources if s.revision}),
    }


def build(args: argparse.Namespace) -> Stats:
    sources = _load_sources(Path(args.sources).resolve())
    if args.validation_fraction < 0.0 or args.validation_fraction >= 1.0:
        raise BuildError("validation fraction must be in [0,1)")
    if args.near_max_hamming < 0 or args.near_max_hamming > 16:
        raise BuildError("near max Hamming distance must be in [0,16]")
    # Weights describe a mixture when a byte budget is supplied. Without a
    # budget, process every input record; this makes a one-source validation
    # build useful without inventing a truncation point.
    weighted = bool(args.target_bytes > 0 and any(source.weight for source in sources))
    target_by_group: dict[str, int] = {}
    if weighted:
        # A glob expands one mixture entry into many files.  Count that
        # entry's weight once; otherwise a 15-shard FineWeb source would
        # receive fifteen times its intended quota.
        group_weights: dict[str, float] = {}
        for source in sources:
            group_weights.setdefault(source.group, source.weight)
        total_weight = sum(group_weights.values())
        for group, weight in group_weights.items():
            target_by_group[group] = int(args.target_bytes * weight / total_weight)

    output = Path(args.output).expanduser().resolve()
    output.parent.mkdir(parents=True, exist_ok=True)
    staging = Path(tempfile.mkdtemp(prefix=f".{output.name}.staging-", dir=output.parent))
    db_path = staging / "dedup.sqlite3"
    index = DedupIndex(db_path, args.near_dedup, args.near_max_hamming)
    stats = Stats()
    documents: dict[str, list[dict[str, Any]]] = {"train": [], "validation": []}
    handles: dict[str, BinaryIO] = {}
    compact_handles: dict[str, BinaryIO] = {}
    metadata_handles: dict[str, TextIO] = {}
    compact_counts = {"train": 0, "validation": 0}
    compact_tokens = {"train": 0, "validation": 0}
    bytes_by_group: dict[str, int] = {}
    try:
        for split in documents:
            handles[split] = (staging / f"{split}.bin").open("wb")
            if args.compact_index:
                compact_path = staging / f"{split}.index"
                compact_handles[split] = compact_path.open("w+b")
                metadata_handles[split] = (staging / f"{split}.documents.jsonl").open(
                    "w", encoding="utf-8", newline="\n"
                )
                compact_handles[split].write(
                    COMPACT_INDEX_HEADER.pack(
                        COMPACT_INDEX_MAGIC, COMPACT_INDEX_VERSION,
                        COMPACT_INDEX_RECORD_SIZE, 0,
                    )
                )
        for source_index, source in enumerate(sources):
            for path in [source.path]:
                if weighted and bytes_by_group.get(source.group, 0) >= target_by_group[source.group]:
                    break
                line_id = 0
                for value in _iter_raw(path, source):
                    line_id += 1
                    stats.seen += 1
                    if not _source_accepts(value, source):
                        stats.filtered += 1
                        continue
                    try:
                        record = _record_from_json(
                            value,
                            source,
                            line_id,
                            bool(args.allow_estimated_token_count or source.allow_estimated_token_count),
                        )
                        encoded = record.text.encode("utf-8", errors="strict")
                    except (UnicodeError, BuildError) as exc:
                        stats.invalid += 1
                        if args.skip_invalid:
                            continue
                        raise BuildError(f"invalid record in {path}:{line_id}: {exc}") from exc
                    if args.target_bytes > 0 and weighted and bytes_by_group.get(source.group, 0) >= target_by_group[source.group]:
                        break
                    reason = index.check_and_add(record.text, record.source.source, record.doc_id)
                    if reason == "exact":
                        stats.duplicate_exact += 1
                        continue
                    if reason == "normalized":
                        stats.duplicate_normalized += 1
                        continue
                    if reason == "near":
                        stats.duplicate_near += 1
                        continue
                    split = "validation" if _split_is_validation(record, args.validation_fraction, args.seed) else "train"
                    offset = handles[split].tell()
                    handles[split].write(encoded)
                    doc = {
                        "id": record.doc_id,
                        "offset": offset,
                        "length": len(encoded),
                        "sha256": hashlib.sha256(encoded).hexdigest(),
                        "token_count": record.token_count,
                        "source": record.source.source,
                        "license": record.source.license,
                        "language": record.source.language,
                        **record.metadata,
                    }
                    if args.compact_index:
                        metadata_handles[split].write(
                            json.dumps(doc, ensure_ascii=False, sort_keys=True, separators=(",", ":")) + "\n"
                        )
                        compact_handles[split].write(
                            struct.pack(
                                "<QQQ32s",
                                int(doc["offset"]),
                                int(doc["length"]),
                                int(doc["token_count"]),
                                bytes.fromhex(doc["sha256"]),
                            )
                        )
                        compact_counts[split] += 1
                        compact_tokens[split] += int(doc["token_count"])
                    else:
                        documents[split].append(doc)
                    stats.written += 1
                    bytes_by_group[source.group] = bytes_by_group.get(source.group, 0) + len(encoded)
                    if split == "train":
                        stats.train_bytes += len(encoded)
                    else:
                        stats.validation_bytes += len(encoded)
            index.commit()
        for handle in handles.values():
            handle.flush()
            os.fsync(handle.fileno())
            handle.close()
        if args.compact_index:
            for split, handle in compact_handles.items():
                handle.seek(0)
                handle.write(
                    COMPACT_INDEX_HEADER.pack(
                        COMPACT_INDEX_MAGIC, COMPACT_INDEX_VERSION,
                        COMPACT_INDEX_RECORD_SIZE, compact_counts[split],
                    )
                )
                handle.flush()
                os.fsync(handle.fileno())
                handle.close()
            for handle in metadata_handles.values():
                handle.flush()
                os.fsync(handle.fileno())
                handle.close()
        index.commit()
        for split in documents:
            payload = staging / f"{split}.bin"
            empty_split = compact_counts[split] == 0 if args.compact_index else not documents[split]
            if empty_split:
                raise BuildError(f"no documents selected for {split}; adjust validation fraction or inputs")
            compact_path = staging / f"{split}.index" if args.compact_index else None
            manifest = _manifest(
                staging / f"{split}.manifest.json", payload, documents[split], split, sources,
                compact_index=compact_path,
                document_count=compact_counts[split],
                total_token_equivalents=compact_tokens[split],
            )
            if args.compact_index:
                metadata_path = staging / f"{split}.documents.jsonl"
                manifest["document_metadata_path"] = metadata_path.name
                manifest["document_metadata_size"] = metadata_path.stat().st_size
                manifest["document_metadata_sha256"] = _sha256_file(metadata_path)
            manifest_path = staging / f"{split}.manifest.json"
            with manifest_path.open("w", encoding="utf-8") as handle:
                json.dump(manifest, handle, ensure_ascii=False, sort_keys=True, separators=(",", ":"))
                handle.write("\n")
                handle.flush()
                os.fsync(handle.fileno())
        output.mkdir(parents=True, exist_ok=True)
        # A run launcher can require this marker before opening any payload;
        # remove an old marker while the new set is being installed.
        ready = output / "READY"
        ready.unlink(missing_ok=True)
        names = ["train.bin", "train.manifest.json", "validation.bin", "validation.manifest.json"]
        if args.compact_index:
            names.extend((
                "train.index", "validation.index",
                "train.documents.jsonl", "validation.documents.jsonl",
            ))
        for name in names:
            os.replace(staging / name, output / name)
        with (output / "build_stats.json").open("w", encoding="utf-8") as handle:
            json.dump({**stats.__dict__, "sources": len(sources)}, handle, indent=2, sort_keys=True)
            handle.write("\n")
        ready.write_text("hnet-byte-corpus/1\n", encoding="ascii")
    except Exception:
        for handle in handles.values():
            with contextlib.suppress(Exception):
                handle.close()
        for handle in metadata_handles.values():
            with contextlib.suppress(Exception):
                handle.close()
        raise
    finally:
        index.close()
        shutil.rmtree(staging, ignore_errors=True)
    return stats


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--sources", required=True, help="JSON list of source specifications")
    parser.add_argument("--output", required=True, help="external output directory for payloads/manifests")
    parser.add_argument("--target-bytes", type=int, default=0, help="total selected bytes; required for weighted mixtures")
    parser.add_argument("--validation-fraction", type=float, default=0.01)
    parser.add_argument("--seed", type=int, default=1729)
    parser.add_argument("--allow-estimated-token-count", action="store_true")
    parser.add_argument("--skip-invalid", action="store_true")
    parser.add_argument("--near-dedup", action="store_true", help="enable conservative SimHash near-duplicate filtering")
    parser.add_argument("--near-max-hamming", type=int, default=3)
    parser.add_argument(
        "--compact-index",
        action="store_true",
        help="write a version-2 manifest with a fixed-width binary document index",
    )
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    try:
        stats = build(parse_args(argv))
    except (BuildError, OSError, sqlite3.Error) as exc:
        print(f"build_byte_corpus: error: {exc}", file=sys.stderr)
        return 2
    print(json.dumps(stats.__dict__, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
