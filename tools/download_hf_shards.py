#!/usr/bin/env python3
"""Fetch and verify immutable Hugging Face dataset shards.

The downloader treats a dataset revision and its tree metadata as part of the
input contract.  It checks the exact byte size and SHA-256 LFS object id from
the Hugging Face API before accepting an existing file, and writes new files
to a same-directory temporary path followed by an atomic rename.  Payloads
are intentionally written outside this repository.

Example::

    python tools/download_hf_shards.py \
      --repo HuggingFaceFW/fineweb \
      --revision 9bb295ddab0e05d785b879661af7260fed5140fc \
      --path sample/100BT \
      --output /home/catid/hnet_data/raw/fineweb/sample100bt \
      --workers 8

Use ``--verify-only`` to audit an existing download without changing it.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path, PurePosixPath
import tempfile
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from typing import Any
from urllib.error import HTTPError, URLError
from urllib.parse import quote
from urllib.request import Request, urlopen


API_BASE = "https://huggingface.co/api/datasets"
DOWNLOAD_BASE = "https://huggingface.co/datasets"
CHUNK_SIZE = 16 * 1024 * 1024


class DownloadError(RuntimeError):
    pass


def _request_json(url: str, attempts: int = 8) -> Any:
    last: Exception | None = None
    for attempt in range(attempts):
        try:
            request = Request(url, headers={"Accept": "application/json", "User-Agent": "hnet-corpus-fetch/1"})
            with urlopen(request, timeout=60) as response:
                return json.load(response)
        except (HTTPError, URLError, TimeoutError, json.JSONDecodeError) as exc:
            last = exc
            if attempt + 1 == attempts:
                break
            time.sleep(min(30.0, 1.5 * (2**attempt)))
    raise DownloadError(f"unable to fetch metadata {url}: {last}") from last


def list_files(repo: str, revision: str, path: str) -> list[dict[str, Any]]:
    """Return regular files below a pinned dataset tree path."""
    if not repo or not revision or not path:
        raise DownloadError("repo, revision, and path are required")
    # The API endpoint accepts slash-separated paths; quote each component so
    # user input cannot alter query/path semantics.
    encoded_repo = "/".join(quote(part, safe="") for part in repo.split("/"))
    encoded_path = "/".join(quote(part, safe="") for part in path.split("/") if part)
    url = f"{API_BASE}/{encoded_repo}/tree/{quote(revision, safe='')}/{encoded_path}?recursive=true"
    value = _request_json(url)
    if not isinstance(value, list):
        raise DownloadError("Hugging Face tree API returned a non-list response")
    files = []
    prefix = PurePosixPath(path)
    for item in value:
        if not isinstance(item, dict) or item.get("type") != "file":
            continue
        raw_path = item.get("path")
        if not isinstance(raw_path, str):
            raise DownloadError("tree response contains a file without a path")
        relative = PurePosixPath(raw_path)
        try:
            relative.relative_to(prefix)
        except ValueError as exc:
            raise DownloadError(f"tree response escaped requested path: {raw_path!r}") from exc
        size = item.get("size")
        lfs = item.get("lfs")
        digest = lfs.get("oid") if isinstance(lfs, dict) else None
        if not isinstance(size, int) or size < 0:
            raise DownloadError(f"missing or invalid size for {raw_path!r}")
        if not isinstance(digest, str) or len(digest) != 64:
            raise DownloadError(f"missing SHA-256 LFS object id for {raw_path!r}")
        files.append({"path": raw_path, "size": size, "sha256": digest.lower()})
    if not files:
        raise DownloadError(f"no files found below {repo}@{revision}:{path}")
    return sorted(files, key=lambda item: item["path"])


def _safe_destination(output: Path, raw_path: str, prefix: str) -> Path:
    relative = PurePosixPath(raw_path)
    root = PurePosixPath(prefix)
    try:
        relative = relative.relative_to(root)
    except ValueError as exc:
        raise DownloadError(f"file path is outside requested prefix: {raw_path!r}") from exc
    if relative.is_absolute() or any(part in ("", ".", "..") for part in relative.parts):
        raise DownloadError(f"unsafe file path: {raw_path!r}")
    destination = (output / Path(*relative.parts)).resolve()
    try:
        destination.relative_to(output.resolve())
    except ValueError as exc:
        raise DownloadError(f"file path escapes output directory: {raw_path!r}") from exc
    return destination


def _hash_file(path: Path) -> tuple[int, str]:
    digest = hashlib.sha256()
    size = 0
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(CHUNK_SIZE), b""):
            size += len(chunk)
            digest.update(chunk)
    return size, digest.hexdigest()


def verify_file(path: Path, expected_size: int, expected_sha256: str) -> bool:
    """Check a regular, non-symlink file against exact API metadata."""
    if not path.is_file() or path.is_symlink():
        return False
    if path.stat().st_size != int(expected_size):
        return False
    size, digest = _hash_file(path)
    return size == int(expected_size) and digest == expected_sha256.lower()


def _download_one(item: dict[str, Any], repo: str, revision: str, output: Path, prefix: str,
                  verify_only: bool, attempts: int) -> tuple[str, str, int]:
    destination = _safe_destination(output, item["path"], prefix)
    if verify_file(destination, item["size"], item["sha256"]):
        return item["path"], "verified", int(item["size"])
    if verify_only:
        return item["path"], "missing-or-hash-mismatch", int(item["size"])
    if destination.is_symlink():
        raise DownloadError(f"refusing to replace symlink: {destination}")
    destination.parent.mkdir(parents=True, exist_ok=True)
    encoded_repo = "/".join(quote(part, safe="") for part in repo.split("/"))
    encoded_path = quote(item["path"], safe="/")
    url = f"{DOWNLOAD_BASE}/{encoded_repo}/resolve/{quote(revision, safe='')}/{encoded_path}"
    last: Exception | None = None
    for attempt in range(attempts):
        temporary = destination.with_name(
            f".{destination.name}.part-{os.getpid()}-{next(tempfile._get_candidate_names())}"
        )
        try:
            request = Request(url, headers={"User-Agent": "hnet-corpus-fetch/1"})
            with urlopen(request, timeout=120) as response, temporary.open("wb") as handle:
                digest = hashlib.sha256()
                size = 0
                while True:
                    chunk = response.read(CHUNK_SIZE)
                    if not chunk:
                        break
                    handle.write(chunk)
                    digest.update(chunk)
                    size += len(chunk)
                handle.flush()
                os.fsync(handle.fileno())
            actual = digest.hexdigest()
            if size != int(item["size"]) or actual != item["sha256"].lower():
                raise DownloadError(
                    f"verification failed for {item['path']}: size {size} (expected {item['size']}), "
                    f"sha256 {actual} (expected {item['sha256']})"
                )
            os.replace(temporary, destination)
            return item["path"], "downloaded", size
        except (HTTPError, URLError, TimeoutError, OSError, DownloadError) as exc:
            last = exc
            try:
                temporary.unlink()
            except FileNotFoundError:
                pass
            if attempt + 1 < attempts:
                time.sleep(min(30.0, 1.5 * (2**attempt)))
    raise DownloadError(f"failed {item['path']}: {last}") from last


def fetch(args: argparse.Namespace) -> dict[str, int]:
    files = list_files(args.repo, args.revision, args.path)
    output = Path(args.output).expanduser().resolve()
    output.mkdir(parents=True, exist_ok=True)
    counts = {"verified": 0, "downloaded": 0, "missing-or-hash-mismatch": 0}
    with ThreadPoolExecutor(max_workers=max(1, int(args.workers))) as pool:
        futures = {
            pool.submit(_download_one, item, args.repo, args.revision, output, args.path,
                        bool(args.verify_only), int(args.attempts)): item["path"]
            for item in files
        }
        for future in as_completed(futures):
            path, status, size = future.result()
            counts[status] += 1
            print(json.dumps({"path": path, "status": status, "size": size}, sort_keys=True), flush=True)
    if counts["missing-or-hash-mismatch"]:
        raise DownloadError(f"{counts['missing-or-hash-mismatch']} files failed verification")
    return {"files": len(files), **counts}


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--repo", required=True, help="dataset repo, e.g. HuggingFaceFW/fineweb")
    parser.add_argument("--revision", required=True, help="immutable commit SHA")
    parser.add_argument("--path", required=True, help="dataset tree path, e.g. sample/100BT")
    parser.add_argument("--output", required=True, help="external payload directory")
    parser.add_argument("--workers", type=int, default=8)
    parser.add_argument("--attempts", type=int, default=8)
    parser.add_argument("--verify-only", action="store_true")
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    try:
        result = fetch(parse_args(argv))
    except DownloadError as exc:
        print(f"download_hf_shards: error: {exc}", file=__import__("sys").stderr)
        return 2
    print(json.dumps(result, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
