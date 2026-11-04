#!/usr/bin/env python3
"""Offline tests for the pinned-shard downloader's safety checks."""

from __future__ import annotations

import hashlib
import tempfile
import unittest
from pathlib import Path

try:
    from .download_hf_shards import DownloadError, _safe_destination, verify_file
except ImportError:
    from download_hf_shards import DownloadError, _safe_destination, verify_file


class DownloadShardTest(unittest.TestCase):
    def test_exact_size_and_sha256_required(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "payload.bin"
            payload = b"immutable shard\n"
            path.write_bytes(payload)
            digest = hashlib.sha256(payload).hexdigest()
            self.assertTrue(verify_file(path, len(payload), digest))
            self.assertFalse(verify_file(path, len(payload) + 1, digest))
            self.assertFalse(verify_file(path, len(payload), "0" * 64))

    def test_symlink_is_never_accepted(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            target = root / "target"
            target.write_bytes(b"x")
            link = root / "link"
            link.symlink_to(target)
            self.assertFalse(verify_file(link, 1, hashlib.sha256(b"x").hexdigest()))

    def test_path_must_stay_under_requested_prefix(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            self.assertEqual(
                _safe_destination(root, "sample/100BT/000.parquet", "sample/100BT"),
                root / "000.parquet",
            )
            with self.assertRaises(DownloadError):
                _safe_destination(root, "sample/other/000.parquet", "sample/100BT")


if __name__ == "__main__":
    unittest.main()
