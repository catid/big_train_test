#!/usr/bin/env python3
"""Tests for document-local byte sampling and rank partitioning."""

from __future__ import annotations

import hashlib
import json
import sys
import tempfile
import unittest
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from train_hnet_600m import ByteStream, _load_manifest  # noqa: E402


class ByteStreamTest(unittest.TestCase):
    def _manifest(self, root: Path) -> Path:
        payload = b"A" * 16 + b"B" * 16 + b"C" * 16 + b"D" * 16
        data = root / "train.bin"
        data.write_bytes(payload)
        documents = []
        for index in range(4):
            body = payload[index * 16 : (index + 1) * 16]
            documents.append(
                {
                    "id": f"doc-{index}",
                    "offset": index * 16,
                    "length": 16,
                    "sha256": hashlib.sha256(body).hexdigest(),
                    "token_count": 4,
                    "source": "test",
                    "license": "test",
                    "language": "en",
                    "repository_id": f"repo-{index}",
                }
            )
        manifest = root / "train.manifest.json"
        manifest.write_text(
            json.dumps(
                {
                    "version": 1,
                    "data_path": data.name,
                    "data_size": len(payload),
                    "data_sha256": hashlib.sha256(payload).hexdigest(),
                    "total_token_equivalents": 16,
                    "documents": documents,
                    "split": "train",
                }
            ),
            encoding="utf-8",
        )
        return manifest

    def test_windows_never_cross_document_boundaries(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            manifest = _load_manifest(self._manifest(Path(directory)))
            stream = ByteStream(manifest, seq=8, batch=2, rank=0, world=1, seed=19)
            x, y, active, loss = stream.get(0, torch.device("cpu"))
            self.assertEqual(tuple(x.shape), (2, 8))
            self.assertTrue(bool(torch.all(active)))
            self.assertTrue(bool(torch.all(loss)))
            self.assertTrue(bool(torch.all(x == y)))
            # A row contains only one repeated document marker. If sampling
            # ever crossed an offset, adjacent rows would expose two markers.
            for row in x:
                self.assertEqual(int(torch.unique(row).numel()), 1)

    def test_world_partition_matches_global_batch_order(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            manifest = _load_manifest(self._manifest(Path(directory)))
            global_stream = ByteStream(manifest, seq=8, batch=2, rank=0, world=1, seed=29)
            rank0 = ByteStream(manifest, seq=8, batch=1, rank=0, world=2, seed=29)
            rank1 = ByteStream(manifest, seq=8, batch=1, rank=1, world=2, seed=29)
            global_x, *_ = global_stream.get(0, torch.device("cpu"))
            rank0_x, *_ = rank0.get(0, torch.device("cpu"))
            rank1_x, *_ = rank1.get(0, torch.device("cpu"))
            self.assertTrue(torch.equal(global_x[0], rank0_x[0]))
            self.assertTrue(torch.equal(global_x[1], rank1_x[0]))


if __name__ == "__main__":
    unittest.main()
