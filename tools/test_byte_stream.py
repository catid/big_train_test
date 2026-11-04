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

    def _weighted_manifest(self, root: Path) -> Path:
        lengths = [10, 20, 30]
        payload = b"".join(bytes([65 + index]) * length for index, length in enumerate(lengths))
        data = root / "train.bin"
        data.write_bytes(payload)
        documents = []
        offset = 0
        for index, length in enumerate(lengths):
            body = payload[offset : offset + length]
            documents.append(
                {
                    "id": f"doc-{index}",
                    "offset": offset,
                    "length": length,
                    "sha256": hashlib.sha256(body).hexdigest(),
                    "token_count": length,
                    "source": "test",
                    "license": "test",
                    "language": "en",
                }
            )
            offset += length
        manifest = root / "weighted.manifest.json"
        manifest.write_text(
            json.dumps(
                {
                    "version": 1,
                    "data_path": data.name,
                    "data_size": len(payload),
                    "data_sha256": hashlib.sha256(payload).hexdigest(),
                    "total_token_equivalents": sum(lengths),
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

    def test_contiguous_payload_uses_byte_weighted_sampling(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            manifest = _load_manifest(self._weighted_manifest(Path(directory)))
            stream = ByteStream(manifest, seq=8, batch=1, rank=0, world=1, seed=41)
            counts = [0, 0, 0]
            for step in range(3000):
                x, *_ = stream.get(step, torch.device("cpu"))
                counts[int(x[0, 0]) - 65] += 1
            # The packed byte ranges have lengths 10:20:30.  Sampling a
            # virtual payload byte should preserve that ratio to within 10%.
            total = sum(counts)
            expected = [total / 6.0, total / 3.0, total / 2.0]
            for actual, target in zip(counts, expected):
                self.assertLess(abs(actual - target), target * 0.10)

    def test_repeated_document_draws_use_distinct_windows(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            payload = bytes(range(100))
            data = root / "train.bin"
            data.write_bytes(payload)
            document = {
                "id": "long-doc",
                "offset": 0,
                "length": len(payload),
                "sha256": hashlib.sha256(payload).hexdigest(),
                "token_count": len(payload),
                "source": "test",
                "license": "test",
                "language": "en",
            }
            manifest_path = root / "long.manifest.json"
            manifest_path.write_text(
                json.dumps(
                    {
                        "version": 1,
                        "data_path": data.name,
                        "data_size": len(payload),
                        "data_sha256": hashlib.sha256(payload).hexdigest(),
                        "total_token_equivalents": len(payload),
                        "documents": [document],
                    }
                ),
                encoding="utf-8",
            )
            stream = ByteStream(
                _load_manifest(manifest_path), seq=8, batch=4, rank=0, world=1, seed=41
            )
            x, y, active, loss = stream.get(0, torch.device("cpu"))
            self.assertGreater(int(torch.unique(x[:, 0]).numel()), 1)
            self.assertTrue(bool(torch.all(y == x + 1)))
            self.assertTrue(bool(torch.all(active)))
            self.assertTrue(bool(torch.all(loss)))

    def test_weighted_sampling_preserves_ddp_global_order(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            manifest = _load_manifest(self._weighted_manifest(Path(directory)))
            global_stream = ByteStream(manifest, seq=8, batch=2, rank=0, world=1, seed=53)
            rank0 = ByteStream(manifest, seq=8, batch=1, rank=0, world=2, seed=53)
            rank1 = ByteStream(manifest, seq=8, batch=1, rank=1, world=2, seed=53)
            for step in range(2):
                global_x, *_ = global_stream.get(step, torch.device("cpu"))
                rank0_x, *_ = rank0.get(step, torch.device("cpu"))
                rank1_x, *_ = rank1.get(step, torch.device("cpu"))
                self.assertTrue(torch.equal(global_x[0], rank0_x[0]))
                self.assertTrue(torch.equal(global_x[1], rank1_x[0]))


if __name__ == "__main__":
    unittest.main()
