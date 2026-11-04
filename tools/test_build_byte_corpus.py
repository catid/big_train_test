#!/usr/bin/env python3
"""Behavioral tests for the external byte-corpus builder."""

from __future__ import annotations

import json
import subprocess
import tempfile
import unittest
from pathlib import Path

from build_byte_corpus import BuildError, build, parse_args


class BuildByteCorpusTest(unittest.TestCase):
    def _source_spec(self, root: Path, *paths: Path) -> Path:
        spec = []
        for index, path in enumerate(paths):
            spec.append({
                "path": str(path), "source": f"source-{index}",
                "revision": f"rev-{index}", "license": "ODC-By 1.0",
                "language": "en", "weight": 1.0, "tier": "base",
            })
        path = root / "sources.json"
        path.write_text(json.dumps(spec), encoding="utf-8")
        return path

    def test_manifest_hashes_split_and_cross_source_dedup(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            first = root / "first.jsonl"
            second = root / "second.jsonl"
            rows = [
                {"id": f"doc-{i}", "text": f"A distinct document number {i} with enough text for testing.", "token_count": 10, "url": f"https://repo-{i}.example/doc"}
                for i in range(80)
            ]
            first.write_text("\n".join(json.dumps(row) for row in rows[:40]) + "\n", encoding="utf-8")
            duplicate = dict(rows[0]); duplicate["id"] = "duplicate-from-second"
            second.write_text("\n".join(json.dumps(row) for row in [duplicate, *rows[40:]]) + "\n", encoding="utf-8")
            output = root / "out"
            stats = build(parse_args([
                "--sources", str(self._source_spec(root, first, second)),
                "--output", str(output), "--target-bytes", "100000",
                "--validation-fraction", "0.25", "--near-dedup",
            ]))
            self.assertEqual(stats.duplicate_exact, 1)
            self.assertEqual(stats.written, 80)
            self.assertTrue((output / "READY").is_file())
            for split in ("train", "validation"):
                manifest = json.loads((output / f"{split}.manifest.json").read_text())
                payload = (output / manifest["data_path"]).read_bytes()
                self.assertEqual(manifest["data_size"], len(payload))
                self.assertEqual(manifest["data_sha256"], __import__("hashlib").sha256(payload).hexdigest())
                self.assertEqual(manifest["total_token_equivalents"], sum(d["token_count"] for d in manifest["documents"]))
                for document in manifest["documents"]:
                    body = payload[document["offset"]:document["offset"] + document["length"]]
                    self.assertEqual(body.decode("utf-8").encode("utf-8"), body)
                    self.assertEqual(document["sha256"], __import__("hashlib").sha256(body).hexdigest())
            self.assertTrue(json.loads((output / "train.manifest.json").read_text())["documents"])
            self.assertTrue(json.loads((output / "validation.manifest.json").read_text())["documents"])

    def test_rejects_missing_token_count_without_explicit_estimate(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            data = root / "data.jsonl"
            data.write_text(json.dumps({"id": "x", "text": "hello"}) + "\n", encoding="utf-8")
            spec = self._source_spec(root, data)
            with self.assertRaises(BuildError):
                build(parse_args(["--sources", str(spec), "--output", str(root / "out"), "--validation-fraction", "0"]))

    def test_source_quality_filter_runs_before_dedup_and_budget(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            data = root / "edu.jsonl"
            rows = [{"id": "low", "text": "low quality document", "token_count": 3, "score": 2.0}]
            rows.extend(
                {"id": f"high-{i}", "text": f"high quality document {i}", "token_count": 3, "score": 3.0}
                for i in range(20)
            )
            data.write_text("\n".join(json.dumps(row) for row in rows) + "\n", encoding="utf-8")
            spec = root / "sources.json"
            spec.write_text(json.dumps([{
                "path": str(data), "source": "fineweb-edu", "revision": "r",
                "license": "ODC-By 1.0", "language": "en", "weight": 1.0,
                "tier": "edu3", "min_score": 3.0,
            }]), encoding="utf-8")
            out = root / "out"
            stats = build(parse_args([
                "--sources", str(spec), "--output", str(out),
                "--validation-fraction", "0.5",
            ]))
            self.assertEqual(stats.filtered, 1)
            self.assertEqual(stats.written, 20)
            for split in ("train", "validation"):
                manifest = json.loads((out / f"{split}.manifest.json").read_text())
                self.assertTrue(manifest["documents"])
                self.assertTrue(all(doc["id"].startswith("high-") for doc in manifest["documents"]))

    def test_zstd_input(self) -> None:
        if not subprocess.run(["sh", "-c", "command -v zstd"], capture_output=True).returncode == 0:
            self.skipTest("zstd executable unavailable")
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            plain = root / "data.jsonl"
            plain.write_text("\n".join(json.dumps({"id": str(i), "text": f"zstd record {i}", "token_count": 2}) for i in range(20)) + "\n", encoding="utf-8")
            compressed = root / "data.jsonl.zst"
            subprocess.run(["zstd", "--quiet", "--rm", "-o", str(compressed), str(plain)], check=True)
            spec = self._source_spec(root, compressed)
            out = root / "out"
            build(parse_args(["--sources", str(spec), "--output", str(out), "--validation-fraction", "0.2"]))
            self.assertTrue((out / "READY").exists())


if __name__ == "__main__":
    unittest.main()
