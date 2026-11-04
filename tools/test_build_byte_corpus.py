#!/usr/bin/env python3
"""Behavioral tests for the external byte-corpus builder."""

from __future__ import annotations

import json
import subprocess
import tempfile
import unittest
from pathlib import Path

try:
    from .build_byte_corpus import BuildError, build, parse_args
except ImportError:  # direct ``python test_build_byte_corpus.py`` execution
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
                "--output", str(output),
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

    def test_compact_index_manifest(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            data = root / "data.jsonl"
            rows = [
                {"id": str(i), "text": f"compact document {i} with enough bytes", "token_count": i + 2}
                for i in range(12)
            ]
            data.write_text("\n".join(json.dumps(row) for row in rows) + "\n", encoding="utf-8")
            spec = self._source_spec(root, data)
            out = root / "out"
            build(parse_args([
                "--sources", str(spec), "--output", str(out),
                "--validation-fraction", "0.25", "--compact-index",
            ]))
            for split in ("train", "validation"):
                manifest = json.loads((out / f"{split}.manifest.json").read_text())
                self.assertEqual(manifest["version"], 2)
                index = manifest["document_index"]
                index_path = out / index["path"]
                self.assertEqual(index["record_size"], 56)
                self.assertEqual(index_path.stat().st_size, 24 + index["count"] * 56)
                self.assertEqual(index["sha256"], __import__("hashlib").sha256(index_path.read_bytes()).hexdigest())
                metadata_path = out / manifest["document_metadata_path"]
                self.assertEqual(manifest["document_metadata_size"], metadata_path.stat().st_size)
                self.assertEqual(
                    manifest["document_metadata_sha256"],
                    __import__("hashlib").sha256(metadata_path.read_bytes()).hexdigest(),
                )
                self.assertNotIn("documents", manifest)

    def test_token_equivalent_budget_uses_token_count_and_source_quotas(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            first = root / "first.jsonl"
            second = root / "second.jsonl"
            first.write_text(
                "\n".join(json.dumps({
                    "id": f"first-{i}", "text": f"first document {i} with distinct content",
                    "token_count": 30, "url": f"https://first.example/doc/{i}",
                    "repository_id": f"first-repo-{i}",
                }) for i in range(10)) + "\n", encoding="utf-8"
            )
            second.write_text(
                "\n".join(json.dumps({
                    "id": f"second-{i}", "text": f"second document {i} with distinct content",
                    "token_count": 30, "url": f"https://second.example/doc/{i}",
                    "repository_id": f"second-repo-{i}",
                }) for i in range(10)) + "\n", encoding="utf-8"
            )
            spec = root / "sources.json"
            spec.write_text(json.dumps([
                {
                    "path": str(first), "source": "first", "revision": "r1",
                    "license": "ODC-By 1.0", "language": "en", "weight": 1.0,
                    "tier": "base",
                },
                {
                    "path": str(second), "source": "second", "revision": "r2",
                    "license": "ODC-By 1.0", "language": "en", "weight": 1.0,
                    "tier": "base",
                },
            ]), encoding="utf-8")
            out = root / "out"
            stats = build(parse_args([
                "--sources", str(spec), "--output", str(out),
                "--target-token-equivalents", "300", "--validation-fraction", "0.5",
                "--compact-index",
            ]))
            self.assertEqual(stats.written, 10)
            metadata = []
            for split in ("train", "validation"):
                metadata.extend(
                    json.loads(line)
                    for line in (out / f"{split}.documents.jsonl").read_text().splitlines()
                )
            self.assertEqual(sum(int(doc["token_count"]) for doc in metadata), 300)
            self.assertEqual({doc["source"] for doc in metadata}, {"first", "second"})
            stats_json = json.loads((out / "build_stats.json").read_text())
            self.assertEqual(stats_json["budget_metric"], "token_equivalents")
            self.assertEqual(stats_json["budget_target"], 300)
            self.assertEqual(
                sum(group["realized"] for group in stats_json["budget_groups"].values()),
                300,
            )

    def test_weighted_budget_underfill_fails_before_ready(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            data = root / "short.jsonl"
            data.write_text(
                "\n".join(
                    json.dumps({
                        "id": f"short-{i}",
                        "text": f"short source document {i}",
                        "token_count": 3,
                        "repository_id": f"repo-{i}",
                    })
                    for i in range(20)
                )
                + "\n",
                encoding="utf-8",
            )
            spec = self._source_spec(root, data)
            out = root / "out"
            with self.assertRaises(BuildError):
                build(parse_args([
                    "--sources", str(spec), "--output", str(out),
                    "--target-token-equivalents", "10000",
                    "--validation-fraction", "0.5", "--compact-index",
                ]))
            self.assertFalse((out / "READY").exists())

    def test_budget_flags_are_mutually_exclusive(self) -> None:
        with self.assertRaises(SystemExit):
            parse_args([
                "--sources", "sources.json", "--output", "out",
                "--target-bytes", "100", "--target-token-equivalents", "100",
            ])


if __name__ == "__main__":
    unittest.main()
