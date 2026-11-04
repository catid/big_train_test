#!/usr/bin/env python3
"""Checks for strict and rank-zero-only manifest verification."""

from __future__ import annotations

import sys
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import train_hnet_600m as trainer


class ManifestLoadingTest(unittest.TestCase):
    manifest = "/home/catid/hnet_data/byte_sample_compact/train.manifest.json"

    def test_default_hashes_and_structural_only_match(self) -> None:
        if not Path(self.manifest).is_file():
            self.skipTest("external compact pilot is not installed")
        strict = trainer._load_manifest(self.manifest)
        structural = trainer._load_manifest(self.manifest, verify_hashes=False)
        self.assertEqual(strict.data_sha256, structural.data_sha256)
        self.assertEqual(strict.manifest_sha256, structural.manifest_sha256)
        self.assertEqual(strict.document_count, structural.document_count)
        self.assertEqual(
            trainer._manifest_integrity_record(strict),
            trainer._manifest_integrity_record(structural),
        )


if __name__ == "__main__":
    unittest.main()
