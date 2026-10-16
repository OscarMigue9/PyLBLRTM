"""Scientific identity checks for the archived spectroscopy input pack."""

from __future__ import annotations

import hashlib
import io
import json

import numpy as np

from pipelines.generation.gpu_pack import (
    PACK_MANIFEST,
    ROOT,
    _pack_continuum,
    _pack_qtab,
    _variants,
)


def _archive_sha(arrays: dict) -> str:
    output = io.BytesIO()
    np.savez_compressed(output, **arrays)
    return hashlib.sha256(output.getvalue()).hexdigest()


def test_partition_snapshot_rebuilds_archived_qtab() -> None:
    manifest = json.loads(PACK_MANIFEST.read_text(encoding="utf-8"))
    variants = _variants(
        ROOT / manifest["tips_directory"],
        ROOT / manifest["tips_manifest"],
        manifest["legacy_co2_mapping"],
    )
    assert [item["qfile"].name for item in variants[15:18]] == [
        "q15.txt",
        "q120.txt",
        "q121.txt",
    ]
    assert [item["isotope"] for item in variants[15:18]] == ["9", "10", "A"]
    assert _archive_sha(_pack_qtab(variants)) == (
        "99c09e782fadc315727d5adbcd3190451ef8ea3d399bdd6ce6f07f45aee92d9a"
    )


def test_four_continuum_snapshots_rebuild_archived_pack() -> None:
    manifest = json.loads(PACK_MANIFEST.read_text(encoding="utf-8"))
    sources = [(item["prefix"], ROOT / item["bundle_path"]) for item in manifest["continuum"]]
    assert _archive_sha(_pack_continuum(sources)) == (
        "e8f18089c28d103e96c12ea2d7aa2ac7190e19cb095ab0a0de8cb7635c1e30f5"
    )
