"""Integrity checks for materializing frozen pipeline inputs."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path

import pytest

from pipelines.assets import Artifact, load_manifest, materialize, verify_file


def _artifact(content: bytes) -> Artifact:
    return Artifact.from_record(
        {
            "id": "sample",
            "role": "source",
            "destination": "inputs/sample.bin",
            "workspace_path": "source/sample.bin",
            "bytes": len(content),
            "sha256": hashlib.sha256(content).hexdigest(),
        }
    )


def test_workspace_copy_is_verified_and_cached(tmp_path: Path) -> None:
    content = b"fixed source bytes\n"
    artifact = _artifact(content)
    source = tmp_path / "workspace" / "source" / "sample.bin"
    source.parent.mkdir(parents=True)
    source.write_bytes(content)
    cache = tmp_path / "cache"

    destination = materialize(artifact, cache, tmp_path / "workspace", "workspace")
    assert destination.read_bytes() == content
    assert materialize(artifact, cache, tmp_path / "workspace", "workspace") == destination

    destination.write_bytes(b"corrupt")
    with pytest.raises(ValueError, match="size"):
        verify_file(destination, artifact)
    with pytest.raises(ValueError, match="size"):
        materialize(artifact, cache, tmp_path / "workspace", "workspace")


def test_bundled_selection_is_available_without_the_original_workspace(tmp_path: Path) -> None:
    artifact = next(
        item
        for item in load_manifest(Path(__file__).resolve().parents[1] / "manifests" / "assets.json")
        if item.identifier == "split_30k_seed42"
    )
    project_root = Path(__file__).resolve().parents[1]
    destination = materialize(artifact, tmp_path / "cache", None, "bundle", project_root)
    verify_file(destination, artifact)


def test_manifest_rejects_unsafe_and_duplicate_destinations(tmp_path: Path) -> None:
    record = {
        "id": "first",
        "role": "source",
        "destination": "inputs/x.bin",
        "workspace_path": "source/x.bin",
        "bytes": 0,
        "sha256": hashlib.sha256(b"").hexdigest(),
    }
    manifest = tmp_path / "assets.json"
    manifest.write_text(
        json.dumps({"schema_version": 1, "hash_algorithm": "sha256", "artifacts": [record]}),
        encoding="utf-8",
    )
    assert len(load_manifest(manifest)) == 1

    bad = dict(record, destination="../outside.bin")
    manifest.write_text(
        json.dumps({"schema_version": 1, "hash_algorithm": "sha256", "artifacts": [bad]}),
        encoding="utf-8",
    )
    with pytest.raises(ValueError, match="safe relative"):
        load_manifest(manifest)

    duplicate = dict(record, id="second")
    manifest.write_text(
        json.dumps(
            {"schema_version": 1, "hash_algorithm": "sha256", "artifacts": [record, duplicate]}
        ),
        encoding="utf-8",
    )
    with pytest.raises(ValueError, match="Duplicate"):
        load_manifest(manifest)
