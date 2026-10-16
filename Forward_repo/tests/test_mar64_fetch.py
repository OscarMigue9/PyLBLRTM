"""Offline checks for byte-pinned ocean scene acquisition."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path

import pytest

from pipelines.mar64_fetch import fetch_mar64_scenes, load_scene_manifest


def _manifest(tmp_path: Path) -> tuple[Path, dict[str, bytes]]:
    content = {f"scene_{index:03d}.npz": f"scene {index}\n".encode() for index in range(64)}
    document = {
        "schema_version": 1,
        "dataset": "iasi_ocean_64",
        "selection": {
            "scene_count": 64,
            "glob": "*.npz",
            "sort": "lexicographic_filename",
        },
        "remote_scene_directory": "fake:final/mar64",
        "scene_files": [
            {
                "name": name,
                "bytes": len(data),
                "sha256": hashlib.sha256(data).hexdigest(),
            }
            for name, data in content.items()
        ],
    }
    manifest = tmp_path / "mar64_inputs.json"
    manifest.write_text(json.dumps(document), encoding="utf-8")
    return manifest, content


def test_fresh_and_partial_cache_download_only_missing_verified_files(tmp_path: Path) -> None:
    manifest, content = _manifest(tmp_path)
    destination = tmp_path / "cache" / "data" / "valstates_full"
    destination.mkdir(parents=True)
    (destination / "scene_000.npz").write_bytes(content["scene_000.npz"])
    calls = []

    def download(uri: str, temporary: Path) -> None:
        calls.append(uri)
        temporary.write_bytes(content[uri.rsplit("/", 1)[1]])

    paths = fetch_mar64_scenes(destination, manifest, downloader=download)
    assert len(paths) == 64
    assert len(calls) == 63
    assert calls[0] == "fake:final/mar64/scene_001.npz"
    assert [path.read_bytes() for path in paths] == list(content.values())
    assert fetch_mar64_scenes(destination, manifest, downloader=download) == paths
    assert len(calls) == 63


def test_existing_wrong_bytes_abort_before_any_download(tmp_path: Path) -> None:
    manifest, _ = _manifest(tmp_path)
    destination = tmp_path / "valstates_full"
    destination.mkdir()
    wrong = destination / "scene_010.npz"
    wrong.write_bytes(b"incorrect")
    calls = []

    def download(uri: str, temporary: Path) -> None:
        calls.append(uri)
        temporary.write_bytes(b"unexpected")

    with pytest.raises(ValueError, match="bytes differ"):
        fetch_mar64_scenes(destination, manifest, downloader=download)
    assert not calls
    assert wrong.read_bytes() == b"incorrect"
    assert sorted(path.name for path in destination.iterdir()) == [wrong.name]


def test_bad_remote_bytes_are_not_published(tmp_path: Path) -> None:
    manifest, _ = _manifest(tmp_path)
    destination = tmp_path / "valstates_full"

    def download(_uri: str, temporary: Path) -> None:
        temporary.write_bytes(b"bad data")

    with pytest.raises(ValueError, match="bytes differ"):
        fetch_mar64_scenes(destination, manifest, downloader=download)
    assert list(destination.iterdir()) == []


def test_file_appearing_during_download_is_checked_not_overwritten(tmp_path: Path) -> None:
    manifest, content = _manifest(tmp_path)
    destination = tmp_path / "valstates_full"

    def download(uri: str, temporary: Path) -> None:
        name = uri.rsplit("/", 1)[1]
        temporary.write_bytes(content[name])
        (destination / name).write_bytes(b"incorrect")

    with pytest.raises(ValueError, match="bytes differ"):
        fetch_mar64_scenes(destination, manifest, downloader=download)
    assert (destination / "scene_000.npz").read_bytes() == b"incorrect"
    assert sorted(path.name for path in destination.iterdir()) == ["scene_000.npz"]


def test_unknown_file_and_unsafe_manifest_name_are_rejected(tmp_path: Path) -> None:
    manifest, _ = _manifest(tmp_path)
    destination = tmp_path / "valstates_full"
    destination.mkdir()
    (destination / "unexpected.npz").write_bytes(b"unknown")
    with pytest.raises(ValueError, match="Unknown ocean scene"):
        fetch_mar64_scenes(destination, manifest, downloader=lambda *_: None)

    document = json.loads(manifest.read_text(encoding="utf-8"))
    document["scene_files"][0]["name"] = "../outside.npz"
    manifest.write_text(json.dumps(document), encoding="utf-8")
    with pytest.raises(ValueError, match="Invalid ocean scene identity"):
        load_scene_manifest(manifest)
