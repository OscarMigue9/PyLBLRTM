"""Tests for byte ranges used to verify the archived radiance parent."""

from __future__ import annotations

import hashlib
import io
import json
from pathlib import Path

import numpy as np
import pytest

from pipelines import radiance_parent_canary


def _sha256(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def _fixture(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> tuple[Path, Path, bytearray]:
    full = np.arange(4 * 5 * 6, dtype=np.float32).reshape(4, 5, 6)
    full_buffer = io.BytesIO()
    np.save(full_buffer, full)
    full_bytes = bytearray(full_buffer.getvalue())
    angles = np.array([0.55, 0.67, 0.8, 0.92, 1.0])
    selected = np.random.default_rng(991).integers(0, 5, size=4)
    nu = np.arange(645.0, 646.5, 0.25)
    meta_buffer = io.BytesIO()
    np.savez(meta_buffer, nu=nu, mus=angles)
    meta_bytes = meta_buffer.getvalue()

    archive_path = tmp_path / "radiance.npz"
    np.savez(
        archive_path,
        L=full[np.arange(4), selected, 2:5],
        mu=angles[selected].astype(np.float32),
        nu=nu[2:5],
    )
    archive_bytes = archive_path.read_bytes()
    manifest = {
        "schema_version": 1,
        "source": {
            "remote_uri": "memory:full.npy",
            "bytes": len(full_bytes),
            "sha256": _sha256(full_bytes),
        },
        "source_metadata": {
            "remote_uri": "memory:meta.npz",
            "bytes": len(meta_bytes),
            "sha256": _sha256(meta_bytes),
        },
        "layout": {
            "dtype": "<f4",
            "shape": [4, 5, 6],
            "npy_header_bytes": 128,
            "channel_start": 2,
            "channel_count": 3,
            "angles": angles.tolist(),
            "selection_seed": 991,
        },
        "frozen_radiance": {
            "bytes": len(archive_bytes),
            "sha256": _sha256(archive_bytes),
            "array": "L",
        },
        "canaries": [
            {
                "profile_index": index,
                "angle_index": int(selected[index]),
                "row_sha256": _sha256(full[index, selected[index], 2:5].tobytes()),
            }
            for index in (0, 3)
        ],
    }
    manifest_path = tmp_path / "manifest.json"
    manifest_path.write_text(json.dumps(manifest), encoding="utf-8")

    def fake_rclone(_binary: str, *args: str) -> bytes:
        if args[:3] == ("lsjson", "--hash", "--stat"):
            return json.dumps(
                {"Size": len(full_bytes), "Hashes": {"sha256": manifest["source"]["sha256"]}}
            ).encode()
        if args[:1] != ("cat",):
            raise AssertionError(args)
        if args[1] == "memory:meta.npz":
            return meta_bytes
        offset = int(args[2])
        count = int(args[4])
        return bytes(full_bytes[offset : offset + count])

    monkeypatch.setattr(radiance_parent_canary, "_rclone", fake_rclone)
    return manifest_path, archive_path, full_bytes


def test_verifies_selected_angle_and_numpy_row_offsets(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    manifest, archive, _source = _fixture(tmp_path, monkeypatch)
    report = radiance_parent_canary.verify_parent(manifest, archive)
    assert report["status"] == "pass"
    assert report["angle_assignment_all_rows_equal"]
    assert [row["profile_index"] for row in report["canaries"]] == [0, 3]
    assert all(row["byte_equal_to_frozen"] for row in report["canaries"])


def test_detects_modified_remote_row(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    manifest, archive, source = _fixture(tmp_path, monkeypatch)
    angle = json.loads(manifest.read_text(encoding="utf-8"))["canaries"][0]["angle_index"]
    source[128 + (angle * 6 + 2) * 4] ^= 1
    with pytest.raises(ValueError, match="Parent row 0 SHA256"):
        radiance_parent_canary.verify_parent(manifest, archive)


def test_full_parent_compares_all_values_in_batches(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    manifest, archive, source = _fixture(tmp_path, monkeypatch)
    parent = tmp_path / "full.npy"
    parent.write_bytes(source)
    report = radiance_parent_canary.verify_full_parent(parent, manifest, archive, batch_size=2)
    assert report["status"] == "pass"
    assert report["profile_count"] == 4
    assert report["value_count"] == 12
    assert report["unequal_count"] == 0
    assert report["rms_difference"] == 0.0
    assert report["max_absolute_difference"] == 0.0
    assert report["batches"] == 2


def test_full_parent_reports_exactly_one_wrong_value(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    manifest, archive, source = _fixture(tmp_path, monkeypatch)
    parent = tmp_path / "full.npy"
    parent.write_bytes(source)
    with np.load(archive) as frozen:
        radiance = frozen["L"].copy()
        mu = frozen["mu"].copy()
        nu = frozen["nu"].copy()
    radiance[2, 1] += 1.0
    np.savez(archive, L=radiance, mu=mu, nu=nu)
    document = json.loads(manifest.read_text(encoding="utf-8"))
    document["frozen_radiance"]["bytes"] = archive.stat().st_size
    document["frozen_radiance"]["sha256"] = _sha256(archive.read_bytes())
    manifest.write_text(json.dumps(document), encoding="utf-8")

    report = radiance_parent_canary.verify_full_parent(parent, manifest, archive, batch_size=2)
    assert report["status"] == "fail"
    assert report["unequal_count"] == 1
    assert report["rms_difference"] == pytest.approx((1 / 12) ** 0.5)
    assert report["max_absolute_difference"] == 1.0
