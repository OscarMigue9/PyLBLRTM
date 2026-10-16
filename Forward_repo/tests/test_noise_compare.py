"""Checks for complete frozen-versus-regenerated noise comparison."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path

import numpy as np
import pytest

from pipelines.noise_compare import compare_noise


def _fixture(tmp_path: Path) -> tuple[Path, Path, Path]:
    frozen = tmp_path / "frozen.npz"
    candidate = tmp_path / "candidate.npz"
    axis = (714 + 0.25 * np.arange(4)).astype(np.float32)
    noise = np.ones((3, 4), dtype=np.float32)
    contents = dict(
        noise=noise,
        sigma_ch=np.full(4, 0.2, dtype=np.float32),
        nu=axis,
        seed=1337,
        grid="iasiang",
        dnu_ch=0.25,
        fwhm=0.5,
        receta="original",
    )
    np.savez_compressed(frozen, **contents)
    changed = noise.copy()
    changed[0, 0] = np.nextafter(changed[0, 0], np.float32(2))
    changed[2, 3] = 2
    np.savez_compressed(candidate, **(contents | {"noise": changed, "receta": "new note"}))
    manifest = tmp_path / "assets.json"
    manifest.write_text(
        json.dumps(
            {
                "schema_version": 1,
                "hash_algorithm": "sha256",
                "artifacts": [
                    {
                        "id": "noise_iasiang",
                        "role": "training_input",
                        "destination": "data/outputs/noise_iasiang.npz",
                        "bytes": frozen.stat().st_size,
                        "sha256": hashlib.sha256(frozen.read_bytes()).hexdigest(),
                        "workspace_path": "frozen.npz",
                    }
                ],
            }
        ),
        encoding="utf-8",
    )
    return candidate, frozen, manifest


def test_reports_all_unequal_cells_and_bounded_examples(tmp_path: Path) -> None:
    candidate, frozen, manifest = _fixture(tmp_path)
    report = compare_noise(
        candidate, frozen, manifest_path=manifest, expected_shape=(3, 4), max_examples=1
    )
    noise = report["arrays"]["noise"]
    assert report["status"] == "numerically_different"
    assert report["frozen"]["manifest_validated"]
    assert noise["cells"] == 12
    assert noise["unequal_cells"] == noise["different_bit_pattern_cells"] == 2
    assert noise["examples"][0]["index"] == [0, 0]
    assert noise["examples_truncated"]
    assert noise["max_abs_error"] == 1.0
    assert report["arrays"]["nu"]["unequal_cells"] == 0
    assert report["arrays"]["sigma_ch"]["unequal_cells"] == 0


def test_descriptive_note_does_not_change_numerical_identity(tmp_path: Path) -> None:
    candidate, frozen, manifest = _fixture(tmp_path)
    with np.load(frozen, allow_pickle=False) as source:
        contents = {key: source[key] for key in source.files}
    np.savez_compressed(candidate, **(contents | {"receta": "new note"}))
    report = compare_noise(candidate, frozen, manifest_path=manifest, expected_shape=(3, 4))
    assert report["status"] == "exact_arrays"
    assert report["arrays"]["noise"]["unequal_cells"] == 0
    assert report["candidate"]["sha256"] != report["frozen"]["sha256"]


def test_rejects_frozen_hash_or_wrong_channel_axis(tmp_path: Path) -> None:
    candidate, frozen, manifest = _fixture(tmp_path)
    with np.load(candidate, allow_pickle=False) as source:
        content = {key: source[key] for key in source.files}
    content["nu"][1] += 0.25
    np.savez_compressed(candidate, **content)
    with pytest.raises(ValueError, match="nu differs"):
        compare_noise(candidate, frozen, manifest_path=manifest, expected_shape=(3, 4))

    frozen.write_bytes(frozen.read_bytes() + b"x")
    with pytest.raises(ValueError, match="size"):
        compare_noise(candidate, frozen, manifest_path=manifest, expected_shape=(3, 4))


def test_build_report_must_name_candidate_hash_and_recipe(tmp_path: Path) -> None:
    candidate, frozen, manifest = _fixture(tmp_path)
    build = tmp_path / "build.json"
    build.write_text(
        json.dumps(
            {
                "sha256": "0" * 64,
                "profiles": 3,
                "channels": 4,
                "seed": 1337,
                "batch_size": 256,
            }
        ),
        encoding="utf-8",
    )
    with pytest.raises(ValueError, match="SHA256 differs"):
        compare_noise(
            candidate,
            frozen,
            manifest_path=manifest,
            expected_shape=(3, 4),
            build_report_path=build,
        )
