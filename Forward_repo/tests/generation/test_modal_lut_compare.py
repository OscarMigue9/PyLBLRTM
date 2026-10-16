"""Checks for full-band LUT sampling and numerical acceptance."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pytest

from pipelines.assets import load_manifest
from pipelines.modal_lut_compare import (
    compare_full_dense,
    compare_sampled_dense,
    generation_source_snapshot,
    hash_file,
    planned_blocks,
    sample_plan,
    verify_candidate,
)
from pipelines.modal_lut_full import source_snapshot


ROOT = Path(__file__).resolve().parents[2]


def _small_plan() -> dict:
    return sample_plan(
        nt=2,
        npres=3,
        block_count=3,
        channels_per_block=10,
        channels_sampled_per_block=3,
        temperature_samples=2,
        pressure_samples=3,
    )


def _write_pair(tmp_path: Path, candidate: np.ndarray) -> tuple[Path, Path]:
    left = tmp_path / "candidate.npy"
    right = tmp_path / "archived.npy"
    np.save(left, candidate)
    np.save(right, np.zeros_like(candidate))
    return left, right


def test_sample_plan_covers_every_block_and_terminal_channel() -> None:
    plan = sample_plan()
    assert plan["temperature_indices"] == [0, 4, 9, 14, 19]
    assert plan["pressure_indices"] == [0, 48, 96, 144]
    assert len(plan["block_starts_cm1"]) == 63
    assert plan["block_starts_cm1"][0] == 635
    assert plan["block_starts_cm1"][-1] == 1255
    assert len(plan["local_channel_indices"]) == 301
    assert plan["local_channel_indices"][0] == 0
    assert plan["local_channel_indices"][-1] == 9999
    assert plan["terminal_global_channel_index"] == 630_000
    assert plan["channel_count"] == 63 * 301 + 1


def test_sampled_comparison_detects_terminal_channel(tmp_path: Path) -> None:
    candidate = np.zeros((2, 3, 31), dtype="<f4")
    candidate[1, 2, 30] = 3e-4
    left, right = _write_pair(tmp_path, candidate)
    plan = _small_plan()
    plan["retrieval_band_global_indices"] = [2, 15]
    report = compare_sampled_dense(left, right, plan)
    assert report["sample_count"] == 2 * 3 * (3 * 3 + 1)
    assert report["terminal_max_abs_log10_sigma"] == pytest.approx(3e-4)
    assert not report["gate"]["passed"]
    assert report["retrieval_band"]["passed"]
    assert report["retrieval_band"]["boundary_max_abs_log10_sigma"] == 0


def test_every_block_must_pass_even_if_overall_rms_passes(tmp_path: Path) -> None:
    plan = sample_plan(
        nt=2,
        npres=3,
        block_count=9,
        channels_per_block=10,
        channels_sampled_per_block=3,
        temperature_samples=2,
        pressure_samples=3,
    )
    candidate = np.zeros((2, 3, 91), dtype="<f4")
    candidate[:, :, [0, 4, 9]] = 5e-5
    left, right = _write_pair(tmp_path, candidate)
    report = compare_sampled_dense(left, right, plan)
    assert report["rms_log10_sigma"] < 2.5e-5
    assert report["max_abs_log10_sigma"] < 2.5e-4
    assert report["blocks"][0]["rms_log10_sigma"] > 2.5e-5
    assert not report["blocks"][0]["passed"]
    assert not report["gate"]["passed"]


def test_identical_dense_arrays_pass(tmp_path: Path) -> None:
    candidate = np.zeros((2, 3, 31), dtype="<f4")
    left, right = _write_pair(tmp_path, candidate)
    report = compare_sampled_dense(left, right, _small_plan())
    assert report["gate"]["passed"]
    assert report["rms_log10_sigma"] == 0
    assert report["max_abs_log10_sigma"] == 0


def test_full_scan_counts_every_cell_and_inclusive_band(tmp_path: Path) -> None:
    candidate = np.zeros((2, 3, 31), dtype="<f4")
    candidate[0, 0, 5] = 0.1  # Just outside the band's lower edge.
    candidate[0, 0, 6] = 0.2  # Band lower edge.
    candidate[1, 2, 21] = 0.3  # Band upper edge.
    candidate[0, 1, 22] = 0.4  # Just outside the band's upper edge.
    candidate[1, 1, 30] = 0.5  # Separate terminal channel.
    left, right = _write_pair(tmp_path, candidate)
    report = compare_full_dense(
        left,
        right,
        nt=2,
        npres=3,
        block_count=3,
        channels_per_block=10,
        chunk_channels=20,
        channel_spacing_cm1=1.0,
        retrieval_band_cm1=(641.0, 656.0),
    )
    difference = candidate.astype(np.float64)
    assert report["sample_count"] == difference.size
    assert report["rms_log10_sigma"] == pytest.approx(np.sqrt(np.mean(difference**2)))
    assert report["max_abs_log10_sigma"] == pytest.approx(0.5)
    assert len(report["blocks"]) == 3
    for index, block in enumerate(report["blocks"]):
        selected = difference[:, :, 10 * index : 10 * (index + 1)]
        assert block["sample_count"] == selected.size
        assert block["rms_log10_sigma"] == pytest.approx(np.sqrt(np.mean(selected**2)))
        assert block["max_abs_log10_sigma"] == pytest.approx(np.max(np.abs(selected)))
    band = report["retrieval_band"]
    exact_band = difference[:, :, 6:22]
    assert band["global_channel_indices_inclusive"] == [6, 21]
    assert band["sample_count"] == exact_band.size
    assert band["rms_log10_sigma"] == pytest.approx(np.sqrt(np.mean(exact_band**2)))
    assert band["max_abs_log10_sigma"] == pytest.approx(0.3)
    terminal = report["terminal_channel"]
    assert terminal["global_channel_index"] == 30
    assert terminal["wavenumber_cm1"] == 665.0
    assert terminal["sample_count"] == 6
    assert terminal["max_abs_log10_sigma"] == pytest.approx(0.5)
    assert not report["gate"]["passed"]


def test_full_scan_detects_a_cell_omitted_by_sampling(tmp_path: Path) -> None:
    candidate = np.zeros((2, 3, 31), dtype="<f4")
    candidate[1, 2, 2] = 3e-4
    left, right = _write_pair(tmp_path, candidate)
    sampled = compare_sampled_dense(left, right, _small_plan())
    full = compare_full_dense(
        left,
        right,
        nt=2,
        npres=3,
        block_count=3,
        channels_per_block=10,
        chunk_channels=20,
        channel_spacing_cm1=1.0,
        retrieval_band_cm1=(636.0, 655.0),
    )
    assert sampled["gate"]["passed"]
    assert full["max_abs_log10_sigma"] == pytest.approx(3e-4)
    assert not full["blocks"][0]["passed"]
    assert not full["gate"]["passed"]


def test_full_scan_exact_714_to_1250_band(tmp_path: Path) -> None:
    candidate = np.zeros((1, 1, 631), dtype="<f4")
    candidate[0, 0, 79] = 1e-5  # 714 cm^-1.
    candidate[0, 0, 615] = 2e-5  # 1250 cm^-1.
    left, right = _write_pair(tmp_path, candidate)
    report = compare_full_dense(
        left,
        right,
        nt=1,
        npres=1,
        block_count=63,
        channels_per_block=10,
        chunk_channels=100,
        channel_spacing_cm1=1.0,
    )
    band = report["retrieval_band"]
    assert band["range_cm1"] == [714.0, 1250.0]
    assert band["global_channel_indices_inclusive"] == [79, 615]
    assert band["sample_count"] == 537
    assert band["rms_log10_sigma"] == pytest.approx(
        np.sqrt((float(candidate[0, 0, 79]) ** 2 + float(candidate[0, 0, 615]) ** 2) / 537)
    )
    assert report["terminal_channel"]["wavenumber_cm1"] == 1265.0
    assert report["terminal_channel"]["sample_count"] == 1
    assert report["gate"]["passed"]


def test_source_snapshot_matches_candidate_builder() -> None:
    assert generation_source_snapshot(ROOT) == source_snapshot(ROOT)


def test_all_block_receipts_and_bytes_are_checked(tmp_path: Path) -> None:
    gas = "CO2"
    records = planned_blocks(ROOT / "manifests/lut_blocks.json", gas)
    assets = {item.identifier: item for item in load_manifest(ROOT / "manifests/assets.json")}
    snapshot = {"sha256": "source", "files": {"generator.py": "hash"}}
    inputs = {
        identifier: assets[identifier].sha256
        for identifier in ("gpu_pack_lines", "gpu_pack_qtab", "gpu_pack_continuum")
    }
    base = tmp_path / gas
    blocks_dir = base / "blocks"
    blocks_dir.mkdir(parents=True)
    dense = base / "dense" / "lut_band645_CO2.npy"
    dense.parent.mkdir()
    dense.write_bytes(b"dense candidate")
    block_hashes = {}
    for record in records:
        block = blocks_dir / record["filename"]
        block.write_bytes(record["filename"].encode())
        digest = hash_file(block)
        block_hashes[block.name] = digest
        receipt = {
            "schema_version": 1,
            "status": "candidate_historical_source_unverified",
            "gas": gas,
            "start_cm1": record["lower_cm1"],
            "end_cm1": record["upper_cm1"],
            "source_snapshot": snapshot,
            "input_sha256": inputs,
            "candidate_sha256": digest,
            "candidate_bytes": block.stat().st_size,
            "archived_reference": record,
            "builder_report": {"blocks": [{"sha256": digest}]},
            "inspection": {
                "gas": gas,
                "blocks": 1,
                "range_cm1": [record["lower_cm1"], record["upper_cm1"]],
                "shape": [20, 145, 10001],
                "rank": 64,
            },
        }
        block.with_suffix(".json").write_text(json.dumps(receipt), encoding="utf-8")
    report = {
        "schema_version": 1,
        "status": "candidate_full_gas_unvalidated",
        "gas": gas,
        "range_cm1": [635, 1265],
        "block_count": 63,
        "source_snapshot": snapshot,
        "input_sha256": inputs,
        "archived_dense_sha256": assets["lut_co2"].sha256,
        "dense_bytes": dense.stat().st_size,
        "dense_sha256": hash_file(dense),
        "expansion_report": {"sha256": hash_file(dense), "shape": [20, 145, 630001], "blocks": 63},
        "block_sha256": block_hashes,
    }
    (base / "report.json").write_text(json.dumps(report), encoding="utf-8")
    assert (
        verify_candidate(tmp_path, gas, records, assets, snapshot)["block_sha256"] == block_hashes
    )
    with pytest.raises(ValueError, match="source provenance differs"):
        verify_candidate(tmp_path, gas, records, assets, {"sha256": "different", "files": {}})
    (blocks_dir / records[37]["filename"]).write_bytes(b"altered")
    with pytest.raises(ValueError, match="block or provenance differs"):
        verify_candidate(tmp_path, gas, records, assets, snapshot)
