"""Local gates for resumable candidate T/U/D and Modal stage planning."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pytest

from pipelines.modal_data_candidate import (
    GASES,
    _comparison_scope,
    _existing_receipt,
    _load_comparisons,
    _sha256,
    _tile_plan,
    _validate_chunk,
)


def test_tud_tiles_cover_every_final_channel_once() -> None:
    tiles = _tile_plan(30_000)
    assert len(tiles) == 12
    assert [tile["start_cm1"] for tile in tiles] == list(range(695, 1250, 50))
    assert tiles[0]["ca"] == 0
    assert tiles[-1]["cb"] == 2145
    assert all(left["cb"] == right["ca"] for left, right in zip(tiles, tiles[1:]))
    with pytest.raises(ValueError, match="count must be"):
        _tile_plan(1)


def test_chunk_validation_rejects_wrong_channel_coordinates(tmp_path: Path) -> None:
    tile = _tile_plan(2)[0]
    width = tile["cb"] - tile["ca"]
    channels = np.arange(714.0, 1250.0 + 1e-9, 0.25)
    path = tmp_path / tile["name"]
    data = {
        "ca": tile["ca"],
        "cb": tile["cb"],
        "nu": channels[tile["ca"] : tile["cb"]],
        "D": np.zeros((2, width), dtype=np.float32),
        "T": np.ones((5, 2, width), dtype=np.float32),
        "U": np.ones((5, 2, width), dtype=np.float32),
    }
    np.savez(path, **data)
    _validate_chunk(path, tile, 2)
    data["nu"] = data["nu"] + 0.25
    np.savez(path, **data)
    with pytest.raises(ValueError, match="Invalid candidate"):
        _validate_chunk(path, tile, 2)


def test_resume_rejects_changed_provenance_or_bytes(tmp_path: Path) -> None:
    artifact = tmp_path / "c695.npz"
    receipt_path = artifact.with_suffix(".json")
    expected = {
        "schema_version": 1,
        "source_snapshot": {"sha256": "source-1"},
        "input_sha256": {"states_30k": "input-1"},
    }
    assert _existing_receipt(artifact, receipt_path, expected) is None
    artifact.write_bytes(b"candidate chunk")
    assert _existing_receipt(artifact, receipt_path, expected) is None
    receipt = {
        **expected,
        "output_sha256": _sha256(artifact),
        "output_bytes": artifact.stat().st_size,
    }
    receipt_path.write_text(json.dumps(receipt), encoding="utf-8")
    assert _existing_receipt(artifact, receipt_path, expected) == receipt
    with pytest.raises(ValueError, match="provenance differs"):
        _existing_receipt(artifact, receipt_path, {**expected, "input_sha256": {"x": "y"}})
    artifact.write_bytes(b"altered")
    with pytest.raises(ValueError, match="provenance differs"):
        _existing_receipt(artifact, receipt_path, expected)


def _passing_report(gas: str) -> dict:
    return {
        "gas": gas,
        "sample_selection": {
            "terminal_global_channel_index": 630_000,
            "channel_spacing_cm1": 0.001,
            "state_count": 20,
            "channel_count": 63 * 301 + 1,
        },
        "comparison": {
            "gate": {
                "passed": True,
                "rms_log10_sigma_limit": 2.5e-5,
                "max_abs_log10_sigma_limit": 2.5e-4,
            },
            "rms_log10_sigma": 1.5e-6,
            "max_abs_log10_sigma": 4e-6,
            "terminal_max_abs_log10_sigma": 1e-6,
            "blocks": [
                {
                    "lower_cm1": start,
                    "passed": True,
                    "rms_log10_sigma": 1e-6,
                    "max_abs_log10_sigma": 4e-6,
                }
                for start in range(635, 1265, 10)
            ],
        },
    }


def test_five_passing_lut_reports_required(tmp_path: Path) -> None:
    for gas in GASES:
        (tmp_path / f"lut_full_{gas.lower()}.json").write_text(
            json.dumps(_passing_report(gas)), encoding="utf-8"
        )
    assert set(_load_comparisons(tmp_path)) == set(GASES)
    failed = _passing_report("O3")
    failed["comparison"]["gate"]["passed"] = False
    (tmp_path / "lut_full_o3.json").write_text(json.dumps(failed), encoding="utf-8")
    with pytest.raises(ValueError, match="Passing O3 LUT report required"):
        _load_comparisons(tmp_path)


def test_terminal_exception_requires_explicit_flag_and_all_blocks() -> None:
    report = _passing_report("CO2")
    report["comparison"]["gate"]["passed"] = False
    report["comparison"]["max_abs_log10_sigma"] = 5.3e-4
    report["comparison"]["terminal_max_abs_log10_sigma"] = 5.3e-4
    with pytest.raises(ValueError, match="isolated 1265"):
        _comparison_scope(report, False)
    assert _comparison_scope(report, True) == (
        "all_block_samples_pass_terminal_1265_excluded_explicitly"
    )
    report["comparison"]["blocks"][27]["passed"] = False
    with pytest.raises(ValueError, match="block samples"):
        _comparison_scope(report, True)


@pytest.mark.parametrize(
    ("gas", "terminal_maximum"),
    (("co2", 0.000522613525390625), ("h2o", 0.0002899169921875)),
)
def test_real_report_isolates_terminal_for_each_gas(gas: str, terminal_maximum: float) -> None:
    path = Path(__file__).resolve().parents[2] / "docs" / "validation" / f"lut_full_{gas}.json"
    report = json.loads(path.read_text(encoding="utf-8"))
    assert report["comparison"]["gate"]["passed"] is False
    assert report["comparison"]["terminal_max_abs_log10_sigma"] == terminal_maximum
    with pytest.raises(ValueError, match="isolated 1265"):
        _comparison_scope(report, False)
    assert _comparison_scope(report, True) == (
        "all_block_samples_pass_terminal_1265_excluded_explicitly"
    )
