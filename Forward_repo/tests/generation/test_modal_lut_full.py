"""Local checks for full candidate LUT planning and safe resumption."""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from pipelines.modal_lut_full import _existing_block, _sha256, planned_blocks


ROOT = Path(__file__).resolve().parents[2]


@pytest.mark.parametrize("gas", ("H2O", "CO2", "O3", "N2O", "CH4"))
def test_planned_blocks_covers_one_gas_without_gaps(gas: str) -> None:
    records = planned_blocks(ROOT / "manifests/lut_blocks.json", gas)
    assert len(records) == 63
    assert [item["lower_cm1"] for item in records] == list(range(635, 1265, 10))
    assert records[-1]["upper_cm1"] == 1265


def test_non_model_gas_is_rejected() -> None:
    with pytest.raises(ValueError, match="gas must be"):
        planned_blocks(ROOT / "manifests/lut_blocks.json", "CO")


def test_resume_rejects_changed_bytes_or_source(tmp_path: Path) -> None:
    block = tmp_path / "lut_CO2_635_645.npz"
    receipt_path = block.with_suffix(".json")
    snapshot = {"sha256": "source-v1", "files": {"one.py": "hash"}}
    inputs = {"gpu_pack_lines": "input-v1"}
    assert _existing_block(block, receipt_path, "CO2", 635, snapshot, inputs) is None
    block.write_bytes(b"candidate block")
    receipt = {
        "status": "candidate_historical_source_unverified",
        "gas": "CO2",
        "start_cm1": 635,
        "end_cm1": 645,
        "source_snapshot": snapshot,
        "input_sha256": inputs,
        "candidate_sha256": _sha256(block),
    }
    assert _existing_block(block, receipt_path, "CO2", 635, snapshot, inputs) is None
    receipt_path.write_text(json.dumps(receipt), encoding="utf-8")
    assert _existing_block(block, receipt_path, "CO2", 635, snapshot, inputs) == receipt
    with pytest.raises(ValueError, match="provenance differs"):
        _existing_block(
            block, receipt_path, "CO2", 635, {**snapshot, "sha256": "source-v2"}, inputs
        )
    with pytest.raises(ValueError, match="provenance differs"):
        _existing_block(block, receipt_path, "CO2", 635, snapshot, {"gpu_pack_lines": "input-v2"})
    block.write_bytes(b"altered")
    with pytest.raises(ValueError, match="provenance differs"):
        _existing_block(block, receipt_path, "CO2", 635, snapshot, inputs)
