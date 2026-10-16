"""The simulated report adapter preserves frozen selection and T4 metrics."""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from pipelines.assets import hash_file
from pipelines.baseline_gate import compare_metrics
from pipelines.report_adapter import adapt_simulated_record, main


ROOT = Path(__file__).resolve().parents[1]


def _record(tmp_path: Path) -> tuple[Path, Path, Path]:
    config = ROOT / "configs" / "final.json"
    rows = ROOT / "frozen" / "nuestro_n300.json"
    reference = json.loads((ROOT / "manifests" / "baseline_sew.json").read_text("utf-8"))
    expected = reference["comparison"]["simulated_300"]
    metrics = {
        "rmse_T_K": expected["historical_rmse_T_K"],
        "rmse_Tskin_K": expected["historical_rmse_Tskin_K"],
        "gases_dex": expected["historical_gases_dex"],
        **{
            f"rmse_{gas}_dex": expected["historical_gases_dex"]
            for gas in ("H2O", "CO2", "O3", "N2O", "CH4")
        },
    }
    record = {
        "stage": "evaluate_simulated",
        "configuration_sha256": hash_file(config),
        "inputs": {"frozen_simulated_rows": {"path": str(rows), "sha256": hash_file(rows)}},
        "details": {
            "selection": "frozen300",
            "count": 300,
            "iterations": 4,
            "metrics": metrics,
        },
    }
    record_path = tmp_path / "evaluate_simulated_frozen300_T4.json"
    record_path.write_text(json.dumps(record), encoding="utf-8")
    return record_path, rows, config


def test_adapter_emits_direct_gate_schema(tmp_path: Path) -> None:
    record_path, rows, config = _record(tmp_path)
    result = adapt_simulated_record(record_path, rows, config)
    reference = json.loads((ROOT / "manifests" / "baseline_sew.json").read_text("utf-8"))
    assert compare_metrics(reference, result, "simulated_300")["passed"] is True
    assert result["source_run_record_sha256"] == hash_file(record_path)
    assert result["frozen_selection_sha256"] == hash_file(rows)

    output = tmp_path / "sim300_T4_metrics.json"
    assert main(["--run-record", str(record_path), "--out", str(output)]) == 0
    assert json.loads(output.read_text("utf-8")) == result


def test_adapter_rejects_unverified_selection_and_wrong_stage(tmp_path: Path) -> None:
    record_path, rows, config = _record(tmp_path)
    record = json.loads(record_path.read_text("utf-8"))
    record["inputs"]["frozen_simulated_rows"]["sha256"] = "0" * 64
    record_path.write_text(json.dumps(record), encoding="utf-8")
    with pytest.raises(ValueError, match="Frozen selection bytes differ"):
        adapt_simulated_record(record_path, rows, config)

    record["stage"] = "evaluate_mar64"
    record_path.write_text(json.dumps(record), encoding="utf-8")
    with pytest.raises(ValueError, match="not a simulated evaluation"):
        adapt_simulated_record(record_path, rows, config)


def test_adapter_rejects_nonfinite_or_inconsistent_metrics(tmp_path: Path) -> None:
    record_path, rows, config = _record(tmp_path)
    record = json.loads(record_path.read_text("utf-8"))
    record["details"]["metrics"]["rmse_T_K"] = float("nan")
    record_path.write_text(json.dumps(record), encoding="utf-8")
    with pytest.raises(ValueError, match="finite T4 state metrics"):
        adapt_simulated_record(record_path, rows, config)

    record["details"]["metrics"]["rmse_T_K"] = 2.0
    record["details"]["metrics"]["gases_dex"] = 0.5
    record_path.write_text(json.dumps(record), encoding="utf-8")
    with pytest.raises(ValueError, match="gas mean disagrees"):
        adapt_simulated_record(record_path, rows, config)
