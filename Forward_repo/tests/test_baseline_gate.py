"""Failure modes for the frozen T4 metric and identity gate."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path

import pytest
import torch

from pipelines.baseline_gate import compare_metrics, evaluate_gate


def _rows_hash(rows: list[int]) -> str:
    payload = json.dumps(rows, separators=(",", ":")).encode("utf-8")
    return hashlib.sha256(payload).hexdigest()


def _reference() -> dict:
    return {
        "schema_version": 1,
        "expected_checkpoint_step": 1000,
        "comparison": {
            name: {
                "historical_rows_sha256": _rows_hash(list(range(count))),
                "historical_gases_dex": 0.1,
                "historical_rmse_T_K": 2.0,
                "historical_rmse_Tskin_K": 1.0,
            }
            for name, count in (("simulated_300", 300), ("sea_64", 64))
        },
        "historical_checkpoint_sha256": None,
        "historical_input_hashes": None,
        "new_run_input_hashes_reported_by_manifest": {"checkpoint": "0" * 64, "data": "1" * 64},
    }


def _result(count: int, gas: float = 0.1, temp: float = 2.0, skin: float = 1.0) -> dict:
    result = {
        "metodo": "nuestro",
        "conjunto": "test3k" if count == 300 else "mar",
        "n_escenas": count,
        "grid": "iasiang",
        "nch": 2145,
        "k": 24,
        "se_corr": "isrf",
        "iters": 4,
        "filas": list(range(count)),
        "rmse_T_K": temp,
        "rmse_Tskin_K": skin,
    }
    result.update({f"rmse_{name}_dex": gas for name in ("H2O", "CO2", "O3", "N2O", "CH4")})
    return result


def test_numeric_agreement_cannot_accredit_missing_historical_identity() -> None:
    report = evaluate_gate(_reference(), _result(300), _result(64))
    assert report["numeric_comparison"]["passed"] is True
    assert report["historical_identity"]["passed"] is False
    assert report["gate_passed"] is False
    assert (
        "Historical checkpoint SHA256 is missing or invalid"
        in report["historical_identity"]["reasons"]
    )


def test_metrics_exceeding_frozen_tolerances_fail() -> None:
    reference = _reference()
    simulated = _result(300, gas=0.1011)
    sea = _result(64, temp=2.051)
    report = evaluate_gate(reference, simulated, sea)
    assert report["numeric_comparison"]["passed"] is False
    assert (
        report["numeric_comparison"]["sets"]["simulated_300"]["metrics"]["gases_dex"]["passed"]
        is False
    )
    assert report["numeric_comparison"]["sets"]["sea_64"]["metrics"]["rmse_T_K"]["passed"] is False


def test_matching_metrics_checkpoint_and_input_hashes_pass(tmp_path: Path) -> None:
    checkpoint = tmp_path / "unroll714_sew_ckpt.pt"
    torch.save({"paso": 1000, "args": {"steps": 1000}, "metricas": {}}, checkpoint)
    digest = hashlib.sha256(checkpoint.read_bytes()).hexdigest()
    reference = _reference()
    reference["historical_checkpoint_sha256"] = digest
    reference["historical_input_hashes"] = {"checkpoint": digest, "data": "1" * 64}
    run_manifest = {"inputs": {"checkpoint": digest, "data": "1" * 64}}
    report = evaluate_gate(
        reference, _result(300, gas=0.1009), _result(64), checkpoint, run_manifest
    )
    assert report["numeric_comparison"]["passed"] is True
    assert report["historical_identity"]["passed"] is True
    assert report["gate_passed"] is True

    checkpoint.write_bytes(b"different checkpoint content")
    changed = evaluate_gate(reference, _result(300), _result(64), checkpoint, run_manifest)
    assert changed["historical_identity"]["passed"] is False
    assert changed["gate_passed"] is False

    reference["historical_input_hashes"]["checkpoint"] = "f" * 64
    inconsistent = evaluate_gate(reference, _result(300), _result(64), None, run_manifest)
    assert any(
        "inventory disagree" in reason for reason in inconsistent["historical_identity"]["reasons"]
    )


def test_different_scene_order_and_incomplete_input_inventory_fail(tmp_path: Path) -> None:
    reference = _reference()
    simulated = _result(300)
    simulated["filas"][0], simulated["filas"][1] = simulated["filas"][1], simulated["filas"][0]
    with pytest.raises(ValueError, match="rows/order differ"):
        compare_metrics(reference, simulated, "simulated_300")

    checkpoint = tmp_path / "model.pt"
    checkpoint.write_bytes(b"weights")
    digest = hashlib.sha256(checkpoint.read_bytes()).hexdigest()
    reference["historical_checkpoint_sha256"] = digest
    reference["historical_input_hashes"] = {"checkpoint": digest}
    incomplete = evaluate_gate(
        reference, _result(300), _result(64), checkpoint, {"inputs": {"checkpoint": digest}}
    )
    assert incomplete["historical_identity"]["passed"] is False
    assert any("incomplete" in reason for reason in incomplete["historical_identity"]["reasons"])


def test_intermediate_checkpoint_cannot_pass_with_matching_hashes(tmp_path: Path) -> None:
    checkpoint = tmp_path / "intermediate.pt"
    torch.save({"paso": 400, "args": {"steps": 1000}}, checkpoint)
    digest = hashlib.sha256(checkpoint.read_bytes()).hexdigest()
    reference = _reference()
    reference["historical_checkpoint_sha256"] = digest
    reference["historical_input_hashes"] = {"checkpoint": digest, "data": "1" * 64}
    run_manifest = {"inputs": {"checkpoint": digest, "data": "1" * 64}}

    report = evaluate_gate(reference, _result(300), _result(64), checkpoint, run_manifest)
    assert report["numeric_comparison"]["passed"] is True
    assert report["historical_identity"]["passed"] is False
    assert report["historical_identity"]["candidate_checkpoint_step"] == 400
    assert any("paso=400" in reason for reason in report["historical_identity"]["reasons"])
    assert any(
        "lacks final evaluation" in reason for reason in report["historical_identity"]["reasons"]
    )


def test_new_stage_records_use_nested_hashes_and_both_evaluations(tmp_path: Path) -> None:
    checkpoint = tmp_path / "step1000.pt"
    torch.save({"paso": 1000, "args": {"steps": 1000}, "metricas": {}}, checkpoint)
    digest = hashlib.sha256(checkpoint.read_bytes()).hexdigest()
    old_path = "/salida/out_iasiang/config/unroll714_sew_ckpt.pt"
    reference = _reference()
    reference["historical_checkpoint_sha256"] = digest
    reference["new_run_input_hashes_reported_by_manifest"] = {
        old_path: digest,
        "data": "1" * 64,
    }
    reference["historical_input_hashes"] = {old_path: digest, "data": "1" * 64}
    simulated_run = {
        "stage": "evaluate_simulated",
        "inputs": {
            "sew": {"path": str(checkpoint), "sha256": digest},
            "data": {"path": "data.npz", "sha256": "1" * 64},
        },
    }
    sea_run = {
        "stage": "evaluate_mar64",
        "inputs": {
            "sew": {"path": str(checkpoint), "sha256": digest},
            "data": {"path": "data.npz", "sha256": "1" * 64},
        },
    }
    accepted = evaluate_gate(
        reference, _result(300), _result(64), checkpoint, [simulated_run, sea_run]
    )
    assert accepted["gate_passed"] is True

    incomplete = evaluate_gate(reference, _result(300), _result(64), checkpoint, [sea_run])
    assert incomplete["historical_identity"]["passed"] is False
    assert any(
        "Both simulated and sea" in reason
        for reason in incomplete["historical_identity"]["reasons"]
    )

    sea_run["inputs"]["data"]["sha256"] = "2" * 64
    conflicting = evaluate_gate(
        reference, _result(300), _result(64), checkpoint, [simulated_run, sea_run]
    )
    assert any(
        "disagree on input" in reason for reason in conflicting["historical_identity"]["reasons"]
    )
