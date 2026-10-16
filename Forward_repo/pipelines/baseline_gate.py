"""Check T4 metrics and provenance against the frozen SEW baseline.

Numerical agreement and historical accreditation are separate outcomes. The gate
passes only when both evaluation sets agree and the historical input identities
are known and match the candidate run.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import sys
from pathlib import Path
from typing import Any


DEFAULT_REFERENCE = Path(__file__).resolve().parents[1] / "manifests" / "baseline_sew.json"
GASES = ("H2O", "CO2", "O3", "N2O", "CH4")
TOLERANCES = {"gases_dex": 0.001, "rmse_T_K": 0.05, "rmse_Tskin_K": 0.05}
SETS = {
    "simulated_300": {"conjunto": "test3k", "scenes": 300},
    "sea_64": {"conjunto": "mar", "scenes": 64},
}
INPUT_ALIASES = {
    "/salida/out_iasi/apae_fix_lat24b1e-4.pt": "apae",
    "/salida/out_iasi/prior_z_deep.pt": "prior",
    "/salida/out_iasiang/s_f_sim_iasiang.npz": "model_error_sf",
    "/salida/out_iasiang/config/unroll714_sew_ckpt.pt": "sew",
    "/salida/out_iasiang/exp24_714b/nuestro_n300.json": "frozen_simulated_rows",
}


def _read_json(path: Path) -> dict[str, Any]:
    document = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(document, dict):
        raise ValueError(f"Expected a JSON object: {path}")
    return document


def _sha256_json(value: object) -> str:
    payload = json.dumps(value, separators=(",", ":"), ensure_ascii=False, allow_nan=False)
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(8 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _valid_sha256(value: object) -> bool:
    return (
        isinstance(value, str)
        and len(value) == 64
        and all(char in "0123456789abcdef" for char in value.lower())
    )


def _finite_number(value: object, label: str) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise ValueError(f"{label} must be a number")
    number = float(value)
    if not math.isfinite(number):
        raise ValueError(f"{label} must be finite")
    return number


def _canonical_input_hashes(source: dict, reasons: list[str], label: str) -> dict[str, str]:
    """Read old flat or new pipeline `{id: {path, sha256}}` inputs."""
    result = {}
    for key, value in source.items():
        canonical = INPUT_ALIASES.get(key, key)
        digest = value.get("sha256") if isinstance(value, dict) else value
        if not _valid_sha256(digest):
            reasons.append(f"{label} input hash is missing or invalid: {key}")
            continue
        digest = digest.lower()
        if canonical in result and result[canonical] != digest:
            reasons.append(f"{label} input hashes conflict for {canonical}")
        result[canonical] = digest
    return result


def compare_metrics(reference: dict[str, Any], result: dict[str, Any], set_name: str) -> dict:
    """Compare one result with frozen rows and absolute metric tolerances."""
    if set_name not in SETS:
        raise ValueError(f"Unknown evaluation set: {set_name}")
    expected_set = SETS[set_name]
    baseline = reference["comparison"][set_name]
    for field, expected in (
        ("conjunto", expected_set["conjunto"]),
        ("n_escenas", expected_set["scenes"]),
        ("grid", "iasiang"),
        ("nch", 2145),
        ("k", 24),
        ("se_corr", "isrf"),
    ):
        if result.get(field) != expected:
            raise ValueError(f"{set_name}: {field}={result.get(field)!r}, expected {expected!r}")

    rows = result.get("filas")
    if (
        not isinstance(rows, list)
        or len(rows) != expected_set["scenes"]
        or any(type(row) is not int for row in rows)
        or len(set(rows)) != len(rows)
    ):
        raise ValueError(f"{set_name}: invalid scene rows")
    if _sha256_json(rows) != baseline["historical_rows_sha256"]:
        raise ValueError(f"{set_name}: scene rows/order differ from historical baseline")

    method = result.get("metodo")
    if method == "nuestro":
        if result.get("iters") != 4 or "pre_oe" in result:
            raise ValueError(f"{set_name}: expected a plain T4 learned result")
        stage = result
        stage_name = "result"
    elif method == "oe_latente":
        if result.get("oe_desde_modelo") is not True or result.get("modelo_iters") != 4:
            raise ValueError(f"{set_name}: expected OE initialized from learned T4")
        stage = result.get("pre_oe")
        if not isinstance(stage, dict) or stage.get("paso") != 0:
            raise ValueError(f"{set_name}: missing pre-OE stage zero")
        stage_name = "pre_oe"
    else:
        raise ValueError(f"{set_name}: unsupported method {method!r}")

    if result.get("estado", "full") != "full":
        raise ValueError(f"{set_name}: expected the Ncol full state")

    gas_values = [_finite_number(stage.get(f"rmse_{gas}_dex"), gas) for gas in GASES]
    observed = {
        "gases_dex": sum(gas_values) / len(gas_values),
        "rmse_T_K": _finite_number(stage.get("rmse_T_K"), "rmse_T_K"),
        "rmse_Tskin_K": _finite_number(stage.get("rmse_Tskin_K"), "rmse_Tskin_K"),
    }
    if "gases_dex" in stage and not math.isclose(
        observed["gases_dex"],
        _finite_number(stage["gases_dex"], "gases_dex"),
        rel_tol=0.0,
        abs_tol=1e-10,
    ):
        raise ValueError(f"{set_name}: gases_dex conflicts with the five gas RMSE values")

    comparisons = {}
    for key, tolerance in TOLERANCES.items():
        historical = _finite_number(baseline[f"historical_{key}"], f"historical_{key}")
        delta = abs(observed[key] - historical)
        comparisons[key] = {
            "historical": historical,
            "observed": observed[key],
            "absolute_delta": delta,
            "tolerance": tolerance,
            "passed": delta <= tolerance,
        }
    return {
        "passed": all(value["passed"] for value in comparisons.values()),
        "stage": stage_name,
        "rows_sha256": _sha256_json(rows),
        "metrics": comparisons,
    }


def check_identity(
    reference: dict[str, Any],
    checkpoint: Path | None,
    run_manifest: dict[str, Any] | list[dict[str, Any]] | None,
) -> dict:
    """Check historical checkpoint and input hashes independently of metrics."""
    reasons = []
    expected_checkpoint = reference.get("historical_checkpoint_sha256")
    expected_step = reference.get("expected_checkpoint_step", 1000)
    if type(expected_step) is not int or expected_step <= 0:
        raise ValueError("expected_checkpoint_step must be a positive integer")
    if not _valid_sha256(expected_checkpoint):
        reasons.append("Historical checkpoint SHA256 is missing or invalid")

    observed_checkpoint = None
    observed_step = None
    configured_steps = None
    if checkpoint is None:
        reasons.append("Candidate checkpoint file was not supplied")
    elif not checkpoint.is_file():
        reasons.append(f"Candidate checkpoint file is missing: {checkpoint}")
    else:
        observed_checkpoint = _sha256_file(checkpoint)
        if (
            _valid_sha256(expected_checkpoint)
            and observed_checkpoint != expected_checkpoint.lower()
        ):
            reasons.append("Candidate checkpoint SHA256 differs from historical checkpoint")
        else:
            try:
                import torch

                payload = torch.load(checkpoint, map_location="cpu", weights_only=True)
                if not isinstance(payload, dict):
                    raise ValueError("checkpoint is not a dictionary")
                observed_step = payload.get("paso")
                args = payload.get("args")
                configured_steps = args.get("steps") if isinstance(args, dict) else None
            except Exception as exc:
                reasons.append(f"Cannot inspect candidate checkpoint: {exc}")
            else:
                if type(observed_step) is not int or observed_step != expected_step:
                    reasons.append(
                        f"Candidate checkpoint paso={observed_step!r}, expected {expected_step}"
                    )
                if type(configured_steps) is not int or configured_steps != expected_step:
                    reasons.append(
                        "Candidate checkpoint args.steps="
                        f"{configured_steps!r}, expected {expected_step}"
                    )
                if "metricas" not in payload:
                    reasons.append("Candidate checkpoint lacks final evaluation metrics")

    required = reference.get("new_run_input_hashes_reported_by_manifest")
    historical_inputs = reference.get("historical_input_hashes")
    if not isinstance(required, dict) or not required:
        reasons.append("Reference has no minimum input inventory")
        required = {}
    required_keys = {INPUT_ALIASES.get(key, key) for key in required}
    if not isinstance(historical_inputs, dict) or not historical_inputs:
        reasons.append("Historical input hashes are missing")
        historical_inputs = {}
    else:
        historical_inputs = _canonical_input_hashes(historical_inputs, reasons, "Historical")
        missing = sorted(required_keys - set(historical_inputs))
        if missing:
            reasons.append(f"Historical input hashes are incomplete: {missing}")
    checkpoint_keys = [
        key
        for key in historical_inputs
        if key == "sew" or "checkpoint" in key.lower() or key.endswith("ckpt.pt")
    ]
    if not checkpoint_keys:
        reasons.append("Historical input inventory has no checkpoint entry")
    for key in checkpoint_keys:
        recorded = historical_inputs[key]
        if _valid_sha256(recorded) and _valid_sha256(expected_checkpoint):
            if recorded.lower() != expected_checkpoint.lower():
                reasons.append(f"Historical checkpoint and input inventory disagree: {key}")

    documents = (
        [run_manifest]
        if isinstance(run_manifest, dict)
        else run_manifest
        if isinstance(run_manifest, list)
        else []
    )
    if not documents:
        reasons.append("Candidate run input manifest is missing")
    candidate_inputs = {}
    stages = set()
    for document in documents:
        if not isinstance(document, dict) or not isinstance(document.get("inputs"), dict):
            reasons.append("Candidate run input manifest is invalid")
            continue
        if document.get("stage") is not None:
            stages.add(document["stage"])
        selected = _canonical_input_hashes(document["inputs"], reasons, "Candidate")
        for key, digest in selected.items():
            if key in candidate_inputs and candidate_inputs[key] != digest:
                reasons.append(f"Candidate run manifests disagree on input: {key}")
            candidate_inputs[key] = digest
    if stages and not {"evaluate_simulated", "evaluate_mar64"}.issubset(stages):
        reasons.append("Both simulated and sea evaluation run records are required")
    for key, expected in historical_inputs.items():
        observed = candidate_inputs.get(key)
        if not _valid_sha256(observed):
            reasons.append(f"Candidate input hash is missing or invalid: {key}")
        elif observed.lower() != expected.lower():
            reasons.append(f"Candidate input hash differs from historical input: {key}")
        elif key in checkpoint_keys and observed_checkpoint is not None:
            if observed.lower() != observed_checkpoint:
                reasons.append(f"Candidate checkpoint file and run manifest disagree: {key}")

    return {
        "passed": not reasons,
        "historical_checkpoint_sha256": expected_checkpoint,
        "candidate_checkpoint_sha256": observed_checkpoint,
        "expected_checkpoint_step": expected_step,
        "candidate_checkpoint_step": observed_step,
        "candidate_configured_steps": configured_steps,
        "reasons": reasons,
    }


def evaluate_gate(
    reference: dict[str, Any],
    simulated: dict[str, Any],
    sea: dict[str, Any],
    checkpoint: Path | None = None,
    run_manifest: dict[str, Any] | list[dict[str, Any]] | None = None,
) -> dict:
    """Return separate numeric and historical identity verdicts."""
    if reference.get("schema_version") != 1:
        raise ValueError("Unsupported baseline manifest schema")
    numeric = {
        "simulated_300": compare_metrics(reference, simulated, "simulated_300"),
        "sea_64": compare_metrics(reference, sea, "sea_64"),
    }
    numeric_passed = all(item["passed"] for item in numeric.values())
    identity = check_identity(reference, checkpoint, run_manifest)
    return {
        "numeric_comparison": {"passed": numeric_passed, "sets": numeric},
        "historical_identity": identity,
        "gate_passed": numeric_passed and identity["passed"],
    }


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--sim-json", type=Path, required=True)
    parser.add_argument("--sea-json", type=Path, required=True)
    parser.add_argument("--checkpoint", type=Path)
    parser.add_argument(
        "--run-manifest",
        type=Path,
        action="append",
        help="Repeat for the simulated and sea stage records",
    )
    parser.add_argument("--reference", type=Path, default=DEFAULT_REFERENCE)
    args = parser.parse_args(argv)

    try:
        reference = _read_json(args.reference)
        simulated = _read_json(args.sim_json)
        sea = _read_json(args.sea_json)
        run_manifest = (
            [_read_json(path) for path in args.run_manifest] if args.run_manifest else None
        )
        report = evaluate_gate(reference, simulated, sea, args.checkpoint, run_manifest)
    except (OSError, json.JSONDecodeError, ValueError, KeyError) as exc:
        print(f"Invalid baseline input: {exc}", file=sys.stderr)
        return 2

    print(json.dumps(report, indent=2, sort_keys=True))
    return 0 if report["gate_passed"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
