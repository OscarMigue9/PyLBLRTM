"""Compare the extracted T4 sea retrieval with the frozen historical dump.

The historical dump stores only the 2,121 valid observed channels. Its
``absol`` indices select those channels from the candidate's 2,145-channel
grid. Input hashes and scene alignment are checked before reporting metrics.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import numpy as np
from iasi_retrieval.state import GASES, N_LEVELS, SKIN_INDEX, gas_slice

from pipelines.assets import hash_file
from pipelines.baseline_gate import compare_metrics


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_CANDIDATE = ROOT / "docs" / "validation" / "mar64_T4.npz"
DEFAULT_RUN_RECORD = ROOT / "docs" / "validation" / "mar64_T4.json"
DEFAULT_METRICS = ROOT / "docs" / "validation" / "mar64_T4_metrics.json"
DEFAULT_HISTORICAL = ROOT / "frozen" / "mar_sew_n4.npz"
DEFAULT_REFERENCE = ROOT / "manifests" / "baseline_sew.json"


def _read_object(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError(f"Expected a JSON object: {path}")
    return value


def _label(path: Path) -> str:
    for base in (ROOT, ROOT.parent):
        try:
            return str(path.resolve().relative_to(base.resolve()))
        except ValueError:
            pass
    return str(path)


def _verify_hash(path: Path, expected: object, label: str) -> str:
    actual = hash_file(path)
    if not isinstance(expected, str) or actual != expected.lower():
        raise ValueError(f"{label} SHA256 differs from its recorded value")
    return actual


def _summary(difference: np.ndarray) -> dict[str, float]:
    delta = np.asarray(difference, dtype=np.float64)
    return {
        "rmse": float(np.sqrt(np.mean(delta**2))),
        "mae": float(np.mean(np.abs(delta))),
        "max_abs": float(np.max(np.abs(delta))),
    }


def _required_array(archive: np.lib.npyio.NpzFile, key: str, shape: tuple[int, ...]) -> np.ndarray:
    if key not in archive:
        raise ValueError(f"{archive.zip.filename} lacks {key}")
    value = np.asarray(archive[key])
    if value.shape != shape or not np.issubdtype(value.dtype, np.number):
        raise ValueError(f"{archive.zip.filename}:{key} must be numeric with shape {shape}")
    if not np.isfinite(value).all():
        raise ValueError(f"{archive.zip.filename}:{key} contains nonfinite values")
    return value


def _scene_block_errors(estimate: np.ndarray, truth: np.ndarray) -> dict[str, np.ndarray]:
    delta = estimate.astype(np.float64) - truth.astype(np.float64)
    errors = {
        "T_K": np.sqrt(np.mean(delta[:, :N_LEVELS] ** 2, axis=1)),
        "Tskin_K": np.abs(delta[:, SKIN_INDEX]),
    }
    for gas in GASES:
        span = gas_slice(gas)
        errors[f"{gas}_dex"] = np.sqrt(np.mean(delta[:, span] ** 2, axis=1))
    return errors


def _check_dump_metrics(estimate: np.ndarray, truth: np.ndarray, metrics: dict) -> None:
    """Allow float32 dump rounding while binding reported metrics to its states."""
    delta = estimate.astype(np.float64) - truth.astype(np.float64)
    slices = {
        "rmse_T_K": (slice(None), slice(0, N_LEVELS)),
        "rmse_Tskin_K": (slice(None), SKIN_INDEX),
        **{f"rmse_{gas}_dex": (slice(None), gas_slice(gas)) for gas in GASES},
    }
    for key, span in slices.items():
        observed = float(np.sqrt(np.mean(delta[span] ** 2)))
        tolerance = 5.0e-5 if key.endswith("_K") else 1.0e-6
        if not np.isclose(observed, metrics[key], rtol=0.0, atol=tolerance):
            raise ValueError(f"Direct {key} differs from the candidate scene dump")


def compare_mar64(
    candidate_path: Path,
    run_record_path: Path,
    metrics_path: Path,
    historical_path: Path,
    reference_path: Path,
) -> dict[str, Any]:
    """Verify provenance/alignment and summarize candidate versus historical T4."""
    record = _read_object(run_record_path)
    metrics = _read_object(metrics_path)
    reference = _read_object(reference_path)
    if record.get("stage") != "evaluate_mar64":
        raise ValueError("Run record is not a sea evaluation")
    details = record.get("details")
    if not isinstance(details, dict) or any(
        details.get(key) != expected for key, expected in (("count", 64), ("iterations", 4))
    ):
        raise ValueError("Run record is not the 64-scene T4 evaluation")
    reported = details.get("metrics")
    if not isinstance(reported, dict) or any(
        metrics.get(key) != value for key, value in reported.items()
    ):
        raise ValueError("Direct metrics differ from the run record")
    outputs = record.get("outputs")
    if not isinstance(outputs, dict):
        raise ValueError("Run record lacks output hashes")
    for key in ("metrics", "scene_dump"):
        if not isinstance(outputs.get(key), dict):
            raise ValueError(f"Run record lacks the {key} output hash")
    candidate_hash = _verify_hash(
        candidate_path, outputs["scene_dump"].get("sha256"), "Candidate scene dump"
    )
    metrics_hash = _verify_hash(metrics_path, outputs["metrics"].get("sha256"), "Direct metrics")
    historical_entry = reference.get("local_artifacts", {}).get("historical_sea_scenes", {})
    if not isinstance(historical_entry, dict):
        raise ValueError("Baseline lacks the historical sea dump identity")
    historical_hash = _verify_hash(
        historical_path, historical_entry.get("sha256"), "Historical scene dump"
    )
    metric_comparison = compare_metrics(reference, metrics, "sea_64")

    with (
        np.load(candidate_path, allow_pickle=False) as candidate,
        np.load(historical_path, allow_pickle=False) as historical,
    ):
        candidate_rows = _required_array(candidate, "filas", (64,))
        historical_rows = _required_array(historical, "filas", (64,))
        if (
            not np.issubdtype(candidate_rows.dtype, np.integer)
            or not np.issubdtype(historical_rows.dtype, np.integer)
            or not np.array_equal(candidate_rows, historical_rows)
            or not np.array_equal(candidate_rows, np.arange(64))
            or metrics.get("filas") != candidate_rows.tolist()
        ):
            raise ValueError("Scene indices or order differ")

        candidate_estimate = _required_array(candidate, "x_hat", (64, 818))
        historical_estimate = _required_array(historical, "x_hat", (64, 818))
        candidate_truth = _required_array(candidate, "x_true", (64, 818))
        historical_truth = _required_array(historical, "x_true", (64, 818))
        candidate_pressure = _required_array(candidate, "P_lay", (64, 136))
        historical_pressure = _required_array(historical, "P_lay", (64, 136))
        candidate_radiance = _required_array(candidate, "y_obs", (64, 2145))
        historical_radiance = _required_array(historical, "y_obs", (64, 2121))
        valid = _required_array(historical, "absol", (2121,))
        if (
            not np.issubdtype(valid.dtype, np.integer)
            or np.any(valid < 0)
            or np.any(valid >= 2145)
            or len(np.unique(valid)) != 2121
        ):
            raise ValueError("Historical valid channel indices are invalid")
        if not np.array_equal(candidate_truth, historical_truth):
            raise ValueError("True states differ from the historical dump")
        if not np.array_equal(candidate_pressure, historical_pressure):
            raise ValueError("Layer pressures differ from the historical dump")
        if not np.array_equal(candidate_radiance[:, valid], historical_radiance):
            raise ValueError("Observed radiance differs on historical valid channels")
        _check_dump_metrics(candidate_estimate, candidate_truth, metrics)

        difference = candidate_estimate.astype(np.float64) - historical_estimate.astype(np.float64)
        candidate_errors = _scene_block_errors(candidate_estimate, candidate_truth)
        historical_errors = _scene_block_errors(historical_estimate, historical_truth)
        block_changes = {
            key: float(np.max(np.abs(candidate_errors[key] - historical_errors[key])))
            for key in historical_errors
        }
        scene_differences = []
        for index in range(64):
            scene_summary = _summary(difference[index])
            scene_differences.append(
                {
                    "scene": int(candidate_rows[index]),
                    "rmse": scene_summary["rmse"],
                    "max_abs": scene_summary["max_abs"],
                }
            )

    return {
        "schema_version": 1,
        "historical_checkpoint_identity_accredited": False,
        "historical_npz": {"path": _label(historical_path), "sha256": historical_hash},
        "new_npz": {"path": _label(candidate_path), "sha256": candidate_hash},
        "new_metrics": {"path": _label(metrics_path), "sha256": metrics_hash},
        "new_stage_record": {"path": _label(run_record_path), "sha256": hash_file(run_record_path)},
        "scene_count": 64,
        "scene_alignment_exact": True,
        "truth_difference": _summary(candidate_truth.astype(np.float64) - historical_truth),
        "pressure_difference": _summary(
            candidate_pressure.astype(np.float64) - historical_pressure
        ),
        "radiance_valid_channel_difference": _summary(
            candidate_radiance[:, valid].astype(np.float64) - historical_radiance
        ),
        "retrieved_state_difference": _summary(difference),
        "retrieved_state_exact_fraction": float(np.mean(difference == 0.0)),
        "scene_retrieved_state_differences": scene_differences,
        "max_scene_block_metric_change": block_changes,
        "block_layout_note": (
            "Official Ncol layout: T 0:137, five gas blocks 137:817, Tskin 817. "
            "The earlier comparison used Tskin 137 and gases from 138; "
            "its block-change values were incorrectly indexed."
        ),
        "metric_comparison": metric_comparison,
    }


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--candidate", type=Path, default=DEFAULT_CANDIDATE)
    parser.add_argument("--run-record", type=Path, default=DEFAULT_RUN_RECORD)
    parser.add_argument("--metrics", type=Path, default=DEFAULT_METRICS)
    parser.add_argument("--historical", type=Path, default=DEFAULT_HISTORICAL)
    parser.add_argument("--reference", type=Path, default=DEFAULT_REFERENCE)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args(argv)
    result = compare_mar64(
        args.candidate, args.run_record, args.metrics, args.historical, args.reference
    )
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(
        json.dumps(result, indent=2, sort_keys=True, allow_nan=False) + "\n", encoding="utf-8"
    )
    print(json.dumps({"output": str(args.out), "sha256": hash_file(args.out)}, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
