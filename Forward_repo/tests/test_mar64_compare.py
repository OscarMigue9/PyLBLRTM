"""The sea comparison binds its report to the dumps and aligned scene data."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pytest

from pipelines.assets import hash_file
from pipelines.mar64_compare import (
    DEFAULT_HISTORICAL,
    _scene_block_errors,
    compare_mar64,
    main,
)


ROOT = Path(__file__).resolve().parents[1]


def _write_json(path: Path, value: dict) -> None:
    path.write_text(json.dumps(value), encoding="utf-8")


def _fixture(tmp_path: Path) -> tuple[Path, Path, Path, Path, Path]:
    candidate_path = tmp_path / "candidate.npz"
    historical_path = tmp_path / "historical.npz"
    metrics_path = tmp_path / "metrics.json"
    record_path = tmp_path / "record.json"
    reference_path = tmp_path / "reference.json"
    rows = np.arange(64, dtype=np.int64)
    truth = np.zeros((64, 818), dtype=np.float32)
    estimate = truth.copy()
    estimate[0, 0] = 0.1
    pressure = np.ones((64, 136), dtype=np.float32)
    radiance = np.zeros((64, 2145), dtype=np.float32)
    valid = np.arange(2121, dtype=np.int32)
    np.savez_compressed(
        historical_path,
        filas=rows,
        x_hat=truth,
        x_true=truth,
        P_lay=pressure,
        y_obs=radiance[:, valid],
        absol=valid,
    )
    np.savez_compressed(
        candidate_path,
        filas=rows,
        x_hat=estimate,
        x_true=truth,
        P_lay=pressure,
        y_obs=radiance,
    )
    metrics = {
        "metodo": "nuestro",
        "conjunto": "mar",
        "estado": "full",
        "n_escenas": 64,
        "grid": "iasiang",
        "nch": 2145,
        "k": 24,
        "se_corr": "isrf",
        "iters": 4,
        "filas": rows.tolist(),
        "rmse_T_K": float(np.sqrt(0.1**2 / (64 * 137))),
        "rmse_Tskin_K": 0.0,
        "gases_dex": 0.0,
        **{f"rmse_{gas}_dex": 0.0 for gas in ("H2O", "CO2", "O3", "N2O", "CH4")},
    }
    _write_json(metrics_path, metrics)
    _write_json(
        record_path,
        {
            "stage": "evaluate_mar64",
            "details": {
                "count": 64,
                "iterations": 4,
                "metrics": {
                    key: value
                    for key, value in metrics.items()
                    if key.startswith("rmse_") or key == "gases_dex"
                },
            },
            "outputs": {
                "scene_dump": {"sha256": hash_file(candidate_path)},
                "metrics": {"sha256": hash_file(metrics_path)},
            },
        },
    )
    reference = json.loads((ROOT / "manifests" / "baseline_sew.json").read_text("utf-8"))
    reference["local_artifacts"]["historical_sea_scenes"]["sha256"] = hash_file(historical_path)
    baseline = reference["comparison"]["sea_64"]
    for key in ("gases_dex", "rmse_T_K", "rmse_Tskin_K"):
        baseline[f"historical_{key}"] = 0.0
    _write_json(reference_path, reference)
    return candidate_path, record_path, metrics_path, historical_path, reference_path


def test_compare_mar64_verifies_hashes_and_scene_alignment(tmp_path: Path) -> None:
    paths = _fixture(tmp_path)
    result = compare_mar64(*paths)
    assert result["scene_alignment_exact"] is True
    assert result["metric_comparison"]["passed"] is True
    assert result["retrieved_state_difference"]["max_abs"] == pytest.approx(0.1)
    assert result["retrieved_state_exact_fraction"] == 1 - 1 / (64 * 818)
    assert result["scene_retrieved_state_differences"][1]["rmse"] == 0.0
    assert result["historical_npz"]["sha256"] == hash_file(paths[3])

    output = tmp_path / "comparison.json"
    assert (
        main(
            [
                "--candidate",
                str(paths[0]),
                "--run-record",
                str(paths[1]),
                "--metrics",
                str(paths[2]),
                "--historical",
                str(paths[3]),
                "--reference",
                str(paths[4]),
                "--out",
                str(output),
            ]
        )
        == 0
    )
    assert json.loads(output.read_text("utf-8")) == result


def test_compare_mar64_rejects_changed_hash_and_scene_data(tmp_path: Path) -> None:
    candidate_path, record_path, metrics_path, historical_path, reference_path = _fixture(tmp_path)
    with np.load(candidate_path, allow_pickle=False) as archive:
        arrays = {key: archive[key] for key in archive.files}
    arrays["x_true"][0, 0] = 1.0
    np.savez_compressed(candidate_path, **arrays)
    paths = candidate_path, record_path, metrics_path, historical_path, reference_path
    with pytest.raises(ValueError, match="Candidate scene dump SHA256"):
        compare_mar64(*paths)

    record = json.loads(record_path.read_text("utf-8"))
    record["outputs"]["scene_dump"]["sha256"] = hash_file(candidate_path)
    _write_json(record_path, record)
    with pytest.raises(ValueError, match="True states differ"):
        compare_mar64(*paths)

    arrays["x_true"][0, 0] = 0.0
    arrays["y_obs"][0, 0] = 1.0
    np.savez_compressed(candidate_path, **arrays)
    record["outputs"]["scene_dump"]["sha256"] = hash_file(candidate_path)
    _write_json(record_path, record)
    with pytest.raises(ValueError, match="Observed radiance differs"):
        compare_mar64(*paths)


def test_block_errors_use_official_ncol_indices() -> None:
    truth = np.zeros((64, 818), dtype=np.float32)
    estimate = truth.copy()
    estimate[0, 137] = 1.0
    estimate[0, 817] = 2.0
    errors = _scene_block_errors(estimate, truth)
    assert errors["T_K"][0] == 0.0
    assert errors["H2O_dex"][0] == pytest.approx(1 / np.sqrt(136))
    assert errors["CO2_dex"][0] == 0.0
    assert errors["Tskin_K"][0] == 2.0


def test_historical_dump_is_bundled_and_pinned() -> None:
    reference = json.loads((ROOT / "manifests" / "baseline_sew.json").read_text("utf-8"))
    expected = reference["local_artifacts"]["historical_sea_scenes"]["sha256"]
    assert DEFAULT_HISTORICAL == ROOT / "frozen" / "mar_sew_n4.npz"
    assert hash_file(DEFAULT_HISTORICAL) == expected
