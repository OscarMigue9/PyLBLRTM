"""Convert the final simulated stage record to a baseline-gate metric file.

``evaluate-simulated`` records metrics in ``details`` and the frozen selection
as an input hash. This adapter verifies that selection and emits the direct T4
schema consumed by :mod:`pipelines.baseline_gate`.
"""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
from typing import Any

from pipelines.assets import hash_file


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_ROWS = ROOT / "frozen" / "nuestro_n300.json"
DEFAULT_CONFIG = ROOT / "configs" / "final.json"
GASES = ("H2O", "CO2", "O3", "N2O", "CH4")


def _read_object(path: Path) -> dict[str, Any]:
    document = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(document, dict):
        raise ValueError(f"Expected a JSON object: {path}")
    return document


def adapt_simulated_record(
    record_path: Path, rows_path: Path = DEFAULT_ROWS, config_path: Path = DEFAULT_CONFIG
) -> dict[str, Any]:
    """Return an independently checked direct metric record for frozen sim300."""
    record = _read_object(record_path)
    if record.get("stage") != "evaluate_simulated":
        raise ValueError("Run record is not a simulated evaluation")
    config = _read_object(config_path)
    if record.get("configuration_sha256") != hash_file(config_path):
        raise ValueError("Run record configuration hash differs from final.json")
    if (
        config.get("state", {}).get("representation") != "Ncol"
        or config.get("state", {}).get("dimension") != 818
        or config.get("spectral", {}).get("grid") != "iasiang"
        or config.get("spectral", {}).get("channels") != 2145
        or config.get("apae", {}).get("latent_dimension") != 24
        or config.get("unrolled", {}).get("residual_whitening") != "isrf"
        or config.get("evaluation", {}).get("selected_inference_steps") != 4
        or config.get("evaluation", {}).get("simulated_frozen_selection") != 300
    ):
        raise ValueError("Configuration is not the frozen Ncol/IASI-angle T4 recipe")

    details = record.get("details")
    if not isinstance(details, dict) or any(
        details.get(key) != expected
        for key, expected in (("selection", "frozen300"), ("count", 300), ("iterations", 4))
    ):
        raise ValueError("Run record is not the frozen 300-scene T4 evaluation")
    selection = _read_object(rows_path)
    rows = selection.get("filas")
    if (
        not isinstance(rows, list)
        or len(rows) != 300
        or any(type(row) is not int or row < 0 or row >= 30_000 for row in rows)
        or len(set(rows)) != 300
    ):
        raise ValueError("Frozen selection lacks 300 unique valid row IDs")
    inputs = record.get("inputs")
    selected_input = inputs.get("frozen_simulated_rows") if isinstance(inputs, dict) else None
    if not isinstance(selected_input, dict) or selected_input.get("sha256") != hash_file(rows_path):
        raise ValueError("Frozen selection bytes differ from the run record")

    metrics = details.get("metrics")
    required = ("rmse_T_K", "rmse_Tskin_K", *(f"rmse_{gas}_dex" for gas in GASES))
    if not isinstance(metrics, dict) or any(
        isinstance(metrics.get(key), bool)
        or not isinstance(metrics.get(key), (int, float))
        or not math.isfinite(metrics[key])
        for key in required
    ):
        raise ValueError("Run record lacks finite T4 state metrics")
    gas_mean = sum(float(metrics[f"rmse_{gas}_dex"]) for gas in GASES) / len(GASES)
    if "gases_dex" in metrics:
        reported_mean = metrics["gases_dex"]
        if (
            isinstance(reported_mean, bool)
            or not isinstance(reported_mean, (int, float))
            or not math.isfinite(reported_mean)
            or not math.isclose(reported_mean, gas_mean, rel_tol=0.0, abs_tol=1.0e-10)
        ):
            raise ValueError("Run record gas mean disagrees with individual gas metrics")

    return {
        **{key: float(metrics[key]) for key in required},
        "metodo": "nuestro",
        "prior": "mlp",
        "conjunto": "test3k",
        "estado": "full",
        "k": 24,
        "iters": 4,
        "n_escenas": 300,
        "grid": "iasiang",
        "nch": 2145,
        "se_corr": "isrf",
        "filas": rows,
        "gases_dex": gas_mean,
        "source_run_record_sha256": hash_file(record_path),
        "frozen_selection_sha256": hash_file(rows_path),
    }


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-record", type=Path, required=True)
    parser.add_argument("--frozen-rows", type=Path, default=DEFAULT_ROWS)
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args(argv)
    result = adapt_simulated_record(args.run_record, args.frozen_rows, args.config)
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(
        json.dumps(result, indent=2, sort_keys=True, allow_nan=False) + "\n",
        encoding="utf-8",
    )
    print(json.dumps({"output": str(args.out), "sha256": hash_file(args.out)}, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
