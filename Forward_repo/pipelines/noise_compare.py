"""Compare a regenerated 30,000-scene noise archive with the frozen asset."""

from __future__ import annotations

import argparse
import json
import math
import sys
from pathlib import Path
from typing import Any

import numpy as np

from .assets import hash_file, load_manifest, verify_file


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_MANIFEST = ROOT / "manifests" / "assets.json"
DEFAULT_FROZEN = ROOT / "data_cache" / "data" / "outputs" / "noise_iasiang.npz"
DEFAULT_CANDIDATE = ROOT / "runs" / "noise_rebuild" / "noise_iasiang.npz"


def _check_archive(
    archive: np.lib.npyio.NpzFile, label: str, expected_shape: tuple[int, int]
) -> dict[str, np.ndarray]:
    """Require the fixed physical grid and complete finite float32 arrays."""
    profiles, channels = expected_shape
    arrays = {}
    for key, shape in (
        ("noise", (profiles, channels)),
        ("nu", (channels,)),
        ("sigma_ch", (channels,)),
    ):
        if key not in archive:
            raise ValueError(f"{label}: missing {key}")
        array = archive[key]
        if array.shape != shape or array.dtype != np.dtype("float32"):
            raise ValueError(f"{label}: {key} must be {shape} float32")
        if not np.isfinite(array).all():
            raise ValueError(f"{label}: {key} contains non-finite values")
        arrays[key] = array
    expected_nu = (714.0 + 0.25 * np.arange(channels)).astype(np.float32)
    if not np.array_equal(arrays["nu"], expected_nu):
        raise ValueError(f"{label}: nu differs from the 714 cm^-1, 0.25 cm^-1 grid")
    if not (arrays["sigma_ch"] > 0).all():
        raise ValueError(f"{label}: sigma_ch must be strictly positive")
    expected_metadata: dict[str, Any] = {
        "seed": 1337,
        "grid": "iasiang",
        "dnu_ch": 0.25,
        "fwhm": 0.5,
    }
    for key, expected in expected_metadata.items():
        if key not in archive or archive[key].shape != () or archive[key].item() != expected:
            raise ValueError(f"{label}: {key} differs from the final noise recipe")
    return arrays


def _compare_array(
    candidate: np.ndarray,
    frozen: np.ndarray,
    *,
    max_examples: int,
    row_offset: int = 0,
) -> dict[str, Any]:
    """Count all unequal values and retain only a bounded set of examples."""
    if candidate.shape != frozen.shape:
        raise ValueError("Candidate and frozen array shapes differ")
    unequal = candidate != frozen
    different_bits = candidate.view(np.uint32) != frozen.view(np.uint32)
    delta = candidate.astype(np.float64) - frozen.astype(np.float64)
    count = int(np.count_nonzero(unequal))
    positions = np.flatnonzero(unequal)[:max_examples]
    examples = []
    for flat_index in positions:
        index = tuple(int(value) for value in np.unravel_index(flat_index, candidate.shape))
        if len(index) == 2:
            index = (index[0] + row_offset, index[1])
        examples.append(
            {
                "index": list(index),
                "candidate": float(candidate.flat[flat_index]),
                "frozen": float(frozen.flat[flat_index]),
                "difference": float(delta.flat[flat_index]),
            }
        )
    return {
        "cells": int(candidate.size),
        "unequal_cells": count,
        "different_bit_pattern_cells": int(np.count_nonzero(different_bits)),
        "sum_squared_error": float(np.sum(delta * delta, dtype=np.float64)),
        "max_abs_error": float(np.max(np.abs(delta), initial=0)),
        "examples": examples,
    }


def _summarize_array(parts: list[dict[str, Any]], max_examples: int) -> dict[str, Any]:
    cells = sum(part["cells"] for part in parts)
    squared_error = sum(part["sum_squared_error"] for part in parts)
    examples = [example for part in parts for example in part["examples"]][:max_examples]
    return {
        "cells": cells,
        "unequal_cells": sum(part["unequal_cells"] for part in parts),
        "different_bit_pattern_cells": sum(part["different_bit_pattern_cells"] for part in parts),
        "rms_error": math.sqrt(squared_error / cells),
        "max_abs_error": max(part["max_abs_error"] for part in parts),
        "examples": examples,
        "examples_truncated": sum(part["unequal_cells"] for part in parts) > len(examples),
    }


def compare_noise(
    candidate_path: Path,
    frozen_path: Path,
    *,
    manifest_path: Path = DEFAULT_MANIFEST,
    expected_shape: tuple[int, int] = (30000, 2145),
    max_examples: int = 10,
    build_report_path: Path | None = None,
) -> dict[str, Any]:
    """Validate the frozen hash, then compare every numerical noise cell."""
    if max_examples < 0 or max_examples > 1000:
        raise ValueError("max_examples must be between 0 and 1000")
    artifacts = {item.identifier: item for item in load_manifest(manifest_path)}
    frozen_artifact = artifacts["noise_iasiang"]
    verify_file(frozen_path, frozen_artifact)
    candidate_size = candidate_path.stat().st_size
    candidate_sha256 = hash_file(candidate_path)
    if build_report_path is not None:
        build_report = json.loads(build_report_path.read_text(encoding="utf-8"))
        if build_report["sha256"] != candidate_sha256:
            raise ValueError("Candidate SHA256 differs from the build report")
        if tuple(build_report[key] for key in ("profiles", "channels")) != expected_shape:
            raise ValueError("Candidate shape differs from the build report")
        if build_report["seed"] != 1337 or build_report["batch_size"] != 256:
            raise ValueError("Candidate seed or batch size differs from the final recipe")

    with (
        np.load(candidate_path, allow_pickle=False) as candidate,
        np.load(frozen_path, allow_pickle=False) as frozen,
    ):
        candidate_arrays = _check_archive(candidate, "candidate", expected_shape)
        frozen_arrays = _check_archive(frozen, "frozen", expected_shape)
        if not np.array_equal(candidate_arrays["nu"], frozen_arrays["nu"]):
            raise ValueError("Candidate and frozen frequency axes differ")
        arrays = {}
        for key in ("nu", "sigma_ch"):
            arrays[key] = _summarize_array(
                [
                    _compare_array(
                        candidate_arrays[key], frozen_arrays[key], max_examples=max_examples
                    )
                ],
                max_examples,
            )

        # NumPy inflates each compressed noise member once. Bounded row slices
        # keep temporary float64 differences small even for the complete set.
        candidate_noise = candidate_arrays["noise"]
        frozen_noise = frozen_arrays["noise"]
        chunks = []
        for first in range(0, expected_shape[0], 256):
            last = min(first + 256, expected_shape[0])
            chunks.append(
                _compare_array(
                    candidate_noise[first:last],
                    frozen_noise[first:last],
                    max_examples=max_examples,
                    row_offset=first,
                )
            )
        arrays["noise"] = _summarize_array(chunks, max_examples)

    exact = all(detail["different_bit_pattern_cells"] == 0 for detail in arrays.values())
    return {
        "status": "exact_arrays" if exact else "numerically_different",
        "arrays_exact": exact,
        "frozen": {
            "asset_id": frozen_artifact.identifier,
            "bytes": frozen_artifact.size,
            "sha256": frozen_artifact.sha256,
            "manifest_validated": True,
        },
        "candidate": {
            "bytes": candidate_size,
            "sha256": candidate_sha256,
            "build_report_validated": build_report_path is not None,
        },
        "shape": list(expected_shape),
        "arrays": arrays,
        "comparison": "All numerical arrays; descriptive receta text ignored.",
    }


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--candidate", type=Path, default=DEFAULT_CANDIDATE)
    parser.add_argument("--frozen", type=Path, default=DEFAULT_FROZEN)
    parser.add_argument("--manifest", type=Path, default=DEFAULT_MANIFEST)
    parser.add_argument("--build-report", type=Path)
    parser.add_argument("--max-examples", type=int, default=10)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args(argv)
    try:
        report = compare_noise(
            args.candidate,
            args.frozen,
            manifest_path=args.manifest,
            max_examples=args.max_examples,
            build_report_path=args.build_report,
        )
        args.out.parent.mkdir(parents=True, exist_ok=True)
        args.out.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    except (FileNotFoundError, KeyError, ValueError, OSError) as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        return 1
    print(f"{report['status']}: {report['arrays']['noise']['unequal_cells']} unequal noise cells")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
