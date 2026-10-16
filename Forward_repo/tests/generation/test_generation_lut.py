"""Small, deterministic checks for the archived SVD storage contract."""

from pathlib import Path

import numpy as np
import pytest

from pipelines.generation.lut import blocks_for, expand_lut, inspect_blocks


def _block(path: Path, start: int, end: int, value: float) -> None:
    channels = end - start + 1
    np.savez_compressed(
        path,
        gas="H2O",
        a=float(start),
        b=float(end),
        U=np.zeros((2, 1), dtype=np.float32),
        V=np.zeros((1, channels), dtype=np.float32),
        mu=np.full(channels, value, dtype=np.float32),
        NT=1,
        NP=2,
        r=1,
        dnu=1.0,
    )


def test_expand_lut_assigns_shared_endpoint_to_right_block(tmp_path: Path) -> None:
    _block(tmp_path / "lut_H2O_10_20.npz", 10, 20, 1.0)
    _block(tmp_path / "lut_H2O_20_30.npz", 20, 30, 2.0)

    report = expand_lut(tmp_path, tmp_path / "dense.npy", "H2O", 12, 28)
    dense = np.load(tmp_path / "dense.npy")

    assert report["shape"] == [1, 2, 17]
    np.testing.assert_array_equal(dense[:, :, :8], 1.0)
    np.testing.assert_array_equal(dense[:, :, 8:], 2.0)
    assert report["status"] == "candidate_historical_provenance_unverified"


def test_inspect_blocks_rejects_gap_and_corrupt_metadata(tmp_path: Path) -> None:
    _block(tmp_path / "lut_H2O_10_20.npz", 10, 20, 1.0)
    _block(tmp_path / "lut_H2O_21_30.npz", 21, 30, 2.0)
    with pytest.raises(ValueError, match="gap or overlap"):
        blocks_for(tmp_path, "H2O", 12, 28)

    (tmp_path / "lut_H2O_21_30.npz").unlink()
    _block(tmp_path / "lut_H2O_20_30.npz", 20, 30, 2.0)
    np.savez_compressed(
        tmp_path / "lut_H2O_20_30.npz",
        gas="CO2",
        a=20.0,
        b=30.0,
        U=np.zeros((2, 1)),
        V=np.zeros((1, 11)),
        mu=np.zeros(11),
        NT=1,
        NP=2,
        r=1,
        dnu=1.0,
    )
    with pytest.raises(ValueError, match="embedded gas"):
        inspect_blocks(tmp_path, "H2O", 12, 28)
