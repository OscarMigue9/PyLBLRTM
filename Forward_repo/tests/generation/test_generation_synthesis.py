"""Numerical output canaries distinguish data identity from archive metadata."""

from pathlib import Path

import numpy as np

from pipelines.generation.synthesis import compare_generated_npz


def test_noise_canary_ignores_note_but_detects_sample_change(tmp_path: Path) -> None:
    frozen = tmp_path / "frozen.npz"
    candidate = tmp_path / "candidate.npz"
    noise = np.arange(12, dtype=np.float32).reshape(3, 4)
    coordinates = np.arange(4, dtype=np.float32)
    np.savez(frozen, noise=noise, sigma_ch=coordinates, nu=coordinates, receta="original")
    np.savez(
        candidate,
        noise=noise[:2],
        sigma_ch=coordinates,
        nu=coordinates,
        receta="candidate",
    )

    report = compare_generated_npz(candidate, frozen, kind="noise", profiles=2)
    assert report["arrays_exact"]

    noise[0, 0] += 1
    np.savez(
        candidate,
        noise=noise[:2],
        sigma_ch=coordinates,
        nu=coordinates,
        receta="candidate",
    )
    report = compare_generated_npz(candidate, frozen, kind="noise", profiles=2)
    assert not report["arrays_exact"]
    assert report["max_abs_by_key"]["noise"] == 1.0
