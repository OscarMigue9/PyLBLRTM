"""Check the sea scene contract and shared-LUT retrieval binding."""

from __future__ import annotations

import hashlib
import json
from argparse import Namespace
from pathlib import Path

import numpy as np
import pytest
import torch

from iasi_retrieval.assembly import bind_mar64
from iasi_retrieval.forward_adapter import NU_CHANNEL, PhysicalForward
from iasi_retrieval.mar64 import Mar64Scenes, load_mar64, sea_fresnel_emissivity
from iasi_retrieval.models import APAE, ResidualStepNetwork
from iasi_retrieval.prior import make_prior_network
from iasi_retrieval.state import Standardizer
from iasi_retrieval.unrolled import UnrolledRetrieval
from pipelines import final


def _scene(path: Path, *, mu: float, latitude: float) -> None:
    native = np.arange(645.0, 2760.0 + 0.125, 0.25)
    assert native.shape == (8461,)
    np.savez(
        path,
        T_levels=np.full(137, 250.0),
        Ncol=np.full((5, 136), 1.0e20),
        T_skin=280.0,
        P_lay=np.linspace(0.01, 1.0, 136),
        psurf_hPa=1000.0,
        mu=mu,
        cloud_frac=0.0,
        lat_fov=latitude,
        lon_fov=40.0,
        nu=native,
        L_obs=native * 1.0e-6,
    )


def test_mar_loader_preserves_legacy_state_radiance_and_geo_order(tmp_path: Path) -> None:
    scene_dir = tmp_path / "scenes"
    scene_dir.mkdir()
    _scene(scene_dir / "b_1.npz", mu=0.8, latitude=2.0)
    _scene(scene_dir / "a_0.npz", mu=0.9, latitude=1.0)
    geo = tmp_path / "geo.npz"
    np.savez(
        geo,
        nombres=np.array(["a_0", "b_1"]),
        lat=[1.0, 2.0],
        lon=[40.0, 40.0],
        doy=[80.0, 81.0],
        cloud=[0.0, 0.0],
        mu=[0.9, 0.8],
    )

    scenes = load_mar64(scene_dir, geo, expected_count=2)
    assert scenes.names == ("a_0", "b_1")
    assert scenes.truth.shape == (2, 818)
    assert scenes.truth[0, 137] == pytest.approx(20.0)
    np.testing.assert_allclose(scenes.observed[0], NU_CHANNEL * 0.001)
    np.testing.assert_array_equal(
        scenes.geography, [[1.0, 40.0, 80.0, 1000.0], [2.0, 40.0, 81.0, 1000.0]]
    )
    np.testing.assert_allclose(scenes.emissivity[0], sea_fresnel_emissivity(0.9))

    np.savez(
        geo,
        nombres=np.array(["b_1", "a_0"]),
        lat=[1.0, 2.0],
        lon=[40.0, 40.0],
        doy=[80.0, 81.0],
        cloud=[0.0, 0.0],
        mu=[0.9, 0.8],
    )
    with pytest.raises(ValueError, match="names are not aligned"):
        load_mar64(scene_dir, geo, expected_count=2)


def test_binding_shares_lut_and_applies_train_statistics(tmp_path: Path) -> None:
    forward = PhysicalForward.__new__(PhysicalForward)
    forward.lut = torch.ones(1)
    forward.pressure_layers = np.zeros((3, 136), dtype=np.float32)
    forward.emissivity = np.zeros((3, 2145))
    forward.window_cache = {714.0: object()}
    base = UnrolledRetrieval(
        APAE(818, 6435, 24),
        ResidualStepNetwork(4292, 24),
        forward,
        Standardizer(np.zeros(818), np.ones(818)),
        torch.zeros(3, 2145),
        torch.ones(2145),
        torch.zeros(3, 2147),
        torch.zeros(3, 24),
        torch.ones(3),
        torch.ones(2145),
        torch.eye(2145),
        [],
    )
    simulated = {
        "d": {"P_surf_hPa": np.array([800.0, 900.0, 1000.0])},
        "EPS": np.stack([np.full(2145, value) for value in (0.90, 0.91, 0.92)]),
        "mu": np.array([0.7, 0.8, 0.9]),
        "idx": {"train": np.array([0, 1])},
    }
    prior = make_prior_network()
    for parameter in prior.parameters():
        torch.nn.init.zeros_(parameter)
    prior_path = tmp_path / "prior.pt"
    torch.save(
        {
            "state_dict": prior.state_dict(),
            "capas": [256] * 4,
            "mu_p": np.zeros(4),
            "sd_p": np.ones(4),
        },
        prior_path,
    )
    scenes = Mar64Scenes(
        names=tuple(f"s_{index}" for index in range(64)),
        truth=np.zeros((64, 818)),
        observed=np.zeros((64, 2145)),
        pressure_layers=np.ones((64, 136), dtype=np.float32),
        emissivity=np.full((64, 2145), 0.915),
        mu=np.full(64, 0.85),
        pressure_surface=np.full(64, 950.0),
        geography=np.tile([0.0, 0.0, 80.0, 950.0], (64, 1)),
        cloud=np.zeros(64),
    )

    sea = bind_mar64(base, simulated, scenes, prior_path)
    assert sea.forward is not base.forward
    assert sea.forward.lut is base.forward.lut
    assert sea.forward.window_cache == {}
    np.testing.assert_array_equal(base.forward.pressure_layers, np.zeros((3, 136)))
    np.testing.assert_array_equal(sea.forward.pressure_layers, scenes.pressure_layers)
    np.testing.assert_allclose(sea.auxiliary[0, :3], [2.0, 2.0, 2.0], rtol=1e-6)
    assert sea.initial_latents.shape == (64, 24)
    assert torch.count_nonzero(sea.initial_latents) == 0


def test_pipeline_rejects_a_changed_scene_before_gpu_setup(tmp_path: Path, monkeypatch) -> None:
    directory = tmp_path / "valstates_full"
    directory.mkdir()
    entries = []
    for index in range(64):
        path = directory / f"scene_{index:02d}.npz"
        path.write_bytes(f"scene:{index}".encode())
        entries.append(
            {
                "name": path.name,
                "bytes": path.stat().st_size,
                "sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
            }
        )
    geo = tmp_path / "geo.npz"
    geo.write_bytes(b"verified geography")
    spec = {"schema_version": 1, "selection": {"scene_count": 64}, "scene_files": entries}
    spec_path = tmp_path / "mar64_inputs.json"
    spec_path.write_text(json.dumps(spec), encoding="utf-8")
    monkeypatch.setattr(final, "MAR_MANIFEST", spec_path)
    monkeypatch.setattr(final, "MAR_GEO_SHA256", hashlib.sha256(geo.read_bytes()).hexdigest())
    monkeypatch.setattr(final, "_paths", lambda _manifest, _cache, _ids: {"mar64_geo": geo})
    args = Namespace(mar_scene_dir=directory, cache=tmp_path, manifest=tmp_path / "assets.json")
    inputs, _ = final._mar64_inputs(args)
    assert len(inputs) == 65

    (directory / "scene_02.npz").write_bytes(b"changed!")
    with pytest.raises(ValueError, match="scene_02"):
        final._mar64_inputs(args)
