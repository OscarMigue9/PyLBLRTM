"""Lightweight guards for the frozen 714 cm⁻¹ retrieval contracts."""

from __future__ import annotations

import numpy as np
import pytest
import torch

from iasi_retrieval.assembly import _initial_latents
from iasi_retrieval.models import APAE, ResidualStepNetwork, load_apae
from iasi_retrieval.prior import geographic_features, load_prior, make_prior_network
from iasi_retrieval.state import (
    BLOCKS,
    Standardizer,
    block_mse,
    build_state_matrix_full,
    gas_slice,
    split_indices,
    state_metrics_full,
)
from iasi_retrieval.unrolled import (
    auxiliary_features,
    full_band_windows,
    load_sew,
    prepare_windows,
    residual,
    UnrolledRetrieval,
)


def test_full_ncol_state_order_floor_and_block_loss() -> None:
    columns = np.ones((2, 5, 136)) * 1.0e20
    columns[:, 0, 0] = 1.0
    data = {
        "Ncol": columns,
        "T_levels": np.ones((2, 137)) * 250,
        "T_skin": np.array([270.0, 280.0]),
    }
    state = build_state_matrix_full(data)
    assert state.shape == (2, 818)
    np.testing.assert_array_equal(state[:, 0], [250.0, 250.0])
    np.testing.assert_array_equal(state[:, gas_slice("H2O").start], [6.0, 6.0])
    np.testing.assert_array_equal(state[:, gas_slice("CH4").stop - 1], [20.0, 20.0])
    np.testing.assert_array_equal(state[:, 817], [270.0, 280.0])
    assert len(BLOCKS) == 7
    predicted = torch.zeros(2, 818)
    predicted[:, gas_slice("N2O")] = 2
    assert float(block_mse(predicted, torch.zeros_like(predicted))) == pytest.approx(4 / 7)


def test_split_and_standardizer_are_reproducible() -> None:
    first, second = split_indices(), split_indices()
    assert {key: len(value) for key, value in first.items()} == {
        "train": 24_000,
        "val": 3_000,
        "test": 3_000,
    }
    for key in first:
        np.testing.assert_array_equal(first[key], second[key])
    np.testing.assert_array_equal(np.sort(np.concatenate(list(first.values()))), np.arange(30_000))
    values = np.array([[1.0, 2.0], [3.0, 2.0]])
    standardizer = Standardizer().fit(values)
    np.testing.assert_allclose(standardizer.to_phys(standardizer.to_z(values)), values)


def test_metrics_zero_error_and_feature_order() -> None:
    state = np.zeros((2, 818))
    for gas in ("H2O", "CO2", "O3", "N2O", "CH4"):
        state[:, gas_slice(gas)] = 20.0
    metrics = state_metrics_full(state, state)
    assert metrics["rmse_T_K"] == 0.0
    assert all(metrics[f"rmse_{gas}_dex"] == 0.0 for gas in ("H2O", "CO2", "O3", "N2O", "CH4"))
    features = geographic_features({"lat": [1], "lon": [2], "P_surf_hPa": [4]}, np.array([3]))
    np.testing.assert_array_equal(features, [[1, 2, 3, 4]])


def test_dense_model_and_prior_checkpoint_load_strictly(tmp_path) -> None:
    apae = APAE(818, 6435, 24)
    path = tmp_path / "apae.pt"
    torch.save(
        {
            "model_state": apae.state_dict(),
            "state": "full",
            "decoder": "denso",
            "y_a_dim": 818,
            "y_t_dim": 6435,
            "latent_dim": 24,
            "n_aux": 0,
        },
        path,
    )
    loaded, _ = load_apae(path)
    assert set(loaded.state_dict()) == set(apae.state_dict())
    assert not any(parameter.requires_grad for parameter in loaded.parameters())
    prior = make_prior_network()
    prior_path = tmp_path / "prior.pt"
    torch.save(
        {
            "state_dict": prior.state_dict(),
            "capas": [256] * 4,
            "lat_dim": 24,
            "mu_p": np.zeros(4),
            "sd_p": np.ones(4),
        },
        prior_path,
    )
    loaded_prior, _ = load_prior(prior_path)
    assert set(loaded_prior.state_dict()) == set(prior.state_dict())
    minimal_path = tmp_path / "prior_historical.pt"
    torch.save(
        {
            "state_dict": prior.state_dict(),
            "capas": [256] * 4,
            "mu_p": np.zeros(4),
            "sd_p": np.ones(4),
            "epocas": 400,
        },
        minimal_path,
    )
    minimal_prior, _ = load_prior(minimal_path)
    assert minimal_prior[-1].out_features == 24
    bad_weights = dict(prior.state_dict())
    bad_weights["8.weight"] = torch.zeros(25, 256)
    torch.save(
        {"state_dict": bad_weights, "capas": [256] * 4, "mu_p": np.zeros(4), "sd_p": np.ones(4)},
        minimal_path,
    )
    with pytest.raises(ValueError, match="must map four features"):
        load_prior(minimal_path)


class SmallWindowForward:
    """A differentiable window with the real channel mask contract."""

    def _win(self, lo):
        mask = np.zeros(2145, bool)
        mask[:5] = True
        return (None, None, mask)

    def window(self, state, rows, lo, mu=None):
        return state[:, :1].expand(-1, 5), self._win(lo)[2]


def test_window_residual_and_shared_step_keep_gradients() -> None:
    forward = SmallWindowForward()
    sigma = np.ones(2145, dtype=np.float32)
    sigma[1] = np.nan
    windows = prepare_windows(forward, sigma, windows=[714.0])
    assert windows[0].absolute.tolist() == [0, 2, 3, 4]
    state = torch.ones(2, 818, requires_grad=True)
    observed = torch.zeros(2, 2145)
    valid = torch.as_tensor(np.isfinite(sigma).astype(np.float32))
    whiten = torch.eye(2145)
    result = residual(
        forward, state, np.array([0, 1]), windows, observed, torch.as_tensor(sigma), valid, whiten
    )
    assert result[:, [0, 2, 3, 4]].tolist() == [[-1.0] * 4] * 2
    assert result[:, 1].tolist() == [0.0, 0.0]
    result.sum().backward()
    assert state.grad[:, 0].tolist() == [-4.0, -4.0]
    assert len(full_band_windows()) == 30


def test_auxiliary_and_sew_checkpoint_contract(tmp_path) -> None:
    pressure = np.array([800.0, 900.0, 1000.0])
    emissivity = np.ones((3, 2145)) * 0.95
    mu = np.array([0.7, 0.8, 0.9])
    features, _ = auxiliary_features(pressure, emissivity, mu, np.array([0, 1]))
    assert features.shape == (3, 2147)
    assert features[2, 0] == pytest.approx(3.0)
    assert features[2, 1] == pytest.approx(3.0)
    network = ResidualStepNetwork(4292, 24)
    path = tmp_path / "sew.pt"
    torch.save(
        {
            "mlp": network.state_dict(),
            "latent": 24,
            "in_dim": 4292,
            "R_SD": torch.ones(2145),
            "args": {
                "se_corr": "isrf",
                "angulo": True,
                "za": "clim",
                "labels": "perfect",
                "lam_chi2": 0.0,
            },
        },
        path,
    )
    restored, _ = load_sew(path)
    assert set(restored.state_dict()) == set(network.state_dict())


def test_zero_initialized_step_preserves_prior_at_t3_and_t4() -> None:
    apae = APAE(818, 6435, 24).eval()
    step = ResidualStepNetwork(4292, 24).eval()
    torch.nn.init.zeros_(step.to_mu.weight)
    torch.nn.init.zeros_(step.to_mu.bias)
    prior = torch.randn(2, 24)
    forward = SmallWindowForward()
    windows = prepare_windows(forward, np.ones(2145), [714.0])
    model = UnrolledRetrieval(
        apae,
        step,
        forward,
        Standardizer(np.zeros(818), np.ones(818)),
        torch.zeros(2, 2145),
        torch.ones(2145),
        torch.zeros(2, 2147),
        prior,
        torch.ones(2),
        torch.ones(2145),
        torch.eye(2145),
        windows,
    )
    expected, _ = model.decode(prior)
    for iterations in (3, 4):
        state, _, latent, updates = model.run(np.array([0, 1]), iterations)
        torch.testing.assert_close(state, expected, rtol=0, atol=0)
        torch.testing.assert_close(latent, prior, rtol=0, atol=0)
        assert len(updates) == iterations


def test_training_precomputed_prior_and_live_inference_prior(tmp_path) -> None:
    class ConstantPrior(torch.nn.Module):
        def forward(self, features):
            return torch.full((len(features), 24), 2.0, device=features.device)

    prior = ConstantPrior()
    checkpoint = {"mu_p": np.zeros(4), "sd_p": np.ones(4)}
    states = {"lat": np.zeros(2), "lon": np.zeros(2), "P_surf_hPa": np.ones(2) * 1013.25}
    live = _initial_latents(prior, checkpoint, states, np.ones(2), None, torch.device("cpu"))
    assert live.shape == (2, 24)
    assert torch.all(live == 2.0)

    saved = tmp_path / "prior_z_deep.npz"
    np.savez(saved, za=np.full((30_000, 24), 3.0, dtype=np.float32))
    training = _initial_latents(prior, checkpoint, states, np.ones(2), saved, torch.device("cpu"))
    assert training.shape == (30_000, 24)
    assert torch.all(training == 3.0)

    np.savez(saved, za=np.zeros((30_000, 25), dtype=np.float32))
    with pytest.raises(ValueError, match="shape \\(30000, 24\\)"):
        _initial_latents(prior, checkpoint, states, np.ones(2), saved, torch.device("cpu"))
    invalid = np.zeros((30_000, 24), dtype=np.float32)
    invalid[0, 0] = np.nan
    np.savez(saved, za=invalid)
    with pytest.raises(ValueError, match="finite numeric"):
        _initial_latents(prior, checkpoint, states, np.ones(2), saved, torch.device("cpu"))
