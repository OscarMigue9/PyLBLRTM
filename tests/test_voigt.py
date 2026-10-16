"""
Tests for voigt.py and for the voigt / n_jobs options of the forward model.

All synthetic and offline. Tests that need numba are skipped without it.
"""

import numpy as np
import pytest

from pylblrtm import voigt as vg
from pylblrtm.atmosphere import AtmosphericProfile
from pylblrtm.config import load_config, run_forward
from pylblrtm.forward import compute_transmittance
from pylblrtm.spectral_data import SpectralData

_LN2 = np.log(2.0)
_DOPPLER = 1e-3   # Doppler HWHM [cm⁻¹], about CO2 at 700 cm⁻¹

# Lorentz/Doppler ratios from the upper atmosphere (Doppler) to the surface (Lorentz)
_RATIOS = [1e-4, 1e-3, 1e-2, 0.1, 1.0, 10.0, 100.0]


def _profile_one_line(dnu, lorentz, backend):
    return backend(dnu, np.array([0.0]), np.array([_DOPPLER]), np.array([lorentz]))[:, 0]


def _make_sd(n: int = 40, seed: int = 0) -> SpectralData:
    rng = np.random.default_rng(seed)
    T = np.arange(150.0, 401.0)
    return SpectralData(
        lines={
            "nu0":    np.sort(rng.uniform(690.0, 730.0, n)),
            "Sref":   10.0 ** rng.uniform(-24.0, -21.0, n),
            "g_air":  rng.uniform(0.06, 0.08, n),
            "g_self": rng.uniform(0.08, 0.10, n),
            "Elow":   rng.uniform(0.0, 1500.0, n),
            "n_air":  np.full(n, 0.75),
            "shift":  np.full(n, -0.002),
            "mol_id": np.full(n, 2, dtype=np.int32),
            "iso_id": np.full(n, 1, dtype=np.int32),
        },
        tips={(2, 1): (T, T * 2.0)},
        molar_masses={(2, 1): 43.98983},
        molecule="CO2",
    )


def _make_profile() -> AtmosphericProfile:
    return AtmosphericProfile(
        pressure_hpa=[1.0, 10.0, 100.0, 500.0, 1013.25],
        temperature_k=[220.0, 230.0, 250.0, 270.0, 288.0],
        vmr={"CO2": [4.15e-4] * 5},
    )


def _config(**calculation) -> dict:
    return {
        "spectral": {"nu_min_cm1": 700.0, "nu_max_cm1": 720.0, "step_cm1": 0.01},
        "calculation": calculation,
        "output": {"transmittance": True, "optical_depth_layers": True},
    }


# ── Backends ──────────────────────────────────────────────────────────────────

class TestWofz:

    def test_gaussian_limit(self):
        """Lorentz → 0 recovers the normalised Doppler (Gaussian) profile."""
        dnu = np.linspace(-3 * _DOPPLER, 3 * _DOPPLER, 601)
        gauss = np.sqrt(_LN2 / np.pi) / _DOPPLER * np.exp(-_LN2 * (dnu / _DOPPLER) ** 2)
        phi = _profile_one_line(dnu, 1e-9 * _DOPPLER, vg.voigt_wofz)
        np.testing.assert_allclose(phi, gauss, rtol=1e-6)

    def test_lorentz_limit(self):
        """Lorentz ≫ Doppler recovers the Lorentzian."""
        gamma = 1e3 * _DOPPLER
        dnu = np.linspace(-20 * gamma, 20 * gamma, 801)
        lorentz = gamma / np.pi / (dnu ** 2 + gamma ** 2)
        phi = _profile_one_line(dnu, gamma, vg.voigt_wofz)
        np.testing.assert_allclose(phi, lorentz, rtol=1e-5)

    def test_unit_area(self):
        """Grid area plus the Lorentzian tails beyond ±L add up to one."""
        gamma, half = 0.1 * _DOPPLER, 20 * _DOPPLER
        dnu = np.linspace(-half, half, 40001)
        phi = _profile_one_line(dnu, gamma, vg.voigt_wofz)
        area = (phi.sum() - 0.5 * (phi[0] + phi[-1])) * (dnu[1] - dnu[0])   # trapezoid rule
        tails = 1.0 - 2.0 / np.pi * np.arctan(half / gamma)   # far wings are Lorentzian
        assert abs(area + tails - 1.0) < 1e-5


class TestHumlicek:

    @pytest.fixture(autouse=True)
    def _needs_numba(self):
        pytest.importorskip("numba")

    @pytest.mark.parametrize("ratio", _RATIOS)
    def test_matches_wofz(self, ratio):
        """Humlicek W4 stays within 1e-4 relative of wofz, centre to 25 cm⁻¹."""
        dnu = np.concatenate([np.linspace(-0.05, 0.05, 2001), np.linspace(0.05, 25.0, 4000)])
        lorentz = ratio * _DOPPLER
        ref = _profile_one_line(dnu, lorentz, vg.voigt_wofz)
        got = _profile_one_line(dnu, lorentz, vg.voigt_humlicek)
        np.testing.assert_allclose(got, ref, rtol=1e-4)

    def test_matrix_shape_and_lines(self):
        """[n_nu × n_lines], one column per line, same as wofz."""
        nu = np.linspace(699.0, 701.0, 101)
        centers = np.array([699.5, 700.0, 700.7])
        doppler = np.full(3, _DOPPLER)
        lorentz = np.array([1e-3, 5e-2, 1e-1])
        got = vg.voigt_humlicek(nu, centers, doppler, lorentz)
        assert got.shape == (101, 3)
        np.testing.assert_allclose(
            got, vg.voigt_wofz(nu, centers, doppler, lorentz), rtol=1e-4,
        )


class TestGetVoigt:

    def test_wofz(self):
        assert vg.get_voigt("wofz") is vg.voigt_wofz

    def test_humlicek(self):
        pytest.importorskip("numba")
        assert vg.get_voigt("humlicek") is vg.voigt_humlicek

    def test_unknown_name_raises(self):
        with pytest.raises(ValueError, match="Unknown Voigt backend"):
            vg.get_voigt("lorentz")

    def test_humlicek_without_numba_says_what_to_install(self, monkeypatch):
        monkeypatch.setattr(vg, "numba", None)   # as if `import numba` had failed
        with pytest.raises(ImportError, match=r"pylblrtm\[fast\]"):
            vg.get_voigt("humlicek")
        with pytest.raises(ImportError, match=r"pylblrtm\[fast\]"):
            vg.humlicek_add(np.zeros(3), (np.zeros(1),) * 4, 25.0, np.zeros(3))


class TestHumlicekSumming:
    """humlicek_add (no phi matrix, closed-form far wings) against the matrix path."""

    @pytest.fixture(autouse=True)
    def _needs_numba(self):
        pytest.importorskip("numba")

    @staticmethod
    def _lines(n, ratio, seed=0):
        rng = np.random.default_rng(seed)
        doppler = np.full(n, _DOPPLER) * rng.uniform(0.5, 2.0, n)
        return (
            rng.uniform(690.0, 712.0, n),      # some centres off the grid
            10.0 ** rng.uniform(-3.0, 0.0, n),
            doppler,
            ratio * doppler * rng.uniform(0.5, 2.0, n),
        )

    @pytest.mark.parametrize("ratio", _RATIOS)
    def test_matches_profile_matrix(self, ratio):
        """Same Humlicek regions point by point: only rounding may differ."""
        nu = 700.0 + np.arange(2001) * 0.001
        shifted, strength, doppler, lorentz = self._lines(300, ratio)
        ref = vg.voigt_humlicek(nu, shifted, doppler, lorentz) @ strength
        got = np.zeros_like(nu)
        vg.humlicek_add(nu, vg.humlicek_prepare(shifted, strength, doppler, lorentz),
                        np.inf, got)
        np.testing.assert_allclose(got, ref, rtol=1e-12)

    def test_cutoff_point_by_point(self):
        """Each line only where |nu − centre| ≤ cutoff, the same test as the wofz path."""
        nu = 700.0 + np.arange(2001) * 0.001
        shifted, strength, doppler, lorentz = self._lines(300, 1.0)
        phi = vg.voigt_humlicek(nu, shifted, doppler, lorentz)
        phi[np.abs(nu[:, None] - shifted) > 3.0] = 0.0
        got = np.zeros_like(nu)
        vg.humlicek_add(nu, vg.humlicek_prepare(shifted, strength, doppler, lorentz),
                        3.0, got)
        np.testing.assert_allclose(got, phi @ strength, rtol=1e-12)

    def test_adds_to_what_is_there(self):
        nu = 700.0 + np.arange(101) * 0.01
        lines = vg.humlicek_prepare(*self._lines(20, 1.0))
        once = np.zeros_like(nu)
        vg.humlicek_add(nu, lines, np.inf, once)
        twice = once.copy()
        vg.humlicek_add(nu, lines, np.inf, twice)
        np.testing.assert_allclose(twice, 2.0 * once, rtol=1e-15)


# ── Line cutoff ───────────────────────────────────────────────────────────────

def _backends():
    return ["wofz", pytest.param("humlicek", marks=pytest.mark.skipif(
        vg.numba is None, reason="needs numba"))]


class TestCutoff:

    @pytest.mark.parametrize("voigt", _backends())
    def test_nothing_beyond_cutoff(self, voigt):
        """One line at 700 cm⁻¹: τ > 0 up to 725 cm⁻¹ and exactly 0 past it."""
        T = np.arange(150.0, 401.0)
        one = lambda v: np.array([v], dtype=np.float64)
        sd = SpectralData(
            lines={"nu0": one(700.0), "Sref": one(1e-19), "g_air": one(0.07),
                   "g_self": one(0.1), "Elow": one(100.0), "n_air": one(0.75),
                   "shift": one(0.0), "mol_id": np.array([2]), "iso_id": np.array([1])},
            tips={(2, 1): (T, T * 2.0)}, molar_masses={(2, 1): 44.0}, molecule="CO2",
        )
        cfg = {"spectral": {"nu_min_cm1": 720.0, "nu_max_cm1": 730.0, "step_cm1": 0.001},
               "calculation": {"voigt": voigt},
               "output": {"transmittance": True, "optical_depth": True}}
        res = run_forward(sd, _make_profile(), cfg)
        nu, tau = res.wavenumber_cm1, res.optical_depth
        assert np.all(tau[nu > 725.0 + 1e-9] == 0.0)
        assert np.all(tau[nu < 725.0 - 1e-9] > 0.0)

    @pytest.mark.parametrize("voigt", _backends())
    def test_tile_width_does_not_change_the_result(self, voigt):
        sd, profile = _make_sd(), _make_profile()
        ref = run_forward(sd, profile, _config(voigt=voigt, tile_width_cm1=2.0))
        for width in (0.37, 5.0):
            got = run_forward(sd, profile, _config(voigt=voigt, tile_width_cm1=width))
            np.testing.assert_allclose(got.optical_depth_layers, ref.optical_depth_layers,
                                       rtol=1e-12)


# ── Forward model: voigt and n_jobs ───────────────────────────────────────────

class TestForwardOptions:

    def test_n_jobs_does_not_change_the_result(self):
        """Threads only change who computes each tile: bit-identical stack."""
        sd, profile = _make_sd(), _make_profile()
        serial = run_forward(sd, profile, _config(n_jobs=1))
        threaded = run_forward(sd, profile, _config(n_jobs=3))
        every_core = run_forward(sd, profile, _config(n_jobs=-1))
        np.testing.assert_array_equal(threaded.optical_depth_layers, serial.optical_depth_layers)
        np.testing.assert_array_equal(every_core.optical_depth_layers, serial.optical_depth_layers)

    @pytest.mark.parametrize("n_levels, n_jobs", [
        (5, 2),    # 4 layers, 2 threads
        (2, 4),    # 1 layer (standoff), 4 threads
        (10, 4),   # 9 layers, 4 threads: layers do not split evenly
    ])
    def test_every_layer_and_thread_count_is_bit_identical(self, n_levels, n_jobs):
        sd = _make_sd()
        profile = AtmosphericProfile(
            pressure_hpa=np.geomspace(1.0, 1013.25, n_levels),
            temperature_k=np.linspace(220.0, 288.0, n_levels),
            vmr={"CO2": [4.15e-4] * n_levels},
        )
        serial = run_forward(sd, profile, _config(n_jobs=1))
        threaded = run_forward(sd, profile, _config(n_jobs=n_jobs))
        np.testing.assert_array_equal(threaded.optical_depth_layers, serial.optical_depth_layers)

    def test_humlicek_close_to_wofz(self):
        pytest.importorskip("numba")
        sd, profile = _make_sd(), _make_profile()
        ref = run_forward(sd, profile, _config(voigt="wofz"))
        got = run_forward(sd, profile, _config(voigt="humlicek", n_jobs=2))
        np.testing.assert_allclose(got.optical_depth_layers, ref.optical_depth_layers, rtol=1e-4)

    def test_compute_transmittance_accepts_options(self):
        sd, profile = _make_sd(), _make_profile()
        _, t1 = compute_transmittance(sd, profile, 700.0, 720.0, step_cm1=0.01)
        _, t2 = compute_transmittance(sd, profile, 700.0, 720.0, step_cm1=0.01, n_jobs=2)
        np.testing.assert_array_equal(t1, t2)

    def test_worker_error_reaches_the_caller(self):
        """An error inside a thread is raised, not swallowed by the pool."""
        sd = _make_sd()
        sd.lines["Sref"][0] = -1.0
        with pytest.raises(FloatingPointError, match="negative line strength"):
            compute_transmittance(sd, _make_profile(), 700.0, 720.0, step_cm1=0.01, n_jobs=2)

    def test_unknown_voigt_raises(self):
        with pytest.raises(ValueError, match="voigt must be one of"):
            compute_transmittance(_make_sd(), _make_profile(), 700.0, 720.0, voigt="nope")

    @pytest.mark.parametrize("n_jobs", [0, -2, 1.5, True, "4"])
    def test_invalid_n_jobs_raises(self, n_jobs):
        with pytest.raises(ValueError, match="n_jobs"):
            compute_transmittance(_make_sd(), _make_profile(), 700.0, 720.0, n_jobs=n_jobs)


class TestConfigOptions:

    def test_defaults_keep_current_behaviour(self):
        ca = load_config({})["calculation"]
        assert ca["voigt"] == "wofz"
        assert ca["n_jobs"] == 1

    def test_unknown_voigt_in_yaml_raises(self):
        with pytest.raises(ValueError, match="calculation.voigt"):
            load_config({"calculation": {"voigt": "fast"}})

    def test_invalid_n_jobs_in_yaml_raises(self):
        with pytest.raises(ValueError, match="calculation.n_jobs"):
            load_config({"calculation": {"n_jobs": 0}})


# ── wing_tolerance: far wings on coarser grids ────────────────────────────────

class TestWingTolerance:

    @pytest.fixture(autouse=True)
    def _needs_numba(self):
        pytest.importorskip("numba")

    @staticmethod
    def _profile(p_top, p_bottom, n_levels=3):
        return AtmosphericProfile(
            pressure_hpa=np.geomspace(p_top, p_bottom, n_levels),
            temperature_k=np.linspace(220.0, 288.0, n_levels),
            vmr={"CO2": [4.15e-4] * n_levels},
        )

    @staticmethod
    def _tau(sd, profile, step=0.002, **calculation):
        cfg = {"spectral": {"nu_min_cm1": 700.0, "nu_max_cm1": 720.0, "step_cm1": step},
               "calculation": {"voigt": "humlicek", **calculation},
               "output": {"transmittance": True, "optical_depth_layers": True}}
        return run_forward(sd, profile, cfg).optical_depth_layers

    @pytest.mark.parametrize("tol", [1e-3, 1e-4])
    @pytest.mark.parametrize("p_top, p_bottom", [
        (500.0, 1013.25),   # Lorentz-dominated, near the surface (standoff-like)
        (1.0, 20.0),        # Doppler-dominated, upper atmosphere
    ])
    def test_error_within_tolerance(self, tol, p_top, p_bottom):
        sd, profile = _make_sd(400), self._profile(p_top, p_bottom)
        exact = self._tau(sd, profile)
        got = self._tau(sd, profile, wing_tolerance=tol)
        assert np.max(np.abs(got - exact) / exact) <= tol

    def test_tile_width_and_threads_do_not_change_it(self):
        sd, profile = _make_sd(400), self._profile(500.0, 1013.25)
        ref = self._tau(sd, profile, wing_tolerance=1e-4)
        for extra in ({"tile_width_cm1": 0.37}, {"tile_width_cm1": 5.0}, {"n_jobs": 3}):
            got = self._tau(sd, profile, wing_tolerance=1e-4, **extra)
            np.testing.assert_allclose(got, ref, rtol=1e-12)

    def test_no_coarse_grid_fits_falls_back_to_exact(self):
        assert vg.humlicek_levels(0.5, 25.0, 1e-4) == []
        sd, profile = _make_sd(), self._profile(500.0, 1013.25)
        np.testing.assert_array_equal(
            self._tau(sd, profile, step=0.5, wing_tolerance=1e-4),
            self._tau(sd, profile, step=0.5),
        )


class TestWingToleranceValidation:

    def test_needs_humlicek(self):
        with pytest.raises(ValueError, match="voigt='humlicek'"):
            compute_transmittance(_make_sd(), _make_profile(), 700.0, 720.0, wing_tolerance=1e-4)

    @pytest.mark.parametrize("bad", [0.0, 1.0, -1e-4, True, "1e-4"])
    def test_out_of_range(self, bad):
        with pytest.raises(ValueError, match="wing_tolerance"):
            compute_transmittance(_make_sd(), _make_profile(), 700.0, 720.0,
                                  voigt="humlicek", wing_tolerance=bad)

    def test_config(self):
        assert load_config({})["calculation"]["wing_tolerance"] is None
        with pytest.raises(ValueError, match="calculation.wing_tolerance"):
            load_config({"calculation": {"wing_tolerance": 1e-4}})
