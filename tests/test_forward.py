"""
Tests for forward.py — physics with a known answer, on synthetic lines.

Offline and fast: no HAPI, a single H2O-labelled line or two, and profiles
built by hand, so every expected value follows from the formulas.
"""

import math

import numpy as np
import pytest
from scipy.integrate import trapezoid

from pylblrtm.atmosphere import AtmosphericProfile
from pylblrtm.forward import compute_transmittance
from pylblrtm.spectral_data import SpectralData

_C2 = 1.4387752    # second radiation constant [cm K], as in forward.py
_T_REF = 296.0


def _lines(nu0, sref=1e-20, g_air=0.05, g_self=0.05, elow=500.0, n_air=0.7):
    nu0 = np.atleast_1d(np.asarray(nu0, dtype=float))
    n = nu0.size
    return {
        "nu0": nu0,
        "Sref": np.full(n, sref),
        "g_air": np.full(n, g_air),
        "g_self": np.full(n, g_self),
        "Elow": np.full(n, elow),
        "n_air": np.full(n, n_air),
        "shift": np.zeros(n),
        "mol_id": np.ones(n, dtype=np.int32),
        "iso_id": np.ones(n, dtype=np.int32),
    }


def _q(T):
    """A partition function that changes with T, so the Q ratio is tested."""
    return 50.0 * (np.asarray(T, dtype=float) / _T_REF) ** 1.5


def _sd(lines):
    T = np.arange(100.0, 401.0)
    return SpectralData(lines=lines, tips={(1, 1): (T, _q(T))},
                        molar_masses={(1, 1): 18.0106}, molecule="H2O")


def _one_layer(T_k, vmr=1e-6, p_top=500.0, p_bot=510.0):
    """Two levels at the same temperature: one layer at exactly T_k."""
    return AtmosphericProfile(pressure_hpa=[p_top, p_bot],
                              temperature_k=[T_k, T_k], vmr={"H2O": [vmr, vmr]})


def _tau(sd, profile, nu_min=880.0, nu_max=920.0, step=0.001, **kw):
    nu, T = compute_transmittance(sd, profile, nu_min, nu_max, step_cm1=step, **kw)
    return nu, -np.log(T)


def _expected_strength(nu0, sref, elow, T):
    """HITRAN line strength at T [cm⁻¹/(molec cm⁻²)]."""
    return (sref * float(_q(_T_REF) / _q(T))
            * math.exp(-_C2 * elow * (1.0 / T - 1.0 / _T_REF))
            * (-math.expm1(-_C2 * nu0 / T)) / (-math.expm1(-_C2 * nu0 / _T_REF)))


def _column(profile):
    return float(profile.build_layers().column_cm2["H2O"][0])


class TestLineArea:
    """∫τ dν of an isolated line = S(T) · N_col (minus the wings past the cutoff)."""

    @pytest.mark.parametrize("T_k", [296.0, 220.0, 280.0, 320.0])
    def test_area_matches_strength_times_column(self, T_k):
        nu0, sref, elow = 900.0, 1e-20, 500.0
        profile = _one_layer(T_k)
        nu, tau = _tau(_sd(_lines(nu0, sref=sref, elow=elow)), profile)
        area = trapezoid(tau, nu)
        expected = _expected_strength(nu0, sref, elow, T_k) * _column(profile)
        # The Lorentz wings beyond 25 cm⁻¹ hold ~ (2/π)·γ/25 of the area (< 0.1 %)
        assert area == pytest.approx(expected, rel=2e-3)

    def test_cold_layer_weakens_high_elow_line(self):
        """Boltzmann factor: a high-E'' line is ~8× weaker at 220 K than at 296 K."""
        line = _lines(900.0, elow=1500.0)
        a_warm = trapezoid(*_tau(_sd(line), _one_layer(296.0))[::-1])
        a_cold = trapezoid(*_tau(_sd(line), _one_layer(220.0))[::-1])
        expected = (_expected_strength(900.0, 1e-20, 1500.0, 220.0)
                    / _expected_strength(900.0, 1e-20, 1500.0, 296.0))
        assert expected < 0.13
        assert a_cold / a_warm == pytest.approx(expected, rel=2e-3)


class TestScaling:

    def test_tau_doubles_with_vmr(self):
        """g_self = g_air: the line shape does not depend on VMR, so τ ∝ column."""
        sd = _sd(_lines([895.0, 905.0]))
        _, tau1 = _tau(sd, _one_layer(260.0, vmr=1e-3))
        _, tau2 = _tau(sd, _one_layer(260.0, vmr=2e-3))
        np.testing.assert_allclose(tau2, 2.0 * tau1, rtol=1e-10)

    def test_mu_view_half_doubles_slant_tau(self):
        # τ ~ 1e-4 or more: -ln(T) of a T within 1e-7 of 1 loses ~1e-9 to rounding
        sd = _sd(_lines([895.0, 905.0]))
        profile = _one_layer(260.0, vmr=1e-3)
        _, tau_nadir = _tau(sd, profile, mu_view=1.0)
        _, tau_slant = _tau(sd, profile, mu_view=0.5)
        np.testing.assert_allclose(tau_slant, 2.0 * tau_nadir, rtol=1e-10)

    def test_more_pressure_broadens_lower_peak_same_area(self):
        sd = _sd(_lines(900.0))
        thin = _one_layer(296.0, p_top=100.0, p_bot=102.0)
        thick = _one_layer(296.0, p_top=1000.0, p_bot=1002.0)   # same column
        nu, tau_thin = _tau(sd, thin)
        _, tau_thick = _tau(sd, thick)
        assert _column(thin) == pytest.approx(_column(thick), rel=1e-9)
        assert tau_thick.max() < tau_thin.max()
        assert trapezoid(tau_thick, nu) == pytest.approx(trapezoid(tau_thin, nu), rel=5e-3)


class TestBounds:

    def test_transmittance_in_unit_interval(self):
        sd = _sd(_lines(np.linspace(881.0, 919.0, 40), sref=1e-18))
        _, T = compute_transmittance(sd, _one_layer(280.0, vmr=1e-3), 880.0, 920.0,
                                     step_cm1=0.005)
        assert np.all(np.isfinite(T)) and np.all((T >= 0.0) & (T <= 1.0))
        assert T.min() < 0.5          # strong enough to absorb

    def test_no_lines_within_cutoff_is_transparent(self):
        sd = _sd(_lines(800.0))       # 80 cm⁻¹ away, beyond the 25 cm⁻¹ cutoff
        _, T = compute_transmittance(sd, _one_layer(280.0), 880.0, 920.0, step_cm1=0.01)
        assert np.all(T == 1.0)

    @pytest.mark.filterwarnings("ignore:Gas 'H2O'")
    def test_gas_missing_from_profile_does_not_absorb(self):
        sd = _sd(_lines(900.0))
        profile = AtmosphericProfile(pressure_hpa=[500.0, 510.0], temperature_k=[280.0, 280.0],
                                     vmr={"CO2": [4e-4, 4e-4]})
        _, T = compute_transmittance(sd, profile, 880.0, 920.0, step_cm1=0.01)
        assert np.all(T == 1.0)


# ── Camino standoff homogéneo ─────────────────────────────────────────────────

from pylblrtm.atmosphere import HomogeneousPath  # noqa: E402
from pylblrtm.config import run_forward  # noqa: E402

_K_B = 1.380649e-23


class TestHomogeneousPath:

    def test_column_is_number_density_times_length(self):
        path = HomogeneousPath(100.0, {"H2O": 1e-6}, pressure_hpa=1013.25, temperature_k=296.0)
        layers = path.build_layers()
        expected = 1e-6 * 101325.0 / (_K_B * 296.0) * 100.0 / 1e4
        assert layers.n_layers == 1
        assert layers.column_cm2["H2O"][0] == pytest.approx(expected, rel=1e-12)
        assert layers.pressure_atm[0] == pytest.approx(1.0)
        assert layers.temperature_k[0] == 296.0

    def test_line_area_matches_strength_times_column(self):
        nu0, sref, elow = 900.0, 1e-20, 500.0
        path = HomogeneousPath(100.0, {"H2O": 1e-6}, temperature_k=280.0)
        nu, tau = _tau(_sd(_lines(nu0, sref=sref, elow=elow)), path)
        N = path.build_layers().column_cm2["H2O"][0]
        assert trapezoid(tau, nu) == pytest.approx(
            _expected_strength(nu0, sref, elow, 280.0) * N, rel=2e-3)

    def test_tau_doubles_with_length(self):
        sd = _sd(_lines([895.0, 905.0]))
        _, t1 = _tau(sd, HomogeneousPath(100.0, {"H2O": 1e-3}))
        _, t2 = _tau(sd, HomogeneousPath(200.0, {"H2O": 1e-3}))
        np.testing.assert_allclose(t2, 2.0 * t1, rtol=1e-10)

    def test_concentration_times_length_with_equal_broadening(self):
        """g_self = g_air: only ppm·m matters (2 ppm × 50 m = 1 ppm × 100 m)."""
        sd = _sd(_lines([895.0, 905.0]))
        _, a = _tau(sd, HomogeneousPath(50.0, {"H2O": 2e-3}))
        _, b = _tau(sd, HomogeneousPath(100.0, {"H2O": 1e-3}))
        np.testing.assert_allclose(a, b, rtol=1e-10)

    def test_works_with_run_forward(self):
        sd = _sd(_lines(900.0))
        path = HomogeneousPath(100.0, {"H2O": 1e-3})
        cfg = {"spectral": {"nu_min_cm1": 880.0, "nu_max_cm1": 920.0, "step_cm1": 0.01},
               "output": {"transmittance": True, "optical_depth_layers": True}}
        res = run_forward(sd, path, cfg)
        assert res.optical_depth_layers.shape[0] == 1
        _, T = compute_transmittance(sd, path, 880.0, 920.0, step_cm1=0.01)
        np.testing.assert_allclose(res.transmittance, T, rtol=1e-12)

    @pytest.mark.parametrize("kwargs, match", [
        ({"path_length_m": 0.0, "vmr": {"H2O": 1e-6}}, "path_length_m"),
        ({"path_length_m": 100.0, "vmr": {"H2O": [1e-6, 2e-6]}}, "single value"),
        ({"path_length_m": 100.0, "vmr": {"H2O": 1.5}}, r"\[0, 1\]"),
        ({"path_length_m": 100.0, "vmr": {}}, "at least one gas"),
        ({"path_length_m": 100.0, "vmr": {"H2O": 0.6, "CO2": 0.6}}, "more than 1"),
        ({"path_length_m": 100.0, "vmr": {"H2O": 1e-6}, "temperature_k": -5.0}, "temperature_k"),
    ])
    def test_invalid_input_raises(self, kwargs, match):
        with pytest.raises(ValueError, match=match):
            HomogeneousPath(**kwargs)
