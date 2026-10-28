"""
Tests for transfer.py — T, U and D with known answers.

Offline: one synthetic H2O line and profiles built by hand. In an isothermal
atmosphere every Planck choice collapses to B(T0), so U and D follow from T.
"""

import numpy as np
import pytest

from pylblrtm.atmosphere import AtmosphericProfile, HomogeneousPath
from pylblrtm.forward import compute_transmittance
from pylblrtm.spectral_data import SpectralData
from pylblrtm.transfer import LAYER_PLANCK, _mass_weighted_temperature, compute_tud, planck


def _sd(sref=1e-21):
    """One H2O line at 900 cm⁻¹."""
    one = lambda v: np.array([v], dtype=float)
    lines = {"nu0": one(900.0), "Sref": one(sref), "g_air": one(0.07), "g_self": one(0.35),
             "Elow": one(200.0), "n_air": one(0.7), "shift": one(0.0),
             "mol_id": np.ones(1, np.int32), "iso_id": np.ones(1, np.int32)}
    T = np.arange(100.0, 401.0)
    return SpectralData(lines=lines, tips={(1, 1): (T, 50.0 * (T / 296.0) ** 1.5)},
                        molar_masses={(1, 1): 18.0106}, molecule="H2O")


def _profile(t=(230.0, 270.0, 295.0), vmr=(1e-4, 5e-3, 2e-2)):
    return AtmosphericProfile(pressure_hpa=[300.0, 700.0, 1013.25],
                              temperature_k=list(t), vmr={"H2O": list(vmr)})


def _tud(sd=None, profile=None, **kw):
    return compute_tud(sd or _sd(), profile or _profile(), 880.0, 920.0, step_cm1=0.01, **kw)


class TestPlanck:

    def test_value(self):
        """Against B = 2hc²ν³/(e^(hcν/kT) − 1) in SI, converted to W/(cm² sr cm⁻¹)."""
        h, c, k = 6.62607015e-34, 2.99792458e8, 1.380649e-23
        nu = np.array([714.0, 870.0, 1250.0])
        nu_m = nu * 100.0                                   # cm⁻¹ → m⁻¹
        b_si = 2 * h * c ** 2 * nu_m ** 3 / np.expm1(h * c * nu_m / (k * 290.0))  # W/(m² sr m⁻¹)
        # rtol: forward.py's c₂ = 1.4387752 cm K (CODATA 1998) against hc/k = 1.438776877
        # moves B by ~7e-6 relative
        np.testing.assert_allclose(planck(nu, 290.0), b_si * 100.0 / 1e4, rtol=2e-5)

    def test_mass_weighted_temperature(self):
        """∫T dP / ΔP with T linear in ln P, against a fine trapezoid."""
        p = np.array([100.0, 1000.0])
        t = np.array([200.0, 300.0])
        pp = np.geomspace(p[0], p[1], 200001)
        tt = t[0] + (t[1] - t[0]) * np.log(pp / p[0]) / np.log(p[1] / p[0])
        expected = np.trapezoid(tt, pp) / (p[1] - p[0])
        assert _mass_weighted_temperature(p, t)[0] == pytest.approx(expected, rel=1e-9)
        assert 250.0 < expected < 300.0          # weighted towards the denser bottom


class TestKnownAnswers:

    def test_transparent(self):
        nu, T, U, D = _tud(profile=_profile(vmr=(0.0, 0.0, 0.0)))
        assert np.all(T == 1.0) and np.all(U == 0.0) and np.all(D == 0.0)

    @pytest.mark.parametrize("layer_planck", LAYER_PLANCK)
    def test_isothermal(self, layer_planck):
        """U = B(T0)·(1 − T) and D = B(T0)·(1 − e^(−τ·diffusivity)), whatever the layer Planck."""
        t0, mu, diff = 260.0, 0.6, 1.67
        prof = _profile(t=(t0, t0, t0))
        nu, T, U, D = _tud(profile=prof, mu_view=mu, diffusivity=diff, layer_planck=layer_planck)
        with np.errstate(divide="ignore"):              # T = 0 at the opaque centre: tau = inf
            tau = -np.log(T) * mu
        b = planck(nu, t0)
        np.testing.assert_allclose(U, b * -np.expm1(-tau / mu), rtol=1e-12)
        np.testing.assert_allclose(D, b * -np.expm1(-tau * diff), rtol=1e-12)

    def test_opaque_line_centre(self):
        """Where every layer is opaque, U is B at the top level and D is B at the surface."""
        nu, T, U, D = _tud(sd=_sd(sref=1e-16))
        i = np.argmin(np.abs(nu - 900.0))
        assert U[i] == pytest.approx(planck(nu[i:i + 1], 230.0)[0], rel=1e-2)
        assert D[i] == pytest.approx(planck(nu[i:i + 1], 295.0)[0], rel=1e-2)


class TestAngle:

    def test_t_and_u_follow_mu_d_does_not(self):
        _, T1, U1, D1 = _tud(mu_view=1.0)
        _, T2, U2, D2 = _tud(mu_view=0.5)
        np.testing.assert_allclose(T2, T1 ** 2, rtol=1e-12)
        np.testing.assert_array_equal(D2, D1)
        # Optically thin wings: U ∝ 1/μ. At the opaque centre it is the other way round:
        # the longer path sees higher, colder layers.
        thin = T1 > 0.999
        np.testing.assert_allclose(U2[thin] / U1[thin], 2.0, rtol=2e-3)

    def test_same_t_as_compute_transmittance(self):
        nu, T, _, _ = _tud(mu_view=0.7, h2o_continuum=True)
        _, T_ref = compute_transmittance(_sd(), _profile(), 880.0, 920.0, step_cm1=0.01,
                                         mu_view=0.7, h2o_continuum=True)
        np.testing.assert_array_equal(T, T_ref)


class TestErrors:

    def test_homogeneous_path(self):
        with pytest.raises(TypeError, match="vertical"):
            compute_tud(_sd(), HomogeneousPath(100.0, {"H2O": 0.01}), 880.0, 920.0)

    def test_layer_planck(self):
        with pytest.raises(ValueError, match="layer_planck"):
            _tud(layer_planck="mean")

    def test_diffusivity(self):
        with pytest.raises(ValueError, match="diffusivity"):
            _tud(diffusivity=0.6)
