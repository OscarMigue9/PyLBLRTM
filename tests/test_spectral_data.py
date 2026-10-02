"""
Tests for spectral_data.py — SpectralData container.
"""

import numpy as np
import pytest
from pylblrtm.spectral_data import SpectralData


def _make_lines(n: int, mol_id: int = 1, iso_id: int = 1) -> dict:
    return {
        "nu0":    np.linspace(700.0, 1400.0, n),
        "Sref":   np.ones(n) * 1e-23,
        "g_air":  np.ones(n) * 0.07,
        "g_self": np.ones(n) * 0.35,
        "Elow":   np.ones(n) * 100.0,
        "n_air":  np.ones(n) * 0.67,
        "shift":  np.zeros(n),
        "mol_id": np.full(n, mol_id, dtype=np.int32),
        "iso_id": np.full(n, iso_id, dtype=np.int32),
    }


def _make_tips(T_min=150.0, T_max=400.0) -> tuple:
    T = np.arange(T_min, T_max + 1.0)
    Q = T * 0.5  # dummy linear Q(T)
    return T, Q


@pytest.fixture
def single_iso_sd():
    T, Q = _make_tips()
    return SpectralData(
        lines=_make_lines(20, mol_id=1, iso_id=1),
        tips={(1, 1): (T, Q)},
        molar_masses={(1, 1): 18.010565},
        molecule="H2O",
    )


@pytest.fixture
def multi_iso_sd():
    lines = {k: np.concatenate([_make_lines(10, 1, 1)[k], _make_lines(8, 1, 2)[k]])
             for k in _make_lines(1)}
    T1, Q1 = _make_tips()
    T2, Q2 = _make_tips(T_min=100.0)
    return SpectralData(
        lines=lines,
        tips={(1, 1): (T1, Q1), (1, 2): (T2, Q2)},
        molar_masses={(1, 1): 18.010565, (1, 2): 20.014811},
        molecule="H2O",
    )


class TestSpectralDataProperties:

    def test_n_lines(self, single_iso_sd):
        assert single_iso_sd.n_lines == 20

    def test_nu0_returns_array(self, single_iso_sd):
        assert len(single_iso_sd.nu0) == 20

    def test_Sref_returns_array(self, single_iso_sd):
        assert len(single_iso_sd.Sref) == 20

    def test_multi_iso_n_lines(self, multi_iso_sd):
        assert multi_iso_sd.n_lines == 18


class TestSpectralDataQ:

    def test_Q_interpolates_correctly(self, single_iso_sd):
        """Q(T=200) for linear Q=T*0.5 should return ~100."""
        result = single_iso_sd.Q(200.0, mol_id=1, iso_id=1)
        assert abs(result - 100.0) < 1e-6

    def test_Q_accepts_numpy_int(self, single_iso_sd):
        """Q() should accept numpy int types for mol_id and iso_id."""
        result = single_iso_sd.Q(200.0, np.int32(1), np.int32(1))
        assert isinstance(result, float)

    def test_Q_unknown_key_raises_key_error(self, single_iso_sd):
        with pytest.raises(KeyError, match="mol_id=1, iso_id=99"):
            single_iso_sd.Q(200.0, mol_id=1, iso_id=99)

    def test_Q_multi_iso_correct_key(self, multi_iso_sd):
        """Each isotopologue returns Q from its own TIPS table."""
        q1 = multi_iso_sd.Q(200.0, 1, 1)
        q2 = multi_iso_sd.Q(200.0, 1, 2)
        # Both use Q=T*0.5 → Q(200)=100, but T2 starts at 100 so same formula
        assert abs(q1 - 100.0) < 1e-6
        assert abs(q2 - 100.0) < 1e-6


class TestSpectralDataRepr:

    def test_repr_contains_molecule(self, single_iso_sd):
        assert "H2O" in repr(single_iso_sd)

    def test_repr_contains_iso_key(self, single_iso_sd):
        assert "(1,1)" in repr(single_iso_sd)

    def test_repr_multi_iso(self, multi_iso_sd):
        r = repr(multi_iso_sd)
        assert "(1,1)" in r
        assert "(1,2)" in r

    def test_repr_no_molecule_shows_multi(self):
        T, Q = _make_tips()
        sd = SpectralData(
            lines=_make_lines(5),
            tips={(1, 1): (T, Q)},
            molar_masses={(1, 1): 18.0},
            molecule=None,
        )
        assert "multi-molecule" in repr(sd)
