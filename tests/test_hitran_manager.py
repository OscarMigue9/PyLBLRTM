"""
Tests for hitran_manager.py — multi-isotopologue support.

All tests are offline: HAPI and network calls are mocked.
"""

import numpy as np
import pytest
from unittest.mock import patch, MagicMock
from pathlib import Path

import pylblrtm.hitran_manager as hm


# ── Helpers ───────────────────────────────────────────────────────────────────

def _make_lines(n: int, iso_id: int, mol_id: int = 1) -> dict:
    """Dummy line parameter dict with n entries for a given isotopologue."""
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


@pytest.fixture
def mock_hapi():
    """Minimal HAPI mock. H2O (mol_id=1) has 4 isotopologues."""
    m = MagicMock()
    m.ISO = {(1, 1): None, (1, 2): None, (1, 3): None, (1, 4): None}
    m.HAPI_VERSION = "1.3.0"
    return m


# ── _fetch_single_iso ─────────────────────────────────────────────────────────

class TestFetchSingleIso:

    def test_cache_hit_skips_download(self, tmp_path, mock_hapi):
        """Cached data is returned without calling fetch()."""
        expected = _make_lines(10, 1)
        with patch.object(hm, "_get_hapi", return_value=mock_hapi), \
             patch.object(hm, "is_cached", return_value=True), \
             patch.object(hm, "_extract_line_data", return_value=expected):

            result = hm._fetch_single_iso("H2O", 1, 1, 700.0, 1400.0, tmp_path, False)

            mock_hapi.fetch.assert_not_called()
            np.testing.assert_array_equal(result["iso_id"], expected["iso_id"])

    def test_force_download_ignores_cache(self, tmp_path, mock_hapi):
        """force_download=True calls fetch() even when the table is cached."""
        with patch.object(hm, "_get_hapi", return_value=mock_hapi), \
             patch.object(hm, "is_cached", return_value=True), \
             patch.object(hm, "_can_reach_hitran", return_value=True), \
             patch.object(hm, "_write_metadata"), \
             patch.object(hm, "_extract_line_data", return_value=_make_lines(5, 1)):

            hm._fetch_single_iso("H2O", 1, 1, 700.0, 1400.0, tmp_path, True)

            mock_hapi.fetch.assert_called_once()

    def test_no_internet_no_cache_raises(self, tmp_path, mock_hapi):
        """No internet and no cache → HITRANConnectionError."""
        with patch.object(hm, "_get_hapi", return_value=mock_hapi), \
             patch.object(hm, "is_cached", return_value=False), \
             patch.object(hm, "_can_reach_hitran", return_value=False):

            with pytest.raises(hm.HITRANConnectionError):
                hm._fetch_single_iso("H2O", 1, 1, 700.0, 1400.0, tmp_path, False)

    def test_no_internet_force_download_falls_back_to_cache(self, tmp_path, mock_hapi):
        """force_download=True + no internet + existing cache → uses cache."""
        with patch.object(hm, "_get_hapi", return_value=mock_hapi), \
             patch.object(hm, "is_cached", return_value=True), \
             patch.object(hm, "_can_reach_hitran", return_value=False), \
             patch.object(hm, "_extract_line_data", return_value=_make_lines(5, 1)):

            result = hm._fetch_single_iso("H2O", 1, 1, 700.0, 1400.0, tmp_path, True)

            mock_hapi.fetch.assert_not_called()
            assert len(result["nu0"]) == 5

    def test_download_writes_metadata(self, tmp_path, mock_hapi):
        """A successful download also calls _write_metadata."""
        with patch.object(hm, "_get_hapi", return_value=mock_hapi), \
             patch.object(hm, "is_cached", return_value=False), \
             patch.object(hm, "_can_reach_hitran", return_value=True), \
             patch.object(hm, "_write_metadata") as mock_meta, \
             patch.object(hm, "_extract_line_data", return_value=_make_lines(3, 1)):

            hm._fetch_single_iso("H2O", 1, 1, 700.0, 1400.0, tmp_path, False)

            mock_meta.assert_called_once()


# ── get_line_parameters ───────────────────────────────────────────────────────

class TestGetLineParameters:

    def test_unknown_molecule_raises_value_error(self, tmp_path, mock_hapi):
        with patch.object(hm, "_get_hapi", return_value=mock_hapi):
            with pytest.raises(ValueError, match="Unknown molecule"):
                hm.get_line_parameters("FAKE", 700, 1400, cache_dir=tmp_path)

    def test_invalid_iso_id_string_raises(self, tmp_path, mock_hapi):
        with patch.object(hm, "_get_hapi", return_value=mock_hapi):
            with pytest.raises(ValueError, match="'all'"):
                hm.get_line_parameters("H2O", 700, 1400, iso_id="none", cache_dir=tmp_path)

    def test_empty_iso_list_raises(self, tmp_path, mock_hapi):
        with patch.object(hm, "_get_hapi", return_value=mock_hapi), \
             patch.object(hm, "_fetch_single_iso"):
            with pytest.raises(hm.HITRANDataError, match="No isotopologues"):
                hm.get_line_parameters("H2O", 700, 1400, iso_id=[], cache_dir=tmp_path)

    # ── iso_id = int ──────────────────────────────────────────────────────────

    def test_int_iso_id_calls_fetch_once(self, tmp_path, mock_hapi):
        """iso_id=2 → _fetch_single_iso called once with iso_id=2."""
        expected = _make_lines(10, 2)
        with patch.object(hm, "_get_hapi", return_value=mock_hapi), \
             patch.object(hm, "_fetch_single_iso", return_value=expected) as mock_fetch:

            result = hm.get_line_parameters("H2O", 700, 1400, iso_id=2, cache_dir=tmp_path)

            mock_fetch.assert_called_once()
            assert mock_fetch.call_args[0][2] == 2  # iso_id argument
            np.testing.assert_array_equal(result["iso_id"], expected["iso_id"])

    def test_default_iso_id_is_1(self, tmp_path, mock_hapi):
        """Default iso_id=1 when not specified."""
        with patch.object(hm, "_get_hapi", return_value=mock_hapi), \
             patch.object(hm, "_fetch_single_iso", return_value=_make_lines(5, 1)) as mock_fetch:

            hm.get_line_parameters("H2O", 700, 1400, cache_dir=tmp_path)

            assert mock_fetch.call_args[0][2] == 1

    # ── iso_id = list ─────────────────────────────────────────────────────────

    def test_list_iso_id_concatenates_arrays(self, tmp_path, mock_hapi):
        """iso_id=[1, 2] → arrays concatenated, total length = sum of both."""
        lines_1 = _make_lines(10, 1)
        lines_2 = _make_lines(8, 2)

        def _side(mol, mol_id, iso_id, *a, **kw):
            return lines_1 if iso_id == 1 else lines_2

        with patch.object(hm, "_get_hapi", return_value=mock_hapi), \
             patch.object(hm, "_fetch_single_iso", side_effect=_side):

            result = hm.get_line_parameters("H2O", 700, 1400, iso_id=[1, 2], cache_dir=tmp_path)

            assert len(result["nu0"]) == 18
            assert set(result["iso_id"].tolist()) == {1, 2}

    def test_all_result_keys_have_same_length(self, tmp_path, mock_hapi):
        """All arrays in the result dict must be the same length."""
        lines_1 = _make_lines(7, 1)
        lines_2 = _make_lines(13, 2)

        def _side(mol, mol_id, iso_id, *a, **kw):
            return lines_1 if iso_id == 1 else lines_2

        with patch.object(hm, "_get_hapi", return_value=mock_hapi), \
             patch.object(hm, "_fetch_single_iso", side_effect=_side):

            result = hm.get_line_parameters("H2O", 700, 1400, iso_id=[1, 2], cache_dir=tmp_path)

            lengths = {k: len(v) for k, v in result.items()}
            assert len(set(lengths.values())) == 1, f"Inconsistent lengths: {lengths}"
            assert list(lengths.values())[0] == 20

    def test_single_item_list_returns_without_concatenation(self, tmp_path, mock_hapi):
        """iso_id=[3] (single-element list) returns the same dict as iso_id=3."""
        expected = _make_lines(5, 3)

        with patch.object(hm, "_get_hapi", return_value=mock_hapi), \
             patch.object(hm, "_fetch_single_iso", return_value=expected):

            result = hm.get_line_parameters("H2O", 700, 1400, iso_id=[3], cache_dir=tmp_path)

            np.testing.assert_array_equal(result["iso_id"], expected["iso_id"])

    # ── iso_id = "all" ────────────────────────────────────────────────────────

    def test_all_discovers_isotopologues_from_hapi_iso(self, tmp_path, mock_hapi):
        """iso_id='all' uses mock_hapi.ISO to discover iso_ids [1,2,3,4]."""
        per_iso = {i: _make_lines(5, i) for i in [1, 2, 3, 4]}

        def _side(mol, mol_id, iso_id, *a, **kw):
            return per_iso[iso_id]

        with patch.object(hm, "_get_hapi", return_value=mock_hapi), \
             patch.object(hm, "_fetch_single_iso", side_effect=_side):

            result = hm.get_line_parameters("H2O", 700, 1400, iso_id="all", cache_dir=tmp_path)

            assert len(result["nu0"]) == 20
            assert sorted(np.unique(result["iso_id"]).tolist()) == [1, 2, 3, 4]

    def test_all_case_insensitive(self, tmp_path, mock_hapi):
        """iso_id='ALL' and iso_id='All' are accepted."""
        per_iso = {i: _make_lines(3, i) for i in [1, 2, 3, 4]}

        def _side(mol, mol_id, iso_id, *a, **kw):
            return per_iso[iso_id]

        with patch.object(hm, "_get_hapi", return_value=mock_hapi), \
             patch.object(hm, "_fetch_single_iso", side_effect=_side):

            result = hm.get_line_parameters("H2O", 700, 1400, iso_id="ALL", cache_dir=tmp_path)
            assert len(result["nu0"]) == 12

    def test_all_processes_in_ascending_order(self, tmp_path, mock_hapi):
        """iso_id='all' processes isotopologues sorted in ascending order."""
        call_order = []

        def _side(mol, mol_id, iso_id, *a, **kw):
            call_order.append(iso_id)
            return _make_lines(2, iso_id)

        with patch.object(hm, "_get_hapi", return_value=mock_hapi), \
             patch.object(hm, "_fetch_single_iso", side_effect=_side):

            hm.get_line_parameters("H2O", 700, 1400, iso_id="all", cache_dir=tmp_path)

            assert call_order == sorted(call_order)

    def test_all_no_isotopologues_raises(self, tmp_path, mock_hapi):
        """If HAPI.ISO has no entries for the molecule, raise HITRANDataError."""
        mock_hapi.ISO = {}  # vacío — ningún isotopólogo conocido

        with patch.object(hm, "_get_hapi", return_value=mock_hapi):
            with pytest.raises(hm.HITRANDataError, match="No isotopologues"):
                hm.get_line_parameters("H2O", 700, 1400, iso_id="all", cache_dir=tmp_path)


# ── "Failed to retrieve data" → error específico ──────────────────────────────

class TestNoDataErrors:
    """HITRAN answers 404 both for an empty range and for an unserved isotopologue."""

    def _fetch_failing(self, tmp_path, mock_hapi, serves):
        mock_hapi.ISO = {(1, 1): [1, "H2(16O)", 0.997, 18.01, "H2O"]}
        mock_hapi.fetch.side_effect = Exception("Failed to retrieve data for given parameters.")
        with patch.object(hm, "_get_hapi", return_value=mock_hapi), \
             patch.object(hm, "is_cached", return_value=False), \
             patch.object(hm, "_can_reach_hitran", return_value=True), \
             patch.object(hm, "_hitran_serves", return_value=serves):
            hm._fetch_single_iso("H2O", 1, 1, 714.0, 1250.0, tmp_path, False)

    def test_served_elsewhere_is_no_lines(self, tmp_path, mock_hapi):
        with pytest.raises(hm.HITRANNoLinesError, match="no lines"):
            self._fetch_failing(tmp_path, mock_hapi, serves=True)

    def test_not_served_at_all(self, tmp_path, mock_hapi):
        with pytest.raises(hm.HITRANNotAvailableError, match="does not serve"):
            self._fetch_failing(tmp_path, mock_hapi, serves=False)

    def test_unknown_stays_generic_data_error(self, tmp_path, mock_hapi):
        with pytest.raises(hm.HITRANDataError) as info:
            self._fetch_failing(tmp_path, mock_hapi, serves=None)
        assert not isinstance(info.value, (hm.HITRANNoLinesError, hm.HITRANNotAvailableError))

    def test_new_errors_are_data_errors(self):
        """Code that catches HITRANDataError keeps working."""
        assert issubclass(hm.HITRANNoLinesError, hm.HITRANDataError)
        assert issubclass(hm.HITRANNotAvailableError, hm.HITRANDataError)


# ── TIPS: rango de la tabla y versión de respaldo ─────────────────────────────

@pytest.fixture
def tips_hapi():
    """HAPI mock: (1,1) has TIPS-2021 up to 1000 K; (1,2) only has TIPS-2025."""
    m = MagicMock()
    m.TIPS_2021_ISOT_HASH = {(1, 1): np.arange(1.0, 1001.0)}
    m.TIPS_2025_ISOT_HASH = {(1, 1): np.arange(1.0, 1001.0), (1, 2): np.arange(1.0, 3001.0)}
    m.TIPS_2017_ISOT_HASH = {}
    m.partitionSum.side_effect = lambda M, I, T, version=2021: float(T)
    return m


class TestTipsTable:

    def test_clipped_to_table_max(self, tips_hapi):
        with patch.object(hm, "_get_hapi", return_value=tips_hapi):
            T, Q = hm.get_tips_table("H2O", 1, T_min=150, T_max=3000)
        assert T[0] == 150.0 and T[-1] == 1000.0
        assert np.array_equal(T, Q)

    def test_inside_table_unchanged(self, tips_hapi):
        with patch.object(hm, "_get_hapi", return_value=tips_hapi):
            T, _ = hm.get_tips_table("H2O", 1, T_min=150, T_max=400)
        assert T[0] == 150.0 and T[-1] == 400.0 and len(T) == 251

    def test_fractional_t_min_never_passes_table(self, tips_hapi):
        with patch.object(hm, "_get_hapi", return_value=tips_hapi):
            T, _ = hm.get_tips_table("H2O", 1, T_min=150.5, T_max=3000)
        assert T[-1] <= 1000.0

    def test_no_overlap_raises(self, tips_hapi):
        with patch.object(hm, "_get_hapi", return_value=tips_hapi):
            with pytest.raises(hm.HITRANDataError, match="does not overlap"):
                hm.get_tips_table("H2O", 1, T_min=1500, T_max=3000)

    def test_falls_back_to_other_version(self, tips_hapi):
        with patch.object(hm, "_get_hapi", return_value=tips_hapi):
            T, _ = hm.get_tips_table("H2O", 2, T_min=150, T_max=3000, version=2021)
        assert T[-1] == 3000.0
        assert tips_hapi.partitionSum.call_args.kwargs["version"] == 2025

    def test_no_version_has_table_raises(self, tips_hapi):
        with patch.object(hm, "_get_hapi", return_value=tips_hapi):
            with pytest.raises(hm.HITRANDataError, match="No TIPS table"):
                hm.get_tips_table("H2O", 3)

    def test_non_positive_q_raises(self, tips_hapi):
        tips_hapi.partitionSum.side_effect = lambda M, I, T, version=2021: 0.0
        with patch.object(hm, "_get_hapi", return_value=tips_hapi):
            with pytest.raises(hm.HITRANDataError, match="non-positive"):
                hm.get_tips_table("H2O", 1, T_min=150, T_max=400)

    def test_get_tips_falls_back_too(self, tips_hapi):
        with patch.object(hm, "_get_hapi", return_value=tips_hapi):
            assert hm.get_tips("H2O", 2, 296.0, version=2021) == 296.0
        assert tips_hapi.partitionSum.call_args.kwargs["version"] == 2025


# ── TIPS con el HAPI real (offline: las tablas vienen dentro de HAPI) ─────────

def _real_hapi_available() -> bool:
    try:
        hm._get_hapi()
        return True
    except ImportError:
        return False


@pytest.mark.skipif(not _real_hapi_available(), reason="hitran-api not installed")
class TestTipsRealHapi:
    """The isotopologues that broke from_hapi in the 8–14 µm download test."""

    @pytest.mark.parametrize("mol,iso,version,t_max", [
        ("O3", 1, 2025, 1000.0), ("CH4", 1, 2025, 2500.0), ("O2", 3, 2025, 2010.0),
        ("NO2", 1, 2021, 1000.0),
    ])
    def test_default_from_hapi_range_works(self, mol, iso, version, t_max):
        T, Q = hm.get_tips_table(mol, iso, T_min=150.0, T_max=3000.0, version=version)
        assert T[0] == 150.0 and T[-1] == t_max
        assert np.all(np.isfinite(Q)) and np.all(Q > 0)

    @pytest.mark.parametrize("mol,iso", [("CH3F", 2), ("NO2", 3), ("HCOOH", 2)])
    def test_missing_tips2021_uses_another_version(self, mol, iso):
        T, Q = hm.get_tips_table(mol, iso, T_min=150.0, T_max=400.0, version=2021)
        assert np.all(np.isfinite(Q)) and np.all(Q > 0)

    def test_default_version_is_2025(self):
        """HITRANonline serves HITRAN2024, which goes with TIPS-2025."""
        import inspect
        from pylblrtm.data_loader import from_hapi
        assert inspect.signature(hm.get_tips_table).parameters["version"].default == 2025
        assert inspect.signature(hm.get_tips).parameters["version"].default == 2025
        assert inspect.signature(from_hapi).parameters["tips_version"].default == 2025


@pytest.mark.skipif(not _real_hapi_available(), reason="hitran-api not installed")
def test_molecule_ids_match_hapi_catalogue():
    """MOLECULE_IDS must give the same molecule HAPI downloads for that ID
    (51–56 were shifted by one: 'SO' downloaded CH3F)."""
    hapi_names = {m: rec[4] for (m, _), rec in hm._get_hapi().ISO.items()}
    ours = {mid: name.replace("+", "p") for name, mid in hm.MOLECULE_IDS.items()}
    assert ours == hapi_names


# ── Varios isotopólogos: se saltan los que no tienen líneas ──────────────────

class TestSkipIsotopologuesWithoutLines:

    def _run(self, tmp_path, mock_hapi, outcomes, iso_id="all"):
        """outcomes: iso → 'ok' | 'none' | 'unserved'."""
        def _side(mol, mol_id, iso, *a, **kw):
            if outcomes[iso] == "none":
                raise hm.HITRANNoLinesError(f"no lines iso {iso}")
            if outcomes[iso] == "unserved":
                raise hm.HITRANNotAvailableError(f"not served iso {iso}")
            return _make_lines(3, iso)
        with patch.object(hm, "_get_hapi", return_value=mock_hapi),              patch.object(hm, "_fetch_single_iso", side_effect=_side):
            return hm.get_line_parameters("H2O", 714, 1250, iso_id=iso_id, cache_dir=tmp_path)

    def test_skips_isos_without_lines(self, tmp_path, mock_hapi):
        out = self._run(tmp_path, mock_hapi, {1: "ok", 2: "none", 3: "ok", 4: "unserved"})
        assert sorted(set(out["iso_id"].tolist())) == [1, 3]
        assert len(out["nu0"]) == 6

    def test_single_survivor_returned(self, tmp_path, mock_hapi):
        out = self._run(tmp_path, mock_hapi, {1: "ok", 2: "none", 3: "none", 4: "none"})
        assert set(out["iso_id"].tolist()) == {1}

    def test_none_with_lines_raises_no_lines(self, tmp_path, mock_hapi):
        with pytest.raises(hm.HITRANNoLinesError, match="None of the isotopologues"):
            self._run(tmp_path, mock_hapi, {1: "none", 2: "none", 3: "unserved", 4: "none"})

    def test_all_unserved_raises_not_available(self, tmp_path, mock_hapi):
        with pytest.raises(hm.HITRANNotAvailableError):
            self._run(tmp_path, mock_hapi, {i: "unserved" for i in (1, 2, 3, 4)})

    def test_single_iso_still_raises(self, tmp_path, mock_hapi):
        with pytest.raises(hm.HITRANNoLinesError):
            self._run(tmp_path, mock_hapi, {2: "none"}, iso_id=2)

    def test_other_errors_are_not_skipped(self, tmp_path, mock_hapi):
        def _side(mol, mol_id, iso, *a, **kw):
            raise hm.HITRANConnectionError("down")
        with patch.object(hm, "_get_hapi", return_value=mock_hapi),              patch.object(hm, "_fetch_single_iso", side_effect=_side):
            with pytest.raises(hm.HITRANConnectionError):
                hm.get_line_parameters("H2O", 714, 1250, iso_id="all", cache_dir=tmp_path)


# ── g_air / g_self faltantes (0 en HITRAN) ────────────────────────────────────

class TestFillMissingBroadening:

    def _lines(self, g_air, g_self, iso=None):
        n = len(g_air)
        d = _make_lines(n, 1)
        d["g_air"] = np.asarray(g_air, dtype=float)
        d["g_self"] = np.asarray(g_self, dtype=float)
        if iso is not None:
            d["iso_id"] = np.asarray(iso, dtype=np.int32)
        return d

    def test_zero_g_air_gets_median_of_isotopologue(self):
        out = hm._fill_missing_broadening(self._lines([0.07, 0.0, 0.08, 0.06], [0.3] * 4))
        assert out["g_air"][1] == pytest.approx(0.07)
        assert np.all(out["g_air"] > 0)

    def test_nan_and_negative_count_as_missing(self):
        out = hm._fill_missing_broadening(self._lines([0.07, np.nan, -1.0], [0.3] * 3))
        assert np.all(out["g_air"] == 0.07)

    def test_zero_g_self_gets_median(self):
        out = hm._fill_missing_broadening(self._lines([0.07] * 3, [0.3, 0.0, 0.5]))
        assert out["g_self"][1] == pytest.approx(0.4)

    def test_no_g_air_at_all_uses_g_self(self):
        out = hm._fill_missing_broadening(self._lines([0.0, 0.0], [0.3, 0.2]))
        assert np.array_equal(out["g_air"], [0.3, 0.2])

    def test_median_is_per_isotopologue(self):
        out = hm._fill_missing_broadening(
            self._lines([0.05, 0.0, 0.09, 0.0], [0.3] * 4, iso=[1, 1, 2, 2]))
        assert out["g_air"][1] == pytest.approx(0.05)
        assert out["g_air"][3] == pytest.approx(0.09)

    def test_input_not_modified_and_clean_data_unchanged(self):
        d = self._lines([0.07, 0.0], [0.3, 0.3])
        hm._fill_missing_broadening(d)
        assert d["g_air"][1] == 0.0
        clean = self._lines([0.07, 0.08], [0.3, 0.3])
        assert np.array_equal(hm._fill_missing_broadening(clean)["g_air"], clean["g_air"])

    def test_warns(self, caplog):
        with caplog.at_level("WARNING", logger="pylblrtm.hitran_manager"):
            hm._fill_missing_broadening(self._lines([0.07, 0.0], [0.3, 0.3]))
        assert "1/2 lines without g_air" in caplog.text


# ── Excepciones con molécula e isotopólogo ────────────────────────────────────

class TestIsotopologueErrorAttributes:

    def test_not_available_names_isotopologue_and_cross_sections(self, tmp_path, mock_hapi):
        mock_hapi.ISO = {(1, 1): [1, "H2(16O)", 0.997, 18.01, "H2O"]}
        mock_hapi.fetch.side_effect = Exception("Failed to retrieve data for given parameters.")
        with patch.object(hm, "_get_hapi", return_value=mock_hapi), \
             patch.object(hm, "is_cached", return_value=False), \
             patch.object(hm, "_can_reach_hitran", return_value=True), \
             patch.object(hm, "_hitran_serves", return_value=False):
            with pytest.raises(hm.HITRANNotAvailableError) as info:
                hm._fetch_single_iso("H2O", 1, 1, 714.0, 1250.0, tmp_path, False)
        assert info.value.molecule == "H2O" and info.value.iso_id == 1
        assert "hitran.org/xsc" in str(info.value)

    def test_handled_in_code(self, tmp_path, mock_hapi):
        """The typical pattern: skip molecules HITRANonline does not serve."""
        def _side(*a, **kw):
            raise hm.HITRANNotAvailableError("not served", molecule="H2O", iso_id=1)
        with patch.object(hm, "_get_hapi", return_value=mock_hapi), \
             patch.object(hm, "_fetch_single_iso", side_effect=_side):
            try:
                hm.get_line_parameters("H2O", 714, 1250, cache_dir=tmp_path)
                handled = False
            except hm.HITRANNotAvailableError as err:
                handled = err.molecule == "H2O"
        assert handled


@pytest.mark.skipif(not _real_hapi_available(), reason="hitran-api not installed")
class TestAbundance:
    """Same values as the HITRAN isotopologue page for H2O."""

    @pytest.mark.parametrize("iso,abundance", [
        (1, 9.97317e-1), (2, 1.99983e-3), (3, 3.71884e-4), (4, 3.10693e-4),
        (5, 6.23003e-7), (6, 1.15853e-7), (7, 2.41974e-8),
    ])
    def test_h2o(self, iso, abundance):
        assert hm.get_abundance("H2O", iso) == pytest.approx(abundance, rel=1e-5)

    def test_unknown_isotopologue_raises(self):
        with pytest.raises(hm.HITRANDataError):
            hm.get_abundance("H2O", 99)


# ── Caché con otro isotopólogo (p.ej. 'SO' bajado cuando SO tenía el ID 51) ───

class TestStaleCache:

    def _hapi(self, mol_ids, iso_ids):
        m = MagicMock()
        n = len(mol_ids)
        m.getColumns.return_value = [np.linspace(900, 901, n)] + [np.ones(n) * 0.07] * 6 \
            + [np.asarray(mol_ids), np.asarray(iso_ids)]
        return m

    def test_matching_table_is_used(self, tmp_path):
        h = self._hapi([1, 1], [1, 1])
        with patch.object(hm, "_get_hapi", return_value=h), \
             patch.object(hm, "is_cached", return_value=True):
            out = hm._fetch_single_iso("H2O", 1, 1, 900, 901, tmp_path, False)
        assert len(out["nu0"]) == 2
        h.fetch.assert_not_called()

    def test_table_with_other_molecule_is_downloaded_again(self, tmp_path):
        h = self._hapi([51, 51], [1, 1])            # holds CH3F, named as SO
        calls = []

        def _fetch(*a, **kw):
            calls.append(a)
            h.getColumns.return_value = h.getColumns.return_value[:7] + \
                [np.array([50, 50]), np.array([1, 1])]
        h.fetch.side_effect = _fetch
        with patch.object(hm, "_get_hapi", return_value=h), \
             patch.object(hm, "is_cached", return_value=True), \
             patch.object(hm, "_can_reach_hitran", return_value=True), \
             patch.object(hm, "_write_metadata"):
            out = hm._fetch_single_iso("SO", 50, 1, 900, 901, tmp_path, False)
        assert len(calls) == 1
        assert set(out["mol_id"].tolist()) == {50}

    def test_stale_table_offline_is_not_used(self, tmp_path):
        h = self._hapi([51], [1])
        with patch.object(hm, "_get_hapi", return_value=h), \
             patch.object(hm, "is_cached", return_value=True), \
             patch.object(hm, "_can_reach_hitran", return_value=False):
            with pytest.raises(hm.HITRANConnectionError):
                hm._fetch_single_iso("SO", 50, 1, 900, 901, tmp_path, False)
