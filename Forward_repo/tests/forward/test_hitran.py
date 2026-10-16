"""Malformed-record isolation, spectroscopy IDs, partition bounds, and cache integrity."""

import hashlib
import io
import os

import numpy as np
import pytest

from photonix.spectroscopy import hitran
from photonix.spectroscopy.hitran import (
    LineList,
    fetch_hitran_par,
    load_partition_function,
    read_hitran_par,
)
from photonix.spectroscopy.species import Isotopologue, default_species


def record(isotope="1", molecule=1, wavenumber=1000.0, shift=0.0):
    core = (
        f"{molecule:2d}{isotope}{wavenumber:12.6f}{1e-20:10.3E}{1.0:10.3E}"
        f"{0.07:5.3f}{0.10:5.3f}{20.0:10.4f}{0.7:4.2f}{shift:8.5f}"
    )
    assert len(core) == 67
    return core + " " * 93


def test_atomic_bad_late_field_does_not_misalign_any_column(tmp_path):
    good = record(wavenumber=1000)
    broken = record(wavenumber=1001)
    broken = broken[:59] + " BAD    " + broken[67:]
    source = tmp_path / "lines.par"
    source.write_text(good + "\n" + broken + "\n" + record(wavenumber=1002) + "\n")
    with pytest.raises(ValueError, match=r"lines.par:2:"):
        read_hitran_par(source)
    with pytest.warns(RuntimeWarning, match="Skipping HITRAN row"):
        lines = read_hitran_par(source, strict=False)
    assert len(lines) == 2
    np.testing.assert_array_equal(lines.wavenumber_cm1, [1000, 1002])
    assert all(len(getattr(lines, name)) == 2 for name in hitran._ARRAY_NAMES)
    assert lines.diagnostics[0].line_number == 2
    assert not lines.wavenumber_cm1.flags.writeable


@pytest.mark.parametrize(
    "bad",
    [
        record()[:-1],
        record() + " ",
        "",
        record(isotope="C"),
        record(wavenumber=-1),
        record(molecule=0),
    ],
)
def test_strict_record_validation(tmp_path, bad):
    source = tmp_path / "bad.par"
    source.write_text(bad + "\n")
    with pytest.raises(ValueError, match=r"bad.par:1:"):
        read_hitran_par(source)


def test_ascii_isotope_ids_fortran_numbers_and_signed_shifts(tmp_path):
    source = tmp_path / "co2.par"
    rows = [record(iso, 2, 1000 + i, shift=-0.003) for i, iso in enumerate("90AB")]
    rows[0] = rows[0][:15] + " 1.000D-20" + rows[0][25:]
    rows[1] = rows[1][:15] + " 1.000-020" + rows[1][25:]
    source.write_bytes(("\r\n".join(rows) + "\r\n").encode("ascii"))
    lines = read_hitran_par(source)
    assert list(lines.isotope_id) == ["9", "0", "A", "B"]
    np.testing.assert_allclose(lines.line_strength_ref, 1e-20)
    np.testing.assert_allclose(lines.pressure_shift_cm1_atm, -0.003)


def test_aer100_is_explicit_and_does_not_relax_hitran160(tmp_path):
    source = tmp_path / "aer.par"
    source.write_text(record(isotope="A", molecule=2)[:100] + "\n")
    with pytest.raises(ValueError, match="expected 160 characters"):
        read_hitran_par(source)
    parsed = read_hitran_par(source, record_format="aer100")
    assert len(parsed) == 1
    assert parsed.isotope_id[0] == "A"
    assert parsed.wavenumber_cm1[0] == 1000
    source.write_text(record() + "\n")
    with pytest.raises(ValueError, match="expected 100 characters"):
        read_hitran_par(source, record_format="aer100")


def test_aer100_rejects_short_extended_and_mixed_records(tmp_path):
    source = tmp_path / "mixed.par"
    source.write_text(record()[:100] + "\n" + record() + "\n" + record()[:99] + "\n")
    with pytest.warns(RuntimeWarning):
        result = read_hitran_par(source, strict=False, record_format="aer100")
    assert len(result) == 1
    assert [issue.line_number for issue in result.diagnostics] == [2, 3]
    with pytest.raises(ValueError, match="record_format"):
        read_hitran_par(source, record_format="auto")


@pytest.mark.parametrize("replacement", ["       NaN", "       inf", "-1.000E-20"])
def test_nonfinite_and_negative_strength_rejected(tmp_path, replacement):
    row = record()
    source = tmp_path / "invalid.par"
    source.write_text(row[:15] + replacement + row[25:] + "\n")
    with pytest.raises(ValueError):
        read_hitran_par(source)


def test_non_ascii_and_all_rejected_rows_are_reported(tmp_path):
    source = tmp_path / "invalid.par"
    row = record().encode("ascii")
    source.write_bytes(row[:100] + b"\xff" + row[101:] + b"\n")
    with pytest.warns(RuntimeWarning):
        result = read_hitran_par(source, strict=False)
    assert len(result) == 0
    assert len(result.diagnostics) == 1


def test_direct_linelist_alignment_and_ids(tmp_path):
    source = tmp_path / "one.par"
    source.write_text(record() + "\n")
    parsed = read_hitran_par(source)
    columns = {name: getattr(parsed, name) for name in hitran._ARRAY_NAMES}
    columns["molecule_id"] = [1.5]
    with pytest.raises(ValueError, match="molecule_id"):
        LineList(**columns)
    columns["molecule_id"] = [1]
    columns["temperature_exponent"] = [0.7, 0.8]
    with pytest.raises(ValueError, match="aligned"):
        LineList(**columns)


def test_partition_cache_is_table_not_temperature_and_invalidates_replacements(
    tmp_path, monkeypatch
):
    source = tmp_path / "q.txt"
    source.write_text("100 10\n296 30\n400 40\n")
    iso = Isotopologue("1", 18, source)
    hitran._read_partition_table.cache_clear()
    original = np.loadtxt
    reads = []

    def tracked(*args, **kwargs):
        reads.append(args[0])
        return original(*args, **kwargs)

    monkeypatch.setattr(np, "loadtxt", tracked)
    assert load_partition_function("H2O", iso, 296) == (30, 30)
    qref, vector = load_partition_function("H2O", iso, np.array([100, 348, 400]))
    assert qref == 30
    np.testing.assert_allclose(vector, [10, 35, 40])
    assert len(reads) == 1
    stamp = source.stat().st_mtime_ns
    source.write_text("100 20\n296 60\n400 80\n")
    os.utime(source, ns=(stamp + 1_000_000, stamp + 1_000_000))
    assert load_partition_function("H2O", iso, 296) == (60, 60)
    assert len(reads) == 2


@pytest.mark.parametrize("temperature", [99, 401, float("nan"), float("inf")])
def test_partition_rejects_extrapolation(tmp_path, temperature):
    source = tmp_path / "q.txt"
    source.write_text("100 10\n296 30\n400 40\n")
    with pytest.raises(ValueError, match="outside partition table"):
        load_partition_function("H2O", Isotopologue("1", 18, source), temperature)


@pytest.mark.parametrize(
    "content",
    ["100 1\n100 2\n400 3\n", "100 1\n400 -1\n", "100 1 7\n400 2 7\n", "100 1\n", "100 1\n200 2\n"],
)
def test_partition_invalid_tables_and_reference_outside_table_fail(tmp_path, content):
    source = tmp_path / "q.txt"
    source.write_text(content)
    with pytest.raises(ValueError):
        load_partition_function("H2O", Isotopologue("1", 18, source), 150)


def test_missing_partition_never_defaults_to_one(tmp_path):
    with pytest.raises(FileNotFoundError):
        load_partition_function("H2O", Isotopologue("1", 18, tmp_path / "missing"), 296)


def test_all_bundled_tables_cover_atmospheric_range():
    for gas in default_species():
        for iso in gas.isotopologues:
            qref, values = load_partition_function(gas.name, iso, np.array([180, 296, 330]))
            assert qref > 0
            assert np.all(np.isfinite(values)) and np.all(values > 0)
            assert values[1] == qref
    # partition_file=None also resolves the bundled mapping lazily.
    assert load_partition_function("CO2", Isotopologue("A", 48.001646), 296)[0] > 0


class FakeResponse(io.BytesIO):
    def geturl(self):
        return "https://example.test/spectroscopy.par"


def test_lazy_download_verifies_digest_and_reuses_cache(tmp_path, monkeypatch):
    payload = (record() + "\n").encode()
    digest = hashlib.sha256(payload).hexdigest()
    calls = []

    def request(url, timeout):
        calls.append(url)
        return FakeResponse(payload)

    monkeypatch.setattr(hitran, "urlopen", request)
    path = fetch_hitran_par("https://example.test/a.par", sha256=digest, cache_dir=tmp_path)
    assert path.read_bytes() == payload
    assert fetch_hitran_par("https://example.test/a.par", sha256=digest, cache_dir=tmp_path) == path
    assert len(calls) == 1
    path.write_bytes(b"corrupt")
    fetch_hitran_par("https://example.test/a.par", sha256=digest, cache_dir=tmp_path)
    assert len(calls) == 2
    assert path.read_bytes() == payload


def test_download_failure_does_not_publish_or_leave_partial_file(tmp_path, monkeypatch):
    monkeypatch.setattr(hitran, "urlopen", lambda *args, **kwargs: FakeResponse(b"wrong"))
    with pytest.raises(ValueError, match="SHA-256"):
        fetch_hitran_par("https://example.test/a.par", sha256="0" * 64, cache_dir=tmp_path)
    assert list(tmp_path.iterdir()) == []


@pytest.mark.parametrize(
    "url", ["http://example.test/a", "file:///tmp/a", "https://user:secret@example.test/a"]
)
def test_download_requires_explicit_https_source_without_embedded_credentials(tmp_path, url):
    with pytest.raises(ValueError, match="HTTPS"):
        fetch_hitran_par(url, sha256="0" * 64, cache_dir=tmp_path)
