"""Atomic HITRAN parsing, cached partition tables, and explicit lazy file acquisition."""

from __future__ import annotations

from dataclasses import dataclass, fields
from functools import lru_cache
import hashlib
import os
from pathlib import Path
import re
import tempfile
from urllib.parse import urlsplit
from urllib.request import urlopen
import warnings

import numpy as np

from .species import Isotopologue, default_species, normalize_isotope_id


@dataclass(frozen=True, slots=True)
class ParseDiagnostic:
    """One rejected physical row, numbered from one in the source file."""

    line_number: int
    reason: str


@dataclass(frozen=True, slots=True)
class LineList:
    """Aligned, read-only HITRAN arrays; strengths retain terrestrial abundance."""

    molecule_id: np.ndarray
    isotope_id: np.ndarray
    wavenumber_cm1: np.ndarray
    line_strength_ref: np.ndarray
    einstein_a_s: np.ndarray
    air_width_cm1_atm: np.ndarray
    self_width_cm1_atm: np.ndarray
    lower_energy_cm1: np.ndarray
    temperature_exponent: np.ndarray
    pressure_shift_cm1_atm: np.ndarray
    diagnostics: tuple[ParseDiagnostic, ...] = ()

    def __post_init__(self) -> None:
        size = None
        for field in fields(self):
            name = field.name
            if name == "diagnostics":
                continue
            raw = np.asarray(getattr(self, name))
            if raw.ndim != 1:
                raise ValueError(f"{name} must be one-dimensional")
            if size is None:
                size = raw.size
            elif raw.size != size:
                raise ValueError(f"{name} is not aligned with the other line arrays")
            if name == "isotope_id":
                array = np.array([normalize_isotope_id(item) for item in raw], dtype="U1")
            elif name == "molecule_id":
                if (
                    raw.dtype.kind == "b"
                    or not np.all(np.isfinite(raw.astype(float)))
                    or np.any(raw.astype(float) != raw.astype(np.int32))
                    or np.any(raw.astype(float) < 1)
                    or np.any(raw.astype(float) > 99)
                ):
                    raise ValueError("molecule_id must contain integers from 1 to 99")
                array = np.array(raw, dtype=np.int32, copy=True)
            else:
                array = np.array(raw, dtype=np.float64, copy=True)
                if not np.all(np.isfinite(array)):
                    raise ValueError(f"{name} contains nonfinite values")
                if name == "wavenumber_cm1" and np.any(array <= 0):
                    raise ValueError("wavenumber_cm1 must be positive")
                if name in {
                    "line_strength_ref",
                    "einstein_a_s",
                    "air_width_cm1_atm",
                    "self_width_cm1_atm",
                    "lower_energy_cm1",
                } and np.any(array < 0):
                    raise ValueError(
                        f"{name} must be nonnegative; unknown-energy flags are unsupported"
                    )
            array.flags.writeable = False
            object.__setattr__(self, name, array)
        object.__setattr__(self, "diagnostics", tuple(self.diagnostics))

    def __len__(self) -> int:
        return self.wavenumber_cm1.size

    @property
    def isotope_pairs(self) -> frozenset[tuple[int, str]]:
        """Molecule/isotope pairs present anywhere in this input line list.

        Presence does not certify spectral completeness. In particular, a list
        prefiltered to a window cannot establish coverage of its parent database.
        """
        return frozenset(zip(self.molecule_id.tolist(), self.isotope_id.tolist()))


_ARRAY_NAMES = tuple(field.name for field in fields(LineList) if field.name != "diagnostics")
_FIELD_BOUNDS = (
    (0, 2),
    (2, 3),
    (3, 15),
    (15, 25),
    (25, 35),
    (35, 40),
    (40, 45),
    (45, 55),
    (55, 59),
    (59, 67),
)
_OMITTED_EXPONENT = re.compile(r"^([+-]?(?:\d+(?:\.\d*)?|\.\d+))([+-]\d+)$")


def _parse_float(value: str) -> float:
    value = value.strip().replace("D", "E").replace("d", "e")
    if "e" not in value.lower():
        value = _OMITTED_EXPONENT.sub(r"\1E\2", value)
    parsed = float(value)
    if not np.isfinite(parsed):
        raise ValueError("nonfinite numeric field")
    return parsed


def read_hitran_par(
    path: str | Path,
    *,
    strict: bool = True,
    record_format: str = "hitran160",
) -> LineList:
    """Read explicitly selected HITRAN160 or AER100 records atomically.

    Strict mode fails at the first invalid row. Permissive mode emits a warning
    per skipped row and returns its line number/reason in LineList.diagnostics.
    record_format='hitran160' requires exactly 160 ASCII characters per row;
    'aer100' requires exactly 100. Their first 67 columns share the physical
    field layout. There is no format autodetection, truncation or padding.
    Blank rows are invalid. Quantum labels remain in the source file only.
    """
    lengths = {"hitran160": 160, "aer100": 100}
    if record_format not in lengths:
        raise ValueError("record_format must be 'hitran160' or 'aer100'")
    expected_length = lengths[record_format]
    path = Path(path)
    columns = [[] for _ in _ARRAY_NAMES]
    diagnostics = []
    with path.open("rb") as stream:
        for number, record in enumerate(stream, 1):
            try:
                row = record.rstrip(b"\r\n").decode("ascii")
                if len(row) != expected_length:
                    raise ValueError(
                        f"expected {expected_length} characters for {record_format}, "
                        f"found {len(row)}"
                    )
                parts = [row[left:right] for left, right in _FIELD_BOUNDS]
                molecule = int(parts[0])
                isotope = normalize_isotope_id(parts[1])
                values = (molecule, isotope, *(_parse_float(value) for value in parts[2:]))
                if not 1 <= molecule <= 99:
                    raise ValueError("molecule ID must be from 1 to 99")
                if values[2] <= 0:
                    raise ValueError("wavenumber must be positive")
                if any(value < 0 for value in values[3:8]):
                    raise ValueError(
                        "strength, A, widths and lower-state energy must be nonnegative"
                    )
            except (ValueError, UnicodeDecodeError) as error:
                message = f"{path}:{number}: {error}"
                if strict:
                    raise ValueError(message) from error
                diagnostics.append(ParseDiagnostic(number, str(error)))
                warnings.warn(f"Skipping HITRAN row {message}", RuntimeWarning, stacklevel=2)
                continue
            for column, value in zip(columns, values):
                column.append(value)
    arrays = dict(zip(_ARRAY_NAMES, columns))
    return LineList(**arrays, diagnostics=tuple(diagnostics))


@lru_cache(maxsize=128)
def _read_partition_table(
    resolved_path: str,
    modified_ns: int,
    size_bytes: int,
    changed_ns: int,
) -> tuple[np.ndarray, np.ndarray]:
    # Stat values are part of the cache key: replacing a table invalidates it.
    path = Path(resolved_path)
    try:
        table = np.loadtxt(path, comments="#", ndmin=2)
    except ValueError:
        table = np.loadtxt(path, comments="#", delimiter=",", ndmin=2)
    if table.ndim != 2 or table.shape[1] != 2 or table.shape[0] < 2:
        raise ValueError(f"Partition table must have at least two T,Q rows: {path}")
    if not np.all(np.isfinite(table)) or np.any(table <= 0):
        raise ValueError(f"Partition temperatures and values must be finite and positive: {path}")
    temperatures, partition = table.T
    if np.any(np.diff(temperatures) <= 0):
        raise ValueError(f"Partition temperatures must be strictly increasing: {path}")
    temperatures.flags.writeable = False
    partition.flags.writeable = False
    return temperatures, partition


def load_partition_function(
    species_name: str,
    isotopologue: Isotopologue,
    temperature_k: float | np.ndarray,
) -> tuple[float, float | np.ndarray]:
    """Interpolate Q(296 K), Q(T) within a cached table, without extrapolation.

    An explicit isotopologue.partition_file overrides bundled data. Missing,
    malformed or out-of-range tables raise; there is no Q=1 fallback.
    """
    path = isotopologue.partition_file
    if path is None:
        catalogue_species = default_species([species_name])[0]
        candidates = [
            item
            for item in catalogue_species.isotopologues
            if item.isotope_id == isotopologue.isotope_id
        ]
        if not candidates:
            raise ValueError(
                f"No bundled partition table for {species_name}/{isotopologue.isotope_id}"
            )
        path = candidates[0].partition_file
    path = Path(path).resolve(strict=True)
    stat = path.stat()
    temperatures, partition = _read_partition_table(
        str(path), stat.st_mtime_ns, stat.st_size, stat.st_ctime_ns
    )
    requested = np.asarray(temperature_k, dtype=np.float64)
    if (
        not np.all(np.isfinite(requested))
        or np.any(requested < temperatures[0])
        or np.any(requested > temperatures[-1])
        or not temperatures[0] <= 296 <= temperatures[-1]
    ):
        raise ValueError(
            f"Temperature outside partition table [{temperatures[0]}, "
            f"{temperatures[-1]}] K for {species_name}/{isotopologue.isotope_id}"
        )
    q_ref = float(np.interp(296.0, temperatures, partition))
    q_temperature = np.interp(requested, temperatures, partition)
    return q_ref, float(q_temperature) if requested.ndim == 0 else q_temperature


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def fetch_hitran_par(
    url: str,
    *,
    sha256: str,
    cache_dir: str | Path,
    timeout_s: float = 60.0,
) -> Path:
    """Fetch an explicitly supplied HTTPS source into a content-addressed cache.

    There is no default URL, login, implicit fetch or import-time networking.
    The caller supplies the expected SHA-256. A verified cached file is reused;
    a new download replaces its cache entry only after checksum verification.
    """
    parsed = urlsplit(url)
    if parsed.scheme != "https" or not parsed.hostname or parsed.username or parsed.password:
        raise ValueError("An explicit HTTPS URL without embedded credentials is required")
    if not re.fullmatch(r"[0-9a-fA-F]{64}", sha256):
        raise ValueError("sha256 must contain exactly 64 hexadecimal characters")
    if not np.isfinite(timeout_s) or timeout_s <= 0:
        raise ValueError("timeout_s must be finite and positive")
    expected = sha256.lower()
    directory = Path(cache_dir)
    directory.mkdir(parents=True, exist_ok=True)
    destination = directory / f"{expected}.par"
    if destination.is_file() and _sha256(destination) == expected:
        return destination
    descriptor, temporary_name = tempfile.mkstemp(prefix="hitran-", suffix=".part", dir=directory)
    temporary = Path(temporary_name)
    try:
        with os.fdopen(descriptor, "wb") as output, urlopen(url, timeout=timeout_s) as response:
            if urlsplit(response.geturl()).scheme != "https":
                raise ValueError("The source redirected outside HTTPS")
            for chunk in iter(lambda: response.read(1024 * 1024), b""):
                output.write(chunk)
        if _sha256(temporary) != expected:
            raise ValueError("Downloaded HITRAN file does not match the expected SHA-256")
        os.replace(temporary, destination)
    finally:
        temporary.unlink(missing_ok=True)
    return destination
