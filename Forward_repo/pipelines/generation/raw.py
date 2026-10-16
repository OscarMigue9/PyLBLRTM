"""Rebuild frozen atmospheric states and emissivity from catalogued raw files.

The transformations follow the archived extraction scripts. Byte and numerical
identity must still be checked against the frozen outputs before using a rebuild.
"""

from __future__ import annotations

import json
import re
from pathlib import Path

import numpy as np

from .l137 import full_level_pressures
from .lut import sha256_file


SUBSETS = ("temp", "hum", "co2", "oz", "n2o", "ch4")
GASES = ("H2O", "CO2", "O3", "N2O", "CH4")
NC_VARIABLE = {"H2O": "hum", "CO2": "co2", "O3": "oz", "N2O": "n2o", "CH4": "ch4"}
MOLAR_MASS = {"H2O": 18.01528, "CO2": 44.0095, "O3": 47.9982, "N2O": 44.0128, "CH4": 16.0425}
AVOGADRO = 6.02214086e23
GRAVITY = 9.80665
AIR_MOLAR_MASS = 28.9644
AIR_MOLAR_MASS_KG = 0.0289644
_SPECTRUM_NUMBERS = re.compile(
    r"^\s*([-+]?(?:\d+(?:\.\d*)?|\.\d+)(?:[eE][-+]?\d+)?)\s+"
    r"([-+]?(?:\d+(?:\.\d*)?|\.\d+)(?:[eE][-+]?\d+)?)\s*$",
    re.M,
)


def _catalog(path: Path) -> dict:
    catalog = json.loads(Path(path).read_text(encoding="utf-8"))
    if catalog.get("schema_version") != 1 or catalog.get("hash_algorithm") != "sha256":
        raise ValueError("Unsupported raw catalog schema")
    return catalog


def _directory(root: Path, relative: str) -> Path:
    candidate = Path(relative)
    if candidate.is_absolute() or ".." in candidate.parts or ":" in relative or "\\" in relative:
        raise ValueError(f"Unsafe catalog directory: {relative}")
    return Path(root).joinpath(*candidate.parts)


def _verified(path: Path, record: dict) -> Path:
    name = record["path"]
    if Path(name).name != name or "\\" in name:
        raise ValueError(f"Unsafe catalog filename: {name}")
    source = path / name
    if not source.is_file():
        raise FileNotFoundError(source)
    if source.stat().st_size != record["bytes"] or sha256_file(source) != record["sha256"]:
        raise ValueError(f"Raw source differs from the catalog: {source}")
    return source


def _new_output(path: Path) -> Path:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.exists():
        raise FileExistsError(path)
    return path


def _extract_subset(source: Path, count: int) -> tuple[np.ndarray, ...]:
    """Apply the exact May 2026 vectorized NWP-SAF state transformation."""
    try:
        import netCDF4 as nc
    except ImportError as exc:
        raise RuntimeError("netCDF4 is required to rebuild atmospheric states") from exc
    with nc.Dataset(source) as dataset:
        if len(dataset.dimensions["rgrid"]) < count:
            raise ValueError(f"{source}: fewer than {count} profiles")
        temperature = np.asarray(dataset.variables["temp"][:count, :], np.float64)
        surface_pressure = np.asarray(dataset.variables["psurf"][:count], np.float64)
        mass_ratio = {
            gas: np.asarray(dataset.variables[NC_VARIABLE[gas]][:count, :], np.float64)
            for gas in GASES
        }
        skin = np.asarray(dataset.variables["tskin"][:count], np.float64)
        latitude = np.asarray(dataset.variables["lat"][:count], np.float64)
        longitude = np.asarray(dataset.variables["lon"][:count], np.float64)
        longitude = np.where(longitude > 180, longitude - 360, longitude)
        land_mask = np.asarray(dataset.variables["lsm"][:count], np.float64)
    if temperature.shape != (count, 137):
        raise ValueError(f"{source}: expected 137 temperature levels")
    pressure_hpa = full_level_pressures(surface_pressure) / 100.0
    volume_ratio = {}
    for gas in GASES:
        ratio = mass_ratio[gas]
        if gas == "H2O":
            ratio = ratio / np.maximum(1.0 - ratio, 1e-12)
        volume_ratio[gas] = ratio * (AIR_MOLAR_MASS / MOLAR_MASS[gas])
    layer_temperature = 0.5 * (temperature[:, :-1] + temperature[:, 1:])
    layer_pressure = np.sqrt(pressure_hpa[:, :-1] * pressure_hpa[:, 1:]) / 1013.25
    pressure_difference = (pressure_hpa[:, 1:] - pressure_hpa[:, :-1]) * 100.0
    coefficient = AVOGADRO / (GRAVITY * AIR_MOLAR_MASS_KG)
    columns = np.zeros((count, len(GASES), 136))
    for index, gas in enumerate(GASES):
        effective_ratio = 0.5 * (volume_ratio[gas][:, :-1] + volume_ratio[gas][:, 1:])
        columns[:, index] = coefficient * effective_ratio * pressure_difference / 1e4
    water_ratio = 0.5 * (volume_ratio["H2O"][:, :-1] + volume_ratio["H2O"][:, 1:])
    return (
        layer_temperature,
        layer_pressure,
        columns,
        water_ratio,
        temperature,
        skin,
        latitude,
        longitude,
        land_mask,
        surface_pressure / 100.0,
    )


def compare_states(candidate: Path, archived: Path, per_subset: int) -> dict:
    """Compare each sampled row with the same row in the frozen 30k archive."""
    largest = {}
    exactly_equal = True
    with (
        np.load(candidate, allow_pickle=False) as built,
        np.load(archived, allow_pickle=False) as frozen,
    ):
        if set(built.files) != set(frozen.files):
            raise ValueError("State archive keys differ")
        for key in built.files:
            a = built[key]
            b = frozen[key]
            if key == "gases":
                selected = b
            else:
                selected = np.concatenate(
                    [b[i * 5000 : i * 5000 + per_subset] for i in range(len(SUBSETS))], axis=0
                )
            if a.shape != selected.shape:
                raise ValueError(f"State array {key} has an unexpected shape")
            equal = np.array_equal(a, selected)
            exactly_equal &= equal
            if equal:
                largest[key] = 0.0
            elif key == "gases":
                largest[key] = None
            else:
                largest[key] = float(np.max(np.abs(a.astype(float) - selected.astype(float))))
    return {"arrays_exact": exactly_equal, "max_abs_by_key": largest}


def build_states(
    workspace_root: Path,
    catalog_path: Path,
    output: Path,
    *,
    per_subset: int = 5000,
    archived: Path | None = None,
) -> dict:
    """Rebuild the ordered 6 × 5000 five-gas states from catalogued CAMS files."""
    if not 1 <= per_subset <= 5000:
        raise ValueError("per_subset must be between 1 and 5000")
    catalog = _catalog(catalog_path)["cams_l137"]
    directory = _directory(workspace_root, catalog["workspace_directory"])
    records = {item["path"]: item for item in catalog["files"]}
    count = len(SUBSETS) * per_subset
    shapes = {
        "T_lay": (count, 136),
        "P_lay": (count, 136),
        "Ncol": (count, 5, 136),
        "VMR_H2O": (count, 136),
        "T_levels": (count, 137),
        "T_skin": (count,),
        "lat": (count,),
        "lon": (count,),
        "lsm": (count,),
        "P_surf_hPa": (count,),
    }
    arrays = {key: np.zeros(shape, np.float32) for key, shape in shapes.items()}
    arrays["subset_id"] = np.zeros(count, np.int16)
    arrays["profile_index"] = np.zeros(count, np.int32)
    source_hashes = {}
    fields = tuple(shapes)
    for subset_id, name in enumerate(SUBSETS):
        filename = f"cams_5000_{name}_2025.nc"
        source = _verified(directory, records[filename])
        source_hashes[filename] = records[filename]["sha256"]
        rows = slice(subset_id * per_subset, (subset_id + 1) * per_subset)
        for key, value in zip(fields, _extract_subset(source, per_subset)):
            arrays[key][rows] = value
        arrays["subset_id"][rows] = subset_id
        arrays["profile_index"][rows] = np.arange(per_subset)
    output = _new_output(output)
    temporary = output.with_name(output.name + ".partial")
    try:
        with temporary.open("wb") as stream:
            np.savez(stream, gases=np.array(GASES), **arrays)
        temporary.replace(output)
    finally:
        temporary.unlink(missing_ok=True)
    report = {
        "status": "candidate_historical_parity_unverified",
        "output": str(output),
        "sha256": sha256_file(output),
        "profiles": count,
        "per_subset": per_subset,
        "source_sha256": source_hashes,
    }
    if archived is not None:
        report["canary"] = compare_states(output, archived, per_subset)
        report["canary"]["archived_sha256"] = sha256_file(archived)
        if per_subset == 5000:
            report["canary"]["byte_identical"] = (
                report["sha256"] == report["canary"]["archived_sha256"]
            )
            if report["canary"]["byte_identical"]:
                report["status"] = "archived_byte_identity_verified"
    return report


def _read_spectrum(path: Path) -> tuple[np.ndarray, np.ndarray]:
    pairs = _SPECTRUM_NUMBERS.findall(path.read_text(encoding="utf-8", errors="ignore"))
    if not pairs:
        raise ValueError(f"No numerical spectrum rows in {path}")
    data = np.array(pairs, dtype=np.float64)
    order = np.argsort(data[:, 0])
    wavelength, reflectance = data[order, 0], data[order, 1]
    if np.nanmax(reflectance) > 1.5:
        reflectance = reflectance / 100.0
    return wavelength, np.clip(reflectance, 0.0, 1.0)


def build_emissivity(
    workspace_root: Path,
    catalog_path: Path,
    record_ids_path: Path,
    output: Path,
    *,
    count: int | None = None,
    archived: Path | None = None,
) -> dict:
    """Interpolate the ordered ECOSTRESS records to the historical 8461 channels."""
    catalog = _catalog(catalog_path)["ecostress_selected"]
    directory = _directory(workspace_root, catalog["workspace_directory"])
    if sha256_file(record_ids_path) != catalog["ordered_record_ids_sha256"]:
        raise ValueError("record_ids.npy differs from the raw catalog")
    record_ids = np.load(record_ids_path, allow_pickle=False)
    if record_ids.shape != (1419,):
        raise ValueError("Expected 1419 ordered ECOSTRESS record IDs")
    n = len(record_ids) if count is None else count
    if not 1 <= n <= len(record_ids):
        raise ValueError("count is outside the selected ECOSTRESS catalog")
    records = {item["path"]: item for item in catalog["files"]}
    expected = {f"{record_id}.spectrum.txt" for record_id in record_ids}
    if set(records) != expected:
        raise ValueError("Raw catalog does not match the ordered record IDs")
    channels = np.arange(645.0, 2760.001, 0.25, dtype=np.float64)
    wavelength = 1e4 / channels
    emissivity = np.zeros((n, channels.size), np.float32)
    held_edges = np.zeros(n, np.int32)
    for index, record_id in enumerate(record_ids[:n]):
        filename = f"{record_id}.spectrum.txt"
        source = _verified(directory, records[filename])
        native_wavelength, reflectance = _read_spectrum(source)
        emissivity[index] = (1.0 - np.interp(wavelength, native_wavelength, reflectance)).astype(
            np.float32
        )
        held_edges[index] = int(
            ((wavelength < native_wavelength.min()) | (wavelength > native_wavelength.max())).sum()
        )
    if not np.isfinite(emissivity).all() or (emissivity < 0).any() or (emissivity > 1).any():
        raise ValueError("Emissivity is outside the physical [0, 1] domain")
    output = _new_output(output)
    temporary = output.with_name(output.name + ".partial")
    try:
        with temporary.open("wb") as stream:
            np.save(stream, emissivity)
        temporary.replace(output)
    finally:
        temporary.unlink(missing_ok=True)
    report = {
        "status": "candidate_historical_parity_unverified",
        "output": str(output),
        "sha256": sha256_file(output),
        "records": n,
        "channels": channels.size,
        "record_ids_sha256": catalog["ordered_record_ids_sha256"],
        "catalogued_spectra": len(records),
        "held_edge_channels_median": int(np.median(held_edges)),
    }
    if archived is not None:
        reference = np.load(archived, mmap_mode="r")
        if reference.shape != (1419, channels.size):
            raise ValueError("Archived emissivity has an unexpected shape")
        comparison = reference[:n]
        report["canary"] = {
            "arrays_exact": bool(np.array_equal(emissivity, comparison)),
            "max_abs": float(np.max(np.abs(emissivity.astype(float) - comparison.astype(float)))),
            "archived_sha256": sha256_file(archived),
        }
        if n == len(record_ids):
            report["canary"]["byte_identical"] = (
                report["sha256"] == report["canary"]["archived_sha256"]
            )
            if report["canary"]["byte_identical"]:
                report["status"] = "archived_byte_identity_verified"
    return report
