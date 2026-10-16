"""Load the frozen IASI ocean scenes in the final Ncol/IASI-angle layout.

The scene loader follows ``exp22_recon.carga_mar``. Geography is a separate
artifact because the day of year is absent from each scene archive.
"""

from __future__ import annotations

import datetime as dt
from dataclasses import dataclass
from pathlib import Path

import numpy as np

from .forward_adapter import NU_CHANNEL
from .state import GASES, N_LAYERS, N_LEVELS, build_state_matrix_full


# The tabulated complex refractive index used by exp0_sigma_forward.py.
_WAVELENGTH_UM = np.array([8.0, 8.5, 9.0, 9.5, 10.0, 10.5, 11.0, 11.5, 12.0, 12.5, 13.0])
_REFRACTIVE_REAL = np.array(
    [1.291, 1.302, 1.262, 1.232, 1.218, 1.185, 1.153, 1.126, 1.111, 1.123, 1.146]
)
_REFRACTIVE_IMAG = np.array(
    [0.0339, 0.0461, 0.0392, 0.0427, 0.0508, 0.0662, 0.0968, 0.146, 0.199, 0.259, 0.305]
)


@dataclass(frozen=True)
class Mar64Scenes:
    """Row-aligned observations, truth, geometry and forward tables."""

    names: tuple[str, ...]
    truth: np.ndarray
    observed: np.ndarray
    pressure_layers: np.ndarray
    emissivity: np.ndarray
    mu: np.ndarray
    pressure_surface: np.ndarray
    geography: np.ndarray
    cloud: np.ndarray


def sea_fresnel_emissivity(mu: float, frequencies: np.ndarray = NU_CHANNEL) -> np.ndarray:
    """Return the legacy Fresnel sea emissivity at one scan cosine."""
    if not np.isfinite(mu) or not 0.0 < mu <= 1.0:
        raise ValueError("Scan cosine must be finite and in (0, 1]")
    wavelengths = 1.0e4 / np.asarray(frequencies, dtype=np.float64)
    real = np.interp(wavelengths, _WAVELENGTH_UM, _REFRACTIVE_REAL)
    imag = np.interp(wavelengths, _WAVELENGTH_UM, _REFRACTIVE_IMAG)
    index = real + 1j * imag
    transmitted = np.sqrt(1.0 - (np.sqrt(1.0 - mu**2) / index) ** 2)
    s_reflection = (mu - index * transmitted) / (mu + index * transmitted)
    p_reflection = (index * mu - transmitted) / (index * mu + transmitted)
    return 1.0 - 0.5 * (np.abs(s_reflection) ** 2 + np.abs(p_reflection) ** 2)


def _scene_value(
    archive: np.lib.npyio.NpzFile, key: str, shape: tuple[int, ...], path: Path
) -> np.ndarray:
    if key not in archive:
        raise ValueError(f"{path.name} lacks {key}")
    value = np.asarray(archive[key])
    if value.shape != shape or not np.issubdtype(value.dtype, np.number):
        raise ValueError(f"{path.name}:{key} must be numeric with shape {shape}")
    if not np.isfinite(value).all():
        raise ValueError(f"{path.name}:{key} contains nonfinite values")
    return value.astype(np.float64)


def load_mar64(
    scene_dir: str | Path,
    geo_path: str | Path,
    *,
    expected_count: int = 64,
    max_cloud: float = 99.0,
) -> Mar64Scenes:
    """Read verified scene files and validate their alignment with geography.

    Byte hashes are checked by the pipeline before this function is called.
    Observed radiance uses the original FOV and the historical factor of 1000.
    """
    paths = sorted(Path(scene_dir).glob("*.npz"))
    if len(paths) != expected_count:
        raise ValueError(f"Expected {expected_count} sea archives, found {len(paths)}")
    names = tuple(path.stem for path in paths)
    with np.load(geo_path, allow_pickle=False) as archive:
        required = ("lat", "lon", "doy", "cloud", "mu", "nombres")
        if not set(required).issubset(archive.files):
            raise ValueError("Sea geography lacks a required array")
        geo = {key: np.asarray(archive[key]) for key in required}
    if any(geo[key].shape != (expected_count,) for key in required):
        raise ValueError("Sea geography arrays must have one entry per scene")
    if geo["nombres"].dtype.kind != "U" or tuple(map(str, geo["nombres"])) != names:
        raise ValueError("Sea geography names are not aligned with scene filenames")
    for key in ("lat", "lon", "doy", "cloud", "mu"):
        if not np.issubdtype(geo[key].dtype, np.number) or not np.isfinite(geo[key]).all():
            raise ValueError(f"Sea geography {key} must be finite and numeric")
    if np.any((geo["doy"] < 1) | (geo["doy"] > 366)) or not np.all(
        geo["doy"] == np.floor(geo["doy"])
    ):
        raise ValueError("Sea geography day of year is invalid")

    temperatures, columns, skin = [], [], []
    observed, layers, cosine, pressure_surface, clouds = [], [], [], [], []
    latitudes, longitudes = [], []
    for path in paths:
        with np.load(path, allow_pickle=False) as archive:
            temperatures.append(_scene_value(archive, "T_levels", (N_LEVELS,), path))
            columns.append(_scene_value(archive, "Ncol", (len(GASES), N_LAYERS), path))
            skin.append(float(_scene_value(archive, "T_skin", (), path)))
            layers.append(_scene_value(archive, "P_lay", (N_LAYERS,), path))
            pressure_surface.append(float(_scene_value(archive, "psurf_hPa", (), path)))
            cosine.append(float(_scene_value(archive, "mu", (), path)))
            clouds.append(float(_scene_value(archive, "cloud_frac", (), path)))
            latitudes.append(float(_scene_value(archive, "lat_fov", (), path)))
            longitudes.append(float(_scene_value(archive, "lon_fov", (), path)))
            frequencies = _scene_value(archive, "nu", (8461,), path)
            radiance = _scene_value(archive, "L_obs", (8461,), path)
            if not np.all(np.diff(frequencies) > 0):
                raise ValueError(f"{path.name}:nu must increase strictly")
            if frequencies[0] > NU_CHANNEL[0] or frequencies[-1] < NU_CHANNEL[-1]:
                raise ValueError(f"{path.name}:nu does not cover the final band")
            observed.append(np.interp(NU_CHANNEL, frequencies, radiance * 1000.0))

    mu = np.asarray(cosine)
    cloud = np.asarray(clouds)
    for key, expected in (("lat", latitudes), ("lon", longitudes), ("cloud", cloud), ("mu", mu)):
        if not np.allclose(geo[key], expected, rtol=1e-8, atol=1e-8):
            raise ValueError(f"Sea geography {key} differs from scene archives")
    if np.any(cloud > max_cloud):
        raise ValueError("Cloud selection would exclude a frozen sea scene")
    if np.any((mu <= 0) | (mu > 1)):
        raise ValueError("Sea scan cosines must be in (0, 1]")
    geography = np.column_stack((geo["lat"], geo["lon"], geo["doy"], pressure_surface))
    emissivity = np.stack([sea_fresnel_emissivity(value) for value in mu])
    truth = build_state_matrix_full(
        {
            "T_levels": np.stack(temperatures),
            "Ncol": np.stack(columns),
            "T_skin": np.asarray(skin),
        }
    )
    return Mar64Scenes(
        names=names,
        truth=truth,
        observed=np.stack(observed),
        pressure_layers=np.asarray(layers, dtype=np.float32),
        emissivity=emissivity,
        mu=mu,
        pressure_surface=np.asarray(pressure_surface),
        geography=geography,
        cloud=cloud,
    )


def derive_geo_from_catalogs(scene_dir: str | Path, catalog_dir: str | Path) -> dict:
    """Recreate val64 geography deterministically from verified NWP-SAF catalogs.

    The caller verifies catalog and scene hashes before using this function.
    The returned numeric arrays can be saved as a small NPZ; semantic identity
    should be checked from canonical records rather than ZIP bytes.
    """
    try:
        import netCDF4
    except ImportError as exc:
        raise RuntimeError("netCDF4 is required to derive sea geography") from exc

    paths = sorted(Path(scene_dir).glob("*.npz"))
    if len(paths) != 64:
        raise ValueError("Expected exactly 64 scene files for geography derivation")
    result: dict[str, list] = {key: [] for key in ("lat", "lon", "doy", "cloud", "mu", "nombres")}
    catalogs = {}
    try:
        for path in paths:
            subset, index_text = path.stem.rsplit("_", 1)
            index = int(index_text)
            if index < 0:
                raise ValueError(f"Negative catalog index in {path.name}")
            if subset not in catalogs:
                catalogs[subset] = netCDF4.Dataset(
                    Path(catalog_dir) / f"cams_5000_{subset}_2025.nc"
                )
            catalog = catalogs[subset]
            day = dt.date(
                int(catalog["year"][index]),
                int(catalog["month"][index]),
                int(catalog["day"][index]),
            )
            with np.load(path, allow_pickle=False) as archive:
                for key, source in (
                    ("lat", "lat_fov"),
                    ("lon", "lon_fov"),
                    ("cloud", "cloud_frac"),
                    ("mu", "mu"),
                ):
                    result[key].append(float(_scene_value(archive, source, (), path)))
            result["doy"].append(float(day.timetuple().tm_yday))
            result["nombres"].append(path.stem)
    finally:
        for catalog in catalogs.values():
            catalog.close()
    return {
        key: np.asarray(values, dtype="U" if key == "nombres" else np.float64)
        for key, values in result.items()
    }
