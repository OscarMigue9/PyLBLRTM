"""Read explicit, versioned metadata and interpolate SVD cross-section blocks.

The storage identity is ``log10(sigma) = mu + U @ V``. Rows of U are flattened
in temperature-major, pressure-minor order; pressure interpolation uses log10(P).
Only the requested layer coefficients and spectral columns are reconstructed.

Historical NPZ files do not identify their physical recipe or isotope mixture.
They therefore require an explicit manifest; filenames never establish provenance.
The schema (version 1) is shown in ``CrossSectionLUT``'s docstring.
"""

from __future__ import annotations

from collections import OrderedDict
from collections.abc import Mapping
from copy import deepcopy
from dataclasses import dataclass
import hashlib
import json
from pathlib import Path
import re
from types import MappingProxyType
import zipfile

import numpy as np


_FORMAT = "svd_log10_cross_section"
_EPS = np.finfo(np.float64).eps


def _positive_integer(value, name, minimum=1):
    if isinstance(value, bool) or not isinstance(value, (int, np.integer)) or value < minimum:
        raise ValueError(f"{name} must be an integer >= {minimum}")
    return int(value)


def _vector(value, name, *, positive=False):
    array = np.atleast_1d(np.asarray(value, dtype=np.float64))
    if array.ndim != 1 or not np.all(np.isfinite(array)):
        raise ValueError(f"{name} must be a finite one-dimensional array")
    if positive and np.any(array <= 0):
        raise ValueError(f"{name} must be positive")
    return array


def _close_coordinate(left, right):
    return abs(left - right) <= 64 * _EPS * max(1.0, abs(left), abs(right))


def _scalar(archive, name, path):
    if name not in archive or archive[name].shape != ():
        raise ValueError(f"{path.name}: missing or non-scalar metadata {name!r}")
    return archive[name].item()


def _array_header(archive, name):
    with archive.open(f"{name}.npy") as stream:
        version = np.lib.format.read_magic(stream)
        if version == (1, 0):
            shape, _, dtype = np.lib.format.read_array_header_1_0(stream)
        elif version == (2, 0):
            shape, _, dtype = np.lib.format.read_array_header_2_0(stream)
        else:
            raise ValueError(f"unsupported NPY header version {version} for {name}")
    if dtype.kind != "f":
        raise ValueError(f"{name} must contain real floating-point factors")
    return shape


@dataclass(frozen=True)
class _Block:
    gas: str
    path: Path
    start_cm1: float
    end_cm1: float
    spectral_size: int
    rank: int
    start_index: int
    end_index: int
    size_bytes: int
    mtime_ns: int
    sha256: str | None


class CrossSectionLUT:
    """Read log10 cross sections in cm²/molecule from compressed SVD blocks.

    ``manifest`` is a mapping or JSON path; omitted means ``directory/lut_manifest.json``.
    Required manifest fields::

        {
          "schema_version": 1,
          "storage_format": "svd_log10_cross_section",
          "recipe": "caller-declared recipe identifier",
          "provenance": {"status": "unverified", "description": "origin of these files"},
          "includes_h2o_continuum": false,
          "includes_co2_continuum": true,
          "gases": {
            "H2O": {"molecule_id": 1, "isotopologues": ["1", "2"], "vmr_ref": 0.01}
          },
          "blocks": [{"gas": "H2O", "file": "lut_H2O_705_715.npz"}],
          "self_broadening": {"method": "effective_pressure", "ratios_file": "r_self.npz"}
        }

    ``self_broadening.method`` may explicitly be ``"none"``. With
    ``"effective_pressure"``, callers must supply ``vmr`` to ``sigma``; the ratio
    table has ``bordes`` (block starts) and one nonnegative ratio array per gas.
    Optional per-block ``sha256`` is checked when the block is first loaded.
    Provenance is a caller declaration, never a validation result inferred here.

    ``bounds='raise'`` rejects out-of-domain temperatures/effective pressures;
    ``'clip'`` explicitly holds the nearest boundary. Spectral extrapolation and
    spectral interpolation are never allowed. Adjacent blocks share one stored
    endpoint; the block on the right owns that endpoint (the last endpoint is kept).
    """

    def __init__(self, directory, *, manifest=None, bounds="raise", cache_blocks=8):
        self._directory = Path(directory).resolve()
        if not self._directory.is_dir():
            raise FileNotFoundError(self._directory)
        if bounds not in {"raise", "clip"}:
            raise ValueError("bounds must be 'raise' or 'clip'")
        self._bounds = bounds
        self._cache_limit = _positive_integer(cache_blocks, "cache_blocks", minimum=0)
        self._cache = OrderedDict()
        if manifest is None:
            manifest = self._directory / "lut_manifest.json"
        if isinstance(manifest, Mapping):
            metadata = deepcopy(dict(manifest))
        else:
            manifest_path = Path(manifest)
            if not manifest_path.is_file():
                raise FileNotFoundError(
                    f"{manifest_path}: an explicit LUT manifest is required; "
                    "historical blocks do not establish recipe or isotope provenance"
                )
            metadata = json.loads(manifest_path.read_text(encoding="utf-8"))
        self._validate_manifest(metadata)
        self._recipe = metadata["recipe"]
        self._includes_h2o = metadata["includes_h2o_continuum"]
        self._includes_co2 = metadata["includes_co2_continuum"]
        self._provenance = MappingProxyType(deepcopy(metadata["provenance"]))
        self._gas_metadata = deepcopy(metadata["gases"])
        self._gases = tuple(self._gas_metadata)
        self._broadening_method = metadata["self_broadening"]["method"]
        self._read_blocks(metadata["blocks"])
        self._ratios = self._read_ratios(metadata["self_broadening"])

    @property
    def gases(self):
        return self._gases

    @property
    def recipe(self):
        return self._recipe

    @property
    def includes_h2o_continuum(self):
        return self._includes_h2o

    @property
    def includes_co2_continuum(self):
        return self._includes_co2

    @property
    def provenance(self):
        """A copy of caller-declared provenance, not a certification by this reader."""
        return deepcopy(dict(self._provenance))

    def _validate_manifest(self, metadata):
        if not isinstance(metadata, dict):
            raise ValueError("LUT manifest must be a JSON object")
        if metadata.get("schema_version") != 1 or metadata.get("storage_format") != _FORMAT:
            raise ValueError(f"manifest requires schema_version=1 and storage_format={_FORMAT!r}")
        if not isinstance(metadata.get("recipe"), str) or not metadata["recipe"].strip():
            raise ValueError("manifest must declare a nonempty recipe identifier")
        provenance = metadata.get("provenance")
        if (
            not isinstance(provenance, dict)
            or provenance.get("status") not in {"unverified", "verified"}
            or not isinstance(provenance.get("description"), str)
            or not provenance["description"].strip()
        ):
            raise ValueError("manifest provenance requires status and a nonempty description")
        for flag in ("includes_h2o_continuum", "includes_co2_continuum"):
            if type(metadata.get(flag)) is not bool:
                raise ValueError(f"manifest must explicitly declare boolean {flag}")
        gases = metadata.get("gases")
        if not isinstance(gases, dict) or not gases:
            raise ValueError("manifest must declare gas and isotope membership")
        for gas, info in gases.items():
            if not isinstance(gas, str) or not gas or not isinstance(info, dict):
                raise ValueError("manifest gases must map gas names to metadata objects")
            _positive_integer(info.get("molecule_id"), f"{gas}.molecule_id")
            isotopes = info.get("isotopologues")
            if (
                not isinstance(isotopes, list)
                or not isotopes
                or any(not isinstance(i, str) or not i for i in isotopes)
                or len(set(isotopes)) != len(isotopes)
            ):
                raise ValueError(f"{gas}: declare distinct string isotopologue IDs")
            reference = info.get("vmr_ref")
            if (
                isinstance(reference, bool)
                or not isinstance(reference, (float, int))
                or not np.isfinite(reference)
                or not 0 <= reference < 1
            ):
                raise ValueError(f"{gas}: vmr_ref must be finite and in [0, 1)")
        if not isinstance(metadata.get("blocks"), list) or not metadata["blocks"]:
            raise ValueError("manifest must explicitly list its blocks")
        broadening = metadata.get("self_broadening")
        if not isinstance(broadening, dict) or broadening.get("method") not in {
            "none",
            "effective_pressure",
        }:
            raise ValueError("declare self_broadening.method as 'none' or 'effective_pressure'")

    def _data_path(self, name):
        if not isinstance(name, str) or not name:
            raise ValueError("manifest data paths must be nonempty strings")
        path = (self._directory / name).resolve()
        if not path.is_relative_to(self._directory):
            raise ValueError(f"manifest data file is outside LUT directory: {name}")
        if not path.is_file():
            raise FileNotFoundError(path)
        return path

    def _read_blocks(self, entries):
        raw_blocks = {gas: [] for gas in self._gases}
        axes = None
        paths = set()
        for entry in entries:
            if not isinstance(entry, dict) or entry.get("gas") not in raw_blocks:
                raise ValueError("every listed block must name a declared gas")
            gas = entry["gas"]
            path = self._data_path(entry.get("file"))
            if path in paths:
                raise ValueError(f"duplicate block file {path.name}")
            paths.add(path)
            digest = entry.get("sha256")
            if digest is not None and (
                not isinstance(digest, str) or re.fullmatch(r"[0-9a-fA-F]{64}", digest) is None
            ):
                raise ValueError(f"{path.name}: invalid sha256")
            with np.load(path, allow_pickle=False) as archive:
                if _scalar(archive, "gas", path) != gas:
                    raise ValueError(f"{path.name}: block gas disagrees with manifest")
                nt = _positive_integer(_scalar(archive, "NT", path), "NT", 2)
                npres = _positive_integer(_scalar(archive, "NP", path), "NP", 2)
                rank = _positive_integer(_scalar(archive, "r", path), "r")
                values = tuple(
                    float(_scalar(archive, k, path))
                    for k in ("T_LO", "T_HI", "P_LO", "P_HI", "dnu", "a", "b")
                )
            tlo, thi, plo, phi, step, start, end = values
            if (
                not all(np.isfinite(values))
                or not 0 < tlo < thi
                or not 0 < plo < phi
                or step <= 0
                or not 0 < start < end
            ):
                raise ValueError(f"{path.name}: invalid finite grid bounds")
            current_axes = (nt, npres, tlo, thi, plo, phi, step)
            if axes is None:
                axes = current_axes
            elif current_axes != axes:
                raise ValueError(f"{path.name}: all gases and blocks must share T/P/dnu axes")
            intervals = int(round((end - start) / step))
            if intervals < 1 or not _close_coordinate(start + intervals * step, end):
                raise ValueError(f"{path.name}: block endpoints do not align with dnu")
            spectral_size = intervals + 1
            if rank > min(nt * npres, spectral_size):
                raise ValueError(f"{path.name}: impossible SVD rank")
            with zipfile.ZipFile(path) as archive:
                for name, expected in (
                    ("U", (nt * npres, rank)),
                    ("V", (rank, spectral_size)),
                    ("mu", (spectral_size,)),
                ):
                    if _array_header(archive, name) != expected:
                        raise ValueError(f"{path.name}: {name} shape must be {expected}")
            raw_blocks[gas].append((start, end, path, spectral_size, rank, digest))
        ranges = None
        self._blocks = {}
        for gas, items in raw_blocks.items():
            if not items:
                raise ValueError(f"manifest has no blocks for declared gas {gas}")
            items.sort(key=lambda row: row[0])
            current_ranges = tuple((item[0], item[1]) for item in items)
            if ranges is None:
                ranges = current_ranges
                for previous, following in zip(ranges, ranges[1:]):
                    if not _close_coordinate(previous[1], following[0]):
                        raise ValueError(
                            "spectral blocks must cover a contiguous domain; gap/overlap"
                        )
            elif current_ranges != ranges:
                raise ValueError("all gases must cover the same spectral block grid")
            self._blocks[gas] = items
        self._temperature = np.linspace(axes[2], axes[3], axes[0])
        self._log_pressure = np.linspace(np.log10(axes[4]), np.log10(axes[5]), axes[1])
        self._temperature.setflags(write=False)
        self._log_pressure.setflags(write=False)
        self._step = axes[-1]
        self._start = ranges[0][0]
        for gas, items in self._blocks.items():
            blocks = []
            for start, end, path, size, rank, digest in items:
                offset = int(round((start - self._start) / self._step))
                if not _close_coordinate(self._start + offset * self._step, start):
                    raise ValueError(f"{path.name}: block start is off the common spectral grid")
                stat = path.stat()
                blocks.append(
                    _Block(
                        gas,
                        path,
                        start,
                        end,
                        size,
                        rank,
                        offset,
                        offset + size - 1,
                        stat.st_size,
                        stat.st_mtime_ns,
                        digest,
                    )
                )
            self._blocks[gas] = tuple(blocks)
        first_blocks = self._blocks[self._gases[0]]
        self._block_starts = np.array([block.start_index for block in first_blocks])
        self._last_index = first_blocks[-1].end_index

    def _read_ratios(self, broadening):
        if self._broadening_method == "none":
            return {}
        path = self._data_path(broadening.get("ratios_file"))
        expected_digest = broadening.get("sha256")
        if expected_digest is not None:
            if (
                not isinstance(expected_digest, str)
                or re.fullmatch(r"[0-9a-fA-F]{64}", expected_digest) is None
            ):
                raise ValueError("invalid self-broadening ratios sha256")
            digest = hashlib.sha256()
            with path.open("rb") as stream:
                for chunk in iter(lambda: stream.read(1024 * 1024), b""):
                    digest.update(chunk)
            if digest.hexdigest().lower() != expected_digest.lower():
                raise ValueError(f"SHA256 mismatch for self-broadening ratios {path.name}")
        ratios = {}
        with np.load(path, allow_pickle=False) as archive:
            if "bordes" not in archive:
                raise ValueError("self-broadening ratios require block starts in 'bordes'")
            starts = _vector(archive["bordes"], "ratio block starts", positive=True)
            if starts.size == 0 or np.any(np.diff(starts) <= 0):
                raise ValueError("ratio block starts must be strictly increasing")
            for gas in self._gases:
                if gas not in archive:
                    raise ValueError(f"self-broadening ratios missing gas {gas}")
                values = _vector(archive[gas], f"{gas} broadening ratios")
                if values.shape != starts.shape or np.any(values < 0):
                    raise ValueError(f"{gas}: invalid self-broadening ratio table")
                selected = []
                for block in self._blocks[gas]:
                    # The entry belongs to the block START, not to the preceding window.
                    index = int(np.searchsorted(starts, block.start_cm1))
                    if index == starts.size or not _close_coordinate(
                        starts[index], block.start_cm1
                    ):
                        raise ValueError(
                            f"ratio table does not contain block start {block.start_cm1}"
                        )
                    selected.append(values[index])
                ratios[gas] = tuple(selected)
        return ratios

    def _spectral_indices(self, wavenumber_cm1):
        grid = _vector(wavenumber_cm1, "wavenumber_cm1", positive=True)
        if grid.size == 0 or np.any(np.diff(grid) <= 0):
            raise ValueError("wavenumber_cm1 must be nonempty and strictly increasing")
        fractional = (grid - self._start) / self._step
        if np.any(np.abs(fractional) > np.iinfo(np.int64).max / 2):
            raise ValueError("requested spectral grid lies outside LUT coverage")
        indices = np.rint(fractional).astype(np.int64)
        if np.any(indices < 0) or np.any(indices > self._last_index):
            raise ValueError("requested spectral grid lies outside LUT coverage")
        canonical = self._start + indices * self._step
        # Repeated floating-point addition (np.arange) accumulates starting-point ulps.
        tolerance = 8 * _EPS * np.maximum(1.0, np.abs(grid))
        tolerance += 2 * np.spacing(max(1.0, abs(self._start))) * np.abs(indices)
        if np.any(np.abs(grid - canonical) > tolerance) or np.any(np.diff(indices) <= 0):
            raise ValueError("requested wavenumbers must be distinct nodes of the LUT grid")
        owners = np.searchsorted(self._block_starts, indices, side="right") - 1
        return indices, owners

    def _load_factors(self, block):
        cached = self._cache.get(block.path)
        if cached is not None:
            self._cache.move_to_end(block.path)
            return cached
        stat = block.path.stat()
        if (stat.st_size, stat.st_mtime_ns) != (block.size_bytes, block.mtime_ns):
            raise ValueError(f"LUT block changed since metadata validation: {block.path}")
        if block.sha256 is not None:
            digest = hashlib.sha256()
            with block.path.open("rb") as stream:
                for chunk in iter(lambda: stream.read(1024 * 1024), b""):
                    digest.update(chunk)
            if digest.hexdigest().lower() != block.sha256.lower():
                raise ValueError(f"SHA256 mismatch for {block.path.name}")
        with np.load(block.path, allow_pickle=False) as archive:
            factors = tuple(
                np.array(archive[key], dtype=np.float64, copy=True) for key in ("U", "V", "mu")
            )
        for array in factors:
            if not np.all(np.isfinite(array)):
                raise ValueError(f"non-finite SVD factors in {block.path.name}")
            array.setflags(write=False)
        if self._cache_limit:
            self._cache[block.path] = factors
            while len(self._cache) > self._cache_limit:
                self._cache.popitem(last=False)
        return factors

    def _interpolation(self, temperatures, pressures):
        log_pressure = np.log10(pressures)
        # Reference-VMR multiplication/division and log10 round trips can move a
        # boundary node by a few ulps; that is not physical extrapolation.
        temperature_slack = 8 * _EPS * np.max(np.abs(self._temperature))
        pressure_slack = 8 * _EPS * max(1.0, np.max(np.abs(self._log_pressure)))
        outside = (
            (temperatures < self._temperature[0] - temperature_slack)
            | (temperatures > self._temperature[-1] + temperature_slack)
            | (log_pressure < self._log_pressure[0] - pressure_slack)
            | (log_pressure > self._log_pressure[-1] + pressure_slack)
        )
        if self._bounds == "raise" and np.any(outside):
            raise ValueError("temperature or effective pressure lies outside LUT bounds")
        temperatures = np.clip(temperatures, self._temperature[0], self._temperature[-1])
        log_pressure = np.clip(log_pressure, self._log_pressure[0], self._log_pressure[-1])
        it = np.clip(
            np.searchsorted(self._temperature, temperatures) - 1, 0, self._temperature.size - 2
        )
        ip = np.clip(
            np.searchsorted(self._log_pressure, log_pressure) - 1, 0, self._log_pressure.size - 2
        )
        wt = (
            (temperatures - self._temperature[it])
            / (self._temperature[it + 1] - self._temperature[it])
        )[:, None]
        wp = (
            (log_pressure - self._log_pressure[ip])
            / (self._log_pressure[ip + 1] - self._log_pressure[ip])
        )[:, None]
        return it, ip, wt, wp

    def sigma(self, gas, wavenumber_cm1, temperature_k, pressure_atm, *, vmr=None):
        """Return cross sections [layers, nodes], using vertical pressures in atm.

        Temperatures and pressures are equally sized 1-D arrays (or scalar pairs).
        ``vmr`` has the same shape and lies in [0, 1]; it is mandatory when the
        manifest selects effective-pressure self-broadening correction.
        """
        if gas not in self._blocks:
            raise KeyError(f"gas {gas!r} is absent from LUT; available gases: {self._gases}")
        temperatures = _vector(temperature_k, "temperature_k", positive=True)
        pressures = _vector(pressure_atm, "pressure_atm", positive=True)
        if temperatures.shape != pressures.shape:
            raise ValueError("temperature_k and pressure_atm must have identical shapes")
        if vmr is not None:
            vmr = _vector(vmr, "vmr")
            if vmr.shape != pressures.shape or np.any((vmr < 0) | (vmr > 1)):
                raise ValueError("vmr must match layer shape and lie in [0, 1]")
        elif self._broadening_method == "effective_pressure":
            raise ValueError("vmr is required by the effective_pressure LUT manifest")
        indices, owners = self._spectral_indices(wavenumber_cm1)
        output = np.empty((temperatures.size, indices.size), dtype=np.float64)
        npres = self._log_pressure.size
        reference_vmr = self._gas_metadata[gas]["vmr_ref"]
        for owner in np.unique(owners):
            block = self._blocks[gas][owner]
            locations = np.flatnonzero(owners == owner)
            effective_pressure = pressures
            if self._broadening_method == "effective_pressure":
                ratio = self._ratios[gas][owner]
                effective_pressure = pressures * (1 + (ratio - 1) * vmr)
                effective_pressure = effective_pressure / (1 + (ratio - 1) * reference_vmr)
                if np.any(effective_pressure <= 0) or not np.all(np.isfinite(effective_pressure)):
                    raise ValueError("effective pressure must remain positive and finite")
            it, ip, wt, wp = self._interpolation(temperatures, effective_pressure)
            u, v, mean = self._load_factors(block)
            coefficients = (
                u[it * npres + ip] * (1 - wt) * (1 - wp)
                + u[(it + 1) * npres + ip] * wt * (1 - wp)
                + u[it * npres + ip + 1] * (1 - wt) * wp
                + u[(it + 1) * npres + ip + 1] * wt * wp
            )
            columns = indices[locations] - block.start_index
            log_sigma = mean[columns] + coefficients @ v[:, columns]
            with np.errstate(over="ignore", invalid="ignore"):
                reconstructed = np.power(10.0, log_sigma)
            if not np.all(np.isfinite(reconstructed)):
                raise ValueError(f"non-finite reconstructed cross section for {gas}")
            output[:, locations] = reconstructed
        return output

    def optical_depth(self, wavenumber_cm1, layers, species):
        """Return vertical dimensionless OD for exactly the declared isotope mixtures.

        The result contains only continuum terms already embedded in the LUT. The
        caller adds missing continua once, using the exposed inclusion flags.
        """
        layers, species = tuple(layers), tuple(species)
        indices, _ = self._spectral_indices(wavenumber_cm1)
        names = [item.name for item in species]
        if len(set(names)) != len(names):
            raise ValueError("species must not contain duplicate gas names")
        for item in species:
            if item.name not in self._gas_metadata:
                raise KeyError(f"requested gas {item.name!r} is absent from LUT")
            info = self._gas_metadata[item.name]
            isotopes = [str(isotope.isotope_id) for isotope in item.isotopologues]
            if (
                item.molecule_id != info["molecule_id"]
                or len(isotopes) != len(set(isotopes))
                or set(isotopes) != set(info["isotopologues"])
            ):
                raise ValueError(
                    f"{item.name}: requested isotope mixture differs from aggregated LUT"
                )
        output = np.zeros((len(layers), indices.size), dtype=np.float64)
        temperatures = [layer.temperature_k for layer in layers]
        pressures = [layer.pressure_atm for layer in layers]
        for gas in names:
            if any(gas not in layer.column_density_cm2 or gas not in layer.vmr for layer in layers):
                raise ValueError(f"layers must explicitly provide column density and VMR for {gas}")
            columns = _vector(
                [layer.column_density_cm2[gas] for layer in layers], f"{gas} column_density_cm2"
            )
            if np.any(columns < 0):
                raise ValueError("column densities must be nonnegative")
            cross_sections = self.sigma(
                gas,
                wavenumber_cm1,
                temperatures,
                pressures,
                vmr=[layer.vmr[gas] for layer in layers],
            )
            output += cross_sections * columns[:, None]
        if not np.all(np.isfinite(output)):
            raise ValueError("non-finite LUT optical depth")
        return output
