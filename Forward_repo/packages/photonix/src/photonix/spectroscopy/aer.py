"""Explicit AER CO2 line-coupling input and CONVF4 line-shape correction."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
import re

import numpy as np

from .voigt import voigt_profile
from .species import normalize_isotope_id

_TEMPERATURES_K = np.array([200.0, 250.0, 296.0, 340.0])
_NUMBER = re.compile(r"[+-]?(?:\d+\.\d*|\.\d+|\d+)(?:[EeDd][+-]?\d+)?")


@dataclass(frozen=True)
class AerLineMixing:
    """CO2 coupling records keyed by local HITRAN isotope and unshifted frequency.

    ``coefficients`` has four columns, for 200, 250, 296 and 340 K. A record with
    four zeros is an explicitly uncoupled line. Missing records are errors, rather
    than an assumption of zero coupling. Temperature interpolation clamps at the
    endpoint values, as in the frozen production recipe.
    """

    isotope_id: np.ndarray
    wavenumber_cm1: np.ndarray
    coefficients: np.ndarray
    source: str = "explicit arrays"

    def __post_init__(self):
        raw_isotope = np.asarray(self.isotope_id)
        if raw_isotope.ndim != 1:
            raise ValueError("AER isotope identifiers must be one-dimensional")
        isotope = np.asarray([normalize_isotope_id(value) for value in raw_isotope], dtype="U1")
        nu = np.asarray(self.wavenumber_cm1, dtype=np.float64).copy()
        coefficients = np.asarray(self.coefficients, dtype=np.float64).copy()
        if isotope.ndim != 1 or nu.shape != isotope.shape:
            raise ValueError("AER isotope and frequency arrays must be aligned and 1-D")
        if coefficients.shape != (nu.size, 4):
            raise ValueError("AER coupling requires four temperature coefficients per line")
        if not np.isfinite(nu).all() or not np.isfinite(coefficients).all():
            raise ValueError("AER coupling data must be finite")
        if np.any(nu <= 0) or np.any(np.char.str_len(isotope) != 1):
            raise ValueError("AER records require positive frequency and a local isotope ID")
        for name, value in (
            ("isotope_id", isotope),
            ("wavenumber_cm1", nu),
            ("coefficients", coefficients),
        ):
            value.flags.writeable = False
            object.__setattr__(self, name, value)

    @classmethod
    def load(cls, path: str | Path) -> "AerLineMixing":
        """Read AER ``lncpl_lines`` in a streaming pass, retaining CO2 only.

        The 100-column AER format stores a line record followed by a coefficient
        record when its trailing flag is nonzero. Zero-flag CO2 records are kept,
        so absence of a line can never be mistaken for a physical zero. Fortran
        D exponents and adjacent signed numeric fields are accepted.
        """
        path = Path(path)
        isotopes, frequencies, coefficients = [], [], []
        pending = None
        with path.open("rb") as handle:
            for number, raw in enumerate(handle, 1):
                raw = raw.rstrip(b"\r\n")
                # AER copyright headers can contain UTF-8; numeric records are ASCII.
                if not raw.strip() or raw.startswith((b">", b"%")):
                    continue
                try:
                    row = raw.decode("ascii", errors="strict")
                    if pending is not None:
                        molecule, isotope, nu, record_number = pending
                        pending = None
                        if molecule == 2:
                            values = [
                                float(x.replace("D", "E").replace("d", "e"))
                                for x in _NUMBER.findall(row[2:-2])
                            ]
                            if len(values) != 8:
                                raise ValueError("coefficient row must contain eight values")
                            isotopes.append(isotope)
                            frequencies.append(nu)
                            coefficients.append(values[::2])
                        continue
                    if len(row) < 100:
                        raise ValueError("line record is shorter than 100 columns")
                    molecule = int(row[:2])
                    isotope = row[2]
                    nu = float(row[3:15].replace("D", "E"))
                    flag = int(row[-2:])
                    if flag:
                        pending = molecule, isotope, nu, number
                    elif molecule == 2:
                        isotopes.append(isotope)
                        frequencies.append(nu)
                        coefficients.append([0.0] * 4)
                except (ValueError, UnicodeDecodeError) as exc:
                    raise ValueError(
                        f"{path}:{number}: invalid AER coupling record: {exc}"
                    ) from exc
        if pending is not None:
            raise ValueError(f"{path}:{pending[3]}: missing AER coefficient continuation")
        if not frequencies:
            raise ValueError(f"{path}: no CO2 coupling records")
        return cls(
            np.asarray(isotopes),
            np.asarray(frequencies),
            np.asarray(coefficients),
            source=str(path),
        )

    def coefficients_for(self, lines, *, tolerance_cm1: float = 1.0e-4) -> np.ndarray:
        """Align coupling with a LineList; reject missing or ambiguous CO2 matches.

        Coincident records are accepted only when their coupling coefficients are
        identical. Matching is isotope-specific and never clips search indices.
        """
        if not np.isfinite(tolerance_cm1) or tolerance_cm1 <= 0:
            raise ValueError("Coupling matching tolerance must be finite and positive")
        result = np.zeros((len(lines.wavenumber_cm1), 4), dtype=np.float64)
        co2 = np.asarray(lines.molecule_id) == 2
        for isotope in np.unique(np.asarray(lines.isotope_id)[co2]):
            target_indices = np.flatnonzero(co2 & (np.asarray(lines.isotope_id) == isotope))
            source_indices = np.flatnonzero(self.isotope_id == isotope)
            order = np.argsort(self.wavenumber_cm1[source_indices], kind="stable")
            source_indices = source_indices[order]
            source_nu = self.wavenumber_cm1[source_indices]
            for index in target_indices:
                frequency = lines.wavenumber_cm1[index]
                lo = np.searchsorted(source_nu, frequency - tolerance_cm1, side="left")
                hi = np.searchsorted(source_nu, frequency + tolerance_cm1, side="right")
                candidates = source_indices[lo:hi]
                if candidates.size == 0:
                    raise ValueError(
                        f"Missing AER CO2 coupling record: isotope={isotope}, "
                        f"wavenumber={frequency:.8f} cm^-1 ({self.source})"
                    )
                distance = np.abs(self.wavenumber_cm1[candidates] - frequency)
                nearest = candidates[distance <= distance.min() + 1e-12]
                values = self.coefficients[nearest]
                if not np.all(values == values[0]):
                    raise ValueError(
                        f"Ambiguous AER CO2 coupling: isotope={isotope}, "
                        f"wavenumber={frequency:.8f} cm^-1"
                    )
                result[index] = values[0]
        return result


def coupling_at_temperature(coefficients: np.ndarray, temperature_k: float) -> np.ndarray:
    """Interpolate the four AER coefficients; endpoints reproduce production."""
    if not np.isfinite(temperature_k) or temperature_k <= 0:
        raise ValueError("Coupling temperature must be positive and finite")
    coefficients = np.asarray(coefficients, dtype=np.float64)
    if coefficients.ndim != 2 or coefficients.shape[1] != 4:
        raise ValueError("Coupling coefficients must have shape [line, 4]")
    return np.asarray(
        [np.interp(temperature_k, _TEMPERATURES_K, row) for row in coefficients], dtype=np.float64
    )


def aer_line_profile(
    wavenumber_cm1,
    line_center_cm1,
    doppler_hwhm_cm1,
    lorentz_hwhm_cm1,
    *,
    is_co2=False,
    coupling=None,
    pressure_atm: float = 1.0,
    cutoff_cm1: float = 25.0,
    voigt_method: str = "humlicek",
) -> np.ndarray:
    """Return CONVF4 profiles with finite support, before multiplying by line strength.

    Non-CO2: V(d)-L(C). CO2: [V(d)-(2-d^2/C^2)L(C)]*(1+d*Y*P/alpha_V).
    Signed line-mixing contributions are retained; only the final physical sum
    can be assessed for nonnegativity. No per-line clipping is applied.
    """
    if not np.isfinite(cutoff_cm1) or cutoff_cm1 <= 0:
        raise ValueError("AER line cutoff must be finite and positive")
    if not np.isfinite(pressure_atm) or pressure_atm < 0:
        raise ValueError("AER pressure must be finite and nonnegative")
    nu = np.asarray(wavenumber_cm1, dtype=np.float64)
    center = np.asarray(line_center_cm1, dtype=np.float64)
    doppler = np.asarray(doppler_hwhm_cm1, dtype=np.float64)
    lorentz = np.asarray(lorentz_hwhm_cm1, dtype=np.float64)
    profile = voigt_profile(nu, center, doppler, lorentz, method=voigt_method)
    distance = nu[:, None] - center
    pedestal = lorentz / (np.pi * (cutoff_cm1**2 + lorentz**2))
    co2 = np.broadcast_to(np.asarray(is_co2, dtype=bool), center.shape)
    profile -= pedestal * np.where(co2, 2.0 - distance**2 / cutoff_cm1**2, 1.0)
    if co2.any():
        if coupling is None:
            raise ValueError("CO2 AER profiles require explicit coupling, including zero values")
        y = np.asarray(coupling, dtype=np.float64)
        if y.shape != center.shape or not np.isfinite(y).all():
            raise ValueError("Coupling coefficients must be finite and aligned with line centers")
        voigt_hwhm = 0.5346 * lorentz + np.sqrt(0.2166 * lorentz**2 + doppler**2)
        profile[:, co2] *= 1.0 + distance[:, co2] * y[co2] * pressure_atm / voigt_hwhm[co2]
    profile[np.abs(distance) > cutoff_cm1] = 0.0
    if not np.isfinite(profile).all():
        raise FloatingPointError("AER line profile produced non-finite values")
    return profile
