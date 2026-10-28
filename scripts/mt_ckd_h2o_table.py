"""
mt_ckd_h2o_table.py — Build pylblrtm/data/mt_ckd_h2o.npz from AER's MT_CKD file.

Source: absco-ref_wv-mt-ckd.nc, from https://github.com/AER-RC/MT_CKD_H2O
(data/) or the copy that ships with LBLRTM. It is netCDF3, so scipy reads it.
The arrays are copied unchanged; only the standard foreign continuum is kept
(not for_closure_absco_ref, the alternative fitted to AERI window closure).

Usage (from the repository root):
    py scripts/mt_ckd_h2o_table.py path/to/absco-ref_wv-mt-ckd.nc
"""

import hashlib
import sys
from pathlib import Path

import numpy as np
from scipy.io import netcdf_file

OUT = Path(__file__).resolve().parent.parent / "pylblrtm" / "data" / "mt_ckd_h2o.npz"


def _text(value) -> str:
    return (value.decode("latin-1") if isinstance(value, bytes) else str(value)).strip()


def main(src: Path) -> None:
    digest = hashlib.sha256(src.read_bytes()).hexdigest()
    with netcdf_file(src, "r", mmap=False) as f:
        var = f.variables
        nu = np.array(var["wavenumbers"].data, dtype=np.float64)
        table = {
            "nu": nu,
            "self": np.array(var["self_absco_ref"].data, dtype=np.float64),
            "foreign": np.array(var["for_absco_ref"].data, dtype=np.float64),
            "texp": np.array(var["self_texp"].data, dtype=np.float64),
            "T_ref": np.float64(var["ref_temp"].data),
            "P_ref_hpa": np.float64(var["ref_press"].data),   # mbar = hPa
            "version": np.array(_text(f._attributes["Title"])),
            "copyright": np.array(_text(f._attributes["Copyright"])),
            "reference": np.array(_text(f._attributes["Reference"])),
            "source": np.array(f"{src.name} sha256 {digest}"),
        }

    if np.any(np.diff(nu) <= 0):
        raise ValueError("wavenumbers must increase strictly")
    if np.any(table["self"] <= 0) or np.any(table["foreign"] <= 0):
        raise ValueError("coefficients must be positive (they are interpolated in log)")

    OUT.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(OUT, **table)
    print(f"{table['version']}: {nu[0]:g} to {nu[-1]:g} cm-1, {nu.size} points, "
          f"T_ref {float(table['T_ref']):g} K, P_ref {float(table['P_ref_hpa']):g} hPa")
    print(f"-> {OUT} ({OUT.stat().st_size / 1024:.0f} KiB)")


if __name__ == "__main__":
    if len(sys.argv) != 2:
        sys.exit(__doc__)
    main(Path(sys.argv[1]))
