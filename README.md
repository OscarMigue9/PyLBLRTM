# PyLBLRTM

**Open-source line-by-line radiative transfer model for atmospheric remote sensing.**

PyLBLRTM computes monochromatic atmospheric transmittance and radiance from HITRAN spectral line parameters. It is designed to be transparent, well-documented, and usable as a standalone forward model or as a component in a retrieval system.

> Status: v0.1.1 — data layer complete. Forward model in active development.

---

## What it does

A line-by-line (LBL) radiative transfer model calculates how radiation propagates through the atmosphere by explicitly summing the contribution of each individual spectral line. This is the most accurate approach for simulating atmospheric spectra and is the basis of instruments like IASI, CrIS, and AIRS.

PyLBLRTM is structured in two layers:

```
Data layer (done)
─────────────────
  SpectralData  ←  HITRAN line parameters + TIPS partition sums + molar masses

Forward model (in development)
──────────────────────────────
  SpectralData + atmospheric profile → transmittance / radiance spectrum
```

---

## Installation

```bash
pip install pylblrtm
```

**Requirements:** Python ≥ 3.9, numpy ≥ 1.24, hitran-api ≥ 1.3

---

## Data layer

The data layer loads HITRAN spectral line parameters into a `SpectralData` object — the input the forward model will receive.

### Loading via HAPI (recommended)

```python
from pylblrtm import from_hapi

# Single isotopologue (default: most abundant)
sd = from_hapi('H2O', nu_min=1000, nu_max=2500)

# Specific isotopologues
sd = from_hapi('CO2', nu_min=1000, nu_max=2500, iso_id=[1, 2, 3])

# All available isotopologues
sd = from_hapi('O3', nu_min=1000, nu_max=2500, iso_id='all')
```

Data is downloaded from HITRAN on the first call and cached locally. Subsequent calls read from cache — no internet required.

### Loading from files

```python
from pylblrtm import from_files

# .par may contain multiple molecules and isotopologues
sd = from_files(
    'mezcla.par',
    tips={(1, 1): 'H2O_iso1.txt', (1, 2): 'H2O_iso2.txt',
          (2, 1): 'CO2_iso1.txt'},
)
```

`tips` is a dict keyed by `(mol_id, iso_id)`. Values can be paths to two-column TIPS `.txt` files or `(T_arr, Q_arr)` numpy arrays already in memory. Molar masses are looked up automatically from HAPI if not provided.

### SpectralData

Both paths return the same `SpectralData` object:

```python
sd.lines              # dict of numpy arrays: nu0, Sref, g_air, g_self,
                      #   Elow, n_air, shift, mol_id, iso_id, ...
sd.molar_masses       # {(mol_id, iso_id): float [g/mol]}
sd.tips               # {(mol_id, iso_id): (T_arr, Q_arr)}
sd.Q(T, mol_id, iso_id)  # partition sum interpolated at temperature T [K]
```

Quick access to line arrays:

```python
sd.lines['nu0']       # line centres [cm⁻¹]
sd.lines['Sref']      # intensities at T_ref = 296 K
sd.lines['Elow']      # lower-state energies [cm⁻¹]
sd.lines['g_air']     # air-broadened half-widths [cm⁻¹/atm]
sd.lines['mol_id']    # HITRAN molecule ID per line
sd.lines['iso_id']    # isotopologue ID per line
```

### Supported molecules

55 molecules from the HITRAN database, accessed by name (case-insensitive):

```python
from pylblrtm import list_molecules
list_molecules()
```

---

## Project structure

```
pylblrtm/
├── __init__.py          public API
├── hitran_manager.py    HITRAN access, cache, format conversion, TIPS
├── data_loader.py       from_hapi / from_files → SpectralData
└── spectral_data.py     SpectralData dataclass

tests/
├── test_hitran_manager.py
├── test_data_loader.py
└── test_spectral_data.py

DOCS.md                  full API reference
pyproject.toml
```

---

## References

If you use HITRAN data in your work, please cite:

> I.E. Gordon et al., *The HITRAN2020 molecular spectroscopic database*,
> J. Quant. Spectrosc. Radiat. Transfer **277**, 107949 (2022).
> https://doi.org/10.1016/j.jqsrt.2021.107949

Also cite HAPI, which is used internally:

> R.V. Kochanov et al., *HITRAN Application Programming Interface (HAPI)*,
> J. Quant. Spectrosc. Radiat. Transfer **177**, 15–30 (2016).
> https://doi.org/10.1016/j.jqsrt.2016.03.005

---

## License

MIT
