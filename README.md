# PyLBLRTM

**Open-source line-by-line radiative transfer model for atmospheric remote sensing.**

PyLBLRTM computes monochromatic atmospheric transmittance and radiance from HITRAN spectral line parameters. It is designed to be transparent, well-documented, and usable as a standalone forward model or as a component in a retrieval system.

> Status: v0.1.0 — data layer complete. Forward model in active development.

---

## What it does

A line-by-line (LBL) radiative transfer model calculates how radiation propagates through the atmosphere by explicitly summing the contribution of each individual spectral line. This is the most accurate approach for simulating atmospheric spectra and is the basis of instruments like IASI, CrIS, and AIRS.

PyLBLRTM is structured in two layers:

```
Data layer (done)
─────────────────
  SpectralData  ←  HITRAN line parameters + TIPS partition sums

Forward model (in development)
──────────────────────────────
  SpectralData + atmospheric profile → transmittance / radiance spectrum
```

---

## Installation

```bash
pip install pylblrtm

# Optional: HITRAN download support
pip install "pylblrtm[hapi]"
```

**Requirements:** Python ≥ 3.9, numpy ≥ 1.24

---

## Data layer

The data layer loads HITRAN spectral line parameters into a `SpectralData` object — the input the forward model will receive.

Two ways to get data:

```python
from pylblrtm import from_hapi, from_files

# Download from HITRAN (requires pip install "pylblrtm[hapi]")
data = from_hapi('H2O', nu_min=700, nu_max=1400)

# Or load your own files (.par + TIPS .txt)
data = from_files('H2O_700_1400.par', 'H2O_tips.txt')
```

Both return the same object:

```python
print(data)
# SpectralData(H2O iso1, 3889 lines, nu=[700.0, 1399.9] cm-1, Wg=18.010565 g/mol)

data.nu0          # line centres [cm⁻¹]
data.Sref         # line intensities at T_ref = 296 K
data.g_air        # air-broadened half-widths
data.Elow         # lower state energies
data.molar_mass   # molar mass [g/mol] — for Doppler width
data.Q(T)         # partition sum interpolated to temperature T [K]
```

---

## Project structure

```
pylblrtm/
├── __init__.py          public API
├── hitran_manager.py    HITRAN access, cache, format conversion, TIPS
├── data_loader.py       from_hapi / from_files → SpectralData
└── spectral_data.py     SpectralData dataclass

workflow_test.ipynb      end-to-end test of all public methods
DOCS.md                  full API reference
pyproject.toml
```

---

## References

If you use HITRAN data in your work, please cite:

> I.E. Gordon et al., *The HITRAN2020 molecular spectroscopic database*,
> J. Quant. Spectrosc. Radiat. Transfer **277**, 107949 (2022).
> https://doi.org/10.1016/j.jqsrt.2021.107949

---

## License

MIT
