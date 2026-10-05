# PyLBLRTM

**Open-source line-by-line radiative transfer model for atmospheric remote sensing.**

PyLBLRTM computes monochromatic atmospheric transmittance from HITRAN spectral line parameters. It is designed to be transparent, well-documented, and usable as a standalone forward model or as a component in a retrieval system.

> Status: v0.1.2 — forward model complete. Transmittance LBL with YAML config, US Standard Atmosphere profiles, any molecule/isotopologue from HITRAN.

---

## What it does

A line-by-line (LBL) radiative transfer model calculates how radiation propagates through the atmosphere by explicitly summing the contribution of each individual spectral line. This is the most accurate approach for simulating atmospheric spectra and is the basis of instruments like IASI, CrIS, and AIRS.

PyLBLRTM is structured in two layers:

```
Data layer
──────────
  SpectralData  ←  HITRAN line parameters + TIPS partition sums + molar masses

Forward model
─────────────
  SpectralData + AtmosphericProfile → transmittance spectrum
  Voigt profile · hydrostatic column density · HITRAN temperature scaling
```

---

## Installation

```bash
pip install pylblrtm
```

**Requirements:** Python ≥ 3.9, numpy ≥ 1.24, scipy ≥ 1.10, hitran-api ≥ 1.3, pyyaml ≥ 6.0

---

## Quick start

### YAML-driven workflow (recommended)

Edit `forward_config.yaml` in the repo root, then:

```python
from pylblrtm import from_hapi, us_standard_atmosphere, run_forward

sd      = from_hapi("CO2", nu_min=700, nu_max=750, iso_id=1)
profile = us_standard_atmosphere(gases=["CO2"])
result  = run_forward(sd, profile, "forward_config.yaml")

result.transmittance      # numpy array, shape (n_nu,)
result.save()             # writes to output.save.path in the YAML
```

The YAML controls the spectral range, resolution, viewing angle, and which outputs to compute:

```yaml
spectral:
  nu_min_cm1: 700.0
  nu_max_cm1: 750.0
  step_cm1:   0.001

calculation:
  mu_view: 1.0            # cos(zenith angle), 1.0 = nadir

output:
  transmittance:        true
  optical_depth:        false
  optical_depth_layers: false
  save:
    enabled: false
    path: "results/"
    format: npz           # npz | csv | txt
    prefix: "forward"
```

### Direct API

```python
from pylblrtm import from_hapi, us_standard_atmosphere, compute_transmittance

sd      = from_hapi("CO2", nu_min=700, nu_max=750, iso_id=1)
profile = us_standard_atmosphere(gases=["CO2"])

nu, T = compute_transmittance(sd, profile, 710, 740, step_cm1=0.005)
```

### Custom atmospheric profile

```python
from pylblrtm import AtmosphericProfile, compute_transmittance

profile = AtmosphericProfile(
    pressure_hpa  = [10., 100., 1013.25],   # TOA → surface
    temperature_k = [220., 265.,  288.],
    vmr           = {"CO2": [4.15e-4] * 3},
)
nu, T = compute_transmittance(sd, profile, 710, 740)
```

---

## Data layer

The data layer loads HITRAN spectral line parameters into a `SpectralData` object.

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

sd = from_files(
    'mezcla.par',
    tips={(1, 1): 'H2O_iso1.txt', (1, 2): 'H2O_iso2.txt',
          (2, 1): 'CO2_iso1.txt'},
)
```

### SpectralData

```python
sd.lines              # dict of numpy arrays: nu0, Sref, g_air, g_self,
                      #   Elow, n_air, shift, mol_id, iso_id
sd.molar_masses       # {(mol_id, iso_id): float [g/mol]}
sd.tips               # {(mol_id, iso_id): (T_arr, Q_arr)}
sd.Q(T, mol_id, iso_id)  # partition sum interpolated at temperature T [K]
```

### Supported molecules

55 molecules from the HITRAN database, accessed by name (case-insensitive):

```python
from pylblrtm import list_molecules
list_molecules()
```

---

## Reference atmosphere profiles

```python
from pylblrtm import us_standard_atmosphere, list_profile_gases

list_profile_gases()
# ['H2O', 'CO2', 'O3', 'N2O', 'CH4', 'CO']

profile = us_standard_atmosphere(gases=["CO2", "H2O"])
```

`us_standard_atmosphere` returns a 20-layer `AtmosphericProfile` from USSA76 pressure/temperature tables and AFGL Mid-Latitude Summer VMR profiles (Anderson et al., 1986).

---

## Project structure

```
pylblrtm/
├── __init__.py          public API
├── hitran_manager.py    HITRAN access, cache, format conversion, TIPS
├── data_loader.py       from_hapi / from_files → SpectralData
├── spectral_data.py     SpectralData dataclass
├── atmosphere.py        AtmosphericProfile: levels → hydrostatic layers
├── profiles.py          us_standard_atmosphere (USSA76 + AFGL VMR)
├── forward.py           compute_transmittance: Voigt LBL by layer
└── config.py            run_forward, load_config, ForwardResult

tests/
├── test_hitran_manager.py
├── test_data_loader.py
├── test_spectral_data.py
├── test_atmosphere.py
└── test_forward.py

forward_config.yaml      editable configuration template
prueba.ipynb             end-to-end demo notebook
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
