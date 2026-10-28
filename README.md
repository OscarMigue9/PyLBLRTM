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

From the repository (the current version; PyPI still has 0.1.1, without the forward model):

```bash
git clone https://github.com/OscarMigue9/PyLBLRTM.git
cd PyLBLRTM
git checkout dev
python -m venv .venv
.venv/bin/python -m pip install -e ".[fast,notebook]"   # Windows: .venv\Scripts\python
```

**Python ≥ 3.9** (tested on 3.11 and 3.12). The base install brings everything the
library needs; the extras are optional:

| Install | Adds | Needed for |
|---|---|---|
| `pip install -e .` | numpy ≥ 1.24, scipy ≥ 1.10, hitran-api ≥ 1.3, pyyaml ≥ 6.0 | Downloading HITRAN lines and the forward model (`voigt="wofz"`) |
| `.[fast]` | numba ≥ 0.59 | `voigt="humlicek"` and `wing_tolerance` — tens of times faster |
| `.[plot]` | matplotlib ≥ 3.7 | Plotting |
| `.[notebook]` | matplotlib, ipykernel, nbconvert | Opening and running the notebooks in `notebooks/` |
| `.[dev]` | pytest, sphinx | Running the tests (`python -m pytest tests`) and building docs |

Extras combine: `pip install -e ".[fast,notebook,dev]"`. If that Python ever had
`pip install pylblrtm` (0.1.1 from PyPI), it keeps importing the old version until
`pip install -e .` is run from the repository in that same Python.

**Getting started:** [`notebooks/guia_forward.ipynb`](notebooks/guia_forward.ipynb) walks
through installing, calling the forward model and its parameters. Downloading lines needs
internet access to hitran.org (no account); HITRAN limits the number of queries per
day and per IP, and downloaded tables are cached in `~/.pylblrtm/hitran_cache`.

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

### Speed: Voigt backend and threads

```python
nu, T = compute_transmittance(sd, profile, 710, 740, voigt="humlicek", n_jobs=-1)
```

- `voigt="wofz"` (default) uses scipy's exact Faddeeva function. `voigt="humlicek"` uses
  the Humlicek W4 approximation compiled with numba and adds the lines straight into the
  optical depth: tens of times faster than `wofz` and within 1e-4 relative of it. It
  needs `pip install pylblrtm[fast]`.
- `n_jobs` runs on that many threads (`-1` = every core). Threads take the spectral tiles
  of every layer from one shared queue, so a single-layer standoff path uses them all
  just like a full column. The result is bit-identical to `n_jobs=1`.

- `wing_tolerance=1e-4` (with `voigt="humlicek"`) computes the far wings on grids 4, 16
  and 64 times coarser and interpolates them, keeping the optical depth within that
  relative error of the exact result; line cores and the 25 cm⁻¹ cutoff stay exact.
  Several times faster again. Default `None`: everything on the fine grid.

The same keys exist under `calculation:` in `forward_config.yaml`.

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

### Standoff path (homogeneous)

A horizontal path of length L with a single concentration, pressure and temperature
per gas. The column is `N = VMR · P / (k_B T) · L`; it works wherever a profile does.

```python
from pylblrtm import from_hapi, compute_transmittance, HomogeneousPath

sd   = from_hapi("NH3", 714.29, 1250.0, iso_id="all")   # every isotopologue
path = HomogeneousPath(100.0, {"NH3": 1e-6})             # 1 ppm over 100 m, 1 atm, 296 K
nu, T = compute_transmittance(sd, path, 714.29, 1250.0, step_cm1=0.01)
```

HITRAN line intensities already include the natural abundance of each isotopologue,
so the VMR is the molecule's total; `get_abundance(molecule, iso_id)` returns it if a
single isotopologue is needed.

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

With `iso_id='all'` or a list, isotopologues without lines in the range are skipped with a
warning. Errors are specific: `HITRANNoLinesError` (no lines in that range),
`HITRANNotAvailableError` (HITRANonline does not serve it line by line, e.g. SF6, CF4),
`HITRANQuotaError` (HITRAN's daily API query limit was exceeded). Missing broadening
coefficients (0 in HITRAN) are filled with the isotopologue's median.

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

All 61 molecules of the HITRAN database (IDs 1–61, as in HAPI), accessed by name (case-insensitive):

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
├── voigt.py             Voigt backends: wofz (scipy), humlicek (numba)
└── config.py            run_forward, load_config, ForwardResult

tests/
├── test_hitran_manager.py
├── test_data_loader.py
├── test_spectral_data.py
├── test_voigt.py
├── test_forward.py          forward-model physics with known answers
├── hapi_download_check.py   script: download every molecule (cluster)
├── tips_check.py            script: TIPS tables of every isotopologue
└── forward_check.py         script: forward model with real HITRAN lines

scripts/                 SLURM jobs for the tests and notebooks
forward_config.yaml      editable configuration template
pyproject.toml
```

---

## References

If you use HITRAN data in your work, please cite:

> I.E. Gordon et al., *The HITRAN2024 molecular spectroscopic database*,
> J. Quant. Spectrosc. Radiat. Transfer (2026).
> https://doi.org/10.1016/j.jqsrt.2026.109807

HITRANonline serves the HITRAN2024 edition since January 2026; the partition
sums (TIPS-2025, the default) are:

> R.R. Gamache et al., J. Quant. Spectrosc. Radiat. Transfer **345**, 109568 (2025).
> https://doi.org/10.1016/j.jqsrt.2025.109568

Also cite HAPI, which is used internally:

> R.V. Kochanov et al., *HITRAN Application Programming Interface (HAPI)*,
> J. Quant. Spectrosc. Radiat. Transfer **177**, 15–30 (2016).
> https://doi.org/10.1016/j.jqsrt.2016.03.005

---

## License

MIT
