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

### Water-vapour continuum

```python
nu, T = compute_transmittance(sd, profile, 714, 1250, h2o_continuum=True)
```

`h2o_continuum=True` adds the MT_CKD 4.3 continuum of H₂O (self + foreign) to the optical
depth of every layer, profile or standoff path; it needs `"H2O"` in the profile and a range
within −20…20 000 cm⁻¹. Default `False`: lines only. It is the only continuum (no CO₂, no
O₂/N₂ collision-induced absorption). MT_CKD assumes each line has its value at 25 cm⁻¹
subtracted (AER's pedestal), which pylblrtm does not do, so those pedestals of the H₂O
lines are counted twice. `h2o_continuum_optical_depth(nu, profile.build_layers())` in
`pylblrtm.continuum` returns the continuum alone. Same key under `calculation:` in
`forward_config.yaml`.

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
├── continuum.py         MT_CKD H2O continuum (self + foreign)
├── data/mt_ckd_h2o.npz  MT_CKD 4.3 coefficients (AER)
└── config.py            run_forward, load_config, ForwardResult

tests/
├── test_hitran_manager.py
├── test_data_loader.py
├── test_spectral_data.py
├── test_voigt.py
├── test_forward.py          forward-model physics with known answers
├── test_continuum.py        H2O continuum with known answers
├── hapi_download_check.py   script: download every molecule (cluster)
├── tips_check.py            script: TIPS tables of every isotopologue
└── forward_check.py         script: forward model with real HITRAN lines

scripts/                 SLURM jobs for the tests and notebooks;
                         mt_ckd_h2o_table.py rebuilds data/mt_ckd_h2o.npz
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

With `h2o_continuum=True`, the MT_CKD water-vapour continuum of AER
(https://github.com/AER-RC/MT_CKD_H2O):

> E.J. Mlawer et al., *Development and recent evaluation of the MT_CKD model of
> continuum absorption*, Phil. Trans. R. Soc. A **370**, 2520–2556 (2012).
> https://doi.org/10.1098/rsta.2011.0295

---

## License

MIT. The MT_CKD coefficients in `pylblrtm/data/mt_ckd_h2o.npz` are © AER and are
redistributed under AER's terms, whose notice is kept in the file (key `copyright`).
