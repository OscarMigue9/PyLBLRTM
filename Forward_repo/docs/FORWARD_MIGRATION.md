# Forward migration and physical status

## Source snapshots

The installed `photonix` CPU tree is a byte-for-byte copy of the 63 source and resource files in
`forward clean/photonix/src/photonix` at commit
`b4277191e86c81ff7ca783463ffdf919f8a2071c`. This includes 18 Python files,
42 TIPS partition tables, the compact AER continuum table and their provenance
manifests. The 24 additional files visible in that original tree were generated
caches, not source. The copy was compared by SHA256 for every source file. No external line
list, coupling file, LUT block, atmosphere, training data or model weight is bundled.

On 2026-09-28 `spectroscopy/absorption.py` and `constants.py` were recopied from the
`forward clean` working tree, not yet committed there. The line radiation term
`nu*tanh(c2*nu/2T)` is now applied at every frequency, as LBLRTM does with `JRAD=0`,
instead of at the line centre. The recipe identifier moved to
`aer-3.8.1-convf4-thermal-714-1250-v2`. `torch_legacy`, the dense LUT and the
checkpoint are unaffected.

`photonix.torch_legacy` is a style-normalized extraction of the numerical kernels in
`model/physinv/forward_lut_torch.py` and `model/physinv/continuum_torch.py` at model
commit `9f3e0a440189e54dd5c7ad6adfc56e74f2b04de7`. Their source SHA256 values are
`b770cf5c3d3dd6dc3a582fce6e70ac98066409c95c756455d6bbecb9c6d15248` and
`aa85c8d0c6a411a2016a802c82eddea06d671a349bfa011c199843f1f54761f3`.
The original `continuum_tables.npz` is bundled unchanged, SHA256
`192e5e2771947e371d729df5d0f08af8bed8804f52f3f7c818a12851bdfcef29`.
The old `continuum_torch.py` fallback to an absolute Windows `.pyc` path is removed;
the installed resource is now the sole source of these coefficients.

The `torch_legacy` recipe identifier is `retrieval-z24-ncol-torch-frozen-v1`. It
identifies this extracted runtime. Its use with the frozen dense LUT and
recovered step-1,000 checkpoint passed full-scene T4 comparisons on the
simulated 300 and ocean 64 sets. It does not by itself identify the generator
of every historical LUT block or prove the original training file's hash.

The old module allowed `WT_DIFUS` to override 1.66 at import time. The final
recipe fixes 1.66 explicitly, matching the dataset generator and the historical
module's default. Runs that intentionally set a different `WT_DIFUS` are outside
this recipe and need their own provenance.

## Two explicit numerical contracts

| Component | CPU `photonix` | `photonix.torch_legacy` |
|---|---|---|
| Purpose | AER LBL and manifest-backed SVD LUT; single-atmosphere radiance | Differentiable dense-LUT training/evaluation kernel |
| LUT representation | Compressed NPZ SVD blocks with recipe, isotope and continuum manifest | Dense `log10(sigma)` tensors; axes and gas order supplied by the retrieval adapter |
| LUT boundaries | Reject by default; `bounds="clip"` is explicit | Hold nearest T/logP boundary; both weights clamped |
| Planck constants | `1.19104272e-5` and `1.4387752` in cm^-1 form | `1.19104297e-16` and `1.43877688e-2` in m^-1 form |
| H2O continuum reference | 1013.0 hPa; radiation constant 1.4387752 cm K | 1013.25 hPa; radiation constant 1.4388 cm K |
| Transfer | CPU float64 EMIN, downward diffusivity 1.66 | Torch EMIN with safe unused branch for finite gradients, fixed downward diffusivity 1.66 |
| Surface | Optional ILS after full radiance composition | The retrieval adapter composes and convolves channel windows |

The CPU tree's broader physical corrections are documented in the source
`forward clean/photonix/docs/RECETA.md` and `MIGRACION.md`: CO2 partition mapping,
line-center shift, unique tile/block boundary ownership, strict missing-data
handling and ILS padding. Those corrections must not be silently applied to data
or a checkpoint made with the historical Torch runtime.

### What can be replaced today

- Importing the clean CPU API from the installed `photonix` package replaces the
  former ad hoc CPU source tree for **new** LBL/LUT runs following its declared
  recipe `aer-3.8.1-convf4-thermal-714-1250-v2`.
- Importing `interp_sigma`, `integrate_tud`, `planck_nu`,
  `prep_continuum_tables` and `continuum_od_torch` from
  `photonix.torch_legacy` replaces the two loose Torch kernel files in the final
  retrieval's forward adapter. For float64 synthetic fixtures, the extracted
  Planck, interpolation, TUD and continuum outputs exactly matched the original
  modules (maximum absolute difference `0.0` in the migration comparison at
  both float32 and float64).
- The table resource is portable inside the installed package. No local `.pyc`,
  working-directory assumption or machine path is used by these kernels.

### What cannot yet be claimed

- `CrossSectionLUT` does not load the dense `.npy` arrays consumed by the retrieval,
  and its corrected physical recipe is not the trained checkpoint's recipe.
  Converting SVD blocks to dense tensors requires an explicit artifact conversion
  and checks against the exact training LUT, not just matching dimensions.
- The synthetic kernel fixtures alone do not establish full-scene equivalence.
  The complete extracted retrieval was therefore evaluated against the
  historical T4 results with all five dense LUTs and verified data. Those
  numerical comparisons passed; they apply to the historical Torch recipe,
  not to replacing it with the corrected CPU recipe.
- The clean CPU project's scientific verification covers a three-channel pilot;
  it does not establish full-band agreement with the historical GPU generator or
  justify replacing the existing LUT and training data. The T4 discrepancy was
  resolved numerically by using the recovered step-1,000 checkpoint, while
  the original checkpoint's byte identity remains unrecorded.
- Float64 CPU parity does not imply bitwise GPU training reproducibility. The
  accepted target is matching data/configuration and metrics within declared
  tolerances.

## Verification performed here

`tests/forward/` carries the clean package's independent unit tests that have no
dependency on the source project's `verification` or `tools` directories, plus
frozen Torch outputs generated from the historical files. One opt-in SVD block
smoke test is skipped without an external LUT directory and manifest. The local
run yielded **171 passed, 1 skipped**. Ruff check passes. The Torch parity
comparison covered Planck, inverse Planck, clamped interpolation, vectorized and
loop TUD, continuum table interpolation and continuum optical depth at float64.
A second fixture compared complete forward T/U/D/radiance and temperature
gradients at float32 and float64; each had maximum absolute difference `0.0`
against the historical modules. Finite continuum and TUD gradients were also
checked on the extracted implementation.

This verification covers kernel extraction and package behavior. Full-scene
T4 validation with the archived dense LUT is recorded in
[`validation/README.md`](validation/README.md). The CPU forward's corrected
recipe and regeneration of all historical LUT/data inputs retain their own
scientific gates.
