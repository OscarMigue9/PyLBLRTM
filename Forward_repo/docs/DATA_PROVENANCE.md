# Provenance of the frozen training data

This audit distinguishes stored-byte identity from reproduction of the scientific
calculation. The frozen byte counts and SHA256 values are in
[`manifests/assets.json`](../manifests/assets.json). Checks below were performed
on 2026-09-27 without changing the frozen data.

| Asset | Established provenance | Remaining gate |
|---|---|---|
| `tud_iasi_apae` | The local `tud_iasi_all.npy` is exactly the 772,200,128-byte frozen asset (SHA256 `8ea2dd0cc6454797099b07087430c016f2ec0db5fe2986a0a03a347ee76e5833`). Its array is `(30000, 6435)` float32, or three 2145-channel fields. Small metadata and the 2145-channel axis from the co-located August Modal archive are bundled in [`frozen/apae_tud_provenance`](../frozen/apae_tud_provenance) and pinned in the [provenance manifest](../manifests/apae_tud_provenance.json). | The metadata identifies a 714–1250 cm⁻¹ nadir product and names a LUT revision, but does not embed the T/U/D file hash. Its direct linkage to the frozen array and the exact executed generator/LUT bytes remain unproven. It cannot be treated as the angle-matched T/U/D of `radiances_iasiang`. |
| `radiances_iasiang` | The local NPZ is exactly the 366,309,747-byte frozen asset (SHA256 `dcb03b4002f5657433db1051df3fd0f9696f8273f68f98cdc1397ea08d70cc4d`). Its embedded note names `prep_recorte714.py` and a crop at channel 276 from `radiances_iasi645ang.npz`. The original five-angle `L_30k.npy` was downloaded and independently hashed; angle selection and crop match all 30,000 × 2,145 frozen `L` values exactly. See the [full report](validation/radiance_parent_full.json). | The intermediate 645–1250 NPZ is absent, but is not needed to verify the parent-to-frozen crop. Independent physical regeneration of the upstream five-angle T/U/D and radiance parent remains unaccredited. |
| `noise_iasiang` | The frozen NPZ was downloaded and independently verified against its 238,626,926-byte, SHA256 `efbe1b1295cddf8735bcd86b2ddbc159831fcd5d598274f66640f1c9430745eb` manifest entry. A fresh CPU reconstruction with seed 1337 and 256-row batches was compared across all 30,000 × 2,145 float32 noise samples: 2 values differ, RMS `1.1609829838e-13`, maximum `9.313225746e-10`; the frequency and channel-noise arrays match exactly. See the [full comparison](validation/noise_rebuild.json). | The new NPZ differs in bytes and in two float32 noise values; do not call it an exact reproduction of the frozen input. The 645–1250 parent NPZ remains absent from the checked Drive directory. |
| `r_self` | `build_self_broadening()` regenerated the 214-edge table from frozen `gpu_pack/lineas.npz` on CPU with exact archived SHA256 `9d8a9825454cab1d29083fa68a2fca06638adf97aa93e1f7e28d54a580b84679`. The local historical copy and cache have the same bytes. | The `gpu_pack` was also rebuilt byte-identically from AER, bundled TIPS files and four frozen continuum tables. The original acquisition of the TIPS edition and three continuum source tables remains unknown; see [the generation guide](../pipelines/generation/README.md). |

## T/U/D identity and mismatch

The neighboring `data/derived/tud30k_channels/meta.npz`, `nu_iasi.npy` and
`chunks/{T,U,D}_iasi_*.npy` describe an older **830–1255 cm⁻¹, 1701-channel**
JAX/nadir product dated July. They do not describe the frozen
`tud_iasi_all.npy`, which has 2145 channels per field and a later timestamp.
Aligning the July chunk at 830–1250 cm⁻¹ against the frozen first row does not
give equality (RMS T `0.00257`, U `0.0666`, D `0.1030`).

The older Modal workspace `oscarmigue9325:wikithesis-datos` retained a different,
co-located `outputs/tud30k_channels/meta.npz` and `nu_iasi.npy`, dated
2026-08-10, alongside a `tud_iasi_all.npy` dated 2026-08-12 with the same
reported size and timestamp as the local frozen copy. The two small files were
downloaded, SHA256-checked and bundled. Their axis is exactly 2145 float64
values from 714 to 1250 cm⁻¹ at 0.25 cm⁻¹ steps. The metadata declares
30,000 profiles, 1,000-row chunks, Gaussian IASI FWHM 0.5 cm⁻¹,
`nadir="mu=1 vertical"`, and
`lut="outputs/lut_band_* 2026-08-02, TODOS los isotopologos de los 5 gases (lblrtm_tud E11/E17/E18)"`.
Its `T_skin_K`, `lat`, `lon`, `lsm`, `subset_id` and `profile_index` arrays
match the SHA-verified `states_30k` asset element for element across all
30,000 rows. This is strong evidence for the state ordering and nadir recipe,
but the metadata does not contain a hash of the large T/U/D cache.

The preserved `gen_tud30k_channels.py` (historical path: `experiments/2026-05_diseno_lut/gen_tud30k_channels.py`)
is a candidate implementation: its executable grid is 714–1250 cm⁻¹, it
computes nadir T/U/D with H₂O continuum, five gases, and a separately applied
Gaussian ILS. Its loader reads `outputs/lut_xsec_jax_*.npy`, while the metadata
string it writes names `outputs/lut_band_*`. The archived metadata repeats the
second name. This discrepancy prevents attributing the exact LUT bytes or
executed script revision to the frozen T/U/D. A small numerical next check is
to read row zero from the original 2145-channel
`T_iasi_000000.npy`, `U_iasi_000000.npy` and `D_iasi_000000.npy` chunks and
compare each full row byte for byte with the corresponding fields of the
frozen cache. That requires 25,740 data bytes, plus three NumPy headers, if
the original chunks can be located. No full T/U/D download or GPU regeneration
is needed for this linkage check.

On 2026-09-27, read-only `modal volume ls` against the old
`oscarmigue9325:wikithesis-datos` archive found only the T/U/D cache and the two
small metadata files in `outputs/tud30k_channels`. Its `chunks` path returned
`No such file or directory`; a direct `modal volume get` of
`chunks/T_iasi_000000.npy` returned the same error. Other accessible
`wikithesis-datos` workspaces checked did not retain that directory either.
No chunk was downloaded, so the proposed row canary remains pending.

The frozen T/U/D also fails an angle-matched radiance check. Using its `[T|U|D]`
fields with the archived `eps`, atmospheric `T_skin`, and
`L = eps × B(T_skin) × T + U + (1−eps) × T × D` gives, for the first 16 scenes,
RMS `2.486` and maximum absolute difference `16.978` in the radiance's
mW/(m² sr cm⁻¹) units. Across the first 256 scenes, RMS by archived viewing
cosine is:

| μ | Scenes | Radiance RMS |
|---:|---:|---:|
| 0.55 | 55 | 4.965 |
| 0.67 | 45 | 2.971 |
| 0.80 | 46 | 1.721 |
| 0.92 | 57 | 0.757 |
| 1.00 | 53 | 0.239 |

The trend independently supports the nadir label in the co-located metadata,
but does not prove that the frozen array came from that archived calculation.
The current
[`generation/synthesis.py`](../pipelines/generation/synthesis.py) can build a
**new candidate** selected-angle T/U/D and radiance. That output must not be
identified with the frozen APAE T/U/D without an archived numerical canary.

## Original full-angle radiance parent

The original `modal_radiancias.py` (historical path: `photonix/production/modal_radiancias.py`)
wrote `(30000, 5, 8461)` float32 radiances on 645–2760 cm⁻¹. A Drive backup
remains at `gdrive:WikiThesis/exp_iasi_full/out/radiancias/L_30k.npy` (also
accessible via the local `uis:` alias). Its 5,076,600,128-byte size and SHA256
`7e33eca95e2d1116041d5361d8969e31acdef3dc49e76008b48e0e4707453dd8`
were independently verified from the complete downloaded file and agree with
Google Drive metadata. The adjacent 971,180-byte `meta_30k.npz` was
read in full and independently hashed to
`a635741a5312985c683255caa2c4af4d5145139bad6bd22f67cf68429f3f0c18`.
It records five angles `[0.55, 0.67, 0.80, 0.92, 1.00]` and 8,461 channels.

The archived construction is explicit in
`prep_banda645.py` (historical path: `model/physinv/closure_forward/experimentos/apae_loss_fisica/prep_banda645.py`),
`prep_angulo.py` (historical path: `model/physinv/closure_forward/experimentos/apae_loss_fisica/prep_angulo.py`)
and `prep_recorte714.py` (historical path: `model/physinv/closure_forward/experimentos/apae_loss_fisica/prep_recorte714.py`):
crop the full radiance to 645–1250, select one of the five angles per profile
with seed 991, then crop at channel 276 to 714–1250. The 645–1250 intermediate
is not currently retained, but the 645–2760 parent is enough to verify all
selected `L` rows. The [`radiance parent manifest`](../manifests/radiance_parent.json)
pins the source, metadata, layout, four earlier row canaries and the full
comparison evidence.

The [`range verifier`](../pipelines/radiance_parent_canary.py) checks the Drive
size and SHA256 metadata, the NumPy header, the small metadata file's actual
bytes, the frozen radiance archive's actual bytes, all 30,000 angle assignments,
and four complete selected radiance rows (profiles 0, 1, 15000, 29999). It
requests only 34,320 radiance payload bytes, plus the 128-byte header, from
the 5.08 GB parent. All four rows matched exactly. The range-only check can be
repeated from `thesis final`:

```powershell
python -m pipelines.assets --cache data_cache fetch radiances_iasiang
python -m pipelines.radiance_parent_canary `
  --archive data_cache/data/outputs/radiances_iasiang.npz `
  --report docs/validation/radiance_parent_canary.json
```

The complete parent was then downloaded as `runs/radiance_parent_full.npy`.
The `--full-parent` mode independently verified its 5.08 GB SHA256, memory
mapped the parent and streamed the frozen NPZ member in 118 bounded batches.
All 30,000 angle assignments matched seed 991, and all 64,350,000 archived
`L` values were equal bit for bit: unequal count, RMS and maximum difference
were all zero. Reproduce the [full result](validation/radiance_parent_full.json)
from the verified local parent with:

```powershell
python -m pipelines.radiance_parent_canary `
  --archive data_cache/data/outputs/radiances_iasiang.npz `
  --full-parent runs/radiance_parent_full.npy `
  --report docs/validation/radiance_parent_full.json
```

The current `rclone-wikithesis` Modal secret defines `gdrive:`, `uis:` and
`d400:`; the full parent is on `gdrive:`/`uis:`, not the `d400:` copy. The
source's whole-file SHA256 is now calculated from its 5.08 GB local bytes.
This proves the archived parent-to-frozen selection and crop. It does **not**
prove that the original forward calculation can regenerate `L_30k.npy` from
spectroscopy and five-angle T/U/D. A 43-block five-angle T/U/D product is
preserved under `gdrive:WikiThesis/exp_iasi_full/out/tud30k/chunks`, but its
full numerical link to the parent has not yet been accredited.

## CPU canaries for radiance and noise

The 30,000 archived `mu` values exactly equal
`default_rng(991).integers(0, 5, size=30000)` indexed into
`[0.55, 0.67, 0.80, 0.92, 1.00]`, with counts `[6069, 5949, 5894, 6079, 6009]`.
For the first 16 scenes, land emissivities exactly equal the frozen
`emis_iasi.npy` rows selected with `default_rng(42)` and the archived land mask.
Sea emissivities also match bit for bit after float32 conversion when calculated
from the OPAC water-index table using the original float64 view nodes. Using
the stored float32 `mu` for this recomputation can differ by one float32 ULP.
These checks confirm the angle, material-selection and Fresnel parts of the
recipe; they do not establish the cross sections or physical generation of the
archived radiance `L`. The separate full-parent comparison establishes its
archived selection and crop exactly.

The first two archived `(2145,)` float32 noise rows match
`build_noise(count=2, batch_size=256, seed=1337)` exactly: both row-pair SHA256
values are `3b8aaa17b40230932e13286c118ae16293fc3535cd65e06fe48ffd3a024f2f93`.
The subsequent full CPU reconstruction wrote a 238,626,813-byte NPZ with
SHA256 `a026a7f16292bc6cefcdba943bbcf0e31d03a30af5b2b96ad3ef4f7d277a1899`.
The [independent comparator](../pipelines/noise_compare.py) verified the frozen
file against the manifest and the candidate against its generation report,
checked the `(30000, 2145)` array shape, float32 type, 714–1250 cm⁻¹ axis,
seed, grid and FWHM, then compared all 64,350,000 noise cells. Exactly two
cells differ: `(9733, 596)` by `-2.7755575616e-17`, and `(14108, 1477)` by
`9.313225746e-10`. The noise RMS difference is `1.1609829838e-13`; `nu`
and `sigma_ch` match exactly. The candidate is numerically very close but
not bit identical, and no tolerance was predeclared to accept it as a training
replacement. The JSON [report](validation/noise_rebuild.json) records both
indices and values.

The metadata contains `seed=1337`, `grid=iasiang`, `dnu_ch=0.25`, `fwhm=0.5`,
and a note recording the crop `[276:]` from `iasi645ang`. The generator starts
its random draw on the 645–1250 fine grid before cropping; generating directly
on 714–1250 would produce another realization. Its batch size affects later
draws and is fixed at 256 for this candidate recipe. Reproduce the comparison
after generating and fetching the two archives:

```powershell
python -m pipelines.noise_compare `
  --candidate runs/noise_rebuild/noise_iasiang.npz `
  --frozen data_cache/data/outputs/noise_iasiang.npz `
  --build-report runs/noise_full_report.json `
  --out docs/validation/noise_rebuild.json
```

The [`final configuration`](../configs/final.json) records the same assignment
seeds 991 and 42 and noise seed 1337. `generation/synthesis.py` uses these
values, but its direct 714–1250 T/U/D calculation is a candidate reconstruction
of a historically wider run. Exact parent-to-frozen radiance identity and the
CPU assignment/noise checks do not prove that this candidate physically
regenerates the five-angle parent or its T/U/D.
