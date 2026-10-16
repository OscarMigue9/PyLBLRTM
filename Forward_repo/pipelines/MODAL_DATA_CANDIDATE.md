# Candidate synthetic data on Modal

`modal_data_candidate.py` runs the 714–1250 cm⁻¹ synthesis recipe in a separate
`thesis-final-data-candidates` volume. It never writes to the frozen input paths.
The three stages are T/U/D chunks, selected-angle radiance with matching T/U/D,
and fixed-seed instrument noise. All outputs retain
`candidate_historical_parity_unverified` status.

From the `thesis final` directory, first inspect the plan. This builds a Modal
image but starts no remote generation function or GPU work:

```powershell
$env:MODAL_PROFILE = '<your-profile>'
$env:PYTHONIOENCODING = 'utf-8'
modal run pipelines/modal_data_candidate.py --stage all --count 30000 --dry-run
```

Before T/U/D, the Modal runner expects its own
`lut_full_{h2o,co2,o3,n2o,ch4}.json` sampled-comparison reports and all five
candidate dense LUTs in `thesis-final-lut-candidates`. It verifies the 315
factor blocks and receipts against those reports, including source and input
SHA256. A failed sampled comparison is rejected by default.

Current evidence is mixed. [O3](../docs/validation/lut_full_o3.json) passes
its sampled full-range gate. [CO2](../docs/validation/lut_full_co2.json),
[H2O](../docs/validation/lut_full_h2o.json) and
[N2O](../docs/validation/lut_full_n2o.json) pass every one of their 63 sampled
blocks and the sampled 714–1250 model band, but exceed the maximum limit at
the 1265 cm⁻¹ terminal channel:

| Gas | Terminal maximum log10 cross-section difference | Limit |
| --- | ---: | ---: |
| CO2 | `5.22613525390625e-4` | `2.5e-4` |
| H2O | `2.899169921875e-4` | `2.5e-4` |
| N2O | `8.697509765625e-4` | `2.5e-4` |

CH4 generated all 63 blocks and receipts before Modal workspace
`ac-wcltPGHZuWysY1tH2xbeL8` was disabled. The blocks were verified by SHA256
and source snapshot, then expanded locally with NumPy 2.2.5. The
[local CH4 report](../docs/validation/lut_full_ch4_local.json) passes its
sampled gate and exhaustively scans the full dense array. All blocks and the
exact 714–1250 model band pass; the 1265 cm⁻¹ terminal **RMS** `3.65e-5`
exceeds `2.5e-5`, although the terminal maximum passes. The pinned Drive
reference was used; a stale same-name `/outputs/` file was excluded. The
local report is a distinct schema and the local dense candidate is not in the
Modal candidate volume, so it cannot be supplied as this runner's fifth
report. Full-cell scans of the other four gases have not been performed.

If an enabled Modal workspace becomes available and compatible five-gas
reports/artifacts are published, the explicit
`--allow-terminal-outside-model-band` option can allow the isolated sampled
terminal-maximum pattern for CO2/H2O/N2O. It checks all sampled blocks and
the overall sampled RMS; each report retains its failed full-band result.
That allowance does not resolve the CH4 exhaustive terminal-RMS failure or
establish historical radiance identity. Candidate data generation remains
pending until these scientific gates are reviewed.

The workspace above is disabled, so the following commands cannot currently
start GPU candidate generation there. They describe the intended stages once
an enabled workspace and compatible LUT receipts are available.

```powershell
modal run pipelines/modal_data_candidate.py --stage tud --count 30000 `
  --allow-terminal-outside-model-band
modal run pipelines/modal_data_candidate.py --stage assemble --count 30000
modal run pipelines/modal_data_candidate.py --stage noise --count 30000
```

`--stage all` runs those stages in order. The T/U/D call uses an A100 and can be
long. It writes twelve 50 cm⁻¹ tiles, each with all five viewing cosines, into
`n30000/tud_chunks/`. Completed tiles are SHA256 checked against receipts and
copied into a fresh local work directory when the command is restarted. Each new
tile is committed to the candidate volume as soon as its atomic generator file
appears. An incomplete publication is preserved under `orphans/` and rebuilt;
a complete output with changed code, inputs or bytes causes a hard failure.

`assemble` verifies the T/U/D receipts, frozen states, emissivity and water index
before producing `n30000/radiance/radiances_iasiang.npz` and
`tud_iasiang_selected.npy`. It numerically compares the radiance arrays to the
SHA256 verified archived radiance and records the result. The selected-angle
T/U/D is **not** the frozen APAE training T/U/D. `noise` uses seed 1337 and batch
size 256, compares the arrays to the verified archived noise and writes
`n30000/noise/noise_iasiang.npz`. Both stages store output hashes and receipts.

`--count 2` runs a small smoke candidate in the separate `n2/` namespace. It
cannot be confused with the 30,000-profile data. No candidate artifact should
replace a frozen training asset until the relevant numerical and downstream
model parity gates are resolved and documented.

For example, inspect a completed stage receipt with:

```powershell
modal volume get thesis-final-data-candidates n30000/tud_report.json `
  runs/candidate_tud_report.json
```
