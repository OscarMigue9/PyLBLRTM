# Verified raw input retrieval

Run these commands from `thesis final`. The default destination is
`data_cache/raw`, with a `data/` tree beneath it. The source URLs and archive digests are recorded in
`manifests/raw_sources.json`; individual CAMS and ECOSTRESS byte counts and SHA256
digests are in `manifests/raw_catalogs.json`. Existing files are verified and an
incorrect file causes an error without being replaced. Each new file is written
to a temporary path, verified, and published atomically.

```powershell
python -m pipelines.raw_fetch cams
python -m pipelines.raw_fetch ecostress
python -m pipelines.raw_fetch aer
```

`cams` requests only six 2025 NWP-SAF CAMS L137 NetCDF members from the official
993,280,000-byte TAR. Each TAR header and member is requested by HTTP Range.
Only about 293 MB of NC data plus six 512-byte headers cross the network. All
six members were checked against the local raw catalog by remote Range streaming
on 2026-09-27. The complete TAR SHA256 is unavailable because the full TAR was
not downloaded; the six member SHA256 checks are the reproducibility boundary.

`ecostress` downloads the 41,526,412-byte Zenodo ZIP and extracts the selected
1419 spectra only. All 1419 ZIP members were compared byte for byte with the raw
catalog on 2026-09-27. The Zenodo record is an archived third-party mirror of
JPL ECOSTRESS spectra, while the JPL download interface requires an order request.
The exact selection is still governed by the frozen `record_ids.npy` asset.

`aer` downloads the 442,551,040-byte AER v3.8.1 TAR from Zenodo record 4019178.
It extracts the TAR root `lncpl_lines` and reconstructs the seven-molecule
620–2785 cm⁻¹ `.par` file using the archived line-selection recipe. The result
was checked locally against the frozen asset: 730,759 lines, 73,806,659 bytes,
SHA256 `898c856b07b298d85a8e0bc788ab388ce7391417fb9a0276b44d62c874c457b2`.
The coupling member also matches its frozen SHA256. This TAR is the original
Zenodo record 4019178; the later reorganized record has different bytes.

Use `--archive PATH` after the `ecostress` or `aer` command to read a local
archive. The same archive size and SHA256 checks still apply. The `all` command
fetches all three source groups and can transfer roughly 0.8 GB. Allow temporary
disk space for the ZIP and TAR, and memory for sorting the AER `.par` lines.

These raw inputs enable `pipelines.generation build-states` and
`build-emissivity`. The AER outputs, bundled TIPS files and four small frozen
continuum tables rebuild all three `gpu_pack` archives byte-identically. Run
`pipelines.assets --cache data_cache --workspace-root data_cache/raw fetch
aer_lines aer_coupling` to place the verified AER outputs at their expected
cache paths. The original acquisition of the TIPS files and three continuum
tables is not recorded; their exact bytes are bundled and verified.
