"""Rebuild the archived GPU spectroscopy pack from verified final inputs.

The packed CO2 isotope assignments are deliberately historical. They reproduce
the archived LUT inputs and must not be substituted with the corrected Photonix
CPU catalogue without regenerating and revalidating every downstream artifact.
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np

from pipelines.assets import Artifact, hash_file, verify_file


ROOT = Path(__file__).resolve().parents[2]
PACK_MANIFEST = ROOT / "manifests" / "gpu_pack_sources.json"
GASES = ("H2O", "CO2", "O3", "N2O", "CO", "CH4", "O2")
TREF = 296.0
TGRID = np.arange(150.0, 350.01, 0.25)


def _read_manifest(path: Path) -> dict:
    manifest = json.loads(path.read_text(encoding="utf-8"))
    if manifest["schema_version"] != 1 or manifest["hash_algorithm"] != "sha256":
        raise ValueError("Unsupported GPU pack source manifest")
    return manifest


def _verify_small(path: Path, record: dict) -> None:
    if not path.is_file():
        raise FileNotFoundError(path)
    if path.stat().st_size != record["bytes"] or hash_file(path) != record["sha256"]:
        raise ValueError(f"Source identity differs from frozen manifest: {path}")


def _variants(tips_dir: Path, tips_manifest: Path, legacy_mapping: dict) -> list[dict]:
    """Load all 42 historical variants and verify their Q(T) file bytes."""
    records = json.loads(tips_manifest.read_text(encoding="utf-8"))["files"]
    if len(records) != 42:
        raise ValueError("Expected the 42-isotopologue frozen TIPS catalogue")
    expected_co2 = ("9", "0", "A")
    if tuple(records[i]["isotope_id"] for i in (15, 16, 17)) != expected_co2:
        raise ValueError("Unexpected CO2 isotope order in frozen TIPS catalogue")
    variants = []
    for index, record in enumerate(records):
        molecule = int(record["molecule_id"])
        isotope = str(record["isotope_id"])
        gas = str(record["path"]).split("/", 1)[0]
        if gas != GASES[molecule - 1]:
            raise ValueError(f"Unexpected molecule order at TIPS row {index}")
        # Legacy Photonix used '10' rather than HITRAN's one-character '0'.
        if index == 16:
            isotope = "10"
        path_name = legacy_mapping.get(isotope, record["path"]) if molecule == 2 else record["path"]
        source = tips_dir.joinpath(*Path(path_name).parts)
        # Reassigned CO2 files retain their own source hash in the clean manifest.
        matching = next((item for item in records if item["path"] == path_name), None)
        if matching is None:
            raise ValueError(f"Unknown historical partition file {path_name}")
        _verify_small(source, matching)
        variants.append(
            {
                "molecule": molecule,
                "isotope": isotope,
                "molar_mass": float(record["molar_mass_g_mol"]),
                "qfile": source,
            }
        )
    return variants


def _read_hitran_par(path: Path) -> dict[str, np.ndarray]:
    """Parse the ten fields used by the historical Photonix packer."""
    widths = [2, 1, 12, 10, 10, 5, 5, 10, 4, 8, 15, 15, 15, 15, 6, 12, 1, 7, 7]
    offsets = np.cumsum([0] + widths)
    slices = [(offsets[i], offsets[i + 1]) for i in range(10)]
    mol, iso, nu0, sref, _, g_air, g_self, elow, n_air, shift = ([] for _ in range(10))
    with path.open("r", errors="ignore") as stream:
        for line in stream:
            if not line.strip():
                continue
            try:
                fields = [line[start:end] for start, end in slices]
                mol.append(int(fields[0]))
                iso.append(fields[1].strip())
                nu0.append(float(fields[2]))
                sref.append(float(fields[3]))
                _.append(float(fields[4]))
                g_air.append(float(fields[5]))
                g_self.append(float(fields[6]))
                elow.append(float(fields[7]))
                n_air.append(float(fields[8]))
                shift.append(float(fields[9]))
            except Exception:
                continue
    return {
        "mol": np.asarray(mol, dtype=np.int32),
        "iso": np.asarray(iso, dtype="<U10"),
        "nu0": np.asarray(nu0, dtype=np.float64),
        "Sref": np.asarray(sref, dtype=np.float64),
        "g_air": np.asarray(g_air, dtype=np.float64),
        "g_self": np.asarray(g_self, dtype=np.float64),
        "Elow": np.asarray(elow, dtype=np.float64),
        "n_air": np.asarray(n_air, dtype=np.float64),
        "shift": np.asarray(shift, dtype=np.float64),
    }


def _read_coupling(path: Path) -> dict[str, np.ndarray]:
    """Extract the four CO2 line-mixing coefficients from AER lncpl_lines."""
    rows = []
    pending = None
    with path.open("r", errors="replace") as stream:
        for raw in stream:
            line = raw.rstrip("\n")
            if not line or line[0] in ">%":
                continue
            if pending is not None:
                rows.append((pending, line))
                pending = None
                continue
            try:
                flag = int(line[-2:])
            except ValueError:
                flag = 0
            if flag != 0:
                pending = line
    values: dict[str, list] = {"iso": [], "nu0": [], "Y": []}
    for header, coefficients in rows:
        if int(header[0:2]) != 2:
            continue
        parsed = np.fromstring(coefficients[2:].replace("-", " -").replace("E -", "E-"), sep=" ")
        if parsed.size < 8:
            continue
        values["iso"].append(header[2:3])
        values["nu0"].append(float(header[3:15]))
        values["Y"].append(parsed[[0, 2, 4, 6]])
    return {key: np.asarray(value) for key, value in values.items()}


def _pack_lines(par_path: Path, coupling_path: Path, variants: list[dict]) -> dict:
    lines = _read_hitran_par(par_path)
    count = lines["nu0"].size
    variant_index = np.full(count, -1, dtype=np.int32)
    masses, molecule_ids = [], []
    for index, variant in enumerate(variants):
        match = (lines["mol"] == variant["molecule"]) & (lines["iso"] == variant["isotope"])
        variant_index[match] = index
        masses.append(variant["molar_mass"])
        molecule_ids.append(variant["molecule"])

    coupling = _read_coupling(coupling_path)
    mixing = np.zeros((count, 4))
    for isotope in np.unique(coupling["iso"]):
        line_match = (lines["mol"] == 2) & (lines["iso"] == isotope)
        coupling_match = coupling["iso"] == isotope
        if line_match.sum() == 0 or coupling_match.sum() == 0:
            continue
        frequencies = lines["nu0"][line_match]
        order = np.argsort(frequencies)
        positions = np.clip(
            np.searchsorted(frequencies[order], coupling["nu0"][coupling_match]),
            1,
            frequencies.size - 1,
        )
        nearest = np.where(
            np.abs(frequencies[order][positions] - coupling["nu0"][coupling_match])
            < np.abs(frequencies[order][positions - 1] - coupling["nu0"][coupling_match]),
            positions,
            positions - 1,
        )
        valid = np.abs(frequencies[order][nearest] - coupling["nu0"][coupling_match]) < 1e-4
        mixing[np.where(line_match)[0][order][nearest[valid]]] = coupling["Y"][
            np.where(coupling_match)[0][valid]
        ]

    used = variant_index >= 0
    return {
        "nu0": lines["nu0"][used].astype(np.float64),
        "Sref": lines["Sref"][used].astype(np.float64),
        "Elow": lines["Elow"][used].astype(np.float64),
        "g_air": lines["g_air"][used].astype(np.float64),
        "g_self": lines["g_self"][used].astype(np.float64),
        "n_air": lines["n_air"][used].astype(np.float64),
        "shift": lines["shift"][used].astype(np.float64),
        "iso_idx": variant_index[used],
        "Y": mixing[used].astype(np.float64),
        "mol": np.array(molecule_ids, dtype=np.int32),
        "Wg": np.array(masses),
        "gases": np.array(GASES),
    }


def _pack_qtab(variants: list[dict]) -> dict:
    partition_values = []
    references = []
    for variant in variants:
        table = np.loadtxt(variant["qfile"], comments="#")
        temperatures, values = table[:, 0], table[:, 1]
        partition_values.append(np.array([np.interp(t, temperatures, values) for t in TGRID]))
        references.append(np.interp(TREF, temperatures, values))
    return {
        "TGRID": TGRID,
        "Q": np.array(partition_values),
        "Qref": np.array(references),
        "TREF": TREF,
    }


def _pack_continuum(sources: list[tuple[str, Path]]) -> dict:
    arrays = {}
    for prefix, path in sources:
        with np.load(path, allow_pickle=False) as source:
            for key in source.files:
                arrays[f"{prefix}_{key}"] = source[key]
    return arrays


def build_gpu_pack(
    par_path: Path,
    coupling_path: Path,
    output_dir: Path,
    *,
    asset_manifest: Path = ROOT / "manifests" / "assets.json",
    pack_manifest: Path = PACK_MANIFEST,
    bundle_root: Path = ROOT,
) -> dict:
    """Write and SHA-verify the three historical packed files to a fresh directory."""
    manifest = _read_manifest(pack_manifest)
    artifacts = {
        item.identifier: item
        for item in (
            Artifact.from_record(record)
            for record in json.loads(asset_manifest.read_text(encoding="utf-8"))["artifacts"]
        )
    }
    verify_file(par_path, artifacts[manifest["line_file_asset"]])
    verify_file(coupling_path, artifacts[manifest["coupling_asset"]])
    tips_dir = bundle_root / manifest["tips_directory"]
    tips_manifest = bundle_root / manifest["tips_manifest"]
    variants = _variants(tips_dir, tips_manifest, manifest["legacy_co2_mapping"])
    sources = []
    for record in manifest["continuum"]:
        source = bundle_root / record["bundle_path"]
        _verify_small(source, record)
        sources.append((record["prefix"], source))

    if any((output_dir / name).exists() for name in manifest["output_assets"]):
        raise FileExistsError(f"GPU pack output already exists in {output_dir}")
    output_dir.mkdir(parents=True, exist_ok=True)
    arrays = {
        "lineas.npz": _pack_lines(par_path, coupling_path, variants),
        "qtab.npz": _pack_qtab(variants),
        "cont.npz": _pack_continuum(sources),
    }
    report = {"status": "archived_pack_rebuild", "outputs": {}}
    for name, contents in arrays.items():
        path = output_dir / name
        np.savez_compressed(path, **contents)
        artifact = artifacts[manifest["output_assets"][name]]
        verify_file(path, artifact)
        report["outputs"][name] = {
            "path": str(path),
            "sha256": artifact.sha256,
            "bytes": artifact.size,
            "byte_identical_to_archive": True,
        }
    report["co2_partition_assignment"] = "historical_legacy"
    return report
