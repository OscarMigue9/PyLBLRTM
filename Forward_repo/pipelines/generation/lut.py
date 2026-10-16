"""Validate archived SVD factors and expand only requested LUT channels.

Historical row order is temperature-major, pressure-minor. The source factors
hold ``log10(sigma) = mu + U @ V``; dense consumers also expect log10 values.
The initial snapshot of the compatible generator is Photonix commit
``9f93c955779c11f7a8018e9200d63dddfa22ba23``. It does not prove the
identity of the August 14 generation run.
"""

from __future__ import annotations

import hashlib
import json
import re
from dataclasses import dataclass
from pathlib import Path

import numpy as np


GASES = ("H2O", "CO2", "O3", "N2O", "CH4", "CO")
MOLECULE_ID = {"H2O": 1, "CO2": 2, "O3": 3, "N2O": 4, "CO": 5, "CH4": 6}
HISTORICAL_SOURCE_COMMIT = "9f93c955779c11f7a8018e9200d63dddfa22ba23"
PILOT_SHA256 = {
    "lut_H2O_895_905.npz": "704a60cb2f37e047babbd52d2caf0b4de03cfbbbdf2fc61dac3083d415ac3813",
    "lut_CO2_895_905.npz": "b83f6813939124aabc568e6be8d8be09d867fae2ee3c062038ed7d44ca7f3b4d",
    "lut_O3_895_905.npz": "d53978f7cb2b80627eb90ce4c87056604c8d1ad08a97536df2c182f538513a30",
    "lut_N2O_895_905.npz": "394960d085b70312798ee5322bf5732b055f1462f38f6259962f5e0db80e9c3d",
    "lut_CO_895_905.npz": "16c143b89a095de58664a780c7cf56195896b095a0af4eb67de8c0376e6fb196",
    "lut_CH4_895_905.npz": "57c0883c1e72f84638702c66b5273eafad813d332c1ec488d049603a85e0f6b9",
    "r_self.npz": "9d8a9825454cab1d29083fa68a2fca06638adf97aa93e1f7e28d54a580b84679",
}
_BLOCK_NAME = re.compile(r"^lut_([A-Z][A-Z0-9]*)_(\d+)_(\d+)\.npz$")


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(8 * 1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


@dataclass(frozen=True)
class Block:
    gas: str
    start: float
    end: float
    path: Path


def _scalar(archive: np.lib.npyio.NpzFile, key: str) -> float:
    value = np.asarray(archive[key])
    if value.shape != ():
        raise ValueError(f"{key} must be scalar")
    return float(value)


def _block_metadata(block: Block) -> tuple[int, int, int, float]:
    with np.load(block.path, allow_pickle=False) as archive:
        if str(archive["gas"].item()) != block.gas:
            raise ValueError(f"{block.path}: embedded gas differs from its name")
        a, b = _scalar(archive, "a"), _scalar(archive, "b")
        if not np.isclose([a, b], [block.start, block.end], atol=1e-9).all():
            raise ValueError(f"{block.path}: embedded spectral limits differ")
        nt, npres, rank = (int(_scalar(archive, key)) for key in ("NT", "NP", "r"))
        step = _scalar(archive, "dnu")
        nnu = int(round((b - a) / step)) + 1
        if archive["U"].shape != (nt * npres, rank):
            raise ValueError(f"{block.path}: invalid U shape")
        if archive["V"].shape != (rank, nnu) or archive["mu"].shape != (nnu,):
            raise ValueError(f"{block.path}: invalid V or mean shape")
        if not np.isfinite(archive["U"]).all() or not np.isfinite(archive["V"]).all():
            raise ValueError(f"{block.path}: nonfinite factors")
        if not np.isfinite(archive["mu"]).all():
            raise ValueError(f"{block.path}: nonfinite mean")
    return nt, npres, rank, step


def blocks_for(directory: Path, gas: str, start: float, end: float) -> list[Block]:
    """Select a contiguous, nonoverlapping historical block sequence."""
    if gas not in GASES or not 0 < start < end:
        raise ValueError("Unknown gas or invalid spectral interval")
    selected: list[Block] = []
    for path in Path(directory).glob(f"lut_{gas}_*.npz"):
        match = _BLOCK_NAME.fullmatch(path.name)
        if match is None or match.group(1) != gas:
            continue
        a, b = float(match.group(2)), float(match.group(3))
        if b > start and a < end:
            selected.append(Block(gas, a, b, path))
    selected.sort(key=lambda item: (item.start, item.end))
    if not selected or selected[0].start > start or selected[-1].end < end:
        raise ValueError(f"{gas}: factors do not cover {start}..{end} cm^-1")
    for left, right in zip(selected, selected[1:]):
        if left.end != right.start:
            raise ValueError(f"{gas}: gap or overlap at {left.end} / {right.start}")
    return selected


def inspect_blocks(directory: Path, gas: str, start: float, end: float) -> dict:
    """Check metadata and all boundaries without reconstructing a dense LUT."""
    selected = blocks_for(directory, gas, start, end)
    expected: tuple[int, int, int, float] | None = None
    for block in selected:
        metadata = _block_metadata(block)
        if expected is None:
            expected = metadata
        elif metadata != expected:
            raise ValueError(f"{block.path}: inconsistent grid, rank or sample spacing")
    assert expected is not None
    nt, npres, rank, step = expected
    nnu = int(round((end - start) / step)) + 1
    return {
        "gas": gas,
        "blocks": len(selected),
        "range_cm1": [start, end],
        "shape": [nt, npres, nnu],
        "rank": rank,
        "step_cm1": step,
        "status": "candidate_historical_provenance_unverified",
    }


def expand_lut(
    directory: Path, output: Path, gas: str, start: float, end: float, *, overwrite: bool = False
) -> dict:
    """Stream SVD blocks to a float32 NPY, including each shared endpoint once."""
    summary = inspect_blocks(directory, gas, start, end)
    nt, npres, nnu = summary["shape"]
    step = summary["step_cm1"]
    output = Path(output)
    if output.exists() and not overwrite:
        raise FileExistsError(output)
    output.parent.mkdir(parents=True, exist_ok=True)
    temporary = output.with_name(output.name + ".partial")
    if temporary.exists():
        raise FileExistsError(temporary)
    written = np.zeros(nnu, dtype=bool)
    try:
        dense = np.lib.format.open_memmap(temporary, mode="w+", dtype="<f4", shape=(nt, npres, nnu))
        selected = blocks_for(directory, gas, start, end)
        for index, block in enumerate(selected):
            with np.load(block.path, allow_pickle=False) as archive:
                logs = archive["mu"].astype(np.float64)[None, :]
                logs = logs + archive["U"].astype(np.float64) @ archive["V"].astype(np.float64)
            first = max(start, block.start)
            last = min(end, block.end)
            target_start = int(round((first - start) / step))
            target_end = int(round((last - start) / step)) + 1
            source_start = int(round((first - block.start) / step))
            source_end = source_start + target_end - target_start
            if index < len(selected) - 1 and last == block.end:
                target_end -= 1
                source_end -= 1
            if written[target_start:target_end].any():
                raise ValueError(f"{block.path}: overlapping output samples")
            dense[:, :, target_start:target_end] = (
                logs[:, source_start:source_end]
                .reshape(nt, npres, target_end - target_start)
                .astype("<f4")
            )
            written[target_start:target_end] = True
        if not written.all():
            raise ValueError(f"{gas}: {int((~written).sum())} spectral samples remain uncovered")
        dense.flush()
        del dense
        temporary.replace(output)
    finally:
        temporary.unlink(missing_ok=True)
    summary["output"] = str(output)
    summary["sha256"] = sha256_file(output)
    return summary


def build_self_broadening(
    lines_file: Path, output: Path, *, low: int = 635, high: int = 2775
) -> dict:
    """Recreate the historical intensity-weighted median self/air ratio table."""
    if low >= high or (high - low) % 10:
        raise ValueError("Self-broadening grid requires 10 cm^-1 blocks")
    with np.load(lines_file, allow_pickle=False) as lines:
        molecule = lines["mol"][lines["iso_idx"]]
        center = lines["nu0"]
        air = lines["g_air"]
        own = lines["g_self"]
        strength = lines["Sref"]
    edges = np.arange(low, high, 10, dtype=np.float64)
    result: dict[str, np.ndarray] = {"bordes": edges}
    for gas in GASES:
        values = np.ones(edges.size)
        by_gas = molecule == MOLECULE_ID[gas]
        for index, edge in enumerate(edges):
            chosen = by_gas & (center >= edge - 25) & (center <= edge + 35) & (air > 1e-30)
            if not chosen.any():
                continue
            ratios = own[chosen] / air[chosen]
            weights = strength[chosen]
            order = np.argsort(ratios)
            cumulative = np.cumsum(weights[order]) / weights.sum()
            values[index] = float(ratios[order[np.searchsorted(cumulative, 0.5)]])
        result[gas] = values
    output = Path(output)
    output.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(output, **result)
    return {"output": str(output), "sha256": sha256_file(output), "blocks": len(edges)}


def check_pilot_hashes(directory: Path) -> dict:
    """Check archived bytes used by the narrow reader pilot; this is not physics validation."""
    found = {}
    for name, expected in PILOT_SHA256.items():
        path = Path(directory) / name
        digest = sha256_file(path)
        if digest != expected:
            raise ValueError(f"{name}: SHA256 {digest} differs from pilot {expected}")
        found[name] = digest
    return {"status": "storage_identity_only", "sha256": found}


def write_report(path: Path, report: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8")
