"""Verify one complete candidate gas LUT against its frozen dense reference.

Run after ``modal_lut_full.py --gas GAS`` has completed::

    modal run pipelines/modal_lut_compare.py --gas CO2

The verifier reads the candidate volume without writing to it. It checks every
block and receipt, the dense file, the generation source snapshot and the
archived dense LUT SHA256. It then compares deterministic samples in all 63
blocks. The local report remains a numerical validation of the sampled grid,
not proof that every value of the candidate LUT matches the archive.
"""

from __future__ import annotations

import base64
import hashlib
import json
import math
import os
from pathlib import Path

import modal
import numpy as np


ROOT = Path(__file__).resolve().parents[1]
APP_ROOT = Path("/app")
CACHE_ROOT = Path("/cache/data")
CANDIDATE_ROOT = Path("/candidates")
GASES = ("H2O", "CO2", "O3", "N2O", "CH4")
PACK_IDS = ("gpu_pack_lines", "gpu_pack_qtab", "gpu_pack_continuum")
MAX_RMS_LOG10_SIGMA = 2.5e-5
MAX_ABS_LOG10_SIGMA = 2.5e-4

app = modal.App("thesis-final-lut-compare")
image = (
    modal.Image.debian_slim(python_version="3.11")
    .apt_install("rclone", "ca-certificates")
    .pip_install("numpy==2.1.3")
    .add_local_dir(
        ROOT / "pipelines" / "generation",
        "/app/pipelines/generation",
        copy=True,
        ignore=["**/__pycache__/**"],
    )
    .add_local_file(ROOT / "pipelines" / "__init__.py", "/app/pipelines/__init__.py", copy=True)
    .add_local_file(ROOT / "pipelines" / "assets.py", "/app/pipelines/assets.py", copy=True)
    .add_local_file(
        ROOT / "pipelines" / "modal_lut_full.py", "/app/pipelines/modal_lut_full.py", copy=True
    )
    .add_local_file(
        ROOT / "pipelines" / "modal_lut_compare.py",
        "/app/pipelines/modal_lut_compare.py",
        copy=True,
    )
    .add_local_file(ROOT / "configs" / "final.json", "/app/configs/final.json", copy=True)
    .add_local_file(ROOT / "manifests" / "assets.json", "/app/manifests/assets.json", copy=True)
    .add_local_file(
        ROOT / "manifests" / "lut_blocks.json", "/app/manifests/lut_blocks.json", copy=True
    )
)
data_volume = modal.Volume.from_name("wikithesis-datos")
candidate_volume = modal.Volume.from_name("thesis-final-lut-candidates")
rclone_secret = modal.Secret.from_name("rclone-wikithesis")


def hash_file(path: Path) -> str:
    """Stream a SHA256 digest without loading a large dense LUT into memory."""
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(8 * 1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def generation_source_snapshot(root: Path) -> dict:
    """Recompute the exact source selection used by the candidate builder."""
    files = [
        Path("pipelines/__init__.py"),
        Path("pipelines/assets.py"),
        Path("pipelines/modal_lut_full.py"),
        Path("configs/final.json"),
        Path("manifests/assets.json"),
        Path("manifests/lut_blocks.json"),
    ]
    files.extend(
        path.relative_to(root)
        for path in sorted((root / "pipelines" / "generation").iterdir())
        if path.is_file() and path.suffix in (".py", ".cu")
    )
    hashes = {path.as_posix(): hash_file(root / path) for path in files}
    encoded = json.dumps(hashes, sort_keys=True, separators=(",", ":")).encode("utf-8")
    return {"sha256": hashlib.sha256(encoded).hexdigest(), "files": dict(sorted(hashes.items()))}


def planned_blocks(path: Path, gas: str) -> tuple[dict, ...]:
    """Validate and return all aligned 10 cm^-1 archived block records."""
    if gas not in GASES:
        raise ValueError(f"gas must be one of {GASES}")
    document = json.loads(path.read_text(encoding="utf-8"))
    grid = document.get("grid", {})
    if (
        document.get("schema_version") != 1
        or document.get("hash_algorithm") != "sha256"
        or grid.get("start_cm1") != 635
        or grid.get("stop_cm1") != 1265
        or grid.get("block_width_cm1") != 10
        or tuple(document.get("gases", ())) != GASES
        or document.get("total_blocks") != 315
        or len(document.get("blocks", ())) != 315
    ):
        raise ValueError("Unexpected frozen LUT block manifest")
    records = tuple(item for item in document["blocks"] if item.get("gas") == gas)
    if len(records) != 63:
        raise ValueError(f"Expected 63 {gas} blocks")
    for index, record in enumerate(records):
        start = 635 + 10 * index
        digest = record.get("sha256")
        if (
            record.get("lower_cm1") != start
            or record.get("upper_cm1") != start + 10
            or record.get("filename") != f"lut_{gas}_{start}_{start + 10}.npz"
            or not isinstance(record.get("bytes"), int)
            or record["bytes"] <= 0
            or not isinstance(digest, str)
            or len(digest) != 64
            or any(char not in "0123456789abcdef" for char in digest)
        ):
            raise ValueError(f"Invalid archived {gas} block at index {index}")
    return records


def sample_plan(
    nt: int = 20,
    npres: int = 145,
    block_count: int = 63,
    channels_per_block: int = 10_000,
    channels_sampled_per_block: int = 301,
    temperature_samples: int = 5,
    pressure_samples: int = 4,
) -> dict:
    """Choose reproducible physical states and channels across every block."""
    if (
        nt <= 0
        or npres <= 0
        or block_count <= 0
        or channels_per_block <= 0
        or not 1 <= channels_sampled_per_block <= channels_per_block
        or not 1 <= temperature_samples <= nt
        or not 1 <= pressure_samples <= npres
    ):
        raise ValueError("Invalid LUT sampling grid")
    temperature = np.linspace(0, nt - 1, temperature_samples, dtype=np.int64)
    pressure = np.linspace(0, npres - 1, pressure_samples, dtype=np.int64)
    local_channels = np.linspace(
        0, channels_per_block - 1, channels_sampled_per_block, dtype=np.int64
    )
    if (
        len(np.unique(temperature)) != temperature_samples
        or len(np.unique(pressure)) != pressure_samples
        or len(np.unique(local_channels)) != channels_sampled_per_block
    ):
        raise ValueError("Sampling grid contains duplicate indices")
    channel_spacing = 10.0 / channels_per_block
    retrieval_bounds = (
        [int(round((edge - 635.0) / channel_spacing)) for edge in (714.0, 1250.0)]
        if 635 + 10 * block_count >= 1250
        else None
    )
    return {
        "temperature_grid_size": nt,
        "pressure_grid_size": npres,
        "temperature_indices": temperature.tolist(),
        "pressure_indices": pressure.tolist(),
        "block_starts_cm1": list(range(635, 635 + 10 * block_count, 10)),
        "channels_per_block": channels_per_block,
        "local_channel_indices": local_channels.tolist(),
        "terminal_global_channel_index": block_count * channels_per_block,
        "retrieval_band_global_indices": retrieval_bounds,
        "channel_spacing_cm1": channel_spacing,
        "global_index_rule": "block_index * channels_per_block + local_channel_index",
        "state_count": temperature_samples * pressure_samples,
        "channel_count": block_count * channels_sampled_per_block + 1,
    }


def compare_sampled_dense(candidate_path: Path, archived_path: Path, selection: dict) -> dict:
    """Read selected NPY cells by mmap and assess each block and the full grid."""
    candidate = np.load(candidate_path, mmap_mode="r", allow_pickle=False)
    archived = np.load(archived_path, mmap_mode="r", allow_pickle=False)
    temperature = selection["temperature_indices"]
    pressure = selection["pressure_indices"]
    starts = selection["block_starts_cm1"]
    local = np.asarray(selection["local_channel_indices"], dtype=np.int64)
    width = selection["channels_per_block"]
    expected_shape = (
        selection["temperature_grid_size"],
        selection["pressure_grid_size"],
        selection["terminal_global_channel_index"] + 1,
    )
    if candidate.shape != expected_shape or archived.shape != expected_shape:
        raise ValueError(f"Dense LUT shape differs from expected {expected_shape}")
    if candidate.dtype != np.dtype("<f4") or archived.dtype != np.dtype("<f4"):
        raise ValueError("Dense LUT dtype must be little-endian float32")

    sum_squares = 0.0
    total = 0
    maximum = 0.0
    band_squares = 0.0
    band_total = 0
    band_maximum = 0.0
    band_bounds = selection.get("retrieval_band_global_indices")
    blocks = []
    for block_index, start in enumerate(starts):
        columns = block_index * width + local
        differences = []
        for t in temperature:
            for p in pressure:
                left = np.asarray(candidate[t, p, columns], dtype=np.float64)
                right = np.asarray(archived[t, p, columns], dtype=np.float64)
                differences.append(left - right)
        difference = np.stack(differences)
        if not np.isfinite(difference).all():
            raise ValueError(f"Nonfinite log10 cross section in block {start}")
        block_rms = float(np.sqrt(np.mean(np.square(difference))))
        block_max = float(np.max(np.abs(difference)))
        block_sum = float(np.square(difference).sum())
        sum_squares += block_sum
        total += difference.size
        maximum = max(maximum, block_max)
        if band_bounds is not None:
            within_band = (columns >= band_bounds[0]) & (columns <= band_bounds[1])
            selected = difference[:, within_band]
            if selected.size:
                band_squares += float(np.square(selected).sum())
                band_total += selected.size
                band_maximum = max(band_maximum, float(np.max(np.abs(selected))))
        blocks.append(
            {
                "lower_cm1": start,
                "upper_cm1": start + 10,
                "sample_count": int(difference.size),
                "rms_log10_sigma": block_rms,
                "max_abs_log10_sigma": block_max,
                "passed": block_rms <= MAX_RMS_LOG10_SIGMA and block_max <= MAX_ABS_LOG10_SIGMA,
            }
        )
    terminal = selection["terminal_global_channel_index"]
    terminal_samples = [
        {
            "temperature_index": int(t),
            "pressure_index": int(p),
            "candidate_log10_sigma": float(candidate[t, p, terminal]),
            "archive_log10_sigma": float(archived[t, p, terminal]),
        }
        for t in temperature
        for p in pressure
    ]
    for sample in terminal_samples:
        sample["difference_log10_sigma"] = (
            sample["candidate_log10_sigma"] - sample["archive_log10_sigma"]
        )
    terminal_difference = np.asarray(
        [sample["difference_log10_sigma"] for sample in terminal_samples], dtype=np.float64
    )
    if not np.isfinite(terminal_difference).all():
        raise ValueError("Nonfinite log10 cross section at terminal channel")
    sum_squares += float(np.square(terminal_difference).sum())
    total += terminal_difference.size
    terminal_max = float(np.max(np.abs(terminal_difference)))
    maximum = max(maximum, terminal_max)
    rms = math.sqrt(sum_squares / total)
    band_result = None
    if band_bounds is not None and band_total:
        boundary_samples = []
        for global_channel in band_bounds:
            for t in temperature:
                for p in pressure:
                    difference = float(candidate[t, p, global_channel]) - float(
                        archived[t, p, global_channel]
                    )
                    boundary_samples.append(difference)
        boundaries = np.asarray(boundary_samples, dtype=np.float64)
        if not np.isfinite(boundaries).all():
            raise ValueError("Nonfinite log10 cross section at retrieval band boundary")
        band_squares += float(np.square(boundaries).sum())
        band_total += boundaries.size
        band_maximum = max(band_maximum, float(np.max(np.abs(boundaries))))
        band_rms = math.sqrt(band_squares / band_total)
        band_result = {
            "global_channel_indices": band_bounds,
            "range_cm1": [714, 1250],
            "sample_count": band_total,
            "rms_log10_sigma": band_rms,
            "max_abs_log10_sigma": band_maximum,
            "boundary_max_abs_log10_sigma": float(np.max(np.abs(boundaries))),
            "passed": band_rms <= MAX_RMS_LOG10_SIGMA and band_maximum <= MAX_ABS_LOG10_SIGMA,
        }
    passed = (
        all(block["passed"] for block in blocks)
        and rms <= MAX_RMS_LOG10_SIGMA
        and maximum <= MAX_ABS_LOG10_SIGMA
    )
    return {
        "sample_count": total,
        "rms_log10_sigma": rms,
        "max_abs_log10_sigma": maximum,
        "terminal_max_abs_log10_sigma": terminal_max,
        "terminal_samples": terminal_samples,
        "retrieval_band": band_result,
        "blocks": blocks,
        "gate": {
            "passed": passed,
            "rms_log10_sigma_limit": MAX_RMS_LOG10_SIGMA,
            "max_abs_log10_sigma_limit": MAX_ABS_LOG10_SIGMA,
            "rule": "Every block RMS and maximum, and overall RMS and maximum, must pass",
        },
    }


def compare_full_dense(
    candidate_path: Path,
    archived_path: Path,
    *,
    nt: int = 20,
    npres: int = 145,
    block_count: int = 63,
    channels_per_block: int = 10_000,
    chunk_channels: int = 100_000,
    spectral_start_cm1: float = 635.0,
    channel_spacing_cm1: float = 0.001,
    retrieval_band_cm1: tuple[float, float] = (714.0, 1250.0),
) -> dict:
    """Compare every float32 cell using bounded row and channel chunks.

    Each block owns its lower boundary and excludes its upper boundary. The
    final 1265 cm^-1 sample is counted separately, so every dense cell is read
    exactly once. The retrieval band includes both 714 and 1250 cm^-1.
    """
    if (
        nt <= 0
        or npres <= 0
        or block_count <= 0
        or channels_per_block <= 0
        or chunk_channels <= 0
        or chunk_channels % channels_per_block
        or channel_spacing_cm1 <= 0
    ):
        raise ValueError("Invalid full-scan grid or chunk size")
    terminal_index = block_count * channels_per_block
    terminal_cm1 = spectral_start_cm1 + terminal_index * channel_spacing_cm1

    def channel_index(wavenumber: float) -> int:
        index = round((wavenumber - spectral_start_cm1) / channel_spacing_cm1)
        if not math.isclose(
            wavenumber,
            spectral_start_cm1 + index * channel_spacing_cm1,
            rel_tol=0.0,
            abs_tol=1e-9,
        ):
            raise ValueError(f"Band edge {wavenumber} is not on the dense LUT grid")
        return index

    band_first, band_last = (channel_index(edge) for edge in retrieval_band_cm1)
    if not 0 <= band_first < band_last < terminal_index:
        raise ValueError("Retrieval band must lie inside the half-open block grid")

    candidate = np.load(candidate_path, mmap_mode="r", allow_pickle=False)
    archived = np.load(archived_path, mmap_mode="r", allow_pickle=False)
    expected_shape = (nt, npres, terminal_index + 1)
    if candidate.shape != expected_shape or archived.shape != expected_shape:
        raise ValueError(f"Dense LUT shape differs from expected {expected_shape}")
    if candidate.dtype != np.dtype("<f4") or archived.dtype != np.dtype("<f4"):
        raise ValueError("Dense LUT dtype must be little-endian float32")

    block_squares = np.zeros(block_count, dtype=np.float64)
    block_maxima = np.zeros(block_count, dtype=np.float64)
    band_squares = 0.0
    band_maximum = 0.0
    terminal_squares = 0.0
    terminal_maximum = 0.0
    for t in range(nt):
        for p in range(npres):
            for first in range(0, terminal_index, chunk_channels):
                last = min(first + chunk_channels, terminal_index)
                difference = np.asarray(candidate[t, p, first:last], dtype=np.float64) - np.asarray(
                    archived[t, p, first:last], dtype=np.float64
                )
                if not np.isfinite(difference).all():
                    raise ValueError(
                        f"Nonfinite log10 cross section at state {(t, p)}, {first}:{last}"
                    )
                block_view = difference.reshape(-1, channels_per_block)
                block_first = first // channels_per_block
                block_last = block_first + block_view.shape[0]
                block_squares[block_first:block_last] += np.einsum(
                    "ij,ij->i", block_view, block_view
                )
                block_maxima[block_first:block_last] = np.maximum(
                    block_maxima[block_first:block_last], np.max(np.abs(block_view), axis=1)
                )
                within_first = max(first, band_first)
                within_last = min(last, band_last + 1)
                if within_first < within_last:
                    segment = difference[within_first - first : within_last - first]
                    band_squares += float(np.dot(segment, segment))
                    band_maximum = max(band_maximum, float(np.max(np.abs(segment))))
            terminal_difference = float(candidate[t, p, terminal_index]) - float(
                archived[t, p, terminal_index]
            )
            if not math.isfinite(terminal_difference):
                raise ValueError(f"Nonfinite log10 cross section at terminal state {(t, p)}")
            terminal_squares += terminal_difference**2
            terminal_maximum = max(terminal_maximum, abs(terminal_difference))

    state_count = nt * npres
    block_samples = state_count * channels_per_block
    blocks = []
    for index in range(block_count):
        rms = math.sqrt(float(block_squares[index]) / block_samples)
        maximum = float(block_maxima[index])
        lower = spectral_start_cm1 + index * channels_per_block * channel_spacing_cm1
        upper = lower + channels_per_block * channel_spacing_cm1
        blocks.append(
            {
                "lower_cm1": lower,
                "upper_cm1": upper,
                "global_channel_indices": [
                    index * channels_per_block,
                    (index + 1) * channels_per_block - 1,
                ],
                "sample_count": block_samples,
                "rms_log10_sigma": rms,
                "max_abs_log10_sigma": maximum,
                "passed": rms <= MAX_RMS_LOG10_SIGMA and maximum <= MAX_ABS_LOG10_SIGMA,
            }
        )
    band_count = state_count * (band_last - band_first + 1)
    band_rms = math.sqrt(band_squares / band_count)
    band_passed = band_rms <= MAX_RMS_LOG10_SIGMA and band_maximum <= MAX_ABS_LOG10_SIGMA
    terminal_rms = math.sqrt(terminal_squares / state_count)
    terminal_passed = (
        terminal_rms <= MAX_RMS_LOG10_SIGMA and terminal_maximum <= MAX_ABS_LOG10_SIGMA
    )
    total_count = state_count * (terminal_index + 1)
    total_rms = math.sqrt((float(block_squares.sum()) + terminal_squares) / total_count)
    total_maximum = max(float(block_maxima.max()), terminal_maximum)
    passed = (
        all(block["passed"] for block in blocks)
        and band_passed
        and terminal_passed
        and total_rms <= MAX_RMS_LOG10_SIGMA
        and total_maximum <= MAX_ABS_LOG10_SIGMA
    )
    return {
        "scan": {
            "mode": "all_dense_float32_cells",
            "shape": list(expected_shape),
            "dtype": "<f4",
            "chunk_channels": chunk_channels,
            "endpoint_policy": "Each block is half-open; terminal channel is counted separately",
        },
        "sample_count": total_count,
        "rms_log10_sigma": total_rms,
        "max_abs_log10_sigma": total_maximum,
        "blocks": blocks,
        "retrieval_band": {
            "range_cm1": list(retrieval_band_cm1),
            "global_channel_indices_inclusive": [band_first, band_last],
            "sample_count": band_count,
            "rms_log10_sigma": band_rms,
            "max_abs_log10_sigma": band_maximum,
            "passed": band_passed,
        },
        "terminal_channel": {
            "wavenumber_cm1": terminal_cm1,
            "global_channel_index": terminal_index,
            "sample_count": state_count,
            "rms_log10_sigma": terminal_rms,
            "max_abs_log10_sigma": terminal_maximum,
            "passed": terminal_passed,
        },
        "gate": {
            "passed": passed,
            "rms_log10_sigma_limit": MAX_RMS_LOG10_SIGMA,
            "max_abs_log10_sigma_limit": MAX_ABS_LOG10_SIGMA,
            "rule": "Every block, exact retrieval band, terminal channel, and entire array must pass",
        },
    }


def verify_candidate(
    root: Path, gas: str, records: tuple[dict, ...], assets: dict, snapshot: dict
) -> dict:
    """Check the dense receipt and all 63 source block receipts and bytes."""
    base = root / gas
    report_path = base / "report.json"
    report = json.loads(report_path.read_text(encoding="utf-8"))
    input_hashes = {identifier: assets[identifier].sha256 for identifier in PACK_IDS}
    dense_asset = assets[f"lut_{gas.lower()}"]
    dense_path = base / "dense" / f"lut_band645_{gas}.npy"
    if (
        report.get("schema_version") != 1
        or report.get("status") != "candidate_full_gas_unvalidated"
        or report.get("gas") != gas
        or report.get("range_cm1") != [635, 1265]
        or report.get("block_count") != 63
        or report.get("source_snapshot") != snapshot
        or report.get("input_sha256") != input_hashes
        or report.get("archived_dense_sha256") != dense_asset.sha256
        or report.get("dense_bytes") != dense_path.stat().st_size
    ):
        raise ValueError("Candidate report metadata or source provenance differs")
    dense_hash = hash_file(dense_path)
    if (
        report.get("dense_sha256") != dense_hash
        or report.get("expansion_report", {}).get("sha256") != dense_hash
        or report.get("expansion_report", {}).get("shape") != [20, 145, 630001]
        or report.get("expansion_report", {}).get("blocks") != 63
    ):
        raise ValueError("Candidate dense LUT differs from expansion report")
    block_hashes = {}
    receipt_hashes = {}
    for record in records:
        name = record["filename"]
        path = base / "blocks" / name
        receipt_path = path.with_suffix(".json")
        receipt = json.loads(receipt_path.read_text(encoding="utf-8"))
        digest = hash_file(path)
        if (
            receipt.get("schema_version") != 1
            or receipt.get("status") != "candidate_historical_source_unverified"
            or receipt.get("gas") != gas
            or receipt.get("start_cm1") != record["lower_cm1"]
            or receipt.get("end_cm1") != record["upper_cm1"]
            or receipt.get("source_snapshot") != snapshot
            or receipt.get("input_sha256") != input_hashes
            or receipt.get("candidate_sha256") != digest
            or receipt.get("candidate_bytes") != path.stat().st_size
            or receipt.get("archived_reference") != record
            or receipt.get("builder_report", {}).get("blocks", [{}])[0].get("sha256") != digest
            or receipt.get("inspection", {}).get("gas") != gas
            or receipt.get("inspection", {}).get("blocks") != 1
            or receipt.get("inspection", {}).get("range_cm1")
            != [record["lower_cm1"], record["upper_cm1"]]
            or receipt.get("inspection", {}).get("shape") != [20, 145, 10001]
            or receipt.get("inspection", {}).get("rank") != 64
        ):
            raise ValueError(f"Candidate block or provenance differs: {name}")
        block_hashes[name] = digest
        receipt_hashes[receipt_path.name] = hash_file(receipt_path)
    if report.get("block_sha256") != block_hashes:
        raise ValueError("Candidate report block hashes differ from verified blocks")
    return {
        "report_sha256": hash_file(report_path),
        "dense_sha256": dense_hash,
        "dense_bytes": dense_path.stat().st_size,
        "block_sha256": block_hashes,
        "receipt_sha256": receipt_hashes,
        "input_sha256": input_hashes,
        "source_snapshot": snapshot,
        "dense_path": dense_path,
    }


def _setup_remote() -> None:
    encoded = os.environ.get("RCLONE_CONF_B64")
    plain = os.environ.get("RCLONE_CONF")
    if not encoded and not plain:
        raise RuntimeError("Modal secret must supply RCLONE_CONF_B64 or RCLONE_CONF")
    directory = Path.home() / ".config" / "rclone"
    directory.mkdir(parents=True, exist_ok=True)
    (directory / "rclone.conf").write_bytes(
        base64.b64decode(encoded) if encoded else plain.encode("utf-8")
    )


@app.function(
    image=image,
    cpu=8.0,
    memory=65536,
    timeout=3600,
    volumes={str(CACHE_ROOT): data_volume, str(CANDIDATE_ROOT): candidate_volume},
    secrets=[rclone_secret],
)
def run_comparison(gas: str, full_scan: bool = False) -> dict:
    """Verify both volumes and return a local-saveable comparison report."""
    import sys

    sys.path.insert(0, str(APP_ROOT))
    from pipelines.assets import load_manifest, materialize, verify_file

    data_volume.reload()
    candidate_volume.reload()
    records = planned_blocks(APP_ROOT / "manifests/lut_blocks.json", gas)
    manifest_path = APP_ROOT / "manifests/assets.json"
    artifacts = {item.identifier: item for item in load_manifest(manifest_path)}
    snapshot = generation_source_snapshot(APP_ROOT)
    candidate = verify_candidate(CANDIDATE_ROOT, gas, records, artifacts, snapshot)
    _setup_remote()
    reference_asset = artifacts[f"lut_{gas.lower()}"]
    # Reuse a verified legacy cache copy; isolate any stale file from the pinned download.
    legacy_reference = CACHE_ROOT / "outputs" / f"lut_band645_{gas}.npy"
    stale_legacy_sha256 = None
    if legacy_reference.is_file():
        try:
            verify_file(legacy_reference, reference_asset)
            reference_path = legacy_reference
        except ValueError:
            stale_legacy_sha256 = hash_file(legacy_reference)
            reference_path = materialize(reference_asset, CACHE_ROOT, None, "auto", APP_ROOT)
    else:
        reference_path = materialize(reference_asset, CACHE_ROOT, None, "auto", APP_ROOT)
    verify_file(reference_path, reference_asset)
    data_volume.commit()
    selection = sample_plan()
    comparison = compare_sampled_dense(candidate["dense_path"], reference_path, selection)
    full_comparison = (
        compare_full_dense(candidate["dense_path"], reference_path) if full_scan else None
    )
    del candidate["dense_path"]
    result = {
        "schema_version": 1,
        "status": "candidate_full_gas_sampled_comparison",
        "gas": gas,
        "candidate": candidate,
        "archive": {
            "sha256": reference_asset.sha256,
            "bytes": reference_asset.size,
            "volume_path": str(reference_path),
            "ignored_stale_legacy_sha256": stale_legacy_sha256,
            "manifest_sha256": hash_file(manifest_path),
            "block_manifest_sha256": hash_file(APP_ROOT / "manifests/lut_blocks.json"),
        },
        "sample_selection": selection,
        "comparison": comparison,
    }
    if full_scan:
        result["status"] = "candidate_full_gas_full_scan_comparison"
        result["full_comparison"] = full_comparison
    return result


@app.local_entrypoint()
def main(gas: str = "CO2", output: str | None = None, full_scan: bool = False) -> None:
    """Run the read-only remote verifier and save its complete report locally."""
    if gas not in GASES:
        raise ValueError(f"gas must be one of {GASES}")
    result = run_comparison.remote(gas, full_scan)
    default_name = (
        f"runs/lut_full_scan_{gas.lower()}.json"
        if full_scan
        else f"runs/lut_full_compare_{gas.lower()}.json"
    )
    destination = Path(output or default_name)
    destination.parent.mkdir(parents=True, exist_ok=True)
    with destination.open("x", encoding="utf-8") as stream:
        json.dump(result, stream, indent=2, sort_keys=True)
        stream.write("\n")
    summary = result["full_comparison"] if full_scan else result["comparison"]
    print(
        json.dumps(
            {
                "report": str(destination),
                "gas": gas,
                "mode": "full_scan" if full_scan else "sampled",
                "rms_log10_sigma": summary["rms_log10_sigma"],
                "max_abs_log10_sigma": summary["max_abs_log10_sigma"],
                "passed": summary["gate"]["passed"],
            },
            sort_keys=True,
        )
    )
    if not summary["gate"]["passed"]:
        raise RuntimeError("Full candidate gas exceeds the fixed numerical limits")
