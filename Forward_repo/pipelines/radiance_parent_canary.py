"""Verify the original full-band radiance with range reads or a local full file."""

from __future__ import annotations

import argparse
import hashlib
import io
import json
import subprocess
import zipfile
from pathlib import Path
from typing import Any

import numpy as np


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_MANIFEST = ROOT / "manifests" / "radiance_parent.json"
DEFAULT_ARCHIVE = ROOT / "data_cache" / "data" / "outputs" / "radiances_iasiang.npz"


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(8 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _rclone(binary: str, *args: str) -> bytes:
    result = subprocess.run([binary, *args], capture_output=True, check=False)
    if result.returncode:
        detail = result.stderr.decode("utf-8", errors="replace")[-1200:]
        raise RuntimeError(f"rclone {' '.join(args[:2])} failed: {detail}")
    return result.stdout


def _read_range(binary: str, uri: str, offset: int, count: int) -> bytes:
    data = _rclone(binary, "cat", "--offset", str(offset), "--count", str(count), uri)
    if len(data) != count:
        raise ValueError(f"Short range read at {offset}: expected {count} bytes, got {len(data)}")
    return data


def _check_header(data: bytes, layout: dict[str, Any]) -> None:
    stream = io.BytesIO(data)
    if np.lib.format.read_magic(stream) != (1, 0):
        raise ValueError("Source must have a NumPy v1.0 header")
    shape, fortran_order, dtype = np.lib.format.read_array_header_1_0(stream)
    if tuple(shape) != tuple(layout["shape"]):
        raise ValueError(f"Source shape {shape} differs from manifest {layout['shape']}")
    if fortran_order or dtype.str != layout["dtype"]:
        raise ValueError("Source array storage order or dtype differs from manifest")
    if stream.tell() != layout["npy_header_bytes"]:
        raise ValueError("Source header length differs from manifest")


def verify_parent(
    manifest_path: Path = DEFAULT_MANIFEST,
    archive_path: Path = DEFAULT_ARCHIVE,
    rclone: str = "rclone",
) -> dict[str, Any]:
    """Check Drive identity, full angle selection, and four byte-exact row canaries."""
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    source = manifest["source"]
    metadata = manifest["source_metadata"]
    layout = manifest["layout"]
    frozen = manifest["frozen_radiance"]
    if manifest["schema_version"] != 1:
        raise ValueError("Unsupported radiance parent manifest version")

    remote = json.loads(_rclone(rclone, "lsjson", "--hash", "--stat", source["remote_uri"]))
    if remote["Size"] != source["bytes"]:
        raise ValueError("Remote source byte count differs from manifest")
    if remote.get("Hashes", {}).get("sha256", "").lower() != source["sha256"]:
        raise ValueError("Remote source SHA256 metadata differs from manifest")
    header = _read_range(rclone, source["remote_uri"], 0, layout["npy_header_bytes"])
    _check_header(header, layout)

    meta_bytes = _rclone(rclone, "cat", metadata["remote_uri"])
    if len(meta_bytes) != metadata["bytes"]:
        raise ValueError("Source metadata byte count differs from manifest")
    if hashlib.sha256(meta_bytes).hexdigest() != metadata["sha256"]:
        raise ValueError("Source metadata SHA256 differs from manifest")
    with np.load(io.BytesIO(meta_bytes), allow_pickle=False) as meta:
        full_channels = np.asarray(meta["nu"])
        angles = np.asarray(meta["mus"])
        if len(full_channels) != layout["shape"][2]:
            raise ValueError("Source metadata channel count differs from source")
        if not np.array_equal(angles, np.asarray(layout["angles"])):
            raise ValueError("Source viewing angles differ from manifest")

    if archive_path.stat().st_size != frozen["bytes"]:
        raise ValueError("Frozen radiance archive byte count differs from manifest")
    archive_sha256 = _sha256_file(archive_path)
    if archive_sha256 != frozen["sha256"]:
        raise ValueError("Frozen radiance archive SHA256 differs from manifest")

    n_profiles, n_angles, n_channels = layout["shape"]
    start, count = layout["channel_start"], layout["channel_count"]
    if start < 0 or count <= 0 or start + count > n_channels:
        raise ValueError("Invalid source channel crop")
    expected_size = np.dtype(layout["dtype"]).itemsize * count
    selected = np.random.default_rng(layout["selection_seed"]).integers(
        0, n_angles, size=n_profiles
    )
    with np.load(archive_path, allow_pickle=False) as archive:
        radiance = np.asarray(archive[frozen["array"]])
        mu = np.asarray(archive["mu"])
        nu = np.asarray(archive["nu"])
        if radiance.shape != (n_profiles, count) or mu.shape != (n_profiles,):
            raise ValueError("Frozen radiance or viewing-angle shape differs from manifest")
        if not np.array_equal(nu, full_channels[start : start + count]):
            raise ValueError("Frozen channel axis differs from source crop")
        expected_mu = angles[selected].astype(mu.dtype)
        if not np.array_equal(mu, expected_mu):
            raise ValueError("Frozen viewing-angle assignment differs from seed 991")

        rows = []
        for canary in manifest["canaries"]:
            profile = canary["profile_index"]
            angle = canary["angle_index"]
            if not (0 <= profile < n_profiles and angle == selected[profile]):
                raise ValueError("Canary row or selected angle differs from manifest")
            offset = (
                layout["npy_header_bytes"]
                + ((profile * n_angles + angle) * n_channels + start)
                * np.dtype(layout["dtype"]).itemsize
            )
            parent_bytes = _read_range(rclone, source["remote_uri"], offset, expected_size)
            row_sha256 = hashlib.sha256(parent_bytes).hexdigest()
            if row_sha256 != canary["row_sha256"]:
                raise ValueError(f"Parent row {profile} SHA256 differs from manifest")
            equal = parent_bytes == radiance[profile].tobytes()
            rows.append(
                {
                    "profile_index": profile,
                    "angle_index": angle,
                    "row_sha256": row_sha256,
                    "byte_equal_to_frozen": equal,
                }
            )
    return {
        "status": "pass" if all(row["byte_equal_to_frozen"] for row in rows) else "fail",
        "source_uri": source["remote_uri"],
        "source_sha256": source["sha256"],
        "source_hash_evidence": "remote_metadata",
        "source_metadata_sha256": metadata["sha256"],
        "frozen_radiance_sha256": archive_sha256,
        "angle_assignment_all_rows_equal": True,
        "canaries": rows,
    }


def verify_full_parent(
    parent_path: Path,
    manifest_path: Path = DEFAULT_MANIFEST,
    archive_path: Path = DEFAULT_ARCHIVE,
    batch_size: int = 256,
) -> dict[str, Any]:
    """Compare every frozen radiance value to a SHA-verified, memory-mapped parent."""
    if batch_size < 1:
        raise ValueError("Batch size must be positive")
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    if manifest["schema_version"] != 1:
        raise ValueError("Unsupported radiance parent manifest version")
    source = manifest["source"]
    layout = manifest["layout"]
    frozen = manifest["frozen_radiance"]
    n_profiles, n_angles, n_channels = layout["shape"]
    start, count = layout["channel_start"], layout["channel_count"]
    dtype = np.dtype(layout["dtype"])
    if start < 0 or count <= 0 or start + count > n_channels:
        raise ValueError("Invalid source channel crop")

    if parent_path.stat().st_size != source["bytes"]:
        raise ValueError("Local full parent byte count differs from manifest")
    parent_sha256 = _sha256_file(parent_path)
    if parent_sha256 != source["sha256"]:
        raise ValueError("Local full parent SHA256 differs from manifest")
    with parent_path.open("rb") as stream:
        _check_header(stream.read(layout["npy_header_bytes"]), layout)
    parent = np.load(parent_path, mmap_mode="r", allow_pickle=False)

    if archive_path.stat().st_size != frozen["bytes"]:
        raise ValueError("Frozen radiance archive byte count differs from manifest")
    archive_sha256 = _sha256_file(archive_path)
    if archive_sha256 != frozen["sha256"]:
        raise ValueError("Frozen radiance archive SHA256 differs from manifest")
    selected = np.random.default_rng(layout["selection_seed"]).integers(
        0, n_angles, size=n_profiles
    )
    with np.load(archive_path, allow_pickle=False) as archive:
        mu = np.asarray(archive["mu"])
        nu = np.asarray(archive["nu"])
    if mu.shape != (n_profiles,) or nu.shape != (count,):
        raise ValueError("Frozen viewing-angle or channel-axis shape differs from manifest")
    angles = np.asarray(layout["angles"])
    if angles.shape != (n_angles,) or not np.array_equal(mu, angles[selected].astype(mu.dtype)):
        raise ValueError("Frozen viewing-angle assignment differs from manifest seed")

    unequal_count = 0
    sum_squared_error = 0.0
    max_absolute_error = 0.0
    batches = 0
    with zipfile.ZipFile(archive_path) as container:
        with container.open(f"{frozen['array']}.npy") as stream:
            if np.lib.format.read_magic(stream) != (1, 0):
                raise ValueError("Frozen radiance array must have a NumPy v1.0 header")
            shape, fortran_order, archive_dtype = np.lib.format.read_array_header_1_0(stream)
            if shape != (n_profiles, count) or fortran_order or archive_dtype != dtype:
                raise ValueError("Frozen radiance array layout differs from manifest")
            for first in range(0, n_profiles, batch_size):
                last = min(first + batch_size, n_profiles)
                rows = last - first
                raw = stream.read(rows * count * dtype.itemsize)
                if len(raw) != rows * count * dtype.itemsize:
                    raise ValueError(f"Frozen radiance ended early at profile {first}")
                frozen_rows = np.frombuffer(raw, dtype=dtype).reshape(rows, count)
                profiles = np.arange(first, last)
                parent_rows = parent[profiles, selected[first:last], start : start + count]
                mismatched = frozen_rows.view(np.uint32) != parent_rows.view(np.uint32)
                changed = int(np.count_nonzero(mismatched))
                unequal_count += changed
                if changed:
                    difference = frozen_rows.astype(np.float64) - parent_rows.astype(np.float64)
                    if not np.isfinite(difference).all():
                        raise ValueError("Non-finite difference in frozen or source radiance")
                    sum_squared_error += float(np.square(difference).sum(dtype=np.float64))
                    max_absolute_error = max(max_absolute_error, float(np.abs(difference).max()))
                batches += 1
            if stream.read(1):
                raise ValueError("Frozen radiance array contains trailing payload bytes")
    value_count = n_profiles * count
    return {
        "status": "pass" if unequal_count == 0 else "fail",
        "mode": "full_parent",
        "source_path": str(parent_path),
        "source_sha256": parent_sha256,
        "source_hash_evidence": "local_file",
        "frozen_radiance_sha256": archive_sha256,
        "angle_assignment_all_rows_equal": True,
        "profile_count": n_profiles,
        "channel_count": count,
        "value_count": value_count,
        "unequal_count": unequal_count,
        "rms_difference": (sum_squared_error / value_count) ** 0.5,
        "max_absolute_difference": max_absolute_error,
        "batch_size": batch_size,
        "batches": batches,
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", type=Path, default=DEFAULT_MANIFEST)
    parser.add_argument("--archive", type=Path, default=DEFAULT_ARCHIVE)
    parser.add_argument("--rclone", default="rclone")
    parser.add_argument("--full-parent", type=Path)
    parser.add_argument("--batch-size", type=int, default=256)
    parser.add_argument("--report", type=Path)
    args = parser.parse_args()
    try:
        report = (
            verify_full_parent(args.full_parent, args.manifest, args.archive, args.batch_size)
            if args.full_parent
            else verify_parent(args.manifest, args.archive, args.rclone)
        )
    except (OSError, ValueError, RuntimeError, KeyError, zipfile.BadZipFile) as exc:
        parser.exit(2, f"Radiance parent verification failed: {exc}\n")
    rendered = json.dumps(report, indent=2) + "\n"
    if args.report:
        args.report.parent.mkdir(parents=True, exist_ok=True)
        args.report.write_text(rendered, encoding="utf-8")
    print(rendered, end="")
    return 0 if report["status"] == "pass" else 1


if __name__ == "__main__":
    raise SystemExit(main())
