"""Materialize and verify the frozen assets used by the final retrieval pipeline."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import shutil
import subprocess
import sys
import tempfile
from dataclasses import dataclass
from pathlib import Path, PurePosixPath
from typing import Any


DEFAULT_MANIFEST = Path(__file__).resolve().parents[1] / "manifests" / "assets.json"


@dataclass(frozen=True)
class Artifact:
    """One immutable file with a verified destination and at least one source."""

    identifier: str
    destination: PurePosixPath
    size: int
    sha256: str
    remote_uri: str | None
    workspace_path: PurePosixPath | None
    bundle_path: PurePosixPath | None
    role: str

    @classmethod
    def from_record(cls, record: dict[str, Any]) -> Artifact:
        destination = _relative_path(record["destination"])
        workspace_path = record.get("workspace_path")
        sha256 = str(record["sha256"]).lower()
        if len(sha256) != 64 or any(char not in "0123456789abcdef" for char in sha256):
            raise ValueError(f"Invalid SHA256 for {record['id']}")
        size = int(record["bytes"])
        if size < 0:
            raise ValueError(f"Invalid byte count for {record['id']}")
        remote_uri = record.get("remote_uri")
        if remote_uri is not None and (":" not in remote_uri or not remote_uri.strip()):
            raise ValueError(f"Invalid remote URI for {record['id']}")
        bundle_path = record.get("bundle_path")
        if remote_uri is None and workspace_path is None and bundle_path is None:
            raise ValueError(f"No source for {record['id']}")
        return cls(
            identifier=str(record["id"]),
            destination=destination,
            size=size,
            sha256=sha256,
            remote_uri=remote_uri,
            workspace_path=_relative_path(workspace_path) if workspace_path else None,
            bundle_path=_relative_path(bundle_path) if bundle_path else None,
            role=str(record["role"]),
        )


def _relative_path(raw: str) -> PurePosixPath:
    path = PurePosixPath(raw)
    if path.is_absolute() or not path.parts or any(part in (".", "..") for part in path.parts):
        raise ValueError(f"Expected a safe relative path: {raw!r}")
    if ":" in path.parts[0] or "\\" in raw:
        raise ValueError(f"Expected a portable POSIX path: {raw!r}")
    return path


def load_manifest(path: Path) -> list[Artifact]:
    """Read a manifest and reject duplicate IDs or destinations."""
    document = json.loads(path.read_text(encoding="utf-8"))
    if document.get("schema_version") != 1 or document.get("hash_algorithm") != "sha256":
        raise ValueError("Unsupported asset manifest schema")
    artifacts = [Artifact.from_record(record) for record in document["artifacts"]]
    ids = [artifact.identifier for artifact in artifacts]
    destinations = [artifact.destination for artifact in artifacts]
    if len(set(ids)) != len(ids) or len(set(destinations)) != len(destinations):
        raise ValueError("Duplicate asset ID or destination")
    return artifacts


def select_artifacts(artifacts: list[Artifact], identifiers: list[str]) -> list[Artifact]:
    if not identifiers:
        return artifacts
    by_id = {artifact.identifier: artifact for artifact in artifacts}
    unknown = sorted(set(identifiers) - set(by_id))
    if unknown:
        raise ValueError(f"Unknown assets: {', '.join(unknown)}")
    return [by_id[identifier] for identifier in identifiers]


def hash_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(8 * 1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def verify_file(path: Path, artifact: Artifact) -> None:
    """Raise if the file size or SHA256 differs from the frozen manifest."""
    if not path.is_file():
        raise FileNotFoundError(path)
    size = path.stat().st_size
    if size != artifact.size:
        raise ValueError(f"{artifact.identifier}: size {size} != {artifact.size}")
    digest = hash_file(path)
    if digest != artifact.sha256:
        raise ValueError(f"{artifact.identifier}: SHA256 {digest} != {artifact.sha256}")


def verify_remote(artifact: Artifact) -> None:
    """Compare remote metadata with the manifest without downloading file bytes."""
    if artifact.remote_uri is None:
        raise ValueError(f"{artifact.identifier} has no remote URI")
    result = subprocess.run(
        ["rclone", "--log-level", "ERROR", "lsjson", "--hash", "--stat", artifact.remote_uri],
        check=True,
        capture_output=True,
        text=True,
    )
    metadata = json.loads(result.stdout)
    if int(metadata["Size"]) != artifact.size:
        raise ValueError(f"{artifact.identifier}: remote size differs")
    digest = metadata.get("Hashes", {}).get("sha256")
    if digest is None:
        raise ValueError(f"{artifact.identifier}: remote SHA256 unavailable")
    if digest.lower() != artifact.sha256:
        raise ValueError(f"{artifact.identifier}: remote SHA256 differs")


def materialize(
    artifact: Artifact,
    cache_root: Path,
    workspace_root: Path | None,
    source: str,
    bundle_root: Path | None = None,
) -> Path:
    """Copy an asset to the cache atomically and validate its bytes."""
    destination = cache_root.joinpath(*artifact.destination.parts)
    if destination.exists():
        verify_file(destination, artifact)
        return destination

    workspace_source = (
        workspace_root.joinpath(*artifact.workspace_path.parts)
        if workspace_root is not None and artifact.workspace_path is not None
        else None
    )
    bundle_source = (
        bundle_root.joinpath(*artifact.bundle_path.parts)
        if bundle_root is not None and artifact.bundle_path is not None
        else None
    )
    if source == "bundle" or (
        source == "auto" and bundle_source is not None and bundle_source.is_file()
    ):
        local_source = bundle_source
    elif source == "workspace" or (
        source == "auto" and workspace_source is not None and workspace_source.is_file()
    ):
        local_source = workspace_source
    else:
        local_source = None
    if local_source is not None:
        verify_file(local_source, artifact)
    elif source not in ("auto", "remote") or artifact.remote_uri is None:
        raise ValueError(f"{artifact.identifier} has no usable source")

    destination.parent.mkdir(parents=True, exist_ok=True)
    fd, temporary_name = tempfile.mkstemp(prefix=f".{destination.name}.", dir=destination.parent)
    os.close(fd)
    temporary = Path(temporary_name)
    try:
        if local_source is not None:
            shutil.copyfile(local_source, temporary)
        else:
            subprocess.run(["rclone", "copyto", artifact.remote_uri, str(temporary)], check=True)
        verify_file(temporary, artifact)
        os.replace(temporary, destination)
    finally:
        temporary.unlink(missing_ok=True)
    return destination


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", type=Path, default=DEFAULT_MANIFEST)
    parser.add_argument("--cache", type=Path, default=Path("data_cache"))
    parser.add_argument("--workspace-root", type=Path)
    subcommands = parser.add_subparsers(dest="command", required=True)
    for name in ("list", "verify", "fetch", "verify-remote"):
        command = subcommands.add_parser(name)
        command.add_argument("ids", nargs="*")
        if name == "fetch":
            command.add_argument(
                "--source", choices=("auto", "bundle", "workspace", "remote"), default="auto"
            )
            command.add_argument("--dry-run", action="store_true")
    args = parser.parse_args(argv)
    try:
        artifacts = select_artifacts(load_manifest(args.manifest), args.ids)
        for artifact in artifacts:
            destination = args.cache.joinpath(*artifact.destination.parts)
            if args.command == "list":
                source = artifact.bundle_path or artifact.remote_uri or artifact.workspace_path
                print(f"{artifact.identifier:26} {artifact.size:>12}  {source}")
            elif args.command == "verify":
                verify_file(destination, artifact)
                print(f"PASS {artifact.identifier}: {destination}")
            elif args.command == "verify-remote":
                if artifact.remote_uri is None:
                    print(f"SKIP {artifact.identifier}: no remote source")
                    continue
                verify_remote(artifact)
                print(f"PASS {artifact.identifier}: {artifact.remote_uri}")
            elif args.dry_run:
                print(f"WOULD FETCH {artifact.identifier}: {destination}")
            else:
                result = materialize(
                    artifact,
                    args.cache,
                    args.workspace_root,
                    args.source,
                    args.manifest.resolve().parent.parent,
                )
                print(f"PASS {artifact.identifier}: {result}")
    except (FileNotFoundError, ValueError, subprocess.CalledProcessError) as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
