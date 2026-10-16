"""Fetch the 64 frozen ocean scenes with per-file byte verification."""

from __future__ import annotations

import argparse
import json
import os
import re
import subprocess
import sys
import tempfile
from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path

from pipelines.assets import hash_file


DEFAULT_MANIFEST = Path(__file__).resolve().parents[1] / "manifests" / "mar64_inputs.json"
DEFAULT_SCENE_DIR = Path("data_cache/data/valstates_full")


@dataclass(frozen=True)
class SceneFile:
    """One frozen scene and its remote source."""

    name: str
    size: int
    sha256: str
    remote_uri: str


def _remote_directory(value: object) -> str:
    if not isinstance(value, str):
        raise ValueError("Missing ocean scene remote directory")
    remote, separator, directory = value.partition(":")
    if (
        not separator
        or not re.fullmatch(r"[A-Za-z0-9_-]+", remote)
        or not directory
        or any(part in ("", ".", "..") for part in directory.split("/"))
        or any(char == "\\" or ord(char) < 32 for char in directory)
    ):
        raise ValueError("Unsafe ocean scene remote directory")
    return value


def load_scene_manifest(manifest_path: Path = DEFAULT_MANIFEST) -> list[SceneFile]:
    """Validate the exact 64-file inventory before accessing the destination."""
    document = json.loads(manifest_path.read_text(encoding="utf-8"))
    selection = document.get("selection", {})
    if (
        document.get("schema_version") != 1
        or document.get("dataset") != "iasi_ocean_64"
        or selection.get("scene_count") != 64
        or selection.get("glob") != "*.npz"
        or selection.get("sort") != "lexicographic_filename"
    ):
        raise ValueError("Unsupported ocean scene manifest")
    remote_directory = _remote_directory(document.get("remote_scene_directory"))
    records = document.get("scene_files")
    if not isinstance(records, list) or len(records) != 64:
        raise ValueError("Ocean scene manifest must contain 64 files")
    scenes = []
    for record in records:
        if not isinstance(record, dict):
            raise ValueError("Invalid ocean scene record")
        name, size, digest = (
            record.get("name"),
            record.get("bytes"),
            record.get("sha256"),
        )
        if (
            not isinstance(name, str)
            or name.startswith(".")
            or not name.endswith(".npz")
            or "/" in name
            or "\\" in name
            or ":" in name
            or any(ord(char) < 32 for char in name)
            or type(size) is not int
            or size <= 0
            or not isinstance(digest, str)
            or re.fullmatch(r"[0-9a-f]{64}", digest) is None
        ):
            raise ValueError(f"Invalid ocean scene identity: {name!r}")
        scenes.append(SceneFile(name, size, digest, f"{remote_directory}/{name}"))
    names = [scene.name for scene in scenes]
    if names != sorted(set(names)):
        raise ValueError("Ocean scene names must be unique and sorted")
    return scenes


def _verify(path: Path, scene: SceneFile) -> None:
    if path.is_symlink() or not path.is_file():
        raise ValueError(f"Ocean scene is not a regular file: {path}")
    if path.stat().st_size != scene.size or hash_file(path) != scene.sha256:
        raise ValueError(f"Ocean scene bytes differ from manifest: {path}")


def _rclone_copyto(remote_uri: str, temporary: Path) -> None:
    subprocess.run(
        ["rclone", "--log-level", "ERROR", "copyto", remote_uri, str(temporary)], check=True
    )


def fetch_mar64_scenes(
    scene_dir: Path,
    manifest_path: Path = DEFAULT_MANIFEST,
    *,
    downloader: Callable[[str, Path], None] = _rclone_copyto,
) -> list[Path]:
    """Download missing scenes; reject existing wrong bytes and unknown NPZs."""
    scenes = load_scene_manifest(manifest_path)
    scene_dir = Path(scene_dir)
    if scene_dir.is_symlink() or (scene_dir.exists() and not scene_dir.is_dir()):
        raise ValueError(f"Ocean scene destination is not a directory: {scene_dir}")
    expected_names = {scene.name for scene in scenes}
    unexpected = sorted(
        path.name for path in scene_dir.glob("*.npz") if path.name not in expected_names
    )
    if unexpected:
        raise ValueError(f"Unknown ocean scene files: {unexpected}")

    missing = []
    for scene in scenes:
        destination = scene_dir / scene.name
        if destination.exists() or destination.is_symlink():
            _verify(destination, scene)
        else:
            missing.append(scene)

    if missing:
        scene_dir.mkdir(parents=True, exist_ok=True)
    for scene in missing:
        destination = scene_dir / scene.name
        descriptor, temporary_name = tempfile.mkstemp(
            prefix=f".{scene.name}.", suffix=".part", dir=scene_dir
        )
        os.close(descriptor)
        temporary = Path(temporary_name)
        try:
            downloader(scene.remote_uri, temporary)
            _verify(temporary, scene)
            if destination.exists() or destination.is_symlink():
                _verify(destination, scene)
            else:
                os.replace(temporary, destination)
            _verify(destination, scene)
        finally:
            temporary.unlink(missing_ok=True)

    paths = [scene_dir / scene.name for scene in scenes]
    if sorted(path.name for path in scene_dir.glob("*.npz")) != [scene.name for scene in scenes]:
        raise ValueError("Ocean scene directory differs from the frozen inventory")
    for path, scene in zip(paths, scenes, strict=True):
        _verify(path, scene)
    return paths


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", type=Path, default=DEFAULT_MANIFEST)
    parser.add_argument("--scene-dir", type=Path, default=DEFAULT_SCENE_DIR)
    args = parser.parse_args(argv)
    try:
        paths = fetch_mar64_scenes(args.scene_dir, args.manifest)
    except (OSError, ValueError, subprocess.CalledProcessError) as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        return 1
    print(f"PASS mar64: {len(paths)} verified scenes in {args.scene_dir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
