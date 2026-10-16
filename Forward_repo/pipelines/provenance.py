"""Capture the exact source tree and runtime used by a reproduction stage."""

from __future__ import annotations

import hashlib
import importlib.metadata
import json
import platform
from pathlib import Path

import torch

from .assets import hash_file


SOURCE_DIRECTORIES = ("packages", "pipelines", "configs", "manifests", "frozen")
EXCLUDED_PARTS = {"__pycache__", "build", "dist", ".git", ".pytest_cache", ".ruff_cache"}
EXCLUDED_SUFFIXES = {".pyc", ".pyo", ".nbc", ".nbi"}


def source_snapshot(root: Path) -> dict:
    """Hash all bundled executable sources, configuration and small resources.

    The explicit directory list excludes downloaded data and run outputs. The
    snapshot is computed inside the stage process, after its Modal image is
    built, so it describes the code that actually produced the report.
    """
    paths = [root / "pyproject.toml"]
    for directory in SOURCE_DIRECTORIES:
        paths.extend((root / directory).rglob("*"))
    files = {}
    for path in paths:
        if not path.is_file() or path.suffix in EXCLUDED_SUFFIXES:
            continue
        relative = path.relative_to(root)
        if any(part in EXCLUDED_PARTS or part.endswith(".egg-info") for part in relative.parts):
            continue
        files[relative.as_posix()] = hash_file(path)
    ordered = dict(sorted(files.items()))
    encoded = json.dumps(ordered, sort_keys=True, separators=(",", ":")).encode("utf-8")
    return {
        "sha256": hashlib.sha256(encoded).hexdigest(),
        "file_count": len(ordered),
        "files": ordered,
    }


def runtime_snapshot() -> dict:
    """Record the package and accelerator versions relevant to GPU results."""
    packages = {}
    for name in ("numpy", "scipy", "numba", "cupy-cuda12x", "torch", "photonix", "iasi-retrieval"):
        try:
            packages[name] = importlib.metadata.version(name)
        except importlib.metadata.PackageNotFoundError:
            packages[name] = None
    gpu_available = torch.cuda.is_available()
    return {
        "python": platform.python_version(),
        "platform": platform.platform(),
        "packages": packages,
        "torch_cuda": torch.version.cuda,
        "cudnn": torch.backends.cudnn.version() if gpu_available else None,
        "gpu": torch.cuda.get_device_name() if gpu_available else None,
    }
