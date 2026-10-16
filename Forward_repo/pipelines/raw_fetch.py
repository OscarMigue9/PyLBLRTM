"""Fetch and verify the frozen raw inputs used by the generation pipeline.

The catalog pins individual CAMS and ECOSTRESS file bytes. The source manifest
pins archive URLs and metadata. Existing files are verified and never replaced.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import re
import sys
import tarfile
import tempfile
import urllib.error
import urllib.request
import zipfile
from contextlib import ExitStack, contextmanager
from dataclasses import dataclass
from pathlib import Path, PurePosixPath
from typing import BinaryIO, Iterator


MANIFESTS = Path(__file__).resolve().parents[1] / "manifests"
DEFAULT_CATALOG = MANIFESTS / "raw_catalogs.json"
DEFAULT_SOURCES = MANIFESTS / "raw_sources.json"
DEFAULT_ASSETS = MANIFESTS / "assets.json"
_CONTENT_RANGE = re.compile(r"bytes (\d+)-(\d+)/(\d+)")
_CHUNK = 4 * 1024 * 1024


@dataclass(frozen=True)
class ExpectedFile:
    """A file pinned by size and SHA256."""

    path: Path
    size: int
    sha256: str


def _document(path: Path, *, hash_algorithm: bool = False) -> dict:
    document = json.loads(Path(path).read_text(encoding="utf-8"))
    if document.get("schema_version") != 1:
        raise ValueError(f"Unsupported manifest schema: {path}")
    if hash_algorithm and document.get("hash_algorithm") != "sha256":
        raise ValueError(f"Unsupported hash algorithm: {path}")
    return document


def _relative(value: str) -> PurePosixPath:
    if not isinstance(value, str) or "\\" in value:
        raise ValueError(f"Expected a portable relative path: {value!r}")
    path = PurePosixPath(value)
    if path.is_absolute() or not path.parts or any(part in (".", "..") for part in path.parts):
        raise ValueError(f"Expected a safe relative path: {value!r}")
    if ":" in path.parts[0]:
        raise ValueError(f"Expected a safe relative path: {value!r}")
    return path


def _expected(root: Path, directory: str, record: dict) -> ExpectedFile:
    folder = _relative(directory)
    name = _relative(record["path"])
    if len(name.parts) != 1:
        raise ValueError(f"Catalog entry must be a filename: {record['path']}")
    size = int(record["bytes"])
    digest = str(record["sha256"]).lower()
    if size < 0 or not re.fullmatch(r"[0-9a-f]{64}", digest):
        raise ValueError(f"Invalid catalog digest or byte count: {record['path']}")
    return ExpectedFile(root.joinpath(*folder.parts, name.name), size, digest)


def _hash_file(path: Path) -> tuple[int, str]:
    digest = hashlib.sha256()
    size = 0
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(_CHUNK), b""):
            digest.update(chunk)
            size += len(chunk)
    return size, digest.hexdigest()


def _verify(path: Path, size: int, sha256: str) -> None:
    if not path.is_file():
        raise FileNotFoundError(path)
    actual_size, actual_hash = _hash_file(path)
    if (actual_size, actual_hash) != (size, sha256):
        raise ValueError(f"File differs from the pinned bytes: {path}")


def _missing(files: list[ExpectedFile]) -> list[ExpectedFile]:
    missing = []
    for item in files:
        if item.path.exists():
            _verify(item.path, item.size, item.sha256)
        else:
            missing.append(item)
    return missing


@contextmanager
def _temporary(parent: Path) -> Iterator[Path]:
    parent.mkdir(parents=True, exist_ok=True)
    descriptor, filename = tempfile.mkstemp(prefix=".raw-fetch-", dir=parent)
    os.close(descriptor)
    path = Path(filename)
    try:
        yield path
    finally:
        path.unlink(missing_ok=True)


def _publish(temporary: Path, expected: ExpectedFile) -> None:
    _verify(temporary, expected.size, expected.sha256)
    try:
        # A same-directory hard link is atomic and fails if another writer won.
        os.link(temporary, expected.path)
    except FileExistsError:
        _verify(expected.path, expected.size, expected.sha256)


def _copy_verified(source: BinaryIO, destination: Path, expected: ExpectedFile) -> None:
    digest = hashlib.sha256()
    size = 0
    with destination.open("wb") as output:
        for chunk in iter(lambda: source.read(_CHUNK), b""):
            output.write(chunk)
            digest.update(chunk)
            size += len(chunk)
    if (size, digest.hexdigest()) != (expected.size, expected.sha256):
        raise ValueError(f"Source member differs from the raw catalog: {expected.path}")


def _range_chunks(url: str, start: int, end: int, archive_size: int) -> Iterator[bytes]:
    request = urllib.request.Request(
        url, headers={"Range": f"bytes={start}-{end}", "Accept-Encoding": "identity"}
    )
    with urllib.request.urlopen(request, timeout=120) as response:
        match = _CONTENT_RANGE.fullmatch(response.headers.get("Content-Range", ""))
        if response.status != 206 or match is None:
            raise ValueError("HTTP server did not honor the requested byte range")
        reported = tuple(int(value) for value in match.groups())
        if reported != (start, end, archive_size):
            raise ValueError(f"Unexpected HTTP Content-Range: {reported}")
        if response.headers.get("Content-Encoding", "identity").lower() != "identity":
            raise ValueError("HTTP server encoded a byte range")
        remaining = end - start + 1
        while remaining:
            chunk = response.read(min(_CHUNK, remaining))
            if not chunk:
                raise ValueError("Truncated HTTP byte range")
            remaining -= len(chunk)
            yield chunk
        if response.read(1):
            raise ValueError("HTTP server returned extra bytes")


def _tar_header_name(header: bytes) -> tuple[str, int]:
    if len(header) != 512:
        raise ValueError("Truncated TAR member header")
    stored = int(header[148:156].split(b"\0", 1)[0].strip() or b"0", 8)
    computed = sum(header[:148]) + 8 * 32 + sum(header[156:])
    if stored != computed:
        raise ValueError("TAR member header checksum differs")
    name = header[:100].split(b"\0", 1)[0].decode("ascii")
    prefix = header[345:500].split(b"\0", 1)[0].decode("ascii")
    size = int(header[124:136].split(b"\0", 1)[0].strip() or b"0", 8)
    return (f"{prefix}/{name}" if prefix else name), size


def fetch_cams(
    workspace_root: Path, catalog_path: Path = DEFAULT_CATALOG, sources_path: Path = DEFAULT_SOURCES
) -> list[Path]:
    """Download only the six required CAMS TAR members by verified HTTP ranges."""
    catalog = _document(catalog_path, hash_algorithm=True)["cams_l137"]
    source = _document(sources_path)["cams_l137"]
    records = {record["path"]: record for record in catalog["files"]}
    locations = {member["path"]: int(member["header_offset"]) for member in source["members"]}
    if len(records) != 6 or set(records) != set(locations):
        raise ValueError("CAMS source offsets do not match the six-file raw catalog")
    expected = [
        _expected(workspace_root, catalog["workspace_directory"], record)
        for record in catalog["files"]
    ]
    needed = _missing(expected)
    url = source["url"]
    archive_size = int(source["archive_bytes"])
    for item in needed:
        offset = locations[item.path.name]
        if offset < 0 or offset + 512 + item.size > archive_size:
            raise ValueError(f"CAMS member offset exceeds TAR size: {item.path.name}")
        header = b"".join(_range_chunks(url, offset, offset + 511, archive_size))
        name, size = _tar_header_name(header)
        if (name, size) != (item.path.name, item.size):
            raise ValueError(f"Unexpected CAMS TAR member at byte {offset}: {name}")
        with _temporary(item.path.parent) as temporary:
            digest = hashlib.sha256()
            count = 0
            with temporary.open("wb") as output:
                start = offset + 512
                for chunk in _range_chunks(url, start, start + item.size - 1, archive_size):
                    output.write(chunk)
                    digest.update(chunk)
                    count += len(chunk)
            if (count, digest.hexdigest()) != (item.size, item.sha256):
                raise ValueError(f"CAMS HTTP member differs from the catalog: {item.path.name}")
            _publish(temporary, item)
    return [item.path for item in expected]


@contextmanager
def _archive(source: dict, local_archive: Path | None) -> Iterator[Path]:
    size = int(source["archive_bytes"])
    digest = str(source["archive_sha256"]).lower()
    if not re.fullmatch(r"[0-9a-f]{64}", digest):
        raise ValueError("An archive SHA256 is required")
    if local_archive is not None:
        path = Path(local_archive)
        _verify(path, size, digest)
        yield path
        return
    with tempfile.TemporaryDirectory(prefix="thesis-raw-source-") as directory:
        path = Path(directory) / "source.archive"
        request = urllib.request.Request(source["url"], headers={"Accept-Encoding": "identity"})
        with urllib.request.urlopen(request, timeout=120) as response, path.open("wb") as output:
            if response.status != 200:
                raise ValueError(f"Archive download returned HTTP {response.status}")
            while chunk := response.read(_CHUNK):
                output.write(chunk)
        _verify(path, size, digest)
        yield path


def fetch_ecostress(
    workspace_root: Path,
    catalog_path: Path = DEFAULT_CATALOG,
    sources_path: Path = DEFAULT_SOURCES,
    *,
    local_archive: Path | None = None,
) -> list[Path]:
    """Extract only the 1419 selected ECOSTRESS spectra from a pinned ZIP."""
    catalog = _document(catalog_path, hash_algorithm=True)["ecostress_selected"]
    source = _document(sources_path)["ecostress_selected"]
    records = catalog["files"]
    if len(records) != 1419 or len({item["path"] for item in records}) != 1419:
        raise ValueError("Expected 1419 unique selected ECOSTRESS spectra")
    expected = [
        _expected(workspace_root, catalog["workspace_directory"], record) for record in records
    ]
    needed = _missing(expected)
    if not needed:
        return [item.path for item in expected]
    with _archive(source, local_archive) as archive, zipfile.ZipFile(archive) as bundle:
        names: dict[str, zipfile.ZipInfo] = {}
        selected = {item.path.name for item in expected}
        for info in bundle.infolist():
            basename = PurePosixPath(info.filename.replace("\\", "/")).name
            if info.is_dir() or basename not in selected:
                continue
            if basename in names:
                raise ValueError(f"Duplicate selected ZIP member: {basename}")
            names[basename] = info
        if set(names) != selected:
            raise ValueError(f"ZIP is missing {len(selected - set(names))} selected spectra")
        for item in needed:
            info = names[item.path.name]
            if info.file_size != item.size:
                raise ValueError(f"ECOSTRESS ZIP member has wrong size: {item.path.name}")
            with _temporary(item.path.parent) as temporary, bundle.open(info) as member:
                _copy_verified(member, temporary, item)
                _publish(temporary, item)
    return [item.path for item in expected]


def _asset(assets_path: Path, identifier: str, root: Path) -> ExpectedFile:
    assets = _document(assets_path, hash_algorithm=True)["artifacts"]
    matches = [item for item in assets if item["id"] == identifier]
    if len(matches) != 1 or not matches[0].get("workspace_path"):
        raise ValueError(f"Missing frozen AER asset: {identifier}")
    item = matches[0]
    relative = _relative(item["workspace_path"])
    digest = str(item["sha256"]).lower()
    if not re.fullmatch(r"[0-9a-f]{64}", digest):
        raise ValueError(f"Invalid asset SHA256: {identifier}")
    return ExpectedFile(root.joinpath(*relative.parts), int(item["bytes"]), digest)


def _aer_records(member: BinaryIO, molecule_id: int, records: list[tuple[float, str]]) -> None:
    waiting = 0
    for raw in member:
        line = raw.decode("utf-8", errors="ignore").rstrip("\r\n")
        if len(line) < 68:
            continue
        if waiting:
            waiting -= 1
            continue
        try:
            parsed_id = int(line[:2])
            wavenumber = float(line[3:15])
        except ValueError:
            continue
        if parsed_id != molecule_id:
            continue
        try:
            flag = int(line[98:100]) if line[98:100].strip() else 0
        except ValueError:
            flag = 0
        if 620.0 <= wavenumber <= 2785.0:
            records.append((wavenumber, line))
        if flag == -1:
            waiting = 1


def _write_par(path: Path, records: list[tuple[float, str]]) -> None:
    records.sort(key=lambda pair: pair[0])
    with path.open("wb") as output:
        for _, line in records:
            padded = line[:100].ljust(100)
            normalized = padded[:3] + padded[3:67].replace("D", "E") + padded[67:]
            output.write(normalized.encode("utf-8") + b"\n")


def fetch_aer(
    workspace_root: Path,
    sources_path: Path = DEFAULT_SOURCES,
    assets_path: Path = DEFAULT_ASSETS,
    *,
    local_archive: Path | None = None,
) -> list[Path]:
    """Rebuild the final AER line file and extract coupling from the pinned TAR."""
    source = _document(sources_path)["aer_v381"]
    par = _asset(assets_path, source["par_asset_id"], workspace_root)
    coupling = _asset(assets_path, source["coupling_asset_id"], workspace_root)
    needed = _missing([par, coupling])
    if not needed:
        return [par.path, coupling.path]
    line_members = list(source["line_members"])
    if len(line_members) != 7 or len(set(line_members)) != 7:
        raise ValueError("Expected seven distinct AER molecule members")
    molecule_ids = {name: index for index, name in enumerate(line_members, start=1)}
    want_par = par in needed
    want_coupling = coupling in needed
    with _archive(source, local_archive) as archive, ExitStack() as stack:
        par_temp = stack.enter_context(_temporary(par.path.parent)) if want_par else None
        coupling_temp = (
            stack.enter_context(_temporary(coupling.path.parent)) if want_coupling else None
        )
        records: list[tuple[float, str]] = []
        seen: set[str] = set()
        with tarfile.open(archive, mode="r|gz") as bundle:
            for member in bundle:
                if member.name == source["coupling_member"] and want_coupling:
                    if member.name in seen or not member.isfile():
                        raise ValueError("Invalid AER coupling TAR member")
                    seen.add(member.name)
                    with bundle.extractfile(member) as content:
                        _copy_verified(content, coupling_temp, coupling)
                elif member.name in molecule_ids and want_par:
                    if member.name in seen or not member.isfile():
                        raise ValueError(f"Invalid AER line TAR member: {member.name}")
                    seen.add(member.name)
                    with bundle.extractfile(member) as content:
                        _aer_records(content, molecule_ids[member.name], records)
        required = set(line_members) if want_par else set()
        if want_coupling:
            required.add(source["coupling_member"])
        if seen != required:
            raise ValueError(f"AER TAR is missing expected members: {sorted(required - seen)}")
        if want_par:
            if len(records) != int(source["par_line_count"]):
                raise ValueError("AER selection produced a different number of lines")
            _write_par(par_temp, records)
            _verify(par_temp, par.size, par.sha256)
        if want_coupling:
            _verify(coupling_temp, coupling.size, coupling.sha256)
        if want_par:
            _publish(par_temp, par)
        if want_coupling:
            _publish(coupling_temp, coupling)
    return [par.path, coupling.path]


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--workspace-root", type=Path, default=Path("data_cache/raw"))
    parser.add_argument("--catalog", type=Path, default=DEFAULT_CATALOG)
    parser.add_argument("--sources", type=Path, default=DEFAULT_SOURCES)
    parser.add_argument("--assets", type=Path, default=DEFAULT_ASSETS)
    commands = parser.add_subparsers(dest="command", required=True)
    commands.add_parser("cams", help="Fetch six CAMS NC members by HTTP Range")
    for name in ("ecostress", "aer"):
        command = commands.add_parser(name)
        command.add_argument("--archive", type=Path, help="Use a verified local source archive")
    commands.add_parser("all", help="Fetch CAMS, ECOSTRESS and AER inputs")
    args = parser.parse_args(argv)
    try:
        if args.command in ("cams", "all"):
            paths = fetch_cams(args.workspace_root, args.catalog, args.sources)
            print(f"PASS CAMS: {len(paths)} verified files")
        if args.command in ("ecostress", "all"):
            paths = fetch_ecostress(
                args.workspace_root,
                args.catalog,
                args.sources,
                local_archive=getattr(args, "archive", None),
            )
            print(f"PASS ECOSTRESS: {len(paths)} verified files")
        if args.command in ("aer", "all"):
            paths = fetch_aer(
                args.workspace_root,
                args.sources,
                args.assets,
                local_archive=getattr(args, "archive", None),
            )
            print(f"PASS AER: {len(paths)} verified files")
    except (
        OSError,
        ValueError,
        tarfile.TarError,
        zipfile.BadZipFile,
        urllib.error.URLError,
    ) as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
