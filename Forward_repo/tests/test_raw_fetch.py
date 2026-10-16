"""Offline checks for bounded raw downloads and archive materialization."""

from __future__ import annotations

import hashlib
import io
import json
import tarfile
import threading
import zipfile
from contextlib import contextmanager
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path

import pytest

from pipelines.raw_fetch import fetch_aer, fetch_cams, fetch_ecostress


def _sha(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def _json(path: Path, content: dict) -> Path:
    path.write_text(json.dumps(content), encoding="utf-8")
    return path


@contextmanager
def _range_server(archive: bytes):
    requests: list[tuple[int, int]] = []

    class Handler(BaseHTTPRequestHandler):
        def do_GET(self):
            assert self.path == "/source.tar"
            start, end = (int(part) for part in self.headers["Range"][6:].split("-"))
            requests.append((start, end))
            self.send_response(206)
            self.send_header("Content-Range", f"bytes {start}-{end}/{len(archive)}")
            self.send_header("Content-Length", str(end - start + 1))
            self.end_headers()
            self.wfile.write(archive[start : end + 1])

        def log_message(self, *args):
            pass

    server = ThreadingHTTPServer(("127.0.0.1", 0), Handler)
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    try:
        yield f"http://127.0.0.1:{server.server_port}/source.tar", requests
    finally:
        server.shutdown()
        thread.join()
        server.server_close()


def test_cams_downloads_only_six_verified_tar_members(tmp_path: Path) -> None:
    names = ("temp", "hum", "co2", "oz", "n2o", "ch4")
    contents = {f"cams_5000_{name}_2025.nc": name.encode() * 11 for name in names}
    stream = io.BytesIO()
    with tarfile.open(fileobj=stream, mode="w", format=tarfile.USTAR_FORMAT) as bundle:
        for name, data in contents.items():
            info = tarfile.TarInfo(name)
            info.size = len(data)
            bundle.addfile(info, io.BytesIO(data))
        unused = tarfile.TarInfo("unused.nc")
        unused.size = 100_000
        bundle.addfile(unused, io.BytesIO(b"x" * unused.size))
    archive = stream.getvalue()
    with tarfile.open(fileobj=io.BytesIO(archive)) as bundle:
        offsets = {item.name: item.offset for item in bundle if item.name in contents}
    catalog = {
        "schema_version": 1,
        "hash_algorithm": "sha256",
        "cams_l137": {
            "workspace_directory": "data/cams",
            "files": [
                {"path": name, "bytes": len(data), "sha256": _sha(data)}
                for name, data in contents.items()
            ],
        },
    }
    catalog_path = _json(tmp_path / "catalog.json", catalog)
    with _range_server(archive) as (url, requests):
        sources = {
            "schema_version": 1,
            "cams_l137": {
                "url": url,
                "archive_bytes": len(archive),
                "members": [{"path": name, "header_offset": offsets[name]} for name in contents],
            },
        }
        source_path = _json(tmp_path / "sources.json", sources)
        paths = fetch_cams(tmp_path, catalog_path, source_path)
        assert len(paths) == 6
        assert all(path.read_bytes() == contents[path.name] for path in paths)
        assert len(requests) == 12
        assert sum(end - start + 1 for start, end in requests) == 6 * 512 + sum(
            map(len, contents.values())
        )
        fetch_cams(tmp_path, catalog_path, source_path)
        assert len(requests) == 12
        paths[0].write_bytes(b"wrong")
        with pytest.raises(ValueError, match="pinned bytes"):
            fetch_cams(tmp_path, catalog_path, source_path)
        assert paths[0].read_bytes() == b"wrong"
        assert len(requests) == 12


def test_ecostress_extracts_only_catalogued_spectra(tmp_path: Path) -> None:
    selected = {f"record{i}.spectrum.txt": f"spectrum {i}\n".encode() for i in range(1419)}
    archive = tmp_path / "source.zip"
    with zipfile.ZipFile(archive, "w", compression=zipfile.ZIP_DEFLATED) as bundle:
        for name, data in selected.items():
            bundle.writestr(f"ecostress_raw/materials/{name}", data)
        bundle.writestr("ecostress_raw/other.spectrum.txt", b"unselected")
    catalog_path = _json(
        tmp_path / "catalog.json",
        {
            "schema_version": 1,
            "hash_algorithm": "sha256",
            "ecostress_selected": {
                "workspace_directory": "data/spectra",
                "files": [
                    {"path": name, "bytes": len(data), "sha256": _sha(data)}
                    for name, data in selected.items()
                ],
            },
        },
    )
    source_path = _json(
        tmp_path / "sources.json",
        {
            "schema_version": 1,
            "ecostress_selected": {
                "url": "https://invalid.example/source.zip",
                "archive_bytes": archive.stat().st_size,
                "archive_sha256": _sha(archive.read_bytes()),
            },
        },
    )
    paths = fetch_ecostress(tmp_path, catalog_path, source_path, local_archive=archive)
    assert len(paths) == 1419
    assert all(path.read_bytes() == selected[path.name] for path in paths)
    assert not (tmp_path / "data/spectra/other.spectrum.txt").exists()
    paths[0].write_bytes(b"wrong")
    with pytest.raises(ValueError, match="pinned bytes"):
        fetch_ecostress(tmp_path, catalog_path, source_path, local_archive=archive)
    assert paths[0].read_bytes() == b"wrong"


def _aer_line(molecule: int, wavenumber: float, *, coupling: bool = False) -> bytes:
    line = bytearray(b" " * 100)
    line[:2] = f"{molecule:02d}".encode()
    line[3:15] = f"{wavenumber:12.6f}".encode()
    line[20:29] = b"1.23D-04 "
    if coupling:
        line[98:100] = b"-1"
    return bytes(line) + b"\n"


def test_aer_rebuilds_pinned_par_and_root_coupling(tmp_path: Path) -> None:
    members = [
        f"line_files_By_Molecule/{index:02d}_{name}/{index:02d}_{name}"
        for index, name in enumerate(("H2O", "CO2", "O3", "N2O", "CO", "CH4", "O2"), 1)
    ]
    first = _aer_line(1, 900.0, coupling=True)
    second = _aer_line(2, 700.0)
    skipped_coupling = b"x" * 100 + b"\n"
    coupling = b"coupling table\n"
    archive = tmp_path / "aer.tar.gz"
    with tarfile.open(archive, "w:gz") as bundle:
        for name, data in [
            ("lncpl_lines", coupling),
            *[
                (name, first + skipped_coupling if index == 1 else second if index == 2 else b"")
                for index, name in enumerate(members, 1)
            ],
        ]:
            info = tarfile.TarInfo(name)
            info.size = len(data)
            bundle.addfile(info, io.BytesIO(data))

    def normalized(line: bytes) -> bytes:
        return line[:3] + line[3:67].replace(b"D", b"E") + line[67:]

    expected_par = normalized(second) + normalized(first)
    assets_path = _json(
        tmp_path / "assets.json",
        {
            "schema_version": 1,
            "hash_algorithm": "sha256",
            "artifacts": [
                {
                    "id": "aer_lines",
                    "workspace_path": "data/aer/selected.par",
                    "bytes": len(expected_par),
                    "sha256": _sha(expected_par),
                },
                {
                    "id": "aer_coupling",
                    "workspace_path": "data/aer/lncpl_lines",
                    "bytes": len(coupling),
                    "sha256": _sha(coupling),
                },
            ],
        },
    )
    source_path = _json(
        tmp_path / "sources.json",
        {
            "schema_version": 1,
            "aer_v381": {
                "url": "https://invalid.example/aer.tar.gz",
                "archive_bytes": archive.stat().st_size,
                "archive_sha256": _sha(archive.read_bytes()),
                "coupling_member": "lncpl_lines",
                "line_members": members,
                "par_asset_id": "aer_lines",
                "coupling_asset_id": "aer_coupling",
                "par_line_count": 2,
            },
        },
    )
    par, coupling_path = fetch_aer(tmp_path, source_path, assets_path, local_archive=archive)
    assert par.read_bytes() == expected_par
    assert coupling_path.read_bytes() == coupling
    par.unlink()
    archive.write_bytes(b"not the pinned archive")
    with pytest.raises(ValueError, match="pinned bytes"):
        fetch_aer(tmp_path, source_path, assets_path, local_archive=archive)
    assert not par.exists()
