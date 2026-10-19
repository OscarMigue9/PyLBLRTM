"""
hapi_download_check.py — Test de descarga HITRAN/HAPI para todas las moléculas.

Descarga las líneas de cada isotopólogo de cada molécula del catálogo de
pylblrtm (MOLECULE_IDS) en la ventana 8–14 µm (714.29–1250 cm⁻¹) y valida
cada paso de la cadena de datos. El test NUNCA se detiene por un error: cada
fallo se registra y se continúa con el siguiente isotopólogo. Al final se
escribe un reporte con todos los errores para solucionarlos después.

Uso:
    py tests/hapi_download_check.py                       # todo, 8–14 µm
    py tests/hapi_download_check.py --molecules H2O CO2   # solo algunas
    py tests/hapi_download_check.py --iso first           # solo iso 1
    py tests/hapi_download_check.py --um 3 5              # otra ventana

Pasos por isotopólogo:
    1. download     get_line_parameters (descarga a una caché nueva)
    2. validate     arrays no vacíos, finitos, dentro del rango, Sref > 0
    3. molar_mass   get_molar_mass
    4. tips         get_tips_table 150–3000 K (lo mismo que usa from_hapi)
    5. from_hapi    construir SpectralData desde la caché
Por molécula, al final: from_hapi(iso_id="all"), que debe saltar los isotopólogos
sin líneas y devolver exactamente los que pasaron (o fallar si ninguno pasó).
Antes de empezar se verifica que MOLECULE_IDS coincida con el catálogo de HAPI.

Salida (una carpeta por corrida, no se borra):
    resultados_test_hapi/<fecha_hora>/
        RESUMEN.md          resumen legible con la tabla de errores
        resultados.csv      una fila por isotopólogo (y por molécula)
        errores.json        cada error con su traceback completo
        test.log            log completo de la corrida
        hapi_stdout/        lo que HAPI imprimió en cada descarga
        cache/<MOL>/        tablas descargadas (una carpeta por molécula)
"""

from __future__ import annotations

import argparse
import contextlib
import csv
import io
import json
import logging
import platform
import socket
import sys
import time
import traceback
from dataclasses import asdict, dataclass, field
from datetime import datetime
from pathlib import Path
from typing import List, Optional

import numpy as np

# Permite correr el script sin instalar el paquete (desde la raíz del repo)
_REPO_ROOT = Path(__file__).resolve().parents[1]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

import pylblrtm  # noqa: E402
from pylblrtm import hitran_manager as hm  # noqa: E402
from pylblrtm.data_loader import from_hapi  # noqa: E402


STAGES = ("catalog", "download", "validate", "molar_mass", "tips", "from_hapi")

# Estados posibles de cada fila
OK = "OK"
WARN = "ADVERTENCIA"        # pasó, pero los datos tienen algo raro
NO_LINES = "SIN_LINEAS"     # HITRAN no tiene líneas en el rango (no es bug)
NOT_SERVED = "NO_DISPONIBLE"  # HITRANonline no sirve el isotopólogo por la API
ERROR = "ERROR"


@dataclass
class Result:
    molecule: str
    mol_id: int
    iso_id: str                 # "1", "2", ... o "all" para la fila de molécula
    status: str = OK
    failed_stage: str = ""
    error_type: str = ""
    error_message: str = ""
    n_lines: int = 0
    nu_min_found: float = float("nan")
    nu_max_found: float = float("nan")
    molar_mass: float = float("nan")
    seconds: float = 0.0
    warnings: List[str] = field(default_factory=list)
    traceback: str = ""


# ── Utilidades ────────────────────────────────────────────────────────────────

def _setup_logging(log_path: Path) -> logging.Logger:
    log = logging.getLogger("hapi_check")
    log.setLevel(logging.DEBUG)
    log.handlers.clear()
    fh = logging.FileHandler(log_path, encoding="utf-8")
    fh.setLevel(logging.DEBUG)
    fh.setFormatter(logging.Formatter("%(asctime)s  %(levelname)-7s  %(message)s"))
    ch = logging.StreamHandler(sys.stdout)
    ch.setLevel(logging.INFO)
    ch.setFormatter(logging.Formatter("%(message)s"))
    log.addHandler(fh)
    log.addHandler(ch)
    # Los mensajes internos de pylblrtm también van al archivo
    pl = logging.getLogger("pylblrtm")
    pl.setLevel(logging.DEBUG)
    pl.addHandler(fh)
    return log


def _short(exc: BaseException) -> str:
    """Mensaje de una línea, incluyendo la causa original si la hay."""
    msg = " ".join(str(exc).split())
    cause = exc.__cause__ or exc.__context__
    if cause is not None and str(cause) and str(cause) not in msg:
        msg += f"  [causa: {type(cause).__name__}: {' '.join(str(cause).split())}]"
    return msg[:500]


def _download_status(exc: BaseException) -> str:
    """Estado de un fallo de descarga: los que no son bugs no cuentan como ERROR."""
    if isinstance(exc, hm.HITRANNoLinesError):
        return NO_LINES
    if isinstance(exc, hm.HITRANNotAvailableError):
        return NOT_SERVED
    return ERROR


def _validate(data: dict, nu_min: float, nu_max: float) -> List[str]:
    """Chequeos de sanidad sobre las líneas descargadas. Devuelve advertencias."""
    warns = []
    expected = ("nu0", "Sref", "g_air", "g_self", "Elow", "n_air", "shift",
                "mol_id", "iso_id")
    missing = [k for k in expected if k not in data]
    if missing:
        raise KeyError(f"Faltan columnas: {missing}")
    n = len(data["nu0"])
    sizes = {k: len(data[k]) for k in expected}
    if len(set(sizes.values())) != 1:
        raise ValueError(f"Columnas con tamaños distintos: {sizes}")
    for k in ("nu0", "Sref", "g_air", "g_self", "Elow", "n_air", "shift"):
        bad = int(np.count_nonzero(~np.isfinite(data[k])))
        if bad:
            warns.append(f"{k}: {bad}/{n} valores no finitos")
    tol = 1e-6
    out = int(np.count_nonzero((data["nu0"] < nu_min - tol) | (data["nu0"] > nu_max + tol)))
    if out:
        warns.append(f"nu0: {out}/{n} líneas fuera de [{nu_min:.3f}, {nu_max:.3f}]")
    if np.any(data["Sref"] <= 0):
        warns.append(f"Sref: {int(np.count_nonzero(data['Sref'] <= 0))}/{n} valores <= 0")
    if np.any(data["g_air"] <= 0):
        warns.append(f"g_air: {int(np.count_nonzero(data['g_air'] <= 0))}/{n} valores <= 0")
    if np.any(data["g_self"] <= 0):
        warns.append(f"g_self: {int(np.count_nonzero(data['g_self'] <= 0))}/{n} valores <= 0 "
                     "(HITRAN usa 0 cuando no hay dato)")
    if np.any(np.diff(data["nu0"]) < 0):
        warns.append("nu0 no está ordenado de forma creciente")
    return warns


# ── Test de un isotopólogo ────────────────────────────────────────────────────

def check_isotopologue(mol: str, mol_id: int, iso: int, nu_min: float, nu_max: float,
                       cache_dir: Path, stdout_dir: Path, log: logging.Logger) -> Result:
    res = Result(molecule=mol, mol_id=mol_id, iso_id=str(iso))
    t0 = time.perf_counter()
    stage = "download"
    hapi_out = io.StringIO()
    try:
        # HAPI imprime mucho por pantalla: se guarda en un archivo aparte
        with contextlib.redirect_stdout(hapi_out):
            data = hm.get_line_parameters(mol, nu_min, nu_max, iso_id=iso,
                                          cache_dir=cache_dir)
        res.n_lines = int(len(data["nu0"]))
        res.nu_min_found = float(np.min(data["nu0"]))
        res.nu_max_found = float(np.max(data["nu0"]))

        stage = "validate"
        res.warnings = _validate(data, nu_min, nu_max)

        stage = "molar_mass"
        res.molar_mass = hm.get_molar_mass(mol, iso)
        if not (np.isfinite(res.molar_mass) and res.molar_mass > 0):
            raise ValueError(f"Masa molar inválida: {res.molar_mass}")

        stage = "tips"
        with contextlib.redirect_stdout(hapi_out):
            T, Q = hm.get_tips_table(mol, iso, T_min=150.0, T_max=3000.0)
        if not np.all(np.isfinite(Q)) or np.any(Q <= 0):
            raise ValueError(
                f"TIPS con valores inválidos: {int(np.count_nonzero(~np.isfinite(Q) | (Q <= 0)))}"
                f"/{Q.size} puntos no finitos o <= 0"
            )

        stage = "from_hapi"
        with contextlib.redirect_stdout(hapi_out):
            sd = from_hapi(mol, nu_min, nu_max, iso_id=iso, cache_dir=cache_dir)
        q296 = sd.Q(296.0, mol_id, iso)
        if not (np.isfinite(q296) and q296 > 0):
            raise ValueError(f"SpectralData.Q(296 K) inválido: {q296}")

        res.status = WARN if res.warnings else OK

    except KeyboardInterrupt:
        raise
    except BaseException as exc:  # noqa: BLE001 — el test no debe pararse nunca
        res.failed_stage = stage
        res.error_type = type(exc).__name__
        res.error_message = _short(exc)
        res.traceback = traceback.format_exc()
        res.status = _download_status(exc) if stage == "download" else ERROR
        if res.status != ERROR:
            res.traceback = ""
    finally:
        res.seconds = round(time.perf_counter() - t0, 2)
        text = hapi_out.getvalue()
        if text.strip():
            (stdout_dir / f"{mol}_iso{iso}.txt").write_text(text, encoding="utf-8")

    _log_result(log, res)
    return res


def check_molecule_all(mol: str, mol_id: int, isos: List[int], nu_min: float,
                       nu_max: float, cache_dir: Path, log: logging.Logger) -> Result:
    """from_hapi(iso_id="all"): salta los isotopólogos sin líneas y concatena.

    *isos* son los isotopólogos que pasaron uno por uno: deben ser exactamente
    los que devuelve. Si no pasó ninguno, debe fallar con HITRANNoLinesError o
    HITRANNotAvailableError (y entonces la fila hereda ese estado).
    """
    res = Result(molecule=mol, mol_id=mol_id, iso_id="all")
    t0 = time.perf_counter()
    try:
        with contextlib.redirect_stdout(io.StringIO()):
            sd = from_hapi(mol, nu_min, nu_max, iso_id="all", cache_dir=cache_dir)
        res.n_lines = int(len(sd.lines["nu0"]))
        found = sorted({int(i) for i in np.unique(sd.lines["iso_id"])})
        if found != sorted(isos):
            raise AssertionError(f"isotopólogos esperados {sorted(isos)}, devueltos {found}")
        res.status = OK
    except KeyboardInterrupt:
        raise
    except BaseException as exc:  # noqa: BLE001
        res.failed_stage = "from_hapi"
        res.error_type = type(exc).__name__
        res.error_message = _short(exc)
        res.status = _download_status(exc) if not isos else ERROR
        res.traceback = traceback.format_exc() if res.status == ERROR else ""
    res.seconds = round(time.perf_counter() - t0, 2)
    _log_result(log, res)
    return res


def _log_result(log: logging.Logger, r: Result) -> None:
    tag = f"{r.molecule:>7} iso {r.iso_id:>3}"
    if r.status in (OK, WARN):
        log.info(f"  [{r.status:^11}] {tag}  {r.n_lines:>8} líneas  {r.seconds:7.1f} s")
        for w in r.warnings:
            log.info(f"                {w}")
    else:
        log.info(f"  [{r.status:^11}] {tag}  etapa={r.failed_stage}  "
                 f"{r.error_type}: {r.error_message[:160]}")
    if r.traceback:
        log.debug(r.traceback)


# ── Reportes ──────────────────────────────────────────────────────────────────

_CSV_FIELDS = ["molecule", "mol_id", "iso_id", "status", "failed_stage", "error_type",
               "error_message", "n_lines", "nu_min_found", "nu_max_found",
               "molar_mass", "seconds", "warnings"]


def _append_csv(path: Path, r: Result) -> None:
    new = not path.exists()
    with path.open("a", newline="", encoding="utf-8") as fp:
        w = csv.DictWriter(fp, fieldnames=_CSV_FIELDS)
        if new:
            w.writeheader()
        row = {k: getattr(r, k) for k in _CSV_FIELDS}
        row["warnings"] = " | ".join(r.warnings)
        w.writerow(row)


def _write_reports(out: Path, results: List[Result], meta: dict, interrupted: bool) -> None:
    problems = [r for r in results if r.status == ERROR]
    warned = [r for r in results if r.status == WARN]
    empty = [r for r in results if r.status == NO_LINES]
    not_served = [r for r in results if r.status == NOT_SERVED]
    ok = [r for r in results if r.status == OK]

    (out / "errores.json").write_text(
        json.dumps({"meta": meta,
                    "errores": [asdict(r) for r in problems],
                    "advertencias": [asdict(r) for r in warned],
                    "sin_lineas": [asdict(r) for r in empty],
                    "no_disponible": [asdict(r) for r in not_served]},
                   indent=2, ensure_ascii=False, default=str),
        encoding="utf-8",
    )

    md = [f"# Test de descarga HAPI — {meta['inicio']}", ""]
    if interrupted:
        md += ["> **Corrida interrumpida por el usuario**: los resultados están incompletos.", ""]
    md += [
        f"- Rango: {meta['um'][0]}–{meta['um'][1]} µm = "
        f"{meta['nu_min_cm1']:.3f}–{meta['nu_max_cm1']:.3f} cm⁻¹",
        f"- pylblrtm {meta['pylblrtm']} · hitran-api {meta['hapi']} · "
        f"Python {meta['python']} · {meta['platform']}",
        f"- Duración: {meta.get('duracion_min', '?')} min",
        "",
        "| Estado | Cantidad |",
        "|---|---|",
        f"| OK | {len(ok)} |",
        f"| ADVERTENCIA (pasó con datos raros) | {len(warned)} |",
        f"| SIN_LINEAS (HITRAN no tiene líneas en el rango) | {len(empty)} |",
        f"| NO_DISPONIBLE (HITRANonline no lo sirve por la API) | {len(not_served)} |",
        f"| **ERROR** | **{len(problems)}** |",
        "",
    ]
    if problems:
        md += ["## Errores", "",
               "| Molécula | Iso | Etapa | Tipo | Mensaje |", "|---|---|---|---|---|"]
        for r in problems:
            msg = r.error_message.replace("|", "\\|")
            md.append(f"| {r.molecule} | {r.iso_id} | {r.failed_stage} | "
                      f"{r.error_type} | {msg} |")
        md += ["", "Traceback completo de cada uno en `errores.json` y `test.log`.", ""]

        md += ["### Errores agrupados por mensaje", ""]
        groups: dict = {}
        for r in problems:
            key = f"[{r.failed_stage}] {r.error_type}: {r.error_message[:120]}"
            groups.setdefault(key, []).append(f"{r.molecule}/{r.iso_id}")
        for key, who in sorted(groups.items(), key=lambda kv: -len(kv[1])):
            md.append(f"- **{len(who)}×** `{key}`  \n  {', '.join(who)}")
        md.append("")
    if warned:
        md += ["## Advertencias", ""]
        for r in warned:
            md.append(f"- **{r.molecule} iso {r.iso_id}**: " + "; ".join(r.warnings))
        md.append("")
    if empty:
        md += ["## Sin líneas en el rango", "",
               "HITRAN tiene líneas de estos isotopólogos, pero no en esta ventana.", "",
               ", ".join(f"{r.molecule}/{r.iso_id}" for r in empty), ""]
    if not_served:
        md += ["## No disponibles en HITRANonline", "",
               "La API responde 404 en todo el rango 0–100000 cm⁻¹ (pueden existir solo "
               "como secciones eficaces o en listas suplementarias).", "",
               ", ".join(f"{r.molecule}/{r.iso_id}" for r in not_served), ""]
    md += ["## Moléculas OK", "",
           ", ".join(sorted({r.molecule for r in ok if r.iso_id != "all"})) or "—", ""]
    (out / "RESUMEN.md").write_text("\n".join(md), encoding="utf-8")


# ── Main ──────────────────────────────────────────────────────────────────────

def main(argv: Optional[List[str]] = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--um", nargs=2, type=float, default=(8.0, 14.0),
                    metavar=("UM_MIN", "UM_MAX"), help="Ventana en µm (default 8 14)")
    ap.add_argument("--molecules", nargs="+", default=None,
                    help="Moléculas a probar (default: todas las de MOLECULE_IDS)")
    ap.add_argument("--iso", choices=("all", "first"), default="all",
                    help="Todos los isotopólogos o solo el iso 1 (default all)")
    ap.add_argument("--out", type=Path, default=_REPO_ROOT / "resultados_test_hapi",
                    help="Carpeta base de resultados")
    ap.add_argument("--timeout", type=float, default=180.0,
                    help="Timeout de red por petición [s] (default 180)")
    args = ap.parse_args(argv)

    # La consola de Windows (cp1252) no puede imprimir µ, ⁻¹, ─ …
    for stream in (sys.stdout, sys.stderr):
        with contextlib.suppress(AttributeError, ValueError):
            stream.reconfigure(encoding="utf-8", errors="replace")

    um_lo, um_hi = sorted(args.um)
    nu_min, nu_max = 1e4 / um_hi, 1e4 / um_lo     # µm → cm⁻¹

    stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    out = args.out / stamp
    stdout_dir = out / "hapi_stdout"
    cache_root = out / "cache"
    for d in (out, stdout_dir, cache_root):
        d.mkdir(parents=True, exist_ok=True)
    log = _setup_logging(out / "test.log")
    csv_path = out / "resultados.csv"

    # Una petición colgada no debe congelar el test
    socket.setdefaulttimeout(args.timeout)

    meta = {
        "inicio": datetime.now().isoformat(timespec="seconds"),
        "um": [um_lo, um_hi],
        "nu_min_cm1": nu_min,
        "nu_max_cm1": nu_max,
        "pylblrtm": getattr(pylblrtm, "__version__", "?"),
        "python": platform.python_version(),
        "platform": platform.platform(),
        "hapi": "?",
        "args": {k: str(v) for k, v in vars(args).items()},
    }

    log.info(f"Test de descarga HAPI — {um_lo}–{um_hi} µm = {nu_min:.3f}–{nu_max:.3f} cm⁻¹")
    log.info(f"Resultados en: {out}")

    results: List[Result] = []

    def record(r: Result) -> None:
        results.append(r)
        _append_csv(csv_path, r)

    t_start = time.perf_counter()
    interrupted = False
    try:
        # HAPI en sí: si no carga, no hay nada más que probar
        try:
            hapi = hm._get_hapi()
            meta["hapi"] = getattr(hapi, "HAPI_VERSION", "?")
        except BaseException as exc:  # noqa: BLE001
            r = Result(molecule="HAPI", mol_id=0, iso_id="-", status=ERROR,
                       failed_stage="import", error_type=type(exc).__name__,
                       error_message=_short(exc), traceback=traceback.format_exc())
            record(r)
            _log_result(log, r)
            return 1

        # Catálogo: el ID de cada nombre debe ser la molécula que HAPI descarga
        hapi_names = {m: rec[4] for (m, _), rec in hapi.ISO.items()}
        ours = {mid: name.replace("+", "p") for name, mid in hm.MOLECULE_IDS.items()}
        diffs = [f"{mid}: nuestro={ours.get(mid, '-')} HAPI={hapi_names.get(mid, '-')}"
                 for mid in sorted(set(ours) | set(hapi_names))
                 if ours.get(mid) != hapi_names.get(mid)]
        cat = Result(molecule="CATALOGO", mol_id=0, iso_id="-")
        if diffs:
            cat.status, cat.failed_stage = ERROR, "catalog"
            cat.error_type = "MOLECULE_IDS"
            cat.error_message = "; ".join(diffs)[:500]
        cat.n_lines = len(hapi_names)
        record(cat)
        log.info(f"Catálogo: {len(ours)} moléculas en MOLECULE_IDS, {len(hapi_names)} en HAPI"
                 + (f" — {len(diffs)} diferencias" if diffs else " — coinciden"))

        if not hm._can_reach_hitran():
            log.warning("ATENCIÓN: no se alcanza hitran.org — las descargas van a fallar.")

        if args.molecules:
            molecules = []
            for m in args.molecules:
                try:
                    molecules.append(hm._resolve_mol_name(m))
                except ValueError as exc:
                    r = Result(molecule=m, mol_id=0, iso_id="-", status=ERROR,
                               failed_stage="catalog", error_type="ValueError",
                               error_message=_short(exc))
                    record(r)
                    _log_result(log, r)
        else:
            molecules = list(hm.MOLECULE_IDS)

        for i, mol in enumerate(molecules, 1):
            mol_id = hm.MOLECULE_IDS[mol]
            log.info(f"\n[{i}/{len(molecules)}] {mol} (mol_id={mol_id})")

            # Isotopólogos según el catálogo interno de HAPI
            isos = sorted(iso for (m, iso) in hapi.ISO if m == mol_id)
            if not isos:
                r = Result(molecule=mol, mol_id=mol_id, iso_id="-", status=ERROR,
                           failed_stage="catalog", error_type="HITRANDataError",
                           error_message=f"HAPI no tiene isotopólogos para mol_id={mol_id}")
                record(r)
                _log_result(log, r)
                continue
            if args.iso == "first":
                isos = isos[:1]
            log.info(f"  isotopólogos: {isos}")

            # Caché propia por molécula: db_begin de HAPI recarga todas las tablas
            # de la carpeta en cada llamada, así que una caché única se vuelve lenta
            cache_dir = cache_root / mol
            passed = []
            for iso in isos:
                r = check_isotopologue(mol, mol_id, iso, nu_min, nu_max,
                                       cache_dir, stdout_dir, log)
                record(r)
                if r.status in (OK, WARN):
                    passed.append(iso)

            if len(isos) > 1:
                record(check_molecule_all(mol, mol_id, passed, nu_min, nu_max,
                                          cache_dir, log))

    except KeyboardInterrupt:
        interrupted = True
        log.warning("\nInterrumpido por el usuario — escribiendo reporte parcial…")
    finally:
        meta["duracion_min"] = round((time.perf_counter() - t_start) / 60, 1)
        _write_reports(out, results, meta, interrupted)

    n_err = sum(r.status == ERROR for r in results)
    log.info("\n" + "─" * 70)
    log.info(f"OK: {sum(r.status == OK for r in results)}   "
             f"ADVERTENCIA: {sum(r.status == WARN for r in results)}   "
             f"SIN_LINEAS: {sum(r.status == NO_LINES for r in results)}   "
             f"NO_DISPONIBLE: {sum(r.status == NOT_SERVED for r in results)}   "
             f"ERROR: {n_err}")
    log.info(f"Reporte: {out / 'RESUMEN.md'}")
    return 1 if n_err else 0


if __name__ == "__main__":
    sys.exit(main())
