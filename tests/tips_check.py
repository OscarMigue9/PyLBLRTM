"""
tips_check.py — Test de las sumas de partición (TIPS) para todos los isotopólogos.

No necesita internet: las tablas TIPS vienen dentro de HAPI. Para cada
isotopólogo que HAPI conoce (de las moléculas de MOLECULE_IDS) y cada versión
de TIPS (2017, 2021, 2025) revisa:

    cobertura    ¿existe la tabla para esa versión? (default de la librería: 2025)
    rango        Tmin/Tmax de la tabla; ¿cubre lo que pide from_hapi
                 (150–3000 K por defecto) y el rango atmosférico (150–350 K)?
    q296         Q(296 K) finito y > 0
    monotonia    Q(T) creciente en el rango atmosférico
    from_hapi    hm.get_tips_table con los defaults de from_hapi (lo que
                 realmente corre la librería) — aquí falla O3, por ejemplo
    versiones    diferencia relativa de Q(296 K) entre 2021 y 2025

Nunca se detiene: cada fallo se registra y sigue.

Uso:
    py tests/tips_check.py
    py tests/tips_check.py --molecules O3 CO2
    py tests/tips_check.py --t-min 150 --t-max 3000      # rango a exigir

Salida:
    resultados_test_tips/<fecha_hora>/
        RESUMEN.md        resumen y tabla de problemas
        resultados.csv    una fila por (molécula, isotopólogo, versión)
        test.log          log completo con tracebacks
"""

from __future__ import annotations

import argparse
import contextlib
import inspect
import csv
import io
import logging
import sys
import traceback
from datetime import datetime
from pathlib import Path
from typing import List, Optional

import numpy as np

_REPO_ROOT = Path(__file__).resolve().parents[1]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

from pylblrtm import hitran_manager as hm  # noqa: E402

VERSIONS = (2017, 2021, 2025)
# Lo que usa from_hapi si no se le pasa tips_version
DEFAULT_VERSION = inspect.signature(hm.get_tips_table).parameters["version"].default
ATM_RANGE = (150.0, 350.0)     # temperaturas que puede ver el forward model [K]
VERSION_DIFF_WARN = 0.01       # >1 % entre 2021 y 2025 en Q(296) se reporta

OK, WARN, ERROR, MISSING = "OK", "ADVERTENCIA", "ERROR", "SIN_TABLA"

_FIELDS = ["molecule", "mol_id", "iso_id", "version", "status", "t_min", "t_max",
           "q296", "covers_from_hapi", "covers_atm", "problems"]


def _setup_logging(path: Path) -> logging.Logger:
    log = logging.getLogger("tips_check")
    log.setLevel(logging.DEBUG)
    log.handlers.clear()
    fh = logging.FileHandler(path, encoding="utf-8")
    fh.setFormatter(logging.Formatter("%(asctime)s  %(levelname)-7s  %(message)s"))
    ch = logging.StreamHandler(sys.stdout)
    ch.setLevel(logging.INFO)
    ch.setFormatter(logging.Formatter("%(message)s"))
    log.addHandler(fh)
    log.addHandler(ch)
    return log


def _table(hapi, version: int, M: int, I: int):
    """Grilla de T de la tabla TIPS embebida en HAPI, o None si no existe."""
    hash_ = getattr(hapi, f"TIPS_{version}_ISOT_HASH", None)
    if hash_ is None:
        return None
    T = hash_.get((M, I))
    return None if T is None else np.asarray(T, dtype=float)


def check_one(hapi, mol: str, M: int, I: int, version: int,
              t_req: tuple, log: logging.Logger) -> dict:
    row = {"molecule": mol, "mol_id": M, "iso_id": I, "version": version,
           "status": OK, "t_min": "", "t_max": "", "q296": "",
           "covers_from_hapi": "", "covers_atm": "", "problems": []}
    errors, warns = [], []
    try:
        T = _table(hapi, version, M, I)
        if T is None:
            row["status"] = MISSING
            row["problems"] = ["HAPI no tiene tabla TIPS para esta versión"]
            return row
        row["t_min"], row["t_max"] = float(T[0]), float(T[-1])
        row["covers_from_hapi"] = bool(T[0] <= t_req[0] and T[-1] >= t_req[1])
        row["covers_atm"] = bool(T[0] <= ATM_RANGE[0] and T[-1] >= ATM_RANGE[1])
        if not row["covers_atm"]:
            errors.append(f"no cubre el rango atmosférico {ATM_RANGE[0]:.0f}–"
                          f"{ATM_RANGE[1]:.0f} K (tabla {T[0]:.0f}–{T[-1]:.0f} K)")
        if not row["covers_from_hapi"]:
            warns.append(f"no cubre {t_req[0]:.0f}–{t_req[1]:.0f} K "
                         f"(tabla {T[0]:.0f}–{T[-1]:.0f} K)")

        with contextlib.redirect_stdout(io.StringIO()):
            q296 = float(hapi.partitionSum(M, I, 296.0, version=version))
            Ta = np.arange(max(ATM_RANGE[0], T[0]), min(ATM_RANGE[1], T[-1]) + 1, 1.0)
            Qa = np.asarray(hapi.partitionSum(M, I, list(Ta), version=version), dtype=float)
        row["q296"] = q296
        if not (np.isfinite(q296) and q296 > 0):
            errors.append(f"Q(296 K) inválido: {q296}")
        if Qa.size and (not np.all(np.isfinite(Qa)) or np.any(Qa <= 0)):
            errors.append("Q(T) no finito o <= 0 en el rango atmosférico")
        if Qa.size > 1 and np.any(np.diff(Qa) <= 0):
            warns.append("Q(T) no es creciente en el rango atmosférico")
    except KeyboardInterrupt:
        raise
    except BaseException as exc:  # noqa: BLE001 — el test no se detiene
        errors.append(f"{type(exc).__name__}: {' '.join(str(exc).split())}")
        log.debug(traceback.format_exc())

    row["problems"] = errors + warns
    row["status"] = ERROR if errors else (WARN if warns else OK)
    return row


def check_library_path(mol: str, M: int, I: int, t_req: tuple, version: int,
                       log: logging.Logger) -> dict:
    """Lo que realmente ejecuta from_hapi: hm.get_tips_table con sus defaults."""
    row = {"molecule": mol, "mol_id": M, "iso_id": I, "version": f"{version} (from_hapi)",
           "status": OK, "t_min": t_req[0], "t_max": t_req[1], "q296": "",
           "covers_from_hapi": "", "covers_atm": "", "problems": []}
    try:
        with contextlib.redirect_stdout(io.StringIO()):
            T, Q = hm.get_tips_table(mol, I, T_min=t_req[0], T_max=t_req[1], version=version)
        if not np.all(np.isfinite(Q)) or np.any(Q <= 0):
            row["status"] = ERROR
            row["problems"] = ["tabla con valores no finitos o <= 0"]
    except KeyboardInterrupt:
        raise
    except BaseException as exc:  # noqa: BLE001
        row["status"] = ERROR
        row["problems"] = [f"{type(exc).__name__}: {' '.join(str(exc).split())[:300]}"]
        log.debug(traceback.format_exc())
    return row


def _write_summary(out: Path, rows: List[dict], cross: List[str], t_req: tuple) -> None:
    lib = [r for r in rows if "from_hapi" in str(r["version"])]
    tab = [r for r in rows if "from_hapi" not in str(r["version"])]
    md = [f"# Test TIPS — {datetime.now().isoformat(timespec='seconds')}", "",
          f"- Rango exigido (defaults de from_hapi): {t_req[0]:.0f}–{t_req[1]:.0f} K",
          f"- Rango atmosférico: {ATM_RANGE[0]:.0f}–{ATM_RANGE[1]:.0f} K", "",
          f"## Ruta real de la librería (`from_hapi` → `get_tips_table`, TIPS-{DEFAULT_VERSION})", "",
          f"- OK: {sum(r['status'] == OK for r in lib)} · "
          f"**ERROR: {sum(r['status'] == ERROR for r in lib)}** de {len(lib)} isotopólogos", ""]
    bad = [r for r in lib if r["status"] == ERROR]
    if bad:
        md += ["| Molécula | Iso | Error |", "|---|---|---|"]
        md += [f"| {r['molecule']} | {r['iso_id']} | {'; '.join(r['problems'])} |" for r in bad]
        md.append("")

    md += ["## Tablas TIPS por versión", "", "| Versión | OK | ADVERTENCIA | ERROR | SIN_TABLA |",
           "|---|---|---|---|---|"]
    for v in VERSIONS:
        rv = [r for r in tab if r["version"] == v]
        md.append(f"| {v} | " + " | ".join(str(sum(r["status"] == s for r in rv))
                                           for s in (OK, WARN, ERROR, MISSING)) + " |")
    md.append("")

    probs = [r for r in tab if r["status"] in (ERROR, WARN, MISSING) and r["version"] in (2021, 2025)]
    if probs:
        md += ["### Problemas en 2021 / 2025", "",
               "| Molécula | Iso | Versión | Estado | Tabla [K] | Detalle |", "|---|---|---|---|---|---|"]
        for r in probs:
            rng = f"{r['t_min']:.0f}–{r['t_max']:.0f}" if r["t_min"] != "" else "—"
            md.append(f"| {r['molecule']} | {r['iso_id']} | {r['version']} | {r['status']} | "
                      f"{rng} | {'; '.join(r['problems'])} |")
        md.append("")
    if cross:
        md += [f"## Diferencias 2021 vs 2025 en Q(296 K) > {VERSION_DIFF_WARN:.0%}", ""]
        md += [f"- {c}" for c in cross]
        md.append("")
    (out / "RESUMEN.md").write_text("\n".join(md), encoding="utf-8")


def main(argv: Optional[List[str]] = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--molecules", nargs="+", default=None)
    ap.add_argument("--t-min", type=float, default=150.0, help="default de from_hapi")
    ap.add_argument("--t-max", type=float, default=3000.0, help="default de from_hapi")
    ap.add_argument("--out", type=Path, default=_REPO_ROOT / "resultados_test_tips")
    args = ap.parse_args(argv)

    for stream in (sys.stdout, sys.stderr):
        with contextlib.suppress(AttributeError, ValueError):
            stream.reconfigure(encoding="utf-8", errors="replace")

    out = args.out / datetime.now().strftime("%Y%m%d_%H%M%S")
    out.mkdir(parents=True, exist_ok=True)
    log = _setup_logging(out / "test.log")
    t_req = (args.t_min, args.t_max)

    with contextlib.redirect_stdout(io.StringIO()):
        hapi = hm._get_hapi()
    molecules = ([hm._resolve_mol_name(m) for m in args.molecules]
                 if args.molecules else list(hm.MOLECULE_IDS))
    log.info(f"Test TIPS — {len(molecules)} moléculas, rango exigido {t_req[0]:.0f}–{t_req[1]:.0f} K")
    log.info(f"Resultados en: {out}")

    rows: List[dict] = []
    cross: List[str] = []
    for mol in molecules:
        M = hm.MOLECULE_IDS[mol]
        isos = sorted(i for (m, i) in hapi.ISO if m == M)
        if not isos:
            log.info(f"  [   ERROR   ] {mol:>7}  HAPI no tiene isotopólogos")
            rows.append({"molecule": mol, "mol_id": M, "iso_id": "-", "version": "-",
                         "status": ERROR, "t_min": "", "t_max": "", "q296": "",
                         "covers_from_hapi": "", "covers_atm": "",
                         "problems": ["HAPI no tiene isotopólogos"]})
            continue
        for I in isos:
            by_v = {v: check_one(hapi, mol, M, I, v, t_req, log) for v in VERSIONS}
            rows.extend(by_v.values())
            lib = check_library_path(mol, M, I, t_req, DEFAULT_VERSION, log)
            rows.append(lib)

            q21, q25 = by_v[2021]["q296"], by_v[2025]["q296"]
            if q21 != "" and q25 != "" and q21 > 0:
                d = abs(q25 - q21) / q21
                if d > VERSION_DIFF_WARN:
                    cross.append(f"{mol} iso {I}: Q21={q21:.4g}  Q25={q25:.4g}  ({d:.1%})")

            states = " ".join(f"{v}:{by_v[v]['status'][:4]}" for v in VERSIONS)
            log.info(f"  [{lib['status']:^11}] {mol:>7} iso {I:>2}  from_hapi  |  {states}")
            for p in lib["problems"]:
                log.info(f"                {p}")

    with (out / "resultados.csv").open("w", newline="", encoding="utf-8") as fp:
        w = csv.DictWriter(fp, fieldnames=_FIELDS)
        w.writeheader()
        for r in rows:
            w.writerow({**r, "problems": " | ".join(r["problems"])})
    _write_summary(out, rows, cross, t_req)

    n_lib_err = sum(r["status"] == ERROR for r in rows if "from_hapi" in str(r["version"]))
    log.info(f"\nfrom_hapi con TIPS-{DEFAULT_VERSION} falla en {n_lib_err} isotopólogos.  "
             f"Reporte: {out / 'RESUMEN.md'}")
    return 1 if n_lib_err else 0


if __name__ == "__main__":
    sys.exit(main())
