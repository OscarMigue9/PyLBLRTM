"""
forward_check.py — Test del forward model con datos reales de HITRAN.

Corre compute_transmittance / run_forward en la ventana 8–14 µm con las líneas
de HITRAN2024 de cada molécula, en un camino standoff homogéneo y en la
atmósfera estándar de EE.UU. Nunca se detiene: cada fallo se registra y se
sigue con lo demás.

Partes:
  1. sanidad      cada molécula con líneas (todos sus isotopólogos, Humlicek,
                  n_jobs hilos): T ∈ [0, 1], sin NaN, τ ≥ 0, en
                  · standoff: 1 ppm en 100 m a 1013.25 hPa y 296 K (todas)
                  · atmósfera estándar (los gases que tienen perfil)
  2. consistencia H2O CO2 O3 N2O CH4 en la atmósfera estándar y NH3 standoff,
                  ventana completa:
                  · n_jobs=1 vs n_jobs=N         → τ idéntica bit a bit
                  · tile_width 2 vs 7 cm⁻¹        → τ igual (rtol 1e-12)
                  · wing_tolerance=1e-4 vs exacto → error relativo ≤ 1e-4
                  · humlicek vs wofz (900–950)    → error relativo ≤ 1e-4
  3. física       atmósfera con los 5 gases a la vez:
                  · ventana 800–980 cm⁻¹ transparente (T media > 0.8)
                  · banda de CO2 714–730 cm⁻¹ opaca   (T media < 0.5)
                  · banda de O3 1000–1070 cm⁻¹        (T mín < 0.05)
                  · mu_view=0.5 duplica τ
                  standoff de NH3:
                  · el doble de longitud duplica τ (exacto)
                  · 2 ppm × 50 m ≈ 1 ppm × 100 m  (área de τ, rtol 1e-3)

Usa la caché del último test de descarga (resultados_test_hapi/*/cache) si
existe; si no, descarga a su propia caché.

Uso:
    py tests/forward_check.py                         # todo
    py tests/forward_check.py --n-jobs 16 --step 0.01
    py tests/forward_check.py --parts sanidad --molecules CO2 O3

Salida: resultados_test_forward/<fecha_hora>/ con RESUMEN.md, resultados.csv
y test.log. Termina con código 1 solo si hay algún ERROR.
"""

from __future__ import annotations

import argparse
import contextlib
import csv
import io
import logging
import os
import platform
import sys
import time
import traceback
from datetime import datetime
from pathlib import Path
from typing import Callable, List, Optional

import numpy as np

_REPO_ROOT = Path(__file__).resolve().parents[1]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

import pylblrtm  # noqa: E402
from pylblrtm import hitran_manager as hm  # noqa: E402
from pylblrtm.atmosphere import HomogeneousPath  # noqa: E402
from pylblrtm.config import run_forward  # noqa: E402
from pylblrtm.data_loader import from_hapi  # noqa: E402
from pylblrtm.profiles import list_profile_gases, us_standard_atmosphere  # noqa: E402
from pylblrtm.spectral_data import SpectralData  # noqa: E402

PARTS = ("sanidad", "consistencia", "fisica")
MAIN_GASES = ("H2O", "CO2", "O3", "N2O", "CH4")
STANDOFF_PPM, STANDOFF_M = 1.0, 100.0   # firma standoff de cada molécula
STANDOFF_GAS = "NH3"        # banda fuerte en 10.3–10.7 µm
TOL = 1e-4                  # humlicek vs wofz y wing_tolerance
VOIGT = "humlicek"          # --voigt wofz solo para pruebas locales sin numba

OK, WARN, SKIP, EXC, ERROR = "OK", "ADVERTENCIA", "OMITIDA", "EXCEPCION", "ERROR"

_FIELDS = ["parte", "caso", "status", "valor", "limite", "detalle", "seconds"]


# ── Utilidades ────────────────────────────────────────────────────────────────

def _setup_logging(path: Path) -> logging.Logger:
    log = logging.getLogger("forward_check")
    log.setLevel(logging.DEBUG)
    log.handlers.clear()
    fh = logging.FileHandler(path, encoding="utf-8")
    fh.setFormatter(logging.Formatter("%(asctime)s  %(levelname)-7s  %(message)s"))
    ch = logging.StreamHandler(sys.stdout)
    ch.setLevel(logging.INFO)
    ch.setFormatter(logging.Formatter("%(message)s"))
    log.addHandler(fh)
    log.addHandler(ch)
    pl = logging.getLogger("pylblrtm")
    pl.setLevel(logging.WARNING)
    pl.addHandler(fh)
    return log


def _latest_cache() -> Optional[Path]:
    runs = sorted((_REPO_ROOT / "resultados_test_hapi").glob("*/cache"))
    return runs[-1] if runs else None


def _quiet(fn: Callable, *a, **kw):
    """HAPI imprime mucho por pantalla; se descarta."""
    with contextlib.redirect_stdout(io.StringIO()):
        return fn(*a, **kw)


def _standoff(gas: str, ppm: float = STANDOFF_PPM, length_m: float = STANDOFF_M):
    return HomogeneousPath(length_m, {gas: ppm * 1e-6})


def _merge(sds: List[SpectralData]) -> SpectralData:
    lines = {k: np.concatenate([sd.lines[k] for sd in sds]) for k in sds[0].lines}
    tips, masses = {}, {}
    for sd in sds:
        tips.update(sd.tips)
        masses.update(sd.molar_masses)
    return SpectralData(lines=lines, tips=tips, molar_masses=masses, molecule=None)


def _run(sd, profile, nu_min, nu_max, step, **calc):
    """τ por capa [n_layers, n_nu] y transmitancia, vía run_forward."""
    cfg = {
        "spectral": {"nu_min_cm1": nu_min, "nu_max_cm1": nu_max, "step_cm1": step},
        "calculation": {"voigt": VOIGT, **calc},
        "output": {"transmittance": True, "optical_depth": False,
                   "optical_depth_layers": True},
    }
    res = _quiet(run_forward, sd, profile, cfg)
    return res.wavenumber_cm1, res.optical_depth_layers, res.transmittance


def _rel_err(got: np.ndarray, ref: np.ndarray) -> float:
    """Máximo error relativo donde la referencia es > 0."""
    m = ref > 0
    return float(np.max(np.abs(got[m] - ref[m]) / ref[m])) if m.any() else 0.0


class Recorder:
    def __init__(self, out: Path, log: logging.Logger):
        self.rows: List[dict] = []
        self.csv = out / "resultados.csv"
        self.log = log

    def add(self, parte, caso, status, valor="", limite="", detalle="", seconds=0.0):
        row = {"parte": parte, "caso": caso, "status": status, "valor": valor,
               "limite": limite, "detalle": detalle, "seconds": round(seconds, 1)}
        self.rows.append(row)
        new = not self.csv.exists()
        with self.csv.open("a", newline="", encoding="utf-8") as fp:
            w = csv.DictWriter(fp, fieldnames=_FIELDS)
            if new:
                w.writeheader()
            w.writerow(row)
        extra = f"  {valor} (límite {limite})" if limite != "" else (f"  {valor}" if valor != "" else "")
        self.log.info(f"  [{status:^11}] {parte:<12} {caso:<34}{extra}  {seconds:6.1f} s"
                      + (f"\n                {detalle}" if detalle else ""))

    def guard(self, parte: str, caso: str):
        """Context manager: una excepción inesperada queda como ERROR y se sigue."""
        rec = self

        class _G:
            def __enter__(self):
                self.t0 = time.perf_counter()
                return self

            def __exit__(self, et, ev, tb):
                if et is None or et is KeyboardInterrupt:
                    return False
                rec.log.debug(traceback.format_exc())
                status = EXC if isinstance(ev, hm.HITRANDataError) else ERROR
                rec.add(parte, caso, status, detalle=f"{et.__name__}: {' '.join(str(ev).split())[:300]}",
                        seconds=time.perf_counter() - self.t0)
                return True

        return _G()


# ── Partes ────────────────────────────────────────────────────────────────────

def load(mol: str, nu_min: float, nu_max: float, cache: Optional[Path], own_cache: Path):
    """SpectralData de todos los isotopólogos, o (status, motivo) si no hay líneas."""
    cdir = cache / mol if cache is not None and (cache / mol).is_dir() else own_cache / mol
    try:
        return _quiet(from_hapi, mol, nu_min, nu_max, iso_id="all", cache_dir=cdir), None
    except hm.HITRANNoLinesError:
        return None, "sin líneas en el rango"
    except hm.HITRANNotAvailableError:
        return None, "HITRANonline no la sirve línea por línea"


def _sanity_case(rec, caso, sd, path, nu_min, nu_max, step, n_jobs, t0):
    nu, od, T = _run(sd, path, nu_min, nu_max, step, n_jobs=n_jobs)
    problems = []
    if not np.all(np.isfinite(T)):
        problems.append(f"{int(np.count_nonzero(~np.isfinite(T)))} valores de T no finitos")
    if np.any((T < 0) | (T > 1)):
        problems.append("T fuera de [0, 1]")
    if np.any(od < 0) or not np.all(np.isfinite(od)):
        problems.append("τ negativa o no finita")
    val = (f"{sd.n_lines} líneas, T mín {T.min():.3g} en {nu[T.argmin()]:.2f} cm⁻¹, "
           f"media {T.mean():.4f}")
    status = ERROR if problems else (WARN if T.min() == 1.0 else OK)
    detalle = "; ".join(problems) or ("no absorbe nada (T = 1)" if status == WARN else "")
    rec.add("sanidad", caso, status, valor=val, detalle=detalle,
            seconds=time.perf_counter() - t0)


def part_sanity(rec, molecules, nu_min, nu_max, step, n_jobs, cache, own_cache):
    for mol in molecules:
        sd = None
        with rec.guard("sanidad", mol) as g:
            sd, why = load(mol, nu_min, nu_max, cache, own_cache)
            if sd is None:
                rec.add("sanidad", mol, SKIP, detalle=why, seconds=time.perf_counter() - g.t0)
        if sd is None:
            continue
        caso = f"{mol} standoff {STANDOFF_PPM:g} ppm·{STANDOFF_M:g} m"
        with rec.guard("sanidad", caso) as g:
            _sanity_case(rec, caso, sd, _standoff(mol), nu_min, nu_max, step, n_jobs, g.t0)
        if mol in list_profile_gases():
            caso = f"{mol} atmósfera estándar"
            with rec.guard("sanidad", caso) as g:
                _sanity_case(rec, caso, sd, us_standard_atmosphere(gases=[mol]),
                             nu_min, nu_max, step, n_jobs, g.t0)


def part_consistency(rec, nu_min, nu_max, step, n_jobs, cache, own_cache):
    cases = [(m, m, lambda m=m: us_standard_atmosphere(gases=[m])) for m in MAIN_GASES]
    cases.append((STANDOFF_GAS, f"{STANDOFF_GAS} standoff", lambda: _standoff(STANDOFF_GAS)))
    for gas, mol, make_path in cases:
        sd = None
        with rec.guard("consistencia", f"{mol} carga") as g:
            sd, why = load(gas, nu_min, nu_max, cache, own_cache)
            if sd is None:
                rec.add("consistencia", mol, SKIP, detalle=why)
        if sd is None:
            continue
        prof = make_path()

        with rec.guard("consistencia", f"{mol} n_jobs 1 vs {n_jobs}") as g:
            _, serial, _ = _run(sd, prof, nu_min, nu_max, step, n_jobs=1)
            _, threaded, _ = _run(sd, prof, nu_min, nu_max, step, n_jobs=n_jobs)
            same = np.array_equal(serial, threaded)
            rec.add("consistencia", f"{mol} n_jobs 1 vs {n_jobs}", OK if same else ERROR,
                    valor="idéntica bit a bit" if same else f"difiere: {_rel_err(threaded, serial):.2e}",
                    limite="bit a bit", seconds=time.perf_counter() - g.t0)

        with rec.guard("consistencia", f"{mol} tile 2 vs 7") as g:
            _, other, _ = _run(sd, prof, nu_min, nu_max, step, n_jobs=n_jobs, tile_width_cm1=7.0)
            err = _rel_err(other, threaded)
            rec.add("consistencia", f"{mol} tile 2 vs 7", OK if err <= 1e-12 else ERROR,
                    valor=f"{err:.2e}", limite="1e-12", seconds=time.perf_counter() - g.t0)

        with rec.guard("consistencia", f"{mol} wing_tolerance {TOL:g}") as g:
            _, wing, _ = _run(sd, prof, nu_min, nu_max, step, n_jobs=n_jobs, wing_tolerance=TOL)
            err = _rel_err(wing, threaded)
            # Donde τ < 1e-10·max no importa para T (e^-τ = 1 a doble precisión)
            m = threaded > 1e-10 * threaded.max()
            err_rel = _rel_err(wing[m], threaded[m])
            status = OK if err <= TOL else (WARN if err_rel <= TOL else ERROR)
            rec.add("consistencia", f"{mol} wing_tolerance {TOL:g}", status,
                    valor=f"{err:.2e} (τ relevante: {err_rel:.2e})", limite=f"{TOL:g}",
                    detalle="" if status == OK else "solo excede donde τ < 1e-10·max",
                    seconds=time.perf_counter() - g.t0)

        with rec.guard("consistencia", f"{mol} humlicek vs wofz") as g:
            lo, hi = 900.0, 950.0
            _, hum, _ = _run(sd, prof, lo, hi, step, n_jobs=n_jobs)
            _, wofz, _ = _run(sd, prof, lo, hi, step, n_jobs=n_jobs, voigt="wofz")
            err = _rel_err(hum, wofz)
            rec.add("consistencia", f"{mol} humlicek vs wofz", OK if err <= TOL else ERROR,
                    valor=f"{err:.2e}", limite=f"{TOL:g}", detalle=f"ventana {lo:.0f}–{hi:.0f} cm⁻¹",
                    seconds=time.perf_counter() - g.t0)


def part_physics(rec, nu_min, nu_max, step, n_jobs, cache, own_cache):
    sd = None
    with rec.guard("fisica", "carga") as g:
        sds = []
        for mol in MAIN_GASES:
            sd, _ = load(mol, nu_min, nu_max, cache, own_cache)
            if sd is not None:
                sds.append(sd)
        sd = _merge(sds)
        prof = us_standard_atmosphere(gases=list(MAIN_GASES))
        nu, od, T = _run(sd, prof, nu_min, nu_max, step, n_jobs=n_jobs)
        rec.add("fisica", "atmósfera estándar, 5 gases", OK,
                valor=f"{sd.n_lines} líneas, {od.shape[0]} capas", seconds=time.perf_counter() - g.t0)

        checks = [
            ("ventana 800–980 T media", 800, 980, np.mean, "> 0.8", lambda v: v > 0.8),
            ("CO2 714–730 T media", 714.3, 730, np.mean, "< 0.5", lambda v: v < 0.5),
            ("O3 1000–1070 T mín", 1000, 1070, np.min, "< 0.05", lambda v: v < 0.05),
            ("O3 1000–1070 T media", 1000, 1070, np.mean, "< 0.8", lambda v: v < 0.8),
        ]
        for name, lo, hi, stat, lim, ok in checks:
            sel = T[(nu >= lo) & (nu <= hi)]
            if sel.size == 0:
                rec.add("fisica", name, SKIP, detalle="la banda está fuera del rango")
                continue
            val = float(stat(sel))
            rec.add("fisica", name, OK if ok(val) else ERROR, valor=f"{val:.4f}", limite=lim)

    part_standoff_physics(rec, nu_min, nu_max, step, n_jobs, cache, own_cache)

    if sd is None:
        return
    with rec.guard("fisica", "mu_view 0.5 duplica τ") as g:
        _, _, T_slant = _run(sd, prof, nu_min, nu_max, step, n_jobs=n_jobs, mu_view=0.5)
        # -ln T pierde precisión cuando T ≈ 1 (τ < 1e-6) o T underflow (τ > 600)
        m = (T > 1e-250) & (T < 1 - 1e-6) & (T_slant > 1e-250)
        tau1, tau2 = -np.log(T[m]), -np.log(T_slant[m])
        err = float(np.max(np.abs(tau2 - 2 * tau1) / (2 * tau1)))
        rec.add("fisica", "mu_view 0.5 duplica τ", OK if err <= 1e-8 else ERROR,
                valor=f"{err:.2e}", limite="1e-8", seconds=time.perf_counter() - g.t0)


def part_standoff_physics(rec, nu_min, nu_max, step, n_jobs, cache, own_cache):
    """Beer-Lambert en un camino homogéneo con líneas reales."""
    from scipy.integrate import trapezoid

    sd = None
    with rec.guard("fisica", f"{STANDOFF_GAS} standoff carga"):
        sd, why = load(STANDOFF_GAS, nu_min, nu_max, cache, own_cache)
        if sd is None:
            rec.add("fisica", f"{STANDOFF_GAS} standoff", SKIP, detalle=why)
    if sd is None:
        return

    with rec.guard("fisica", f"{STANDOFF_GAS} 200 m duplica τ de 100 m") as g:
        _, od100, _ = _run(sd, _standoff(STANDOFF_GAS, 1.0, 100.0), nu_min, nu_max, step, n_jobs=n_jobs)
        nu, od200, _ = _run(sd, _standoff(STANDOFF_GAS, 1.0, 200.0), nu_min, nu_max, step, n_jobs=n_jobs)
        err = _rel_err(od200, 2.0 * od100)
        rec.add("fisica", f"{STANDOFF_GAS} 200 m duplica τ de 100 m", OK if err <= 1e-10 else ERROR,
                valor=f"{err:.2e}", limite="1e-10", seconds=time.perf_counter() - g.t0)

    with rec.guard("fisica", f"{STANDOFF_GAS} 2 ppm·50 m ≈ 1 ppm·100 m") as g:
        _, od_b, _ = _run(sd, _standoff(STANDOFF_GAS, 2.0, 50.0), nu_min, nu_max, step, n_jobs=n_jobs)
        a, b = trapezoid(od100[0], nu), trapezoid(od_b[0], nu)
        err = abs(b - a) / a
        # Solo el ensanchamiento propio (g_self·P_self) cambia con la concentración
        rec.add("fisica", f"{STANDOFF_GAS} 2 ppm·50 m ≈ 1 ppm·100 m", OK if err <= 1e-3 else ERROR,
                valor=f"área {err:.2e}", limite="1e-3", seconds=time.perf_counter() - g.t0)


# ── Reporte ───────────────────────────────────────────────────────────────────

def _write_summary(out: Path, rows: List[dict], meta: dict) -> None:
    md = [f"# Test del forward model — {meta['inicio']}", "",
          f"- Rango: {meta['nu_min']:.3f}–{meta['nu_max']:.3f} cm⁻¹, paso {meta['step']} cm⁻¹, "
          f"n_jobs={meta['n_jobs']}",
          f"- Líneas de: {meta['cache']}",
          f"- pylblrtm {pylblrtm.__version__} · Python {platform.python_version()} · "
          f"{platform.node()}",
          f"- Duración: {meta.get('duracion_min', '?')} min", "",
          "| Estado | Cantidad |", "|---|---|"]
    for s in (OK, WARN, SKIP, EXC, ERROR):
        n = sum(r["status"] == s for r in rows)
        md.append(f"| {'**' + s + '**' if s == ERROR else s} | {n} |")
    md.append("")
    for parte in PARTS:
        pr = [r for r in rows if r["parte"] == parte]
        if not pr:
            continue
        md += [f"## {parte.capitalize()}", "", "| Caso | Estado | Valor | Límite | Detalle |",
               "|---|---|---|---|---|"]
        for r in pr:
            md.append(f"| {r['caso']} | {r['status']} | {r['valor']} | {r['limite']} | "
                      f"{str(r['detalle']).replace('|', '/')} |")
        md.append("")
    (out / "RESUMEN.md").write_text("\n".join(md), encoding="utf-8")


def main(argv: Optional[List[str]] = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--um", nargs=2, type=float, default=(8.0, 14.0))
    ap.add_argument("--step", type=float, default=0.01, help="paso espectral [cm⁻¹]")
    ap.add_argument("--n-jobs", type=int,
                    default=int(os.environ.get("SLURM_CPUS_PER_TASK", os.cpu_count() or 1)))
    ap.add_argument("--parts", nargs="+", choices=PARTS, default=list(PARTS))
    ap.add_argument("--molecules", nargs="+", default=None, help="solo para 'sanidad'")
    ap.add_argument("--cache", type=Path, default=None,
                    help="caché con una carpeta por molécula (default: último test de descarga)")
    ap.add_argument("--out", type=Path, default=_REPO_ROOT / "resultados_test_forward")
    ap.add_argument("--voigt", choices=("humlicek", "wofz"), default="humlicek",
                    help="wofz solo para probar sin numba (wing_tolerance fallará)")
    args = ap.parse_args(argv)
    global VOIGT
    VOIGT = args.voigt

    for stream in (sys.stdout, sys.stderr):
        with contextlib.suppress(AttributeError, ValueError):
            stream.reconfigure(encoding="utf-8", errors="replace")

    um_lo, um_hi = sorted(args.um)
    nu_min, nu_max = 1e4 / um_hi, 1e4 / um_lo     # mismos límites que hapi_download_check
    out = args.out / datetime.now().strftime("%Y%m%d_%H%M%S")
    out.mkdir(parents=True, exist_ok=True)
    log = _setup_logging(out / "test.log")
    cache = args.cache or _latest_cache()
    own_cache = out / "cache"
    molecules = ([hm._resolve_mol_name(m) for m in args.molecules]
                 if args.molecules else list(hm.MOLECULE_IDS))

    meta = {"inicio": datetime.now().isoformat(timespec="seconds"), "nu_min": nu_min,
            "nu_max": nu_max, "step": args.step, "n_jobs": args.n_jobs,
            "cache": str(cache) if cache else f"descarga nueva en {own_cache}"}
    log.info(f"Test del forward — {nu_min:.3f}–{nu_max:.3f} cm⁻¹, paso {args.step}, "
             f"n_jobs={args.n_jobs}")
    log.info(f"Líneas de: {meta['cache']}")
    log.info(f"Resultados en: {out}")

    rec = Recorder(out, log)
    t0 = time.perf_counter()
    try:
        # Compila los kernels de numba una vez antes de medir tiempos
        if VOIGT == "humlicek":
            with rec.guard("sanidad", "numba (humlicek)"):
                from pylblrtm import voigt
                voigt._require_numba()
        if "sanidad" in args.parts:
            log.info("\n── 1. Sanidad ─────────────────────────────────────────────")
            part_sanity(rec, molecules, nu_min, nu_max, args.step, args.n_jobs, cache, own_cache)
        if "consistencia" in args.parts:
            log.info("\n── 2. Consistencia ───────────────────────────────────────")
            part_consistency(rec, nu_min, nu_max, args.step, args.n_jobs, cache, own_cache)
        if "fisica" in args.parts:
            log.info("\n── 3. Física ─────────────────────────────────────────────")
            part_physics(rec, nu_min, nu_max, args.step, args.n_jobs, cache, own_cache)
    except KeyboardInterrupt:
        log.warning("Interrumpido: reporte parcial")
    finally:
        meta["duracion_min"] = round((time.perf_counter() - t0) / 60, 1)
        _write_summary(out, rec.rows, meta)

    counts = {s: sum(r["status"] == s for r in rec.rows) for s in (OK, WARN, SKIP, EXC, ERROR)}
    log.info("\n" + "   ".join(f"{k}: {v}" for k, v in counts.items()))
    log.info(f"Reporte: {out / 'RESUMEN.md'}")
    return 1 if counts[ERROR] else 0


if __name__ == "__main__":
    sys.exit(main())
