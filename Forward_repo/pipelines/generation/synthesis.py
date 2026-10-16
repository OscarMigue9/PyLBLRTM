"""Candidate historical T/U/D and radiance synthesis on a CUDA GPU.

This implementation follows the numerical order in Photonix production files at
commit ``9f93c955779c11f7a8018e9200d63dddfa22ba23``. It computes the public
714..1250 cm^-1 channels, retaining the original 645 cm^-1 tile anchor and a
2 cm^-1 ILS guard. Archived 645..2760 T/U/D and radiation were computed before
that commit and no source hash was recorded, so parity remains an explicit gate.
The five-gas 30k states are sufficient inside this band: CO and O2/N2 CIA
contribute outside it. Output chunks hold all five view cosines to preserve the
historical angle assignment when assembling radiance.
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np

from .lut import blocks_for, inspect_blocks, sha256_file


GASES = ("H2O", "CO2", "O3", "N2O", "CH4")
VMR_REFERENCE = {"H2O": 0.01, "CO2": 4e-4, "O3": 2.66e-8, "N2O": 3.2e-7, "CH4": 1.7e-6}
VIEW_COSINES = np.array([0.55, 0.67, 0.80, 0.92, 1.0])
C1 = 1.19104272e-5
C2 = 1.4387752
FINE_STEP = 0.001
CHANNEL_STEP = 0.25
ANCHOR = 645.0
TILE_WIDTH = 50.0
GUARD = 2.0


def _load_states(path: Path) -> dict[str, np.ndarray]:
    with np.load(path, allow_pickle=False) as archive:
        required = ("gases", "T_lay", "P_lay", "Ncol", "VMR_H2O", "T_levels", "T_skin", "lsm")
        if any(key not in archive for key in required):
            raise ValueError("The five-gas states archive is missing required arrays")
        states = {key: archive[key] for key in required}
    if tuple(states["gases"].tolist()) != GASES:
        raise ValueError("The states gas order differs from the historical five-gas order")
    n, layers = states["T_lay"].shape
    if states["P_lay"].shape != (n, layers) or states["Ncol"].shape != (n, 5, layers):
        raise ValueError("Inconsistent state shapes")
    if states["T_levels"].shape != (n, layers + 1):
        raise ValueError("Layer and level counts do not match")
    return states


def _load_broadening(path: Path) -> dict[str, np.ndarray]:
    with np.load(path, allow_pickle=False) as archive:
        return {key: archive[key] for key in archive.files}


def _source_block(block, cache: dict, cp) -> tuple:
    key = block.path
    if key not in cache:
        with np.load(key, allow_pickle=False) as archive:
            cache[key] = (
                cp.asarray(archive["U"].astype(np.float32)),
                cp.asarray(archive["V"].astype(np.float32)),
                cp.asarray(archive["mu"].astype(np.float32)),
                int(archive["NT"]),
                int(archive["NP"]),
                float(archive["T_LO"]),
                float(archive["T_HI"]),
                float(archive["P_LO"]),
                float(archive["P_HI"]),
            )
    return cache[key]


def _sigma(
    gas: str,
    nu,
    temperature,
    pressure,
    vmr,
    block_dir: Path,
    ratios: dict[str, np.ndarray],
    cache: dict,
    cp,
):
    """Interpolate SVD coefficients in T/logP before spectral reconstruction."""
    n, layers = temperature.shape
    start, end = float(nu[0].item()), float(nu[-1].item())
    result = cp.zeros((n * layers, nu.size), dtype=cp.float64)
    for block in blocks_for(block_dir, gas, start, end):
        u, v, mean, nt, npres, t_low, t_high, p_low, p_high = _source_block(block, cache, cp)
        t_axis = cp.linspace(t_low, t_high, nt)
        p_axis = cp.log10(cp.logspace(cp.log10(p_low), cp.log10(p_high), npres))
        ratio_index = int(round((block.start - float(ratios["bordes"][0])) / 10.0))
        ratio = float(ratios[gas][ratio_index])
        effective = pressure * (1 + (ratio - 1) * vmr) / (1 + (ratio - 1) * VMR_REFERENCE[gas])
        log_pressure = cp.log10(cp.clip(effective, 1e-30, None)).ravel()
        t_flat = temperature.ravel()
        it = cp.clip(cp.searchsorted(t_axis, t_flat) - 1, 0, nt - 2)
        ip = cp.clip(cp.searchsorted(p_axis, log_pressure) - 1, 0, npres - 2)
        wt = cp.clip((t_flat - t_axis[it]) / (t_axis[it + 1] - t_axis[it]), 0, 1)[:, None]
        wp = cp.clip((log_pressure - p_axis[ip]) / (p_axis[ip + 1] - p_axis[ip]), 0, 1)[:, None]

        def indexed(i, j):
            return u[i * npres + j]

        interpolated = (
            indexed(it, ip) * (1 - wt) * (1 - wp)
            + indexed(it + 1, ip) * wt * (1 - wp)
            + indexed(it, ip + 1) * (1 - wt) * wp
            + indexed(it + 1, ip + 1) * wt * wp
        )
        cross_section = cp.power(10.0, (mean[None, :] + interpolated @ v).astype(cp.float64))
        first = max(start, block.start)
        last = min(end, block.end)
        source_start = int(round((first - block.start) / FINE_STEP))
        destination_start = int(round((first - start) / FINE_STEP))
        count = int(round((last - first) / FINE_STEP)) + 1
        result[:, destination_start : destination_start + count] = cross_section[
            :, source_start : source_start + count
        ]
    return result.reshape(n, layers, nu.size)


def _h2o_continuum(nu, temperature, pressure, columns, water_vmr, table, cp):
    nu_cpu = cp.asnumpy(nu)
    cs = cp.asarray(np.exp(np.interp(nu_cpu, table["h2o_NU_TABLE"], np.log(table["h2o_CS_REF"]))))
    cf = cp.asarray(np.exp(np.interp(nu_cpu, table["h2o_NU_TABLE"], np.log(table["h2o_CF_REF"]))))
    power = cp.asarray(np.interp(nu_cpu, table["h2o_NU_TABLE"], table["h2o_TEXP_REF"]))
    t_ref = float(table["h2o_T_REF"])
    p_ref = float(table["h2o_P_REF_HP"])
    t, p, vmr = temperature[:, :, None], pressure[:, :, None], water_vmr[:, :, None]
    exponent = nu[None, None, :] * C2 / t
    radiation = cp.where(
        exponent <= 0.01,
        0.5 * exponent * nu[None, None, :],
        cp.where(
            exponent <= 10,
            nu[None, None, :] * (1 - cp.exp(-exponent)) / (1 + cp.exp(-exponent)),
            nu[None, None, :],
        ),
    )
    density = (p * 1013.25 / p_ref) * (t_ref / t)
    depth = (
        cs * (t_ref / t) ** power * vmr * density * radiation + cf * (1 - vmr) * density * radiation
    ) * columns[:, :, None]
    return cp.maximum(depth, 0.0)


def _layer_emission(planck_near, planck_far, trans_near, trans_far, depth, cp):
    change = trans_near - trans_far
    safe_depth = cp.where(depth > 1e-12, depth, 1e-12)
    return cp.where(
        depth > 1e-12,
        planck_near * change + (planck_far - planck_near) * (change / safe_depth - trans_far),
        0.5 * (planck_near + planck_far) * change,
    )


def _up(depth, planck, cp):
    trans = cp.exp(-depth)
    cumulative = cp.concatenate([cp.ones_like(trans[:, :1]), cp.cumprod(trans, axis=1)], axis=1)
    up = sum(
        _layer_emission(
            planck[:, layer],
            planck[:, layer + 1],
            cumulative[:, layer],
            cumulative[:, layer + 1],
            depth[:, layer],
            cp,
        )
        for layer in range(depth.shape[1])
    )
    return up, cumulative[:, -1]


def _down(depth, planck, cp):
    diffuse = depth * 1.66
    trans = cp.exp(-diffuse)
    cumulative = cp.concatenate(
        [cp.flip(cp.cumprod(cp.flip(trans, 1), axis=1), 1), cp.ones_like(trans[:, :1])], axis=1
    )
    return sum(
        _layer_emission(
            planck[:, layer + 1],
            planck[:, layer],
            cumulative[:, layer + 1],
            cumulative[:, layer],
            diffuse[:, layer],
            cp,
        )
        for layer in range(depth.shape[1])
    )


def _tiles(start: float, end: float):
    first = ANCHOR + np.floor((start - ANCHOR) / TILE_WIDTH) * TILE_WIDTH
    for tile_start in np.arange(first, end, TILE_WIDTH):
        tile_end = min(tile_start + TILE_WIDTH, end)
        if tile_end > start:
            yield float(tile_start), float(tile_end)


def build_tud_chunks(
    states_path: Path,
    block_dir: Path,
    ratios_path: Path,
    continuum_path: Path,
    output_dir: Path,
    *,
    start: float = 714.0,
    end: float = 1250.0,
    count: int | None = None,
    batch_size: int = 16,
) -> dict:
    """Write resumable spectral chunks for all historical viewing cosines."""
    if (start, end) != (714.0, 1250.0) or batch_size < 1:
        raise ValueError("The candidate output is defined on 714..1250 cm^-1")
    try:
        import cupy as cp
        from cupyx.scipy.signal import fftconvolve
    except ImportError as exc:
        raise RuntimeError("CuPy and a CUDA GPU are required for T/U/D generation") from exc
    states = _load_states(states_path)
    n = states["T_lay"].shape[0] if count is None else count
    if not 1 <= n <= states["T_lay"].shape[0]:
        raise ValueError("count is outside the states archive")
    ratios = _load_broadening(ratios_path)
    with np.load(continuum_path, allow_pickle=False) as archive:
        continuum = {key: archive[key] for key in archive.files}
    for gas in GASES:
        inspect_blocks(block_dir, gas, 685.0, 1265.0)
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    channel_axis = np.arange(start, end + 1e-9, CHANNEL_STEP)
    width = 0.5 / (2 * np.sqrt(2 * np.log(2))) / FINE_STEP
    kernel_axis = cp.arange(-int(4 * width), int(4 * width) + 1)
    kernel = cp.exp(-0.5 * (kernel_axis / width) ** 2)
    kernel /= kernel.sum()
    written = []
    for tile_start, tile_end in _tiles(start, end):
        mask = (channel_axis >= max(tile_start, start) - 1e-9) & (
            channel_axis <= tile_end + 1e-9 if tile_end >= end else channel_axis < tile_end - 1e-9
        )
        channel_indices = np.where(mask)[0]
        if not channel_indices.size:
            continue
        ca, cb = int(channel_indices[0]), int(channel_indices[-1]) + 1
        path = output_dir / f"c{int(tile_start)}.npz"
        if path.exists():
            with np.load(path, allow_pickle=False) as existing:
                valid = (
                    int(existing["ca"]) == ca
                    and int(existing["cb"]) == cb
                    and existing["T"].shape == (5, n, cb - ca)
                    and existing["U"].shape == (5, n, cb - ca)
                    and existing["D"].shape == (n, cb - ca)
                    and np.array_equal(existing["nu"], channel_axis[ca:cb])
                )
            if not valid:
                raise ValueError(f"Existing chunk has incompatible shape/grid: {path}")
            written.append(str(path))
            continue
        nu = cp.arange(tile_start - GUARD, tile_end + GUARD + FINE_STEP / 2, FINE_STEP)
        channels = cp.asarray(channel_axis[ca:cb])
        d_all = np.empty((n, cb - ca), dtype=np.float32)
        t_all = np.empty((5, n, cb - ca), dtype=np.float32)
        u_all = np.empty((5, n, cb - ca), dtype=np.float32)
        cache = {}
        for row_start in range(0, n, batch_size):
            row_end = min(row_start + batch_size, n)
            temperature = cp.asarray(states["T_lay"][row_start:row_end].astype(np.float64))
            pressure = cp.asarray(states["P_lay"][row_start:row_end].astype(np.float64))
            columns = cp.asarray(states["Ncol"][row_start:row_end].astype(np.float64))
            water_vmr = cp.asarray(states["VMR_H2O"][row_start:row_end].astype(np.float64))
            air_column = cp.where(
                water_vmr > 1e-12, columns[:, 0] / cp.maximum(water_vmr, 1e-30), 0
            )
            depth = cp.zeros((row_end - row_start, temperature.shape[1], nu.size), dtype=cp.float64)
            for gas_index, gas in enumerate(GASES):
                vmr = (
                    water_vmr
                    if gas_index == 0
                    else cp.where(
                        air_column > 0, columns[:, gas_index] / cp.maximum(air_column, 1e-30), 0
                    )
                    .astype(cp.float32)
                    .astype(cp.float64)
                )
                depth += (
                    _sigma(gas, nu, temperature, pressure, vmr, block_dir, ratios, cache, cp)
                    * (columns[:, gas_index, :, None])
                )
            depth += _h2o_continuum(
                nu, temperature, pressure, columns[:, 0], water_vmr, continuum, cp
            )
            cp.maximum(depth, 0, out=depth)
            level_t = cp.asarray(states["T_levels"][row_start:row_end].astype(np.float64))
            planck = (
                C1
                * nu[None, None, :] ** 3
                / (cp.exp(C2 * nu[None, None, :] / level_t[:, :, None]) - 1)
            )
            down = _down(depth, planck, cp)
            norm = fftconvolve(cp.ones_like(down), kernel[None, :], "same", axes=1)

            def smooth(array):
                return fftconvolve(array, kernel[None, :], "same", axes=1) / norm

            d_all[row_start:row_end] = cp.asnumpy(
                cp.array([cp.interp(channels, nu, row) for row in smooth(down)])
            ).astype(np.float32)
            for angle_index, cosine in enumerate(VIEW_COSINES):
                up, trans = _up(depth / cosine, planck, cp)
                t_all[angle_index, row_start:row_end] = cp.asnumpy(
                    cp.array([cp.interp(channels, nu, row) for row in smooth(trans)])
                ).astype(np.float32)
                u_all[angle_index, row_start:row_end] = cp.asnumpy(
                    cp.array([cp.interp(channels, nu, row) for row in smooth(up)])
                ).astype(np.float32)
            cp.get_default_memory_pool().free_all_blocks()
        temporary = path.with_name(path.name + ".partial")
        try:
            with temporary.open("wb") as stream:
                np.savez(stream, ca=ca, cb=cb, nu=channel_axis[ca:cb], D=d_all, T=t_all, U=u_all)
            temporary.replace(path)
        finally:
            temporary.unlink(missing_ok=True)
        written.append(str(path))
    report = {
        "status": "candidate_historical_parity_unverified",
        "source_states_sha256": sha256_file(Path(states_path)),
        "source_ratios_sha256": sha256_file(Path(ratios_path)),
        "source_continuum_sha256": sha256_file(Path(continuum_path)),
        "range_cm1": [start, end],
        "profiles": n,
        "view_cosines": VIEW_COSINES.tolist(),
        "chunks": written,
    }
    (output_dir / "generation.json").write_text(
        json.dumps(report, indent=2) + "\n", encoding="utf-8"
    )
    return report


def sea_emissivity(wavenumber: np.ndarray, cosine: float, water_index: np.ndarray) -> np.ndarray:
    """Return the planar OPAC Fresnel emissivity used in historical radiance."""
    wavelength = 1e4 / wavenumber
    real = np.interp(wavelength, water_index[:, 0], water_index[:, 1])
    imaginary = np.interp(wavelength, water_index[:, 0], np.abs(water_index[:, 2]))
    index = real + 1j * imaginary
    sine = np.sqrt(max(1 - cosine * cosine, 0.0))
    transmitted = np.sqrt(1 - (sine / index) ** 2)
    r_s = (cosine - index * transmitted) / (cosine + index * transmitted)
    r_p = (index * cosine - transmitted) / (index * cosine + transmitted)
    return 1 - 0.5 * (np.abs(r_s) ** 2 + np.abs(r_p) ** 2)


def assemble_radiance(
    chunks_dir: Path,
    states_path: Path,
    emissivity_path: Path,
    water_index_path: Path,
    radiance_output: Path,
    tud_output: Path,
    *,
    view_seed: int = 991,
    emissivity_seed: int = 42,
    tud_view: str = "selected",
    count: int | None = None,
) -> dict:
    """Assemble one frozen viewing angle and its matching T/U/D per profile.

    ``tud_view='nadir'`` is available for a new training candidate. It does not
    claim parity with the older archived ``tud_iasi_all.npy`` input to APAE.
    """
    if tud_view not in {"selected", "nadir"}:
        raise ValueError("tud_view must be selected or nadir")
    states = _load_states(states_path)
    n = states["T_lay"].shape[0] if count is None else count
    if not 1 <= n <= states["T_lay"].shape[0]:
        raise ValueError("count is outside the states archive")
    channels = np.arange(714.0, 1250.0 + 1e-9, CHANNEL_STEP)
    nchannel = channels.size
    selected_angle = np.random.default_rng(view_seed).integers(0, 5, size=n)
    cosines = VIEW_COSINES[selected_angle].astype(np.float32)
    land = states["lsm"][:n] >= 0.5
    emissivity_library = np.load(emissivity_path, mmap_mode="r")
    if emissivity_library.ndim != 2 or emissivity_library.shape[1] != 8461:
        raise ValueError("The land emissivity library must span 645..2760 at 0.25 cm^-1")
    land_indices = np.random.default_rng(emissivity_seed).integers(
        0, emissivity_library.shape[0], size=int(land.sum())
    )
    assigned = np.full(n, -1, dtype=np.int32)
    assigned[land] = land_indices
    water = np.loadtxt(water_index_path)
    if water.ndim != 2 or water.shape[1] < 3:
        raise ValueError("The water index table requires wavelength, n and k columns")
    global_offset = int(round((714.0 - 645.0) / CHANNEL_STEP))
    radiance = np.empty((n, nchannel), dtype=np.float32)
    emissivity = np.empty((n, nchannel), dtype=np.float32)
    tud = np.empty((n, 3 * nchannel), dtype=np.float32)
    coverage = np.zeros(nchannel, dtype=np.int8)
    files = sorted(Path(chunks_dir).glob("c*.npz"), key=lambda p: int(p.stem[1:]))
    if not files:
        raise FileNotFoundError(f"No T/U/D chunks found in {chunks_dir}")
    profile_rows = np.arange(n)
    for path in files:
        with np.load(path, allow_pickle=False) as chunk:
            ca, cb = int(chunk["ca"]), int(chunk["cb"])
            if not 0 <= ca < cb <= nchannel or chunk["nu"].shape != (cb - ca,):
                raise ValueError(f"{path}: invalid channel indices")
            if not np.array_equal(chunk["nu"], channels[ca:cb]):
                raise ValueError(f"{path}: channel coordinates differ")
            if chunk["T"].shape != (5, n, cb - ca) or chunk["U"].shape != (5, n, cb - ca):
                raise ValueError(f"{path}: invalid angle/profile dimensions")
            if chunk["D"].shape != (n, cb - ca):
                raise ValueError(f"{path}: invalid downwelling dimensions")
            if coverage[ca:cb].any():
                raise ValueError(f"{path}: duplicate channel coverage")
            coverage[ca:cb] = 1
            trans = chunk["T"][selected_angle, profile_rows]
            up = chunk["U"][selected_angle, profile_rows]
            down = chunk["D"]
            local_channels = channels[ca:cb]
            eps = np.empty((n, cb - ca), dtype=np.float32)
            for angle_index, cosine in enumerate(VIEW_COSINES):
                sea_rows = np.where(~land & (selected_angle == angle_index))[0]
                if sea_rows.size:
                    eps[sea_rows] = sea_emissivity(local_channels, float(cosine), water)
            land_rows = np.where(land)[0]
            if land_rows.size:
                eps[land_rows] = emissivity_library[
                    assigned[land_rows, None],
                    np.arange(global_offset + ca, global_offset + cb)[None, :],
                ]
            blackbody = (
                C1
                * local_channels[None, :] ** 3
                / (np.exp(C2 * local_channels[None, :] / states["T_skin"][:n, None]) - 1)
            )
            radiance[:, ca:cb] = (eps * blackbody * trans + up + (1 - eps) * trans * down).astype(
                np.float32
            )
            emissivity[:, ca:cb] = eps
            tud_angle = selected_angle if tud_view == "selected" else np.full(n, 4, dtype=int)
            tud[:, ca:cb] = chunk["T"][tud_angle, profile_rows]
            tud[:, nchannel + ca : nchannel + cb] = chunk["U"][tud_angle, profile_rows]
            tud[:, 2 * nchannel + ca : 2 * nchannel + cb] = down
    if not coverage.all() or not np.isfinite(radiance).all() or not np.isfinite(tud).all():
        raise ValueError("Incomplete or nonfinite assembled data")
    if np.any(radiance <= 0) or np.any((emissivity < 0) | (emissivity > 1)):
        raise ValueError("Radiance or emissivity is outside its physical domain")
    radiance_output, tud_output = Path(radiance_output), Path(tud_output)
    for path in (radiance_output, tud_output):
        path.parent.mkdir(parents=True, exist_ok=True)
        if path.exists():
            raise FileExistsError(path)
    np.savez_compressed(
        radiance_output,
        L=radiance,
        eps=emissivity,
        mu=cosines,
        nu=channels.astype(np.float32),
        nota="Candidate historical 714..1250 radiance; original generation source is unverified.",
    )
    np.save(tud_output, tud)
    return {
        "status": "candidate_historical_parity_unverified",
        "profiles": n,
        "channels": nchannel,
        "radiance_sha256": sha256_file(radiance_output),
        "tud_sha256": sha256_file(tud_output),
        "tud_view": tud_view,
        "view_seed": view_seed,
        "emissivity_seed": emissivity_seed,
    }


def build_noise(
    output: Path, *, count: int = 30000, batch_size: int = 256, seed: int = 1337
) -> dict:
    """Regenerate 645..1250 noise, then crop at channel 276 as in the final run."""
    from scipy.signal import fftconvolve

    if count < 1 or batch_size < 1:
        raise ValueError("count and batch_size must be positive")
    step_channel, fwhm, step_fine = 0.25, 0.5, 0.001
    sigma = fwhm / (2 * np.sqrt(2 * np.log(2)))
    channels = np.arange(645.0, 1250.0 + 0.5 * step_channel, step_channel)
    margin = 5 * sigma
    fine = np.arange(645.0 - margin, 1250.0 + margin + 0.5 * step_fine, step_fine)
    channel_index = np.round((channels - fine[0]) / step_fine).astype(int)
    x = 1.43877688 * channels / 280.0
    exponential = np.exp(x)
    shape = channels**3 * x * exponential / (exponential - 1) ** 2
    sigma_channel = shape / shape.mean() * 0.243
    half = int(np.ceil(5 * sigma / step_fine))
    support = np.arange(-half, half + 1) * step_fine
    kernel = np.exp(-0.5 * (support / sigma) ** 2)
    kernel /= kernel.sum()
    scale = sigma_channel / np.sqrt(np.sum(kernel**2))
    random = np.random.default_rng(seed)
    cropped = np.empty((count, 2145), dtype=np.float32)
    for first in range(0, count, batch_size):
        last = min(first + batch_size, count)
        white = random.standard_normal((last - first, fine.size))
        filtered = fftconvolve(white, kernel[None, :], mode="same", axes=-1)
        cropped[first:last] = (filtered[:, channel_index[276:]] * scale[None, 276:]).astype(
            np.float32
        )
    output = Path(output)
    output.parent.mkdir(parents=True, exist_ok=True)
    if output.exists():
        raise FileExistsError(output)
    np.savez_compressed(
        output,
        noise=cropped,
        sigma_ch=sigma_channel[276:].astype(np.float32),
        nu=channels[276:].astype(np.float32),
        seed=seed,
        grid="iasiang",
        dnu_ch=step_channel,
        fwhm=fwhm,
        receta="Candidate historical 645..1250 noise, cropped to 714..1250 at index 276.",
    )
    return {
        "status": "candidate_historical_parity_unverified",
        "sha256": sha256_file(output),
        "profiles": count,
        "channels": 2145,
        "seed": seed,
        "batch_size": batch_size,
    }


def compare_generated_npz(candidate: Path, archived: Path, *, kind: str, profiles: int) -> dict:
    """Compare numerical arrays while ignoring deliberately new provenance notes."""
    if kind == "radiance":
        keys, row_keys = ("L", "eps", "mu", "nu"), {"L", "eps", "mu"}
    elif kind == "noise":
        keys, row_keys = ("noise", "sigma_ch", "nu"), {"noise"}
    else:
        raise ValueError("kind must be radiance or noise")
    differences = {}
    exact = True
    with (
        np.load(candidate, allow_pickle=False) as built,
        np.load(archived, allow_pickle=False) as frozen,
    ):
        for key in keys:
            expected = frozen[key][:profiles] if key in row_keys else frozen[key]
            actual = built[key]
            if actual.shape != expected.shape:
                raise ValueError(f"{key}: candidate and archive shapes differ")
            same = np.array_equal(actual, expected)
            exact &= same
            differences[key] = (
                0.0
                if same
                else float(np.max(np.abs(actual.astype(np.float64) - expected.astype(np.float64))))
            )
    return {
        "arrays_exact": exact,
        "max_abs_by_key": differences,
        "archived_sha256": sha256_file(archived),
        "status": "numerical_canary_only",
    }
