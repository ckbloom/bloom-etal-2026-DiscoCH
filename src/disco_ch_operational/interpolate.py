"""
interpolate.py

Interpolates FORCE's raw Sentinel-2 VI time series (*_TSS.tif: one band per
acquisition) onto a regular 5-day series (*_TSS_TSI.tif), tile by tile,
with FORCE's ensemble RBF filter:

  - for each output date and each sigma in rbf_sigma, a Gaussian-weighted
    mean of the valid observations within that sigma's cutoff radius
    (rbf_cutoff = share of the kernel's area kept; 0.95 -> ~1.96 sigma);
  - the sigmas are combined weighted by how much data each one found, so
    the wide kernels only dominate where data is sparse;
  - no observation within any radius -> nodata.

Near-real-time mode (as_of = a date, normally date.today()):
  - only output dates d with d + lag_days <= as_of are produced;
  - each one uses observations up to d + lag_days and no later (the kernel
    stays symmetric; it just has no data beyond that point);
  - with skip_existing=True (default) dates already in the output are
    frozen and each run only appends newly eligible dates.
as_of=None interpolates the full date_range from every observation in
the file (e.g. for the 2017-2023 training years).

run_full_data_season() does that for a finished season and runs the model
on it with the near-real-time run's per-date min/max, to compare the
near-real-time series with one whose interpolation sees all the data.

Output: int16 (same scaled units as the input), nodata -9999, one band per
output date with its ISO date as the band description, and GeoTIFF tags
recording the settings. A run refuses to append to a file written with
different settings.

Replaces rbf.run_tsi_tiles() (left in place for now) with identical output,
and can append to files rbf.py wrote in as_of mode.

Usage (notebook):

    from src.disco_ch_operational.interpolate import interpolate_tiles

    interpolate_tiles(
        tss_dir=force_directory, out_dir=rbf_output, vi_keys=vi_keys,
        date_range=(date(2026, 1, 1), date(2026, 12, 31)),
        as_of=date.today(), tiles_csv=force_tiles_to_process, max_workers=10,
    )
"""

import os
import time
import math
import contextlib
from datetime import timedelta
from concurrent.futures import ProcessPoolExecutor, as_completed

import numpy as np
import rasterio

from src.disco_ch_operational.utils import (
    resolve_vi_keys, discover_tile_dirs, find_vi_file, band_dates, as_date,
    read_float, mean_bands, chunk_windows, format_duration, write_run_params, limit_threads,
)

DEFAULT_TSS_PATTERN = "{year}*_{vi}_TSS.tif"
DEFAULT_LAG_DAYS = 5
NODATA = -9999


# ---------------------------------------------------------------------
# FORCE helpers (from force_tsi.py, unchanged apart from despike's name)
# ---------------------------------------------------------------------

def _erfinv(x, tol=1e-12, max_iter=100):
    lo, hi = -6.0, 6.0
    for _ in range(max_iter):
        mid = (lo + hi) / 2.0
        if math.erf(mid) < x:
            lo = mid
        else:
            hi = mid
        if hi - lo < tol:
            break
    return (lo + hi) / 2.0


def rbf_cutoff_radius(sigma_days, cutoff):
    """One-sided truncation radius (days) keeping `cutoff` fraction of a
    Gaussian kernel's area, i.e. FORCE's RBF_CUTOFF: cutoff = erf(radius
    / (sigma * sqrt(2))), solved for radius."""
    return sigma_days * math.sqrt(2.0) * _erfinv(cutoff)


def output_dates_from_range(date_range, int_day, doy_range=(1, 365)):
    """Every int_day days across date_range, kept only where day-of-year
    falls in doy_range. doy_range supports FORCE's wrap-around convention
    (e.g. (274, 90) means "Oct-Mar", spanning the year boundary)."""
    start, end = date_range
    doy_lo, doy_hi = doy_range
    dates = []
    d = start
    while d <= end:
        doy = d.timetuple().tm_yday
        keep = (doy_lo <= doy <= doy_hi) if doy_lo <= doy_hi else (doy >= doy_lo or doy <= doy_hi)
        if keep:
            dates.append(d)
        d += timedelta(days=int_day)
    return dates


def _triplet_interp(values, ordinals, idx_arr, n, h, w):
    """Linear interpolation, per pixel, between each position's nearest
    real neighbors on either side in time -- the "triplet" FORCE's
    ABOVE_NOISE/BELOW_NOISE description refers to."""
    active = ~np.isnan(values)

    idx_if_active = np.where(active, idx_arr, -1)
    last_idx = np.maximum.accumulate(idx_if_active, axis=0)
    last_idx = np.concatenate([np.full((1, h, w), -1, dtype=int), last_idx[:-1]], axis=0)

    idx_if_active_rev = np.where(active, idx_arr, n)
    next_idx = np.minimum.accumulate(idx_if_active_rev[::-1], axis=0)[::-1]
    next_idx = np.concatenate([next_idx[1:], np.full((1, h, w), n, dtype=int)], axis=0)

    has_prev, has_next = last_idx >= 0, next_idx < n
    safe_last, safe_next = np.clip(last_idx, 0, n - 1), np.clip(next_idx, 0, n - 1)

    prev_val = np.take_along_axis(values, safe_last, axis=0)
    next_val = np.take_along_axis(values, safe_next, axis=0)
    prev_ord, next_ord = ordinals[safe_last], ordinals[safe_next]
    cur_ord = np.broadcast_to(ordinals[:, None, None], (n, h, w))

    span = (next_ord - prev_ord).astype("float64")
    has_neighbors = has_prev & has_next & (span > 0)
    frac = (cur_ord - prev_ord) / np.where(span > 0, span, 1.0)

    with np.errstate(invalid="ignore"):
        interp = prev_val + (next_val - prev_val) * frac
    return interp, has_neighbors, active


def despike_series(dates, cube, above_noise=3.0, below_noise=1.0, max_iter=20):
    """FORCE's ABOVE_NOISE / BELOW_NOISE outlier filter.

    Per pixel, the RMSE of triplet residuals (real value vs. linear
    interpolation of its real neighbors) is the noise level. The single
    worst-residual observation exceeding `above_noise` times that noise
    is removed per pass, one at a time (removing it changes its
    neighbors' triplets and tightens the noise estimate), repeated up to
    `max_iter` passes. `above_noise = 0` disables filtering entirely.

    Once that converges, a removed point is restored if its ORIGINAL
    value now falls within `below_noise` times the noise estimated from
    its neighbors' current state -- again one best-fit point at a time,
    up to `max_iter` passes. `below_noise = 0` disables restoration.
    """
    if above_noise <= 0:
        return cube.copy(), np.zeros(cube.shape, dtype=bool)

    ordinals = np.array([d.toordinal() for d in dates], dtype="int64")
    n, h, w = cube.shape
    idx_arr = np.broadcast_to(np.arange(n)[:, None, None], (n, h, w))

    cleaned = cube.copy()
    removed = np.zeros((n, h, w), dtype=bool)

    for _ in range(max_iter):
        interp, has_neighbors, active = _triplet_interp(cleaned, ordinals, idx_arr, n, h, w)
        has_triplet = active & has_neighbors
        with np.errstate(invalid="ignore"):
            residual = np.where(has_triplet, cleaned - interp, np.nan)
            noise = np.sqrt(np.nanmean(residual ** 2, axis=0, keepdims=True))
        is_candidate = has_triplet & (np.abs(residual) > above_noise * noise)
        any_candidate = is_candidate.any(axis=0, keepdims=True)
        if not any_candidate.any():
            break
        worst = np.argmax(np.where(is_candidate, np.abs(residual), -np.inf), axis=0, keepdims=True)
        this_pass = (idx_arr == worst) & any_candidate
        cleaned = np.where(this_pass, np.nan, cleaned).astype("float32")
        removed |= this_pass

    if below_noise > 0:
        for _ in range(max_iter):
            if not removed.any():
                break
            interp, has_neighbors, active = _triplet_interp(cleaned, ordinals, idx_arr, n, h, w)
            has_triplet = active & has_neighbors
            with np.errstate(invalid="ignore"):
                residual = np.where(has_triplet, cleaned - interp, np.nan)
                noise = np.sqrt(np.nanmean(residual ** 2, axis=0, keepdims=True))
                candidates = removed & has_neighbors
                restore_residual = np.where(candidates, cube - interp, np.nan)  # original value
            is_candidate = candidates & (np.abs(restore_residual) <= below_noise * noise)
            any_candidate = is_candidate.any(axis=0, keepdims=True)
            if not any_candidate.any():
                break
            best = np.argmin(np.where(is_candidate, np.abs(restore_residual), np.inf), axis=0, keepdims=True)
            this_pass = (idx_arr == best) & any_candidate
            cleaned = np.where(this_pass, cube, cleaned).astype("float32")
            removed &= ~this_pass

    return cleaned, removed


# ---------------------------------------------------------------------


# ---------------------------------------------------------------------
# RBF
# ---------------------------------------------------------------------

def rbf_weights(acq_dates, out_dates, rbf_sigma, rbf_cutoff, max_ahead_days=None):
    """Per output date: (indices of the acquisitions in reach, weights of
    shape (n_sigma, n_in_reach)), or (indices, None) if none are in reach.
    max_ahead_days drops acquisitions more than that many days after the
    output date. Depends only on dates, so it's computed once per file."""
    ordinals = np.array([d.toordinal() for d in acq_dates], dtype="float64")
    sigmas = np.asarray(rbf_sigma, dtype="float64")
    radii = np.array([rbf_cutoff_radius(s, rbf_cutoff) for s in sigmas])

    weights = []
    for od in out_dates:
        delta_all = ordinals - od.toordinal()
        in_reach = np.abs(delta_all) <= radii.max()
        if max_ahead_days is not None:
            in_reach &= delta_all <= max_ahead_days
        idx = np.where(in_reach)[0]
        if idx.size == 0:
            weights.append((idx, None))
            continue
        delta = delta_all[idx]
        w = np.exp(-0.5 * (delta[None, :] / sigmas[:, None]) ** 2)
        weights.append((idx, np.where(np.abs(delta)[None, :] <= radii[:, None], w, 0.0)))
    return weights


def apply_rbf(cube, weights):
    """(n_out, h, w) float32 interpolation of a (n_acq, h, w) cube (NaN =
    no data). Each sigma's weighted sum and weight mass are added up across
    sigmas before dividing -- that's the data-density weighting."""
    valid = ~np.isnan(cube)
    valid_f = valid.astype("float64")
    filled = np.where(valid, cube, 0.0).astype("float64")

    out = np.full((len(weights),) + cube.shape[1:], np.nan, dtype="float32")
    for i, (idx, w) in enumerate(weights):
        if w is None:
            continue
        mass = np.tensordot(w, valid_f[idx], axes=([1], [0])).sum(axis=0)
        wsum = np.tensordot(w, filled[idx], axes=([1], [0])).sum(axis=0)
        has_data = mass > 0
        out[i] = np.where(has_data, wsum / np.where(has_data, mass, 1.0), np.nan)
    return out


def to_int16(arr):
    """Round (not truncate) to int16, NaN -> NODATA."""
    rounded = np.round(arr)
    return np.where(np.isnan(rounded), NODATA, np.clip(rounded, -32768, 32767)).astype("int16")


# ---------------------------------------------------------------------
# One file
# ---------------------------------------------------------------------

def _settings_tags(lag_days, rbf_sigma, rbf_cutoff, int_day):
    """Tags written to every output and checked before appending to one.
    Same names/values as rbf.py's as_of mode; TSI_LAG_DAYS is "none" for a
    full-season (as_of=None) file so the two kinds are never mixed."""
    return {"TSI_LAG_DAYS": str(lag_days) if lag_days is not None else "none",
            "TSI_RBF_SIGMA": ",".join(str(s) for s in rbf_sigma),
            "TSI_RBF_CUTOFF": str(rbf_cutoff),
            "TSI_INT_DAY": str(int_day)}


def interpolate_file(tss_path, out_path, date_range, as_of, lag_days=DEFAULT_LAG_DAYS,
                     int_day=5, doy_range=(1, 365), rbf_sigma=(10, 20, 30), rbf_cutoff=0.95,
                     despike=False, above_noise=3.0, below_noise=1.0,
                     chunk_size=512, skip_existing=True, threads=1):
    """Interpolates one TSS file -- see module docstring and
    interpolate_tiles() for the arguments.

    :return: (number of output dates written, status) -- status "written"
        or "skipped" (nothing new to add).
    """
    as_of = as_date(as_of)
    targets = output_dates_from_range(date_range, int_day, doy_range)
    if as_of is not None:
        targets = [d for d in targets if d + timedelta(days=lag_days) <= as_of]
    tags = _settings_tags(lag_days if as_of is not None else None, rbf_sigma, rbf_cutoff, int_day)

    existing = []
    if skip_existing and os.path.exists(out_path):
        with rasterio.open(out_path) as src:
            existing = [as_date(d) for d in src.descriptions]
            old_tags = src.tags()
        new_dates = [d for d in targets if d not in set(existing)]
        if not new_dates:
            return 0, "skipped"
        if "TSI_LAG_DAYS" not in old_tags:
            raise ValueError(f"{out_path} has no recorded settings (a full-season rbf.py file?); "
                             f"delete it or pass skip_existing=False to rebuild it")
        mismatched = {k: (old_tags.get(k), v) for k, v in tags.items() if old_tags.get(k) != v}
        if mismatched:
            raise ValueError(f"{out_path} was written with different settings {{tag: (file, now)}}: "
                             f"{mismatched}; delete it or pass skip_existing=False to rebuild it")
    else:
        new_dates = targets
    if not new_dates:
        return 0, "skipped"

    all_dates = sorted(set(existing) | set(new_dates))
    band_of = {d: i + 1 for i, d in enumerate(all_dates)}

    bands_by_date = band_dates(tss_path)
    acq_dates = sorted(bands_by_date)
    weights = rbf_weights(acq_dates, new_dates, rbf_sigma, rbf_cutoff,
                          max_ahead_days=lag_days if as_of is not None else None)
    after_as_of = np.array([as_of is not None and d > as_of for d in acq_dates])

    tmp_path = out_path + ".tmp"
    with contextlib.ExitStack() as stack:
        src = stack.enter_context(rasterio.open(tss_path))
        old = stack.enter_context(rasterio.open(out_path)) if existing else None
        dst = stack.enter_context(rasterio.open(
            tmp_path, "w", driver="GTiff", height=src.height, width=src.width, count=len(all_dates),
            dtype="int16", crs=src.crs, transform=src.transform, nodata=NODATA, compress="deflate"))
        dst.update_tags(**tags, **({"TSI_AS_OF": as_of.isoformat()} if as_of is not None else {}))
        for d, b in band_of.items():
            dst.set_band_description(b, d.isoformat())

        for window in chunk_windows(src.width, src.height, chunk_size):
            if old is not None:
                for i, d in enumerate(existing):
                    dst.write(old.read(i + 1, window=window), band_of[d], window=window)

            raw = read_float(src, None, window)  # all bands
            cube = np.stack([mean_bands(raw[[b - 1 for b in bands_by_date[d]]]) for d in acq_dates])
            cube[after_as_of] = np.nan
            with limit_threads(threads):
                if despike:
                    cube, _removed = despike_series(acq_dates, cube, above_noise, below_noise)
                interpolated = to_int16(apply_rbf(cube, weights))
            for i, d in enumerate(new_dates):
                dst.write(interpolated[i], band_of[d], window=window)

    os.replace(tmp_path, out_path)
    return len(new_dates), "written"


def _run_job(tss_path, out_path, kwargs):
    """interpolate_file() for the worker pool; errors become a status."""
    try:
        n, status = interpolate_file(tss_path, out_path, **kwargs)
        return f"{status} ({n} date(s))" if status == "written" else status
    except Exception as e:
        return f"failed: {type(e).__name__}: {e}"


# ---------------------------------------------------------------------
# All tiles
# ---------------------------------------------------------------------

def interpolate_tiles(tss_dir, out_dir, vi_keys, date_range, as_of, lag_days=DEFAULT_LAG_DAYS,
                      int_day=5, doy_range=(1, 365), rbf_sigma=(10, 20, 30), rbf_cutoff=0.95,
                      despike=False, above_noise=3.0, below_noise=1.0,
                      tiles_csv=None, tiles_column="Tile_ID", tiles_csv_mode="include",
                      tss_pattern=DEFAULT_TSS_PATTERN, year=None, suffix="_TSI",
                      skip_existing=True, chunk_size=512, max_workers=1, threads=1):
    """Interpolates every (tile, VI) under tss_dir into
    out_dir/<tile>/<input name><suffix>.tif.

    :param tss_dir: FORCE level-3 folder, one subfolder per tile.
    :param vi_keys: {canonical name: file token}, e.g. 2026 naming
        {"NDVI": "NDVI", "EVI": "EVI", "NDMI": "NDMI", "CIRE": "CIre", "CCI": "CCI"}.
    :param date_range: (first, last) output date; with int_day the output
        grid is first, first + int_day, ... (doy_range can restrict it to
        part of the year, FORCE-style, wrapping allowed e.g. (274, 90)).
    :param as_of: date (or "YYYY-MM-DD") for near-real-time mode, or None
        for the full date_range -- required so it's always a deliberate
        choice. Never later than the day the run happens.
    :param lag_days: days of data after each output date it may use, and
        how long a date waits before being produced (as_of mode only).
    :param rbf_sigma, rbf_cutoff: kernel standard deviations (days) and
        the share of each kernel's area kept.
    :param despike: apply FORCE's outlier filter (above_noise/below_noise)
        first. Off by default -- it can remove real change near the newest
        data. In as_of mode it sees all data up to as_of, not just up to
        each date + lag_days.
    :param tiles_csv, tiles_column, tiles_csv_mode: optional tile allow
        list ("include") or skip list ("exclude").
    :param tss_pattern, year: input file pattern ({year}, {vi}); year
        defaults to date_range[0].year.
    :param skip_existing: True keeps existing output dates and only adds
        missing ones; False rebuilds every file.
    :param chunk_size: pixels per side of the blocks processed at a time
        (bounds memory); None = whole tile at once.
    :param max_workers: files processed in parallel.
    :param threads: BLAS threads per worker.
    :return: {tile_id: {vi: (tss_path, out_path, status)}}
    """
    year = year or date_range[0].year
    call_params = dict(locals())

    tile_dirs = discover_tile_dirs(tss_dir, tiles_csv, tiles_column, tiles_csv_mode)
    print(f"Found {len(tile_dirs)} tile(s) under {tss_dir}")

    jobs = []
    for tile_dir in tile_dirs:
        tile_id = os.path.basename(tile_dir)
        for vi, token in resolve_vi_keys(vi_keys).items():
            try:
                tss_path = find_vi_file(tile_dir, token, year, tss_pattern)
            except (FileNotFoundError, ValueError) as e:
                print(f"  {tile_id}/{vi}: skipped -- {e}")
                continue
            base, ext = os.path.splitext(os.path.basename(tss_path))
            os.makedirs(os.path.join(out_dir, tile_id), exist_ok=True)
            jobs.append((tile_id, vi, tss_path, os.path.join(out_dir, tile_id, f"{base}{suffix}{ext}")))

    kwargs = dict(date_range=date_range, as_of=as_of, lag_days=lag_days, int_day=int_day,
                  doy_range=doy_range, rbf_sigma=rbf_sigma, rbf_cutoff=rbf_cutoff,
                  despike=despike, above_noise=above_noise, below_noise=below_noise,
                  chunk_size=chunk_size, skip_existing=skip_existing, threads=threads)

    results = {}
    start = time.monotonic()

    def report(done, job, status):
        tile_id, vi, tss_path, out_path = job
        results.setdefault(tile_id, {})[vi] = (tss_path, out_path, status)
        elapsed = time.monotonic() - start
        eta = f", ~{format_duration(elapsed / done * (len(jobs) - done))} left" if done < len(jobs) else ""
        print(f"  [{done}/{len(jobs)}, {format_duration(elapsed)}{eta}] {tile_id}/{vi}: {status}", flush=True)

    if max_workers <= 1:
        for i, job in enumerate(jobs, start=1):
            report(i, job, _run_job(job[2], job[3], kwargs))
    else:
        with ProcessPoolExecutor(max_workers=max_workers) as pool:
            futures = {pool.submit(_run_job, job[2], job[3], kwargs): job for job in jobs}
            for i, fut in enumerate(as_completed(futures), start=1):
                try:
                    status = fut.result()
                except Exception as e:  # worker died outright, e.g. out of memory
                    status = f"failed: worker crashed: {e}"
                report(i, futures[fut], status)

    statuses = [s for by_vi in results.values() for _, _, s in by_vi.values()]
    summary = {"written": sum(s.startswith("written") for s in statuses),
               "skipped": sum(s == "skipped" for s in statuses),
               "failed": sum(s.startswith("failed") for s in statuses),
               "elapsed_seconds": time.monotonic() - start}
    print(f"Done: {summary['written']} written, {summary['skipped']} up to date, "
          f"{summary['failed']} failed in {format_duration(summary['elapsed_seconds'])}")
    write_run_params(out_dir, "interpolate.interpolate_tiles", call_params, summary)
    return results


# ---------------------------------------------------------------------
# End of season: all data, near-real-time normalization
# ---------------------------------------------------------------------

def run_full_data_season(tss_dir, rbf_dir, out_dir, nrt_out_dir, year, model_path, vi_keys,
                         date_range=None, int_day=5, doy_range=(1, 365),
                         rbf_sigma=(10, 20, 30), rbf_cutoff=0.95, despike=False,
                         tiles_csv=None, forest_mask_path=None, model_months=(6, 7, 8, 9),
                         skip_existing=True, max_workers=1, threads=1, model_threads=1):
    """Reruns a finished season with every observation, for comparison with
    the near-real-time (as_of + lag_days) series of the same year:

      1. interpolate_tiles(as_of=None) into rbf_dir: each date uses all
         observations within the kernel radius, before and after it;
      2. apply_model.apply_discoch_tiles() on that, normalizing each date
         with the near-real-time run's min/max snapshot of the same date
         (nrt_out_dir/minmax_by_date/, written by its export_minmax=True),
         so the normalization stays stepwise -- nothing after a date enters
         its min/max -- and only the interpolation differs. Outputs go to
         out_dir (<tile>/, mosaic/) as in a normal run.

    Use the near-real-time run's int_day, rbf_sigma, rbf_cutoff and despike
    so the two series differ only in the data each date can see; date_range
    must start where that run's did (default 1 Jan) so the 5-day grids line
    up. rbf_dir and out_dir must be separate from the near-real-time
    folders (interpolate_file() refuses to mix the two kinds of file).

    :param nrt_out_dir: the near-real-time apply_discoch_tiles() out_dir.
    :return: (interpolate_tiles() result, apply_discoch_tiles() summary)
    """
    from src.disco_ch_operational.apply_model import apply_discoch_tiles

    date_range = date_range or (as_date(f"{year}-01-01"), as_date(f"{year}-12-31"))
    minmax_by_date_dir = os.path.join(nrt_out_dir, "minmax_by_date")
    if not os.path.isdir(minmax_by_date_dir):
        raise FileNotFoundError(f"No {minmax_by_date_dir}: run the near-real-time apply_discoch_tiles() "
                                f"with export_minmax=True first")
    if os.path.abspath(out_dir) == os.path.abspath(nrt_out_dir):
        raise ValueError("out_dir must differ from nrt_out_dir")

    print(f"=== 1/2 Interpolating {year} with all observations -> {rbf_dir}")
    interpolated = interpolate_tiles(
        tss_dir=tss_dir, out_dir=rbf_dir, vi_keys=vi_keys, date_range=date_range, as_of=None,
        int_day=int_day, doy_range=doy_range, rbf_sigma=rbf_sigma, rbf_cutoff=rbf_cutoff,
        despike=despike, tiles_csv=tiles_csv, year=year, skip_existing=skip_existing,
        max_workers=max_workers, threads=threads)
    failed = [f"{t}/{vi}" for t, by_vi in interpolated.items()
              for vi, (_, _, s) in by_vi.items() if s.startswith("failed")]
    if failed:
        raise RuntimeError(f"Interpolation failed for {', '.join(failed)}; fix and rerun "
                           f"(finished files are kept)")

    print(f"\n=== 2/2 Model with the near-real-time min/max of each date ({minmax_by_date_dir}) -> {out_dir}")
    summary = apply_discoch_tiles(
        rbf_dir=rbf_dir, out_dir=out_dir, year=year, model_path=model_path, vi_keys=vi_keys,
        tiles_csv=tiles_csv, forest_mask_path=forest_mask_path, model_months=model_months,
        minmax_by_date_dir=minmax_by_date_dir, max_workers=max_workers, model_threads=model_threads)
    return interpolated, summary
