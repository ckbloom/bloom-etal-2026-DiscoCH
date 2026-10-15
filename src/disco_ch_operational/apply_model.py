"""
apply_model.py

Applies the DiscoCH discoloration model to the 5-day RBF time series
written by rbf.run_tsi_tiles() (one *_TSS_TSI.tif per VI per tile).

For each tile, every not-yet-processed date is walked in order:

  1. Running seasonal min/max: each VI's per-pixel min and max over all
     dates of the year processed so far, including this one (pixels
     outside the optional forest mask are ignored).
  2. Normalize: (value - min) / (max - min), so each VI is scaled to its
     range over the season to date.
  3. Model: on dates present in every VI and within model_months, the
     pickled sklearn model's positive-class probability for the five
     normalized VIs -> <out_dir>/<tile>/Disco_Proba_<date>.tif (int16,
     scaled; see utils.write_scaled_int()).
  4. Record the date as processed, so the next run starts after it.

Then each date's tiles are mosaicked into <out_dir>/mosaic/.

With minmax_by_date_dir (an earlier run's per-date min/max snapshots),
step 1 takes that run's snapshot of the date instead of this run's own
running min/max -- see apply_discoch_tiles().

Output layout (all under out_dir):
    <tile>/Disco_Proba_<date>.tif   per-tile probability
    mosaic/Disco_Proba_<date>.tif   all tiles merged
    minmax/<tile>/                  running min/max cache + processed dates
    vi_norm/<tile>/<VI>/            optional per-date normalized VIs
    minmax_by_date/<tile>/<VI>/     optional per-date min/max snapshots
    logs/<tile>.log                 per-tile log when max_workers > 1

Replaces apply_discoch.run_tsa_workflow_tiled() (left in place for now)
with the same calculation, file names and cache format -- it resumes from a
minmax cache the old code wrote. What changed:
  - plain rasterio/numpy on each tile's own grid instead of dask/rioxarray
    (a tile is 3000 x 3000 px, so one date of all five VIs is ~180 MB);
  - the model is loaded once per tile instead of once per date;
  - each date's model output is written BEFORE the date is recorded as
    processed, so a crash just redoes that date on the next run (min/max
    updates are idempotent) -- no recovery step needed;
  - min/max start as NaN, so no national template raster is needed;
  - no import of stac_pull (and so none of its GDAL settings or STAC deps);
  - a date where no pixel can be predicted is skipped with a message
    instead of failing the tile;
  - mosaics are only rebuilt for dates whose tile files changed.

Usage (notebook):

    from src.disco_ch_operational.apply_model import apply_discoch_tiles

    apply_discoch_tiles(
        rbf_dir=rbf_output, out_dir=discoCH_output, year=2026,
        model_path=disco_model, vi_keys=vi_keys,
        tiles_csv=force_tiles_to_process, model_months=(6, 7, 8, 9),
        max_workers=10,
    )
"""

import os
import sys
import json
import time
import pickle
import contextlib
from datetime import datetime, UTC
from concurrent.futures import ProcessPoolExecutor, as_completed

import numpy as np
import pandas as pd
import rasterio

from src.disco_ch_operational.utils import (
    resolve_vi_keys, discover_tile_dirs, find_vi_file, band_dates, as_date,
    read_date_mean, warp_to_grid, write_float, write_scaled_int, read_scaled,
    mosaic_by_filename, format_duration, write_run_params, limit_threads,
)

DEFAULT_VI_PATTERN = "{year}*_{vi}_TSS_TSI.tif"

# Model features, in the column order the model was trained with, keyed by
# canonical VI name.
MODEL_FEATURES = {
    "CCI": "Season_Peak_pct_CCI",
    "EVI": "Season_Peak_pct_EVI",
    "CIRE": "Season_Peak_pct_CRE",
    "NDMI": "Season_Peak_pct_NDM",
    "NDVI": "Season_Peak_pct_NDV",
}

PROBA_PATTERN = "Disco_Proba_*.tif"


# ---------------------------------------------------------------------
# Model
# ---------------------------------------------------------------------

def load_model(model_path):
    with open(model_path, "rb") as f:
        return pickle.load(f)


def predict_probability(model, normalized):
    """Positive-class probability per pixel from {canonical VI: 2-D
    normalized array}. Pixels missing any feature stay NaN."""
    shape = normalized["CCI"].shape
    features = np.column_stack([normalized[vi].ravel() for vi in MODEL_FEATURES])
    valid = ~np.isnan(features).any(axis=1)

    proba = np.full(features.shape[0], np.nan, dtype="float32")
    if valid.any():
        df = pd.DataFrame(features[valid], columns=list(MODEL_FEATURES.values()))
        proba[valid] = model.predict_proba(df)[:, 1]
    return proba.reshape(shape)


def normalize(value, vi_min, vi_max):
    """(value - min) / (max - min); NaN where undefined (e.g. max == min)."""
    with np.errstate(invalid="ignore", divide="ignore"):
        out = (value - vi_min) / (vi_max - vi_min)
    out[~np.isfinite(out)] = np.nan
    return out


# ---------------------------------------------------------------------
# Running min/max cache -- same files as the old apply_discoch/stac_pull
# ---------------------------------------------------------------------

def _meta_path(cache_dir, year):
    return os.path.join(cache_dir, f"vi_minmax_{year}_meta.json")


def _minmax_path(cache_dir, vi, kind, year):
    return os.path.join(cache_dir, f"{vi}_{kind}_{year}.tif")


def load_processed_dates(cache_dir, year):
    path = _meta_path(cache_dir, year)
    if not os.path.exists(path):
        return set()
    with open(path) as f:
        return set(json.load(f)["processed_dates"])


def save_processed_dates(cache_dir, year, processed):
    os.makedirs(cache_dir, exist_ok=True)
    tmp = _meta_path(cache_dir, year) + ".tmp"
    with open(tmp, "w") as f:
        json.dump({"year": year, "processed_dates": sorted(processed),
                   "last_updated": datetime.now(UTC).isoformat()}, f, indent=2)
    os.replace(tmp, _meta_path(cache_dir, year))


def load_minmax(cache_dir, year, vis, shape):
    """{vi: array} min and max from the cache, NaN-initialised for any VI
    without a cached file yet."""
    vi_min, vi_max = {}, {}
    for vi in vis:
        for kind, store in (("min", vi_min), ("max", vi_max)):
            path = _minmax_path(cache_dir, vi, kind, year)
            if os.path.exists(path):
                with rasterio.open(path) as src:
                    arr = src.read(1, masked=True).astype("float32").filled(np.nan)
                if arr.shape != shape:
                    raise ValueError(f"cached {path} is {arr.shape}, tile is {shape}")
                store[vi] = arr
            else:
                store[vi] = np.full(shape, np.nan, dtype="float32")
    return vi_min, vi_max


def save_minmax(cache_dir, year, vi_min, vi_max, transform, crs):
    os.makedirs(cache_dir, exist_ok=True)
    for vi in vi_min:
        write_float(vi_min[vi], _minmax_path(cache_dir, vi, "min", year), transform, crs)
        write_float(vi_max[vi], _minmax_path(cache_dir, vi, "max", year), transform, crs)


# ---------------------------------------------------------------------
# One tile
# ---------------------------------------------------------------------

def _in_range(d, start, end):
    return (start is None or d >= start) and (end is None or d <= end)


def _export(arr, path, transform, crs):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    if write_scaled_int(arr, path, transform, crs) is None:
        print(f"    no valid pixels, not exported: {os.path.basename(path)}")


def _snapshot_path(snap_dir, vi, kind, day):
    """A minmax_by_date/<tile>/ snapshot, as written with export_minmax."""
    return os.path.join(snap_dir, vi, f"{vi}_{kind}_{day}.tif")


def process_tile(tile_dir, out_dir, year, model_path=None, vi_keys=None,
                 vi_pattern=DEFAULT_VI_PATTERN, forest_mask_path=None,
                 start_date=None, end_date=None, model_months=(6, 7, 8, 9), run_model=True,
                 export_normalized_vi=False, export_minmax=False,
                 export_start_date=None, export_end_date=None,
                 plot=False, model_threads=1, minmax_by_date_dir=None):
    """Walks every unprocessed date of one tile (see module docstring).
    Arguments as apply_discoch_tiles().

    :return: (number of dates processed, number of model outputs written)
    """
    tile_id = os.path.basename(tile_dir)
    vi_map = resolve_vi_keys(vi_keys)
    vis = list(vi_map)
    start_date, end_date = as_date(start_date), as_date(end_date)
    export_start_date, export_end_date = as_date(export_start_date), as_date(export_end_date)

    tile_out = os.path.join(out_dir, tile_id)
    cache_dir = os.path.join(out_dir, "minmax", tile_id)
    norm_dir = os.path.join(out_dir, "vi_norm", tile_id) if export_normalized_vi else None
    minmax_dir = os.path.join(out_dir, "minmax_by_date", tile_id) if export_minmax else None
    snap_dir = os.path.join(minmax_by_date_dir, tile_id) if minmax_by_date_dir else None
    os.makedirs(tile_out, exist_ok=True)

    paths = {vi: find_vi_file(tile_dir, token, year, vi_pattern) for vi, token in vi_map.items()}
    dates_by_vi = {vi: band_dates(p) for vi, p in paths.items()}
    for vi, p in paths.items():
        print(f"  {vi}: {os.path.basename(p)} ({len(dates_by_vi[vi])} dates)")

    all_dates = sorted(d for d in set().union(*dates_by_vi.values())
                       if d.year == year and _in_range(d, start_date, end_date))
    if snap_dir:
        # Dated min/max mode: only dates with a snapshot of every VI.
        all_dates = [d for d in all_dates if all(
            os.path.exists(_snapshot_path(snap_dir, vi, k, d.isoformat()))
            for vi in vis for k in ("min", "max"))]
        print(f"  {len(all_dates)} date(s) with min/max snapshots in {snap_dir}")
    processed = load_processed_dates(cache_dir, year)
    todo = [d for d in all_dates if d.isoformat() not in processed]
    print(f"  {len(processed)} date(s) already processed, {len(todo)} to do")
    if not todo:
        return 0, 0

    with contextlib.ExitStack() as stack:
        srcs = {vi: stack.enter_context(rasterio.open(p)) for vi, p in paths.items()}
        first = srcs[vis[0]]
        transform, crs, shape = first.transform, first.crs, first.shape
        for vi, src in srcs.items():
            if src.shape != shape or src.transform != transform or src.crs != crs:
                raise ValueError(f"{vi} file is on a different grid than {vis[0]}: {paths[vi]}")

        forest = None
        if forest_mask_path is not None:
            forest = warp_to_grid(forest_mask_path, transform, crs, shape) == 1

        vi_min, vi_max = load_minmax(cache_dir, year, vis, shape)
        model = load_model(model_path) if run_model and model_path else None
        n_written = 0

        for i, d in enumerate(todo, start=1):
            day = d.isoformat()
            present = [vi for vi in vis if d in dates_by_vi[vi]]
            print(f"  [{i}/{len(todo)}] {day}: {', '.join(present)}")

            values = {}
            for vi in present:
                v = read_date_mean(srcs[vi], dates_by_vi[vi][d])
                if forest is not None:
                    v[~forest] = np.nan
                if snap_dir:
                    # This date's snapshot, widened by this date's value as
                    # the running min/max is -- keeps normalized in 0-1.
                    for kind, store in (("min", vi_min), ("max", vi_max)):
                        snap = read_scaled(_snapshot_path(snap_dir, vi, kind, day))
                        if snap.shape != shape:
                            raise ValueError(f"{vi} {kind} snapshot for {day} is {snap.shape}, tile is {shape}")
                        store[vi] = snap
                np.fmin(vi_min[vi], v, out=vi_min[vi])
                np.fmax(vi_max[vi], v, out=vi_max[vi])
                values[vi] = v

            normalized = {vi: normalize(values[vi], vi_min[vi], vi_max[vi]) for vi in present}

            if _in_range(d, export_start_date, export_end_date):
                for vi in present:
                    if minmax_dir:
                        _export(vi_min[vi], os.path.join(minmax_dir, vi, f"{vi}_min_{day}.tif"), transform, crs)
                        _export(vi_max[vi], os.path.join(minmax_dir, vi, f"{vi}_max_{day}.tif"), transform, crs)
                    if norm_dir:
                        _export(normalized[vi], os.path.join(norm_dir, vi, f"{vi}_{day}_NORM.tif"), transform, crs)

            if model is not None:
                if len(present) < len(vis):
                    print(f"    model skipped: missing {sorted(set(vis) - set(present))}")
                elif model_months is not None and d.month not in model_months:
                    pass
                else:
                    with limit_threads(model_threads):
                        proba = predict_probability(model, normalized)
                    out_path = os.path.join(tile_out, f"Disco_Proba_{day}.tif")
                    if write_scaled_int(proba, out_path, transform, crs) is None:
                        print("    model skipped: no pixel has all five VIs")
                    else:
                        n_written += 1
                        if plot:
                            plot_probability(out_path, f"{tile_id} {day}")

            # Cache first, then mark processed: a crash in between just
            # redoes this date next run, which gives the same min/max.
            if not snap_dir:    # dated mode has no running min/max to carry on
                save_minmax(cache_dir, year, vi_min, vi_max, transform, crs)
            processed.add(day)
            save_processed_dates(cache_dir, year, processed)

    return len(todo), n_written


def _run_tile(tile_dir, out_dir, log_dir, kwargs):
    """process_tile() for the worker pool: optional per-tile log file,
    exceptions turned into a returned error message."""
    tile_id = os.path.basename(tile_dir)
    log_path = os.path.join(log_dir, f"{tile_id}.log") if log_dir else None
    t0 = time.monotonic()
    try:
        with contextlib.ExitStack() as stack:
            if log_path:
                os.makedirs(log_dir, exist_ok=True)
                log = stack.enter_context(open(log_path, "a"))
                stack.enter_context(contextlib.redirect_stdout(log))
                stack.enter_context(contextlib.redirect_stderr(log))
            print(f"--- {tile_id}, started {datetime.now(UTC).isoformat()} ---")
            n_dates, n_written = process_tile(tile_dir, out_dir, **kwargs)
        return tile_id, f"{n_dates} date(s), {n_written} model output(s)", None, time.monotonic() - t0
    except Exception as e:
        return tile_id, None, f"{type(e).__name__}: {e}", time.monotonic() - t0


# ---------------------------------------------------------------------
# All tiles
# ---------------------------------------------------------------------

def apply_discoch_tiles(rbf_dir, out_dir, year, model_path, vi_keys=None,
                        vi_pattern=DEFAULT_VI_PATTERN, tiles_csv=None, tiles_column="Tile_ID",
                        tiles_csv_mode="include", forest_mask_path=None,
                        start_date=None, end_date=None, model_months=(6, 7, 8, 9), run_model=True,
                        export_normalized_vi=False, export_minmax=False,
                        export_start_date=None, export_end_date=None,
                        mosaic=True, plot_mosaic=False, plot_tiles=False,
                        max_workers=1, model_threads=1, minmax_by_date_dir=None):
    """Runs the model over every tile under rbf_dir, then mosaics.
    Re-running only processes dates that are new since the last run.

    :param rbf_dir: rbf.run_tsi_tiles() output folder (one folder per tile).
    :param out_dir: everything is written under here -- see module docstring.
    :param year: season to process; also fills {year} in vi_pattern.
    :param model_path: pickled sklearn model with predict_proba().
    :param vi_keys: {canonical name: file token}, e.g. 2026 naming
        {"NDVI": "NDVI", "EVI": "EVI", "NDMI": "NDMI", "CIRE": "CIre", "CCI": "CCI"}.
        None = all five with token == name.
    :param tiles_csv, tiles_column, tiles_csv_mode: optional tile allow
        list ("include") or skip list ("exclude").
    :param forest_mask_path: optional mask raster (any CRS/extent); only
        pixels == 1 are used. None = every pixel with data.
    :param start_date, end_date: optional inclusive date window (date or
        "YYYY-MM-DD"). Dates before start_date never enter the min/max.
    :param model_months: months in which the model runs (min/max still
        updates for every date). None = every month.
    :param run_model: False only updates min/max (and exports).
    :param export_normalized_vi, export_minmax: also write each date's
        normalized VIs / min-max snapshot (vi_norm/, minmax_by_date/),
        limited to [export_start_date, export_end_date] if given.
    :param mosaic, plot_mosaic, plot_tiles: mosaic per date; plot mosaics /
        per-tile outputs as they're written (leave plot_tiles off with
        max_workers > 1).
    :param max_workers: tiles processed in parallel. Each needs a few GB
        of RAM at most.
    :param model_threads: BLAS/OpenMP threads per tile during prediction.
    :param minmax_by_date_dir: the minmax_by_date/ folder of an earlier run
        (with export_minmax=True), e.g. the near-real-time one. Each date is
        then normalized with that run's min/max snapshot of the same date
        (widened by this date's own value, as the running min/max is)
        instead of this run's running min/max, and only dates with a
        snapshot of every VI are processed. Used by
        interpolate.run_full_data_season() to rerun a season on fully
        interpolated VIs with the near-real-time normalization.
    """
    call_params = dict(locals())
    tile_dirs = discover_tile_dirs(rbf_dir, tiles_csv, tiles_column, tiles_csv_mode)
    print(f"Found {len(tile_dirs)} tile(s) under {rbf_dir}")

    kwargs = dict(year=year, model_path=model_path, vi_keys=vi_keys, vi_pattern=vi_pattern,
                  forest_mask_path=forest_mask_path, start_date=start_date, end_date=end_date,
                  model_months=model_months, run_model=run_model,
                  export_normalized_vi=export_normalized_vi, export_minmax=export_minmax,
                  export_start_date=export_start_date, export_end_date=export_end_date,
                  plot=plot_tiles, model_threads=model_threads,
                  minmax_by_date_dir=minmax_by_date_dir)
    log_dir = os.path.join(out_dir, "logs") if max_workers > 1 else None
    if log_dir:
        print(f"Per-tile logs: {log_dir}")

    start = time.monotonic()
    failed = []

    def report(done, result):
        tile_id, status, error, seconds = result
        elapsed = time.monotonic() - start
        eta = f", ~{format_duration(elapsed / done * (len(tile_dirs) - done))} left" if done < len(tile_dirs) else ""
        line = f"FAILED: {error}" if error else status
        print(f"  [{done}/{len(tile_dirs)}, {format_duration(elapsed)}{eta}] {tile_id}: {line}", flush=True)
        if error:
            failed.append(tile_id)

    if max_workers <= 1:
        for i, tile_dir in enumerate(tile_dirs, start=1):
            report(i, _run_tile(tile_dir, out_dir, None, kwargs))
    else:
        with ProcessPoolExecutor(max_workers=max_workers) as pool:
            futures = {pool.submit(_run_tile, t, out_dir, log_dir, kwargs): t for t in tile_dirs}
            for i, fut in enumerate(as_completed(futures), start=1):
                try:
                    result = fut.result()
                except Exception as e:  # worker died outright, e.g. out of memory
                    result = (os.path.basename(futures[fut]), None, f"worker crashed: {e}", 0)
                report(i, result)

    mosaics = []
    if mosaic and run_model and model_path:
        tile_outs = [os.path.join(out_dir, os.path.basename(t)) for t in tile_dirs]
        mosaics = mosaic_by_filename([t for t in tile_outs if os.path.isdir(t)],
                                     os.path.join(out_dir, "mosaic"), PROBA_PATTERN)
        print(f"Mosaicked {len(mosaics)} new/updated date(s) into {os.path.join(out_dir, 'mosaic')}")
        if plot_mosaic:
            for path in mosaics:
                plot_probability(path, os.path.basename(path)[len("Disco_Proba_"):-len(".tif")])

    summary = {"tiles": len(tile_dirs), "failed": failed, "mosaics_written": len(mosaics),
               "elapsed_seconds": time.monotonic() - start}
    print(f"Done: {len(tile_dirs) - len(failed)} tile(s) ok, {len(failed)} failed "
          f"in {format_duration(summary['elapsed_seconds'])}")
    write_run_params(out_dir, "apply_model.apply_discoch_tiles", call_params, summary)
    return summary


# ---------------------------------------------------------------------
# Plotting
# ---------------------------------------------------------------------

def plot_probability(path, title, save_png=True, show=True, min_pixels=100):
    """Map of one Disco_Proba_*.tif on a topographic basemap; saved next to
    it as .png. Plotting libraries are only imported here."""
    import matplotlib.pyplot as plt
    from matplotlib.colors import LinearSegmentedColormap

    proba = read_scaled(path)
    if np.count_nonzero(~np.isnan(proba)) < min_pixels:
        print(f"  not enough valid pixels to plot {os.path.basename(path)}")
        return
    with rasterio.open(path) as src:
        b, crs = src.bounds, src.crs

    cmap = LinearSegmentedColormap.from_list(
        "disco", ["#0A2F1F", "#228B22", "#9ACD32", "#FFFF00", "#FFFFE0"])
    fig, ax = plt.subplots(figsize=(10, 8))
    im = ax.imshow(proba, cmap=cmap, vmin=0, vmax=1, extent=(b.left, b.right, b.bottom, b.top))
    try:
        import contextily as cx
        cx.add_basemap(ax, crs=crs.to_string(), source=cx.providers.Esri.WorldTopoMap)
    except Exception as e:
        print(f"  basemap unavailable: {e}", file=sys.stderr)
    fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04).set_label("Discoloration Probability", fontweight="bold")
    ax.set_title(f"Discoloration Probability: {title}", fontsize=14)
    ax.axis("off")
    if save_png:
        fig.savefig(os.path.splitext(path)[0] + ".png", dpi=300, bbox_inches="tight")
    if show:
        plt.show()
    plt.close(fig)
