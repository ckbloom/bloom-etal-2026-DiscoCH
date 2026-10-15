"""
disco_statistics.py

Per-date discoloration statistics from a folder of Disco_Proba_<date>.tif
mosaics: how much of the valid (non-nodata) area is at or above a
threshold, over the whole mosaic and, optionally, per polygon of a zone
layer (e.g. forest or political districts).

  - A pixel is discolored when its stored value >= threshold. Thresholds
    are on the raster's stored scale, as in validate.py: 4000 on the
    0-10000 int16 masked mosaics (and on the old uint16 Proba_mosaic_*
    files) = 40 % probability.
  - Valid pixels: not the file's nodata (or the `nodata` override) and not
    NaN. Areas in hectares come from each raster's own pixel size.
  - Zones: polygons are reprojected to the raster's CRS and burned in by
    pixel centre (rasterio.features.rasterize); polygons sharing a
    zone_field value are counted together. Where polygons overlap, the
    later one wins. The "All" rows always cover every valid pixel of the
    raster, inside a zone or not.
  - The mosaics in one call are counted on their most common grid. A file
    on the same pixel grid but another extent is read by its offset
    (pixels outside it count as nodata); a file off that grid is skipped
    with a message.

Output: a long table, one row per date and zone, with valid/discolored
pixels and hectares and the discolored proportion. summarize_discoloration()
also writes it and a wide date x zone table of proportions as CSV (UTF-8
with BOM, so Excel shows the accents in district names).

Replaces the table parts of plot_discoloration_timeseries.py
(compute_discolored_proportions(), compute_discoloration_table(),
save_discoloration_table()); its year-comparison plot is now compare.py.
Differences:
  - one pass over the mosaics gives the total and every zone together;
  - read in chunks with several threads instead of whole rasters, so
    memory stays small for national 10 m mosaics;
  - hectares for every zone, from the raster's pixel size (no
    pixel_size_m argument);
  - nothing is written unless output_dir is given.

Usage (notebook):

    from src.disco_ch_operational.disco_statistics import summarize_discoloration

    stats = summarize_discoloration(masked_output, threshold=4000, zones=districts,
                                    zone_field="bzname", year=2026, output_dir=stats_output)
"""

import os
import re
import glob
import time
import warnings
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime

import numpy as np
import pandas as pd
import rasterio
import shapely
from rasterio.errors import NotGeoreferencedWarning
from rasterio.features import rasterize
from rasterio.windows import Window

from src.disco_ch_operational.utils import chunk_windows, format_duration

TOTAL = "All"
DATE_RE = re.compile(r"(\d{4}-\d{2}-\d{2})")


# ---------------------------------------------------------------------
# Inputs
# ---------------------------------------------------------------------

def _year_matches(d, year):
    """year: None (any), an int, or an iterable of ints."""
    if year is None:
        return True
    if isinstance(year, (int, np.integer)):
        return d.year == year
    return d.year in set(year)


def list_mosaics(mosaic_dir, pattern="Disco_Proba_*.tif", year=None):
    """[(date, path)] of the files matching pattern whose name holds a
    YYYY-MM-DD date (in `year` if given), sorted by date."""
    paths = sorted(glob.glob(os.path.join(mosaic_dir, pattern)))
    if not paths:
        raise FileNotFoundError(f"No files matching {pattern!r} in {mosaic_dir}")
    out = []
    for path in paths:
        m = DATE_RE.search(os.path.basename(path))
        if not m:
            print(f"  No date in {os.path.basename(path)}, skipping")
            continue
        d = datetime.strptime(m.group(1), "%Y-%m-%d").date()
        if _year_matches(d, year):
            out.append((d, path))
    out.sort()
    if not out:
        raise FileNotFoundError(f"No dated {pattern!r} files in {mosaic_dir}"
                                + (f" for {year}" if year is not None else ""))
    return out


def load_zones(zones, crs, zone_field="bzname"):
    """(names, [(geometry, zone index)]) of a polygon layer (path or
    GeoDataFrame) in `crs`; polygons sharing a zone_field value get the
    same index. Polygons without a name or geometry are dropped."""
    import geopandas as gpd

    gdf = gpd.read_file(zones) if isinstance(zones, (str, os.PathLike)) else zones
    if zone_field not in gdf.columns:
        raise ValueError(f"zone_field {zone_field!r} not in {list(gdf.columns)}")
    gdf = gdf[gdf[zone_field].notna() & gdf.geometry.notna() & ~gdf.geometry.is_empty]
    if gdf.crs is not None and gdf.crs != crs:
        gdf = gdf.to_crs(crs)
    names = sorted(gdf[zone_field].astype(str).unique())
    index = {n: i for i, n in enumerate(names)}
    shapes = [(geom, index[str(n)]) for geom, n in zip(gdf.geometry, gdf[zone_field])]
    return names, shapes


# ---------------------------------------------------------------------
# Counting
# ---------------------------------------------------------------------

def _aligned_files(mosaics):
    """The mosaics on the pixel grid (CRS, pixel size, alignment) of the
    most common one, as [(date, path, col_off, row_off, width, height)]
    with offsets on the union of their extents, and that union's (crs,
    transform, width, height). Files off the grid are reported and
    skipped."""
    grids = {}
    for d, path in mosaics:
        with rasterio.open(path) as src:
            grids[path] = (src.crs, src.transform, src.width, src.height)
    keys = [(g[0].to_string() if g[0] else None, tuple(g[1]), g[2], g[3]) for g in grids.values()]
    ref = grids[list(grids)[keys.index(max(set(keys), key=keys.count))]]
    crs, transform = ref[0], ref[1]
    files = []
    for d, path in mosaics:
        g_crs, g_tr, w, h = grids[path]
        col, row = ~transform * (g_tr.c, g_tr.f)
        if (g_crs == crs and (g_tr.a, g_tr.b, g_tr.d, g_tr.e) == (transform.a, transform.b, transform.d, transform.e)
                and abs(col - round(col)) < 1e-6 and abs(row - round(row)) < 1e-6):
            files.append((d, path, int(round(col)), int(round(row)), w, h))
        else:
            print(f"  {os.path.basename(path)} is not on the common pixel grid, skipping")

    c0, r0 = min(f[2] for f in files), min(f[3] for f in files)
    width = max(f[2] + f[4] for f in files) - c0
    height = max(f[3] + f[5] for f in files) - r0
    files = [(d, p, c - c0, r - r0, w, h) for d, p, c, r, w, h in files]
    for d, p, c, r, w, h in files:
        if (c, r, w, h) != (0, 0, width, height):
            print(f"  {os.path.basename(p)} covers {w} x {h} of the {width} x {height} px; "
                  "pixels outside it count as nodata")
    return files, (crs, transform * transform.translation(c0, r0), width, height)


def _count_window(files, window, transform, threshold, nodata, shapes, tree, n_zones):
    """valid/discolored pixel counts (dates, 1 + n_zones) in one window;
    column 0 = all pixels, column 1 + i = zone i. tree: STRtree of the
    shapes' geometries, to rasterize only those touching the window."""
    valid_n = np.zeros((len(files), 1 + n_zones), dtype="int64")
    disc_n = np.zeros_like(valid_n)
    zone = None
    if shapes:
        hits = tree.query(shapely.box(*rasterio.windows.bounds(window, transform)))
        zone = np.empty(0, dtype="int32")          # no zone here: totals only
        if len(hits):
            zone = rasterize([shapes[j] for j in hits], out_shape=(window.height, window.width),
                             transform=rasterio.windows.transform(window, transform),
                             fill=-1, dtype="int32")
            if (zone < 0).all():
                zone = np.empty(0, dtype="int32")
    for i, (_, path, dc, dr, fw, fh) in enumerate(files):
        # Part of the window inside this file (all of it on the common grid).
        c0, r0 = max(window.col_off - dc, 0), max(window.row_off - dr, 0)
        c1, r1 = min(window.col_off + window.width - dc, fw), min(window.row_off + window.height - dr, fh)
        if c1 <= c0 or r1 <= r0:
            continue
        with rasterio.open(path) as src:
            arr = src.read(1, window=Window(c0, r0, c1 - c0, r1 - r0))
            nd = src.nodata if nodata is None else nodata
        ok = np.ones(arr.shape, bool) if nd is None or np.isnan(nd) else arr != nd
        if np.issubdtype(arr.dtype, np.floating):
            ok &= ~np.isnan(arr)
        if arr.shape == (window.height, window.width):
            valid = ok
        else:
            valid = np.zeros((window.height, window.width), bool)
            valid[r0 + dr - window.row_off:r1 + dr - window.row_off,
                  c0 + dc - window.col_off:c1 + dc - window.col_off] = ok
            arr_full = np.zeros(valid.shape, arr.dtype)
            arr_full[valid] = arr[ok]
            arr = arr_full
        disc = valid & (arr >= threshold)
        valid_n[i, 0], disc_n[i, 0] = valid.sum(), disc.sum()
        if zone is not None and zone.size:
            z = zone + 1                            # 0 = outside every zone
            valid_n[i, 1:] = np.bincount(z[valid], minlength=n_zones + 1)[1:]
            disc_n[i, 1:] = np.bincount(z[disc], minlength=n_zones + 1)[1:]
    return valid_n, disc_n


def discoloration_stats(mosaic_dir, threshold=4000, zones=None, zone_field="bzname",
                        pattern="Disco_Proba_*.tif", year=None, nodata=None,
                        chunk_size=2048, max_workers=4):
    """Long DataFrame of discoloration per date and zone (see the module
    docstring for the definitions).

    :param threshold: stored raster value at/above which a pixel is
        discolored (4000 = 40 % on 0-10000 mosaics).
    :param zones: polygon layer (path or GeoDataFrame), e.g. districts;
        None = only the "All" total.
    :param zone_field: zone name column, e.g. "bzname".
    :param pattern: glob of the mosaics, e.g. "Proba_mosaic_*.tif" for the
        2017-2023 Bloom et al. (2026) mosaics.
    :param year: int or list of ints to keep only those years' dates.
    :param nodata: overrides the files' own nodata value.
    :return: columns date, zone, valid_pixels, discolored_pixels,
        proportion, valid_ha, discolored_ha.
    """
    t0 = time.time()
    files, (crs, transform, width, height) = _aligned_files(list_mosaics(mosaic_dir, pattern, year))
    dates = [f[0] for f in files]
    names, shapes = load_zones(zones, crs, zone_field) if zones is not None else ([], [])
    tree = shapely.STRtree([g for g, _ in shapes]) if shapes else None
    print(f"Counting {len(files)} mosaics in {mosaic_dir} ({width} x {height} px"
          + (f", {len(names)} zones" if names else "") + f"), threshold >= {threshold:g} ...")

    valid_n = np.zeros((len(files), 1 + len(names)), dtype="int64")
    disc_n = np.zeros_like(valid_n)
    windows = list(chunk_windows(width, height, chunk_size))
    # rasterize() in worker threads raises a spurious NotGeoreferencedWarning
    # (the counts are identical to a single-threaded run).
    with warnings.catch_warnings(), ThreadPoolExecutor(max_workers=max_workers) as pool:
        warnings.simplefilter("ignore", NotGeoreferencedWarning)
        for v, d in pool.map(lambda w: _count_window(files, w, transform, threshold, nodata,
                                                     shapes, tree, len(names)), windows):
            valid_n += v
            disc_n += d

    pixel_ha = abs(transform.a * transform.e) / 10000
    zone_names = [TOTAL] + names
    df = pd.DataFrame({
        "date": pd.to_datetime(np.repeat(dates, len(zone_names))),
        "zone": np.tile(zone_names, len(dates)),
        "valid_pixels": valid_n.ravel(),
        "discolored_pixels": disc_n.ravel(),
    })
    with np.errstate(invalid="ignore", divide="ignore"):
        df["proportion"] = np.where(df.valid_pixels > 0, df.discolored_pixels / df.valid_pixels, np.nan)
    df["valid_ha"] = df.valid_pixels * pixel_ha
    df["discolored_ha"] = df.discolored_pixels * pixel_ha

    empty = df[(df.zone == TOTAL) & (df.valid_pixels == 0)].date
    if len(empty):
        print(f"  No valid pixels on {', '.join(d.strftime('%Y-%m-%d') for d in empty)}")
    print(f"Done in {format_duration(time.time() - t0)}")
    return df


def wide_table(stats, value="proportion"):
    """date x zone table of one column of a discoloration_stats() table,
    "All" first."""
    wide = stats.pivot(index="date", columns="zone", values=value)
    return wide[[TOTAL] + [c for c in wide.columns if c != TOTAL]]


# ---------------------------------------------------------------------
# Driver
# ---------------------------------------------------------------------

def summarize_discoloration(mosaic_dir, threshold=4000, zones=None, zone_field="bzname",
                            pattern="Disco_Proba_*.tif", year=None, nodata=None,
                            output_dir=None, label=None, chunk_size=2048, max_workers=4):
    """discoloration_stats(), printed as a date x zone proportion table and,
    with output_dir, saved there as discoloration_<label>_t<threshold>.csv
    (long) and ..._proportion_by_zone.csv (wide).

    :param label: file name label; defaults to the year(s) or "all".
    :return: the long DataFrame (pass it to compare.plot_year_comparison()
        to plot without counting again).
    """
    stats = discoloration_stats(mosaic_dir, threshold, zones, zone_field, pattern, year,
                                nodata, chunk_size, max_workers)
    total = stats[stats.zone == TOTAL].set_index("date")
    with pd.option_context("display.width", 200):
        print(total[["valid_ha", "discolored_ha", "proportion"]].round(4).to_string())

    if output_dir:
        os.makedirs(output_dir, exist_ok=True)
        if label is None:
            label = ("all" if year is None else str(year) if isinstance(year, (int, np.integer))
                     else "-".join(str(y) for y in year))
        base = os.path.join(output_dir, f"discoloration_{label}_t{threshold:g}")
        stats.to_csv(base + ".csv", index=False, date_format="%Y-%m-%d", encoding="utf-8-sig")
        wide_table(stats).to_csv(base + "_proportion_by_zone.csv", date_format="%Y-%m-%d",
                                 encoding="utf-8-sig")
        print(f"Saved {base}.csv and {base}_proportion_by_zone.csv")
    return stats
