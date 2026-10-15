"""
utils.py

Helpers shared across the DiscoCH operational scripts: tile discovery,
VI file lookup, band-date parsing, reading/warping rasters onto a tile's
grid, scaled-integer GeoTIFF output, mosaicking, and run bookkeeping.

Plain rasterio + numpy throughout. Importing this module has no side
effects (no GDAL environment changes, no warning filters).
"""

import os
import re
import csv
import glob
import json
import shutil
import contextlib
from datetime import date, datetime

import numpy as np
import rasterio
from rasterio.merge import merge as rio_merge
from rasterio.warp import reproject, Resampling


# ---------------------------------------------------------------------
# VI names
# ---------------------------------------------------------------------

# The five VIs the DiscoCH model uses, by canonical name.
CANONICAL_VIS = ("NDVI", "EVI", "NDMI", "CIRE", "CCI")


def resolve_vi_keys(vi_keys=None):
    """{canonical_name: file_token} from None (all CANONICAL_VIS), a list
    of names (token == name), or a dict. The file token is only used to
    find files on disk -- different years/products abbreviate differently
    (e.g. 2026 "CIre", 2018 "CRE"/"NDV"/"NDM"); everything downstream is
    keyed by canonical name."""
    if vi_keys is None:
        vi_keys = CANONICAL_VIS
    if isinstance(vi_keys, dict):
        return dict(vi_keys)
    return {k: k for k in vi_keys}


# ---------------------------------------------------------------------
# Tiles and files
# ---------------------------------------------------------------------

TILE_ID_RE = r"^X\d{4}_Y\d{4}$"


def read_tile_ids(csv_path, column="Tile_ID"):
    """Set of tile IDs (e.g. "X0068_Y0043") from one CSV column.
    utf-8-sig strips the BOM Excel adds, which would otherwise hide the
    first header name."""
    with open(csv_path, newline="", encoding="utf-8-sig") as f:
        reader = csv.DictReader(f)
        if reader.fieldnames is None or column not in reader.fieldnames:
            raise ValueError(f"Column {column!r} not found in {csv_path} (columns: {reader.fieldnames})")
        return {row[column].strip() for row in reader if row.get(column, "").strip()}


def discover_tile_dirs(root_dir, tiles_csv=None, tiles_column="Tile_ID", tiles_csv_mode="include",
                       tile_regex=TILE_ID_RE):
    """Sorted tile folders (names matching tile_regex) directly under
    root_dir. tiles_csv optionally restricts them: "include" keeps only the
    listed tiles, "exclude" drops them. tile_regex=None accepts any folder."""
    pattern = re.compile(tile_regex) if tile_regex else None
    tile_dirs = sorted(p for p in glob.glob(os.path.join(root_dir, "*"))
                       if os.path.isdir(p) and (pattern is None or pattern.match(os.path.basename(p))))

    if tiles_csv:
        ids = read_tile_ids(tiles_csv, tiles_column)
        before = len(tile_dirs)
        if tiles_csv_mode == "include":
            tile_dirs = [p for p in tile_dirs if os.path.basename(p) in ids]
        elif tiles_csv_mode == "exclude":
            tile_dirs = [p for p in tile_dirs if os.path.basename(p) not in ids]
        else:
            raise ValueError(f"tiles_csv_mode must be 'include' or 'exclude', got {tiles_csv_mode!r}")
        print(f"  tiles_csv ({os.path.basename(tiles_csv)}, {tiles_csv_mode}): "
              f"{len(tile_dirs)} of {before} tile(s) selected")
    return tile_dirs


def find_vi_file(tile_dir, file_token, year, pattern_template):
    """The one file in tile_dir matching pattern_template (placeholders
    {year} and {vi}). Raises if none or several match rather than guessing."""
    pattern = pattern_template.format(year=year, vi=file_token)
    matches = sorted(glob.glob(os.path.join(tile_dir, pattern)))
    if not matches:
        raise FileNotFoundError(f"no file matching {pattern!r} in {tile_dir}")
    if len(matches) > 1:
        raise ValueError(f"several files match {pattern!r} in {tile_dir}: {matches}")
    return matches[0]


# ---------------------------------------------------------------------
# Band dates
# ---------------------------------------------------------------------

_DATE_COMPACT = re.compile(r"^(\d{8})(?:_|$)")   # "20260405_SEN2A" (TSS) or "20260405"
_DATE_ISO = re.compile(r"^(\d{4}-\d{2}-\d{2})$")  # "2026-04-05" (rbf.py TSI)


def parse_band_date(desc):
    """date from a band description, or None if it has no recognised date."""
    if not desc:
        return None
    m = _DATE_COMPACT.match(desc)
    if m:
        return datetime.strptime(m.group(1), "%Y%m%d").date()
    m = _DATE_ISO.match(desc)
    if m:
        return date.fromisoformat(m.group(1))
    return None


def band_dates(path):
    """{date: [1-based band indices]} from band descriptions. Several bands
    on one date (two satellites) are all listed; read_date_mean() averages
    them. FORCE keeps descriptions in a .aux.xml sidecar, so sibling-file
    lookup is forced on in case GDAL_DISABLE_READDIR_ON_OPEN is set."""
    with rasterio.Env(GDAL_DISABLE_READDIR_ON_OPEN="NO"):
        with rasterio.open(path) as src:
            descriptions = src.descriptions
    out = {}
    for i, desc in enumerate(descriptions, start=1):
        d = parse_band_date(desc)
        if d is not None:
            out.setdefault(d, []).append(i)
    return out


def as_date(value):
    """date passthrough; also accepts an ISO "YYYY-MM-DD" string or None."""
    return date.fromisoformat(value) if isinstance(value, str) else value


# ---------------------------------------------------------------------
# Reading
# ---------------------------------------------------------------------

def read_float(src, band, window=None):
    """One band (or a list of bands) as float32 with the file's nodata (and
    any masked pixel) as NaN, optionally limited to a rasterio Window."""
    return src.read(band, window=window, masked=True).astype("float32").filled(np.nan)


def read_date_mean(src, band_indices, window=None):
    """Per-pixel mean of one date's band(s), NaN-skipping -- a pixel valid
    in only one of two same-day bands keeps that value."""
    if len(band_indices) == 1:
        return read_float(src, band_indices[0], window)
    return mean_bands(read_float(src, list(band_indices), window))


def mean_bands(stack):
    """NaN-skipping mean over axis 0 (NaN where every band is NaN)."""
    if stack.shape[0] == 1:
        return stack[0]
    with np.errstate(invalid="ignore"), _quiet_nanmean():
        return np.nanmean(stack, axis=0).astype("float32")


@contextlib.contextmanager
def _quiet_nanmean():
    import warnings
    with warnings.catch_warnings():
        warnings.filterwarnings("ignore", "Mean of empty slice", RuntimeWarning)
        yield


def chunk_windows(width, height, size=None):
    """Row-major Windows tiling a raster into size x size blocks (edge blocks
    smaller); size=None gives one window covering everything."""
    from rasterio.windows import Window
    size = size or max(width, height)
    for row in range(0, height, size):
        for col in range(0, width, size):
            yield Window(col, row, min(size, width - col), min(size, height - row))


def warp_to_grid(path, transform, crs, shape, resampling=Resampling.nearest):
    """Band 1 of `path` reprojected onto a target grid, as float32 with
    nodata/outside-extent as NaN. Only the needed part of the source is read."""
    out = np.full(shape, np.nan, dtype="float32")
    with rasterio.open(path) as src:
        reproject(rasterio.band(src, 1), out, src_nodata=src.nodata,
                  dst_transform=transform, dst_crs=crs, dst_nodata=np.nan,
                  resampling=resampling)
    return out


# ---------------------------------------------------------------------
# Writing
# ---------------------------------------------------------------------

def write_float(arr, path, transform, crs):
    """float32 single-band GeoTIFF, NaN nodata, written to a temporary file
    and swapped into place so a crash never leaves a truncated file."""
    tmp = path + ".tmp"
    with rasterio.open(tmp, "w", driver="GTiff", height=arr.shape[0], width=arr.shape[1], count=1,
                       dtype="float32", crs=crs, transform=transform, nodata=np.nan,
                       compress="deflate", tiled=True) as dst:
        dst.write(arr.astype("float32"), 1)
    os.replace(tmp, path)


def write_scaled_int(arr, path, transform, crs, dtype="int16"):
    """Single-band integer GeoTIFF of a float array (NaN = nodata), scaled
    by the largest of 10000/1000/100/10/1 that fits the dtype -- the same
    convention as mask_and_scale_rasters.scale_and_save_as_int_from_da():
    nodata = dtype minimum, GDAL band scale = 1/scale (true = stored * scale),
    LZW + predictor 2, tiled.

    :return: the scale factor used, or None (nothing written) if arr has no
        valid pixels.
    """
    valid = arr[~np.isnan(arr)]
    if valid.size == 0:
        return None
    info = np.iinfo(dtype)
    max_abs = float(np.max(np.abs(valid)))
    scale = next((c for c in (10000, 1000, 100, 10) if max_abs * c <= info.max), 1)
    out = np.where(np.isnan(arr), info.min, np.round(arr * scale)).astype(dtype)

    tmp = path + ".tmp"
    with rasterio.open(tmp, "w", driver="GTiff", height=arr.shape[0], width=arr.shape[1], count=1,
                       dtype=dtype, crs=crs, transform=transform, nodata=info.min,
                       compress="LZW", predictor=2, tiled=True) as dst:
        dst.write(out, 1)
        dst.scales = (1.0 / scale,)
        dst.offsets = (0.0,)
    os.replace(tmp, path)
    return scale


def read_scaled(path):
    """float32 values of a write_scaled_int() file (nodata -> NaN, band
    scale/offset applied)."""
    with rasterio.open(path) as src:
        arr = read_float(src, 1)
        return arr * src.scales[0] + src.offsets[0]


# ---------------------------------------------------------------------
# Mosaicking
# ---------------------------------------------------------------------

def mosaic_by_filename(tile_dirs, mosaic_dir, pattern, key=None):
    """Merges same-named files (e.g. Disco_Proba_2026-06-15.tif) across
    tile folders into one file per name in mosaic_dir. A mosaic newer than
    all its tile files is left alone, so re-running only redoes dates that
    changed. Band scale/offset carry over.

    key: optional function path -> mosaic file name, for tiles that keep
    the tile id in the name (e.g. Proba_X0051_Y0066_2018-08-14.tif in one
    folder); default os.path.basename.

    :return: list of mosaic paths written this call.
    """
    os.makedirs(mosaic_dir, exist_ok=True)
    by_name = {}
    for d in tile_dirs:
        for f in glob.glob(os.path.join(d, pattern)):
            by_name.setdefault((key or os.path.basename)(f), []).append(f)

    written = []
    for name, paths in sorted(by_name.items()):
        out_path = os.path.join(mosaic_dir, name)
        if os.path.exists(out_path) and os.path.getmtime(out_path) >= max(map(os.path.getmtime, paths)):
            continue
        if len(paths) == 1:
            shutil.copy(paths[0], out_path)
        else:
            with contextlib.ExitStack() as stack:
                srcs = [stack.enter_context(rasterio.open(p)) for p in paths]
                arr, transform = rio_merge(srcs, nodata=srcs[0].nodata)
                profile = srcs[0].profile
                profile.update(height=arr.shape[1], width=arr.shape[2], transform=transform)
                tmp = out_path + ".tmp"
                with rasterio.open(tmp, "w", **profile) as dst:
                    dst.write(arr)
                    dst.scales, dst.offsets = srcs[0].scales, srcs[0].offsets
            os.replace(tmp, out_path)
        written.append(out_path)
    return written


# ---------------------------------------------------------------------
# Run bookkeeping
# ---------------------------------------------------------------------

def format_duration(seconds):
    """H:MM:SS, or M:SS under an hour."""
    seconds = int(seconds)
    h, rem = divmod(seconds, 3600)
    m, s = divmod(rem, 60)
    return f"{h}:{m:02d}:{s:02d}" if h else f"{m}:{s:02d}"


def write_run_params(out_dir, script, params, summary=None):
    """Timestamped JSON record of a run's parameters (and optional result
    summary) in out_dir, so it's always possible to tell what produced a
    folder's contents. Non-JSON values (dates, paths) are stringified."""
    os.makedirs(out_dir, exist_ok=True)
    stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    record = {"script": script, "timestamp": stamp, "params": params}
    if summary is not None:
        record["summary"] = summary
    path = os.path.join(out_dir, f"{script.rsplit('.', 1)[-1]}_params_{stamp}.json")
    with open(path, "w") as f:
        json.dump(record, f, indent=2, default=str)
    print(f"Wrote run parameters to {path}")
    return path


def limit_threads(n):
    """Context manager capping BLAS/OpenMP threads (e.g. inside sklearn
    predict) to n -- keeps parallel worker processes from oversubscribing
    the CPU. No-op if n is None or threadpoolctl isn't installed."""
    if n is None:
        return contextlib.nullcontext()
    try:
        from threadpoolctl import threadpool_limits
    except ImportError:
        return contextlib.nullcontext()
    return threadpool_limits(limits=n)
