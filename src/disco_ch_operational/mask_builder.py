"""
masking.py

Single-date VI layers as masked Swiss mosaics, built from the tiled
outputs with the helpers of utils.py and mask.py:

  - the VI of one date from the interpolated *_TSS_TSI.tif tiles
    (vi_date_tiles()), e.g. EVI on 5 June 2018 and 2026;
  - a per-date min/max snapshot of the near-real-time run
    (snapshot_tiles(): minmax_by_date/<tile>/<VI>/<VI>_min_<date>.tif),
    e.g. the CCI minimum as of 5 July 2026;
  - the difference of two masked layers (difference()), e.g. EVI 2026 - 2018.

Per layer:
  1. one float32 file per tile in true VI units (stored value / 10000) in
     <out_dir>/tiles/<tile>/;
  2. merged into <out_dir>/mosaic/ (EPSG:3035, utils.mosaic_by_filename());
  3. reprojected onto the forest mask grid (EPSG:2056), blanked outside it
     and written as int16 with band scale 0.0001 into <out_dir>/masked/
     (mask.mask_and_scale_dir()).
The difference is computed on the masked layers (same grid for every year,
even when the years have different tiles), block by block to keep memory low.

Usage (notebook or `python -m src.disco_ch_operational.masking`):

    from src.disco_ch_operational.masking import vi_date_tiles, snapshot_tiles, mosaic_and_mask, difference

    vi_date_tiles(level4_rbf_2018, "EVI", "2018-06-05", out_dir)
    vi_date_tiles(level4_rbf_2026, "EVI", "2026-06-05", out_dir)
    mosaic_and_mask(out_dir, mask_path)
    difference(f"{out_dir}/masked/EVI_2026-06-05.tif", f"{out_dir}/masked/EVI_2018-06-05.tif",
               f"{out_dir}/masked/EVI_diff_2026-06-05_minus_2018-06-05.tif")
"""

import os
from pathlib import Path

import numpy as np
import rasterio

from src.disco_ch_operational.apply_model import DEFAULT_VI_PATTERN
from src.disco_ch_operational.mask import mask_and_scale_dir
from src.disco_ch_operational.utils import (
    as_date, band_dates, chunk_windows, discover_tile_dirs, find_vi_file, mosaic_by_filename,
    read_date_mean, read_float, write_float,
)

ROOT = Path(__file__).resolve().parents[2]
MASK_PATH = ROOT / "data" / "DiscoCH_2026_Mask.tif"
VI_SCALE = 10000  # FORCE / RBF stored VI units -> VI


def vi_date_tiles(tsi_dir, vi, day, out_dir, file_token=None, pattern=DEFAULT_VI_PATTERN, name=None):
    """The VI of one date from every tile's interpolated TSI file (band
    description = date; several bands on the date are averaged), written as
    <out_dir>/tiles/<tile>/<name> in VI units.

    :param tsi_dir: level4_rbf/<year> folder with one folder per tile.
    :param vi: VI name for the output file, e.g. "EVI".
    :param day: date (or "YYYY-MM-DD") of the band to take.
    :param file_token: VI as spelled in the file names (default vi; 2018 uses
        e.g. "NDV", "CRE").
    :param name: output file name (default <vi>_<day>.tif).
    :return: number of tiles written.
    """
    day = as_date(day)
    name = name or f"{vi}_{day}.tif"
    n = 0
    for tile_dir in discover_tile_dirs(tsi_dir):
        tile = os.path.basename(tile_dir)
        try:
            path = find_vi_file(tile_dir, file_token or vi, str(day.year), pattern)
        except FileNotFoundError as e:
            print(f"  {tile}: {e}")
            continue
        bands = band_dates(path).get(day)
        if not bands:
            print(f"  {tile}: no band for {day} in {os.path.basename(path)}")
            continue
        with rasterio.open(path) as src:
            arr = read_date_mean(src, bands) / VI_SCALE
            transform, crs = src.transform, src.crs
        os.makedirs(os.path.join(out_dir, "tiles", tile), exist_ok=True)
        write_float(arr, os.path.join(out_dir, "tiles", tile, name), transform, crs)
        n += 1
    print(f"{name}: {n} tile(s) from {tsi_dir}")
    return n


def snapshot_tiles(minmax_dir, vi, kind, day, out_dir, name=None):
    """A min/max snapshot of the near-real-time run (minmax_by_date/<tile>/
    <vi>/<vi>_<kind>_<day>.tif, stored VI units) per tile, written as
    <out_dir>/tiles/<tile>/<name> in VI units.

    :param kind: "min" or "max".
    :return: number of tiles written.
    """
    day = as_date(day)
    name = name or f"{vi}_{kind}_{day}.tif"
    n = 0
    for tile_dir in discover_tile_dirs(minmax_dir):
        tile = os.path.basename(tile_dir)
        path = os.path.join(tile_dir, vi, f"{vi}_{kind}_{day}.tif")
        if not os.path.exists(path):
            print(f"  {tile}: missing {os.path.basename(path)}")
            continue
        with rasterio.open(path) as src:
            arr = (read_float(src, 1) * src.scales[0] + src.offsets[0]) / VI_SCALE
            transform, crs = src.transform, src.crs
        os.makedirs(os.path.join(out_dir, "tiles", tile), exist_ok=True)
        write_float(arr, os.path.join(out_dir, "tiles", tile, name), transform, crs)
        n += 1
    print(f"{name}: {n} tile(s) from {minmax_dir}")
    return n


def mosaic_and_mask(out_dir, mask_path=MASK_PATH, pattern="*.tif", threads=4):
    """Mosaics every per-tile layer in <out_dir>/tiles (-> <out_dir>/mosaic)
    and masks/reprojects the mosaics (-> <out_dir>/masked). Both steps only
    redo files whose inputs are newer.

    :return: {file name: status} of the masking.
    """
    tile_dirs = discover_tile_dirs(os.path.join(out_dir, "tiles"))
    mosaics = mosaic_by_filename(tile_dirs, os.path.join(out_dir, "mosaic"), pattern)
    print(f"{len(mosaics)} mosaic(s) (re)built in {os.path.join(out_dir, 'mosaic')}")
    return mask_and_scale_dir(os.path.join(out_dir, "mosaic"), mask_path, os.path.join(out_dir, "masked"),
                              pattern=pattern, threads=threads)


def difference(a_path, b_path, out_path, scale=VI_SCALE, block=4096):
    """a - b of two masked layers on the same grid (mask_and_scale_dir()
    outputs), NaN where either is nodata, written block by block as int16
    with band scale 1/scale (differences beyond +-32767/scale are clipped).

    :return: out_path.
    """
    with rasterio.open(a_path) as a, rasterio.open(b_path) as b:
        if (a.shape, a.transform, a.crs) != (b.shape, b.transform, b.crs):
            raise ValueError(f"{a_path} and {b_path} are not on the same grid")
        profile = a.profile
        profile.update(dtype="int16", nodata=-32768, compress="LZW", predictor=2, tiled=True,
                       blockxsize=256, blockysize=256)
        tmp = str(out_path) + ".tmp"
        n_clipped = 0
        with rasterio.open(tmp, "w", **profile) as dst:
            for win in chunk_windows(a.width, a.height, block):
                d = (read_float(a, 1, win) * a.scales[0] + a.offsets[0]
                     - (read_float(b, 1, win) * b.scales[0] + b.offsets[0]))
                v = np.round(d * scale)
                n_clipped += int(np.sum(np.abs(v) > 32767))
                out = np.where(np.isnan(d), -32768, np.clip(v, -32767, 32767)).astype("int16")
                dst.write(out, 1, window=win)
            dst.scales, dst.offsets = (1.0 / scale,), (0.0,)
    os.replace(tmp, out_path)
    print(f"Saved {out_path}" + (f" ({n_clipped} pixel(s) clipped)" if n_clipped else ""))
    return out_path


if __name__ == "__main__":
    # EVI on 5 June 2018 and 2026 (raw interpolated EVI, same units in both
    # years), their difference, and the CCI minimum of the near-real-time
    # run as of 5 July 2026 (first date in July)
    level4_2025 = ROOT / "output" / "DiscoCH_rbf_10_20_30" / "level4_rbf" / "2025"  # TSI of the 2026 run
    nrt_2026 = ROOT / "output" / "DiscoCH_rbf_10_20_30" / "2026"
    level4_2026 = ROOT / "output" / "DiscoCH_rbf_10_20_30" / "level4_rbf" / "2026"  # TSI of the 2026 run
    out_dir = ROOT / "output" / "DiscoCH_rbf_10_20_30" / "masking"

    vi_date_tiles(level4_2025, "EVI", "2025-06-05", out_dir)
    vi_date_tiles(level4_2026, "EVI", "2026-06-05", out_dir)
    # snapshot_tiles(nrt_2026 / "minmax_by_date", "CCI", "min", "2026-07-05", out_dir)
    mosaic_and_mask(out_dir)

    masked = out_dir / "masked"
    difference(masked / "EVI_2026-06-05.tif", masked / "EVI_2025-06-05.tif",
               masked / "EVI_diff_2026-06-05_minus_2025-06-05.tif")
