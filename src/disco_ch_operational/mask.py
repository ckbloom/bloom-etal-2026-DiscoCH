"""
mask.py

Masks and reprojects a folder of rasters (normally the Disco_Proba mosaics)
onto a binary mask's grid in the Swiss coordinate system (EPSG:2056,
CH1903+ / LV95), and writes each one as a compact scaled-integer GeoTIFF.

Per raster:
  1. Reproject onto the mask's grid (same CRS, resolution and extent) with
     bilinear resampling. The mask itself is read once, and reprojected to
     the target CRS with nearest-neighbour if it isn't already in it.
  2. Set every pixel where the mask isn't exactly 1 to nodata.
  3. Write as int16 scaled by the largest of 10000/1000/100/10/1 that fits
     (utils.write_scaled_int(): nodata -32768, GDAL band scale 1/scale).

mask_path=None skips the masking: each raster is only reprojected to the
target CRS (at its own resolution) and scaled.

Replaces mask_and_scale_rasters.mask_and_scale_directory() (in src/dep).
Differences:
  - inputs that are themselves scaled integers (the Disco_Proba mosaics:
    int16 with band scale 0.0001) are read as their true values. The old
    code read the raw stored integers, so its outputs held 0-10000 with a
    band scale of 1.0; now they hold the same integers with scale 0.0001,
    i.e. GIS software shows probabilities 0-1;
  - an output is redone when its input (or the mask) is newer than it, not
    only when missing, so rebuilt mosaics are picked up;
  - a raster with no valid pixel inside the mask is reported and skipped
    instead of stopping the whole folder;
  - plain rasterio instead of rioxarray/xarray.

Usage (notebook):

    from src.disco_ch_operational.mask import mask_and_scale_dir

    mask_and_scale_dir(
        input_dir=os.path.join(discoCH_output, 'mosaic'),
        mask_path=forest_mask, output_dir=masked_output,
    )
"""

import os
import glob
import time

import numpy as np
import rasterio
from rasterio.warp import reproject, calculate_default_transform, Resampling

from src.disco_ch_operational.utils import write_scaled_int, format_duration

TARGET_CRS = "EPSG:2056"


# ---------------------------------------------------------------------
# Grids
# ---------------------------------------------------------------------

def target_grid(path, target_crs=TARGET_CRS):
    """(transform, crs, (height, width)) of `path` in target_crs: its own
    grid if already in target_crs, otherwise GDAL's default grid for it
    there (same as rioxarray's .rio.reproject() picks)."""
    with rasterio.open(path) as src:
        if src.crs is None:
            raise ValueError(f"{path} has no CRS defined")
        dst_crs = rasterio.crs.CRS.from_user_input(target_crs)
        if src.crs == dst_crs:
            return src.transform, src.crs, (src.height, src.width)
        transform, width, height = calculate_default_transform(
            src.crs, dst_crs, src.width, src.height, *src.bounds)
        return transform, dst_crs, (height, width)


def warp_bands(path, transform, crs, shape, resampling=Resampling.bilinear, threads=1):
    """Every band of `path` reprojected onto a target grid as float32
    (bands, height, width), with the band scale/offset applied and
    nodata/outside-extent as NaN."""
    with rasterio.open(path) as src:
        out = np.full((src.count,) + tuple(shape), np.nan, dtype="float32")
        for i in range(src.count):
            reproject(rasterio.band(src, i + 1), out[i], src_nodata=src.nodata,
                      dst_transform=transform, dst_crs=crs, dst_nodata=np.nan,
                      resampling=resampling, num_threads=threads)
            scale, offset = src.scales[i], src.offsets[i]
            if scale != 1 or offset != 0:
                out[i] *= scale
                out[i] += offset
    return out


def load_mask(mask_path, target_crs=TARGET_CRS, threads=1):
    """(keep, transform, crs): boolean array, True where the mask is exactly
    1, on the mask's grid in target_crs (nearest-neighbour, so no fractional
    mask values are invented at edges)."""
    transform, crs, shape = target_grid(mask_path, target_crs)
    mask = warp_bands(mask_path, transform, crs, shape, Resampling.nearest, threads)[0]
    return mask == 1, transform, crs


# ---------------------------------------------------------------------
# One raster
# ---------------------------------------------------------------------

def mask_and_scale_file(in_path, out_path, keep=None, transform=None, crs=None,
                        target_crs=TARGET_CRS, resampling=Resampling.bilinear,
                        dtype="int16", threads=1):
    """Reprojects in_path onto the mask grid (keep/transform/crs from
    load_mask()) and blanks pixels outside it -- or, with keep=None, only
    reprojects to target_crs -- then writes out_path as a scaled integer
    GeoTIFF. Single-band input only (the DiscoCH outputs all are).

    :return: the scale factor used, or None if nothing was written (no
        valid pixels).
    """
    if keep is None:
        transform, crs, shape = target_grid(in_path, target_crs)
    else:
        shape = keep.shape
    data = warp_bands(in_path, transform, crs, shape, resampling, threads)
    if data.shape[0] != 1:
        raise ValueError(f"{in_path} has {data.shape[0]} bands; only single-band rasters are supported")
    data = data[0]
    if keep is not None:
        data[~keep] = np.nan
    return write_scaled_int(data, out_path, transform, crs, dtype=dtype)


# ---------------------------------------------------------------------
# Folder driver
# ---------------------------------------------------------------------

def mask_and_scale_dir(input_dir, mask_path, output_dir, pattern="*.tif", dtype="int16",
                       target_crs=TARGET_CRS, resampling=Resampling.bilinear,
                       overwrite=False, threads=4):
    """Masks and scales every file matching `pattern` in input_dir into
    output_dir under the same name.

    :param mask_path: binary mask raster (1 = keep, anything else or nodata
        = discard), any CRS/extent; the outputs take its grid. None = no
        masking, only reproject to target_crs and scale.
    :param pattern: glob for the input files (default all .tif).
    :param dtype: integer dtype for the output ("int16", or "int32" if a
        raster's values need more range).
    :param resampling: for the data rasters (the mask is always nearest).
        Use Resampling.nearest for categorical data.
    :param overwrite: redo every file. Otherwise a file is only (re)done if
        its output is missing or older than the input or the mask.
    :param threads: GDAL warp threads.
    :return: {filename: status}, status "written (scale N)", "skipped",
        "no valid pixels" or "failed: <error>".
    """
    t0 = time.time()
    os.makedirs(output_dir, exist_ok=True)
    files = sorted(glob.glob(os.path.join(input_dir, pattern)))
    if not files:
        print(f"No files matching {pattern!r} in {input_dir}")
        return {}

    keep = transform = crs = None
    mask_time = 0
    if mask_path is not None:
        print(f"Loading mask {mask_path} on its grid in {target_crs} ...")
        keep, transform, crs = load_mask(mask_path, target_crs, threads)
        mask_time = os.path.getmtime(mask_path)
        print(f"  {keep.shape[1]} x {keep.shape[0]} px, {keep.mean():.1%} kept")
    else:
        print(f"No mask -- reprojecting to {target_crs} without masking")

    status = {}
    for in_path in files:
        name = os.path.basename(in_path)
        out_path = os.path.join(output_dir, name)
        if (not overwrite and os.path.exists(out_path)
                and os.path.getmtime(out_path) >= max(os.path.getmtime(in_path), mask_time)):
            status[name] = "skipped"
            continue
        try:
            scale = mask_and_scale_file(in_path, out_path, keep, transform, crs, target_crs,
                                        resampling, dtype, threads)
            status[name] = "no valid pixels" if scale is None else f"written (scale {scale})"
        except Exception as e:
            status[name] = f"failed: {type(e).__name__}: {e}"
        print(f"  {name}: {status[name]}")

    counts = {}
    for s in status.values():
        k = s.split(" (")[0].split(":")[0]
        counts[k] = counts.get(k, 0) + 1
    print(f"Done in {format_duration(time.time() - t0)}: "
          + ", ".join(f"{n} {k}" for k, n in sorted(counts.items())))
    return status
