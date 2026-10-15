"""
hexmap.py

Maps of discoloration aggregated to hexagons: rows = years (e.g. 2018,
2026), columns = months (default June-September, each the last mosaic
date of that month), over a shaded-relief basemap. Also exports each
year's hexagons with one proportion field per mosaic date.

  - One flat-top hexagon grid of hex_area_km2 cells, built in EPSG:2056
    over the boundary (e.g. the districts, i.e. Switzerland) or else over
    the union of the datasets' extents, and clipped to the boundary. The
    same grid is used for every year, so the maps and the exported layers
    line up hexagon by hexagon.
  - A hexagon's proportion on a date is (valid pixels >= threshold) /
    (valid pixels) inside it, counted with disco_statistics.discoloration_stats()
    -- the same nodata, threshold and grid handling as the national and
    district statistics, read in chunks. NaN where it has no valid pixel;
    hexagons without data on any date are dropped.
  - Thresholds are on the mosaics' stored scale (4000 = 40 %).

Replaces plot_hex_discoloration_grid.py. Differences:
  - one hexagon grid for all years (it built one per dataset);
  - any number of years (rows); larger text (font_scale, shared with
    compare.py);
  - counting is chunked and multithreaded instead of whole rasters, and
    mosaics with a smaller extent on the same pixel grid are counted
    instead of skipped;
  - one colour scale for every panel, shown as %, topped at the 99th
    percentile by default (it was the maximum); empty months are
    labelled; the hexagons are written as GeoPackage by default (no
    10-character field limit, but dYYYYMMDD names still fit a shapefile).

Usage (notebook):

    from src.disco_ch_operational.hexmap import plot_hex_maps

    hexes = plot_hex_maps({
        "2018": dict(mosaic_dir=mosaics_2018, pattern="Proba_mosaic_*.tif", year=2018),
        "2026": dict(mosaic_dir=masked_output, year=2026),
    }, threshold=4000, hex_area_km2=3, boundary=districts, output_dir=stats_output)
"""

import os
import time
from datetime import datetime

import numpy as np
import pandas as pd
import rasterio
import shapely
from rasterio.warp import transform_bounds

from src.disco_ch_operational.compare import rc_params
from src.disco_ch_operational.disco_statistics import TOTAL, discoloration_stats, list_mosaics
from src.disco_ch_operational.utils import format_duration, write_run_params

HEX_CRS = "EPSG:2056"


# ---------------------------------------------------------------------
# Hexagon grid
# ---------------------------------------------------------------------

def hex_grid(bounds, hex_area_km2=3, crs=HEX_CRS, boundary=None):
    """GeoDataFrame of flat-top hexagons ("hex_id" from 1) of hex_area_km2
    covering bounds (minx, miny, maxx, maxy in crs), clipped to `boundary`
    (path or GeoDataFrame, any CRS) if given."""
    import geopandas as gpd

    r = np.sqrt(hex_area_km2 * 1e6 / (1.5 * np.sqrt(3)))     # circumradius
    dx, dy = 1.5 * r, np.sqrt(3) * r
    minx, miny, maxx, maxy = bounds
    cols = np.arange(int(np.ceil((maxx - minx) / dx)) + 3)
    rows = np.arange(int(np.ceil((maxy - miny) / dy)) + 3)
    cc, rr = np.meshgrid(cols, rows, indexing="ij")
    cx = minx - dx + cc * dx
    cy = miny - dy + rr * dy + (cc % 2) * dy / 2                # odd columns shifted half a row
    angles = np.radians(np.arange(0, 360, 60))
    ring = np.stack([cx.ravel()[:, None] + r * np.cos(angles),
                     cy.ravel()[:, None] + r * np.sin(angles)], axis=-1)
    hexes = gpd.GeoDataFrame({"hex_id": np.arange(1, len(ring) + 1)},
                             geometry=shapely.polygons(ring), crs=crs)

    if boundary is not None:
        b = gpd.read_file(boundary) if isinstance(boundary, (str, os.PathLike)) else boundary
        shape = b.to_crs(crs).union_all()
        hexes = hexes[hexes.intersects(shape)]
        hexes = hexes.set_geometry(hexes.intersection(shape))
        hexes = hexes[~hexes.is_empty].reset_index(drop=True)
    return hexes


def _dataset_bounds(datasets, crs=HEX_CRS):
    """Union of the first mosaic's extent of every dataset, in crs."""
    boxes = []
    for spec in datasets.values():
        _, path = list_mosaics(spec["mosaic_dir"], spec.get("pattern", "Disco_Proba_*.tif"),
                               spec.get("year"))[0]
        with rasterio.open(path) as src:
            boxes.append(transform_bounds(src.crs, crs, *src.bounds))
    b = np.array(boxes)
    return b[:, 0].min(), b[:, 1].min(), b[:, 2].max(), b[:, 3].max()


# ---------------------------------------------------------------------
# Proportions per hexagon
# ---------------------------------------------------------------------

def date_field(d):
    """'dYYYYMMDD' (fits a shapefile's 10-character field names)."""
    return f"d{d:%Y%m%d}"


def hex_discoloration(mosaic_dir, hexes, threshold=4000, pattern="Disco_Proba_*.tif", year=None,
                      nodata=None, output=None, **stats_kw):
    """hexes with one proportion column (0-1, NaN = no valid pixel) per
    mosaic date, named date_field(date); hexagons without data on any date
    are dropped. Saved to `output` (.gpkg, .shp, ...) if given.

    :param stats_kw: chunk_size, max_workers for discoloration_stats().
    """
    stats = discoloration_stats(mosaic_dir, threshold, zones=hexes, zone_field="hex_id",
                                pattern=pattern, year=year, nodata=nodata, **stats_kw)
    stats = stats[stats.zone != TOTAL]
    wide = stats.pivot(index="zone", columns="date", values="proportion")
    wide.index = wide.index.astype(int)
    wide.columns = [date_field(d) for d in wide.columns]
    out = hexes.merge(wide, left_on="hex_id", right_index=True, how="inner")
    out = out[out[wide.columns].notna().any(axis=1)].reset_index(drop=True)
    print(f"  {len(out):,} of {len(hexes):,} hexagons have data")
    if output:
        os.makedirs(os.path.dirname(output) or ".", exist_ok=True)
        out.to_file(output)
        print(f"Saved {len(out):,} hexagons x {len(wide.columns)} dates to {output}")
    return out


def month_end_fields(hexes, months):
    """{month: date field of the latest date in that month} of a
    hex_discoloration() layer; months without a date are left out."""
    dates = [datetime.strptime(c[1:], "%Y%m%d") for c in hexes.columns
             if len(c) == 9 and c[0] == "d" and c[1:].isdigit()]
    out = {}
    for m in months:
        in_month = [d for d in dates if d.month == m]
        if in_month:
            out[m] = date_field(max(in_month))
    return out


# ---------------------------------------------------------------------
# Figure
# ---------------------------------------------------------------------

def _add_basemap(ax, crs, source):
    try:
        import contextily as cx
        cx.add_basemap(ax, crs=crs, source=source or cx.providers.Esri.WorldShadedRelief,
                       attribution=False, zorder=0)
        return True
    except Exception as e:      # no contextily / no internet: plot without
        print(f"  No basemap ({type(e).__name__}: {e})")
        return False


def plot_hex_maps(datasets, threshold=4000, hex_area_km2=3, boundary=None, months=(6, 7, 8, 9),
                  cmap="YlOrRd", vmax=None, basemap=True, basemap_source=None, font_scale=1.4,
                  output_dir=None, output_name=None, export_hexes=True, show=True, **stats_kw):
    """Grid of hexagon maps: one row per dataset, one column per month
    (that month's last mosaic date).

    :param datasets: {label: dict(mosaic_dir=..., pattern=..., year=...,
        nodata=...)}, e.g. {"2018": ..., "2026": ...}; rows in this order.
    :param threshold: stored value at/above which a pixel is discolored.
    :param hex_area_km2: hexagon size.
    :param boundary: polygons to build the grid over and clip it to (path
        or GeoDataFrame, e.g. the districts); None = the datasets' extent.
    :param months: month numbers, one column each.
    :param vmax: top of the colour scale (proportion); None = the 99th
        percentile of the values shown (higher values get the top colour).
    :param basemap: shaded relief under the hexagons (needs contextily and
        internet; skipped with a message otherwise). basemap_source: a
        contextily provider instead of Esri.WorldShadedRelief.
    :param font_scale: text size relative to 12 pt.
    :param output_dir: saves <output_name>.png (default
        hex_discoloration_<labels>_t<threshold>.png), the run parameters and,
        with export_hexes, hex_<label>_t<threshold>.gpkg per dataset.
    :param stats_kw: chunk_size, max_workers for the counting.
    :return: {label: hex_discoloration() GeoDataFrame}.
    """
    import matplotlib.pyplot as plt
    import matplotlib.ticker as mtick

    t0 = time.time()
    labels = [str(k) for k in datasets]
    boundary_path = boundary if isinstance(boundary, (str, os.PathLike)) else None
    if boundary is None:
        bounds = _dataset_bounds(datasets)
    else:
        import geopandas as gpd
        if isinstance(boundary, (str, os.PathLike)):
            boundary = gpd.read_file(boundary)
        bounds = boundary.to_crs(HEX_CRS).total_bounds
    grid = hex_grid(bounds, hex_area_km2, HEX_CRS, boundary)
    print(f"{len(grid):,} hexagons of {hex_area_km2:g} km2")

    hexes, fields = {}, {}
    for label, spec in zip(labels, datasets.values()):
        print(f"\n{label}:")
        out = (os.path.join(output_dir, f"hex_{label}_t{threshold:g}.gpkg")
               if output_dir and export_hexes else None)
        hexes[label] = hex_discoloration(threshold=threshold, hexes=grid, output=out, **spec, **stats_kw)
        fields[label] = month_end_fields(hexes[label], months)
        missing = [m for m in months if m not in fields[label]]
        if missing:
            print(f"  No {label} mosaic in month(s) {missing}")

    shown = np.concatenate([hexes[lab][f].dropna().to_numpy() for lab in labels
                            for f in fields[lab].values()] or [np.zeros(1)])
    if vmax is None:    # 99th percentile: a few tiny border slivers reach 100 %
        vmax = float(np.percentile(shown, 99)) or 1.0
    norm = plt.Normalize(vmin=0, vmax=vmax)

    with plt.rc_context(rc_params(font_scale)):
        fig, axes = plt.subplots(len(labels), len(months), squeeze=False,
                                 figsize=(4.6 * len(months), 3.4 * len(labels) + 1.2),
                                 layout="constrained")
        minx, miny, maxx, maxy = grid.total_bounds
        pad = 0.02 * (maxx - minx)
        for i, lab in enumerate(labels):
            for j, m in enumerate(months):
                ax = axes[i, j]
                ax.set_xlim(minx - pad, maxx + pad)
                ax.set_ylim(miny - pad, maxy + pad)
                ax.set_aspect("equal")
                ax.set_xticks([])
                ax.set_yticks([])
                field = fields[lab].get(m)
                month = datetime(2000, m, 1).strftime("%B")
                if field is None:
                    ax.set_title(f"{month}: no data")
                else:
                    d = hexes[lab][hexes[lab][field].notna()]
                    d.plot(column=field, ax=ax, cmap=cmap, norm=norm, edgecolor="none", zorder=2)
                    ax.set_title(datetime.strptime(field[1:], "%Y%m%d").strftime("%d %B"))
                if basemap:
                    basemap = _add_basemap(ax, HEX_CRS, basemap_source)
                if j == 0:
                    ax.set_ylabel(lab, fontsize=plt.rcParams["axes.titlesize"], fontweight="bold")

        sm = plt.cm.ScalarMappable(cmap=cmap, norm=norm)
        cbar = fig.colorbar(sm, ax=axes, shrink=0.8, pad=0.01,
                            extend="max" if shown.max() > vmax else "neither")
        cbar.ax.yaxis.set_major_formatter(mtick.PercentFormatter(1.0, decimals=0))
        cbar.set_label(f"Beech forest discolored (threshold ≥ {threshold / 10000:g})")
        fig.suptitle(f"Discoloration per {hex_area_km2:g} km² hexagon, end of month")

        if output_dir:
            os.makedirs(output_dir, exist_ok=True)
            name = output_name or f"hex_discoloration_{'_vs_'.join(labels)}_t{threshold:g}"
            png = os.path.join(output_dir, name + ".png")
            fig.savefig(png, dpi=200, bbox_inches="tight")
            print(f"Saved {png}")
            write_run_params(output_dir, "hexmap.plot_hex_maps",
                             dict(datasets=datasets, threshold=threshold, hex_area_km2=hex_area_km2,
                                  boundary=boundary_path, months=months, vmax=vmax),
                             {lab: {m: f for m, f in fields[lab].items()} for lab in labels})
        if show:
            plt.show()
        plt.close(fig)
    print(f"Done in {format_duration(time.time() - t0)}")
    return hexes
