"""
plot_date_grid.py

Grid of maps of the discoloration probability of every date of a season on
the swisstopo grey national map (WMS ch.swisstopo.landeskarte-grau-10): one
column per month, one row per date within the month (the 5-day dates fall
on different days in each month, so every panel is labelled with its date).

The map extent comes from a shapefile (or GeoDataFrame / bounds), only to
place the maps -- nothing is clipped to it; its outline can be drawn on top (outline=True).
The probabilities are read from the masked mosaics (EPSG:2056, int16 with
band scale 0.0001) of the near-real-time or the retrospective (all-data) run.

Usage (notebook or `python -m src.disco_ch_operational.plot_date_grid`):

    from src.disco_ch_operational.plot_date_grid import plot_date_grid

    plot_date_grid(dataset="retrospective",             # or "nrt", or mosaic_dir=...
                   extent=r"...\\my_area.shp",          # extent only, no clipping
                   alpha=0.6,                           # probability transparency
                   min_proba=0.1,                       # lower values transparent (None = all)
                   output=r"...\\date_grid.png")
"""

import calendar
import io
import os
import re
from pathlib import Path

import numpy as np

OUTPUT_ROOT = Path(__file__).resolve().parents[2] / "output" / "DiscoCH_rbf_10_20_30"
DATASETS = {"nrt": "{year}/masked", "retrospective": "{year}_full_data/masked"}
WMS_URL = "https://wms.geo.admin.ch/"
WMS_LAYER = "ch.swisstopo.landeskarte-grau-10"
CRS = "EPSG:2056"
# probability colours: dark green (0) -> yellow (1), stretched linearly
DISCO_COLORS = ["#0A2F1F", "#228B22", "#5EAC2A", "#9ACD32", "#FFFF00"]


def _disco_cmap():
    from matplotlib.colors import LinearSegmentedColormap
    return LinearSegmentedColormap.from_list("custom_stretch", DISCO_COLORS)


DISCO_CMAP = _disco_cmap()


def get_extent(extent, pad=0.05):
    """(xmin, ymin, xmax, ymax) in EPSG:2056 of a shapefile path, GeoDataFrame
    or bounds tuple, enlarged by `pad` (share of the larger side) on each side."""
    if isinstance(extent, (str, os.PathLike)) or hasattr(extent, "total_bounds"):
        import geopandas as gpd
        gdf = gpd.read_file(extent) if isinstance(extent, (str, os.PathLike)) else extent
        if gdf.crs is not None:
            gdf = gdf.to_crs(CRS)
        xmin, ymin, xmax, ymax = gdf.total_bounds
    else:
        xmin, ymin, xmax, ymax = extent
    d = pad * max(xmax - xmin, ymax - ymin)
    return xmin - d, ymin - d, xmax + d, ymax + d


def fetch_basemap(bounds, width=1600, layer=WMS_LAYER, url=WMS_URL, timeout=120):
    """RGB(A) array of the WMS layer for bounds (EPSG:2056), `width` pixels wide."""
    import requests
    from PIL import Image

    xmin, ymin, xmax, ymax = bounds
    height = max(1, round(width * (ymax - ymin) / (xmax - xmin)))
    params = dict(SERVICE="WMS", VERSION="1.3.0", REQUEST="GetMap", LAYERS=layer, STYLES="",
                  CRS=CRS, BBOX=f"{xmin},{ymin},{xmax},{ymax}",  # EPSG:2056 axis order is E, N
                  WIDTH=width, HEIGHT=height, FORMAT="image/png")
    r = requests.get(url, params=params, timeout=timeout)
    r.raise_for_status()
    if "image" not in r.headers.get("content-type", ""):
        raise RuntimeError(f"WMS returned {r.headers.get('content-type')}: {r.text[:300]}")
    return np.asarray(Image.open(io.BytesIO(r.content)).convert("RGBA"))


def list_dates(mosaic_dir, year=None, months=None, pattern=r"Disco_Proba_(\d{4}-\d{2}-\d{2})\.tif$"):
    """{date string: path} of the mosaics in mosaic_dir, optionally of one year
    and some months."""
    out = {}
    for f in sorted(os.listdir(mosaic_dir)):
        m = re.search(pattern, f)
        if not m:
            continue
        d = m.group(1)
        if year is not None and int(d[:4]) != int(year):
            continue
        if months is not None and int(d[5:7]) not in months:
            continue
        out[d] = os.path.join(mosaic_dir, f)
    return out


def read_proba(path, bounds, max_pixels=1000):
    """Probability (0-1, NaN = nodata) of the raster within bounds, averaged
    down to at most max_pixels on the longer side, and the extent actually
    read (for imshow)."""
    import rasterio
    from rasterio.enums import Resampling
    from rasterio.windows import from_bounds

    with rasterio.open(path) as src:
        win = from_bounds(*bounds, transform=src.transform)
        f = max(1.0, max(win.width, win.height) / max_pixels)
        shape = (max(1, round(win.height / f)), max(1, round(win.width / f)))
        a = src.read(1, window=win, out_shape=shape, masked=True, boundless=True,
                     resampling=Resampling.average if f > 1 else Resampling.nearest)
        p = a.astype("float32").filled(np.nan) * src.scales[0] + src.offsets[0]
        l, b, r, t = rasterio.windows.bounds(win, src.transform)
    return p, (l, r, b, t)


def plot_date_grid(dataset="nrt", extent=None, year=2026, months=(6, 7, 8, 9), alpha=0.6,
                   min_proba=None, cmap=None, pad=0.05, mosaic_dir=None, output_root=OUTPUT_ROOT,
                   outline=False, basemap_width=1600, max_pixels=1000, panel_size=3.2,
                   title=None, font_scale=1.0, output=None, show=True):
    """Grid of probability maps: one column per month, one row per date in it.

    :param dataset: "nrt" (near real time, <year>/masked) or "retrospective"
        (all data, <year>_full_data/masked) under output_root; ignored if
        mosaic_dir is given.
    :param extent: shapefile path, GeoDataFrame or (xmin, ymin, xmax, ymax)
        in EPSG:2056 -- only places the maps, nothing is clipped. None = the
        whole mosaic.
    :param alpha: opacity of the probability layer (0 = invisible, 1 = opaque).
    :param min_proba: probabilities below this are not drawn (basemap shows
        through); None = draw every valid pixel.
    :param cmap: matplotlib colormap (or name) of the probability (0-1); None =
        DISCO_CMAP, dark green -> yellow.
    :param pad: margin around the extent, share of its larger side.
    :param outline: draw the extent shapefile's outline (if extent is one).
    :param basemap_width: WMS image width in pixels (fetched once for all panels).
    :param max_pixels: probability pixels on the longer side per panel
        (averaged down; 10 m pixels over a large extent would be slow).
    :param font_scale: size of all text relative to the default (month
        headers 22 pt, date labels 18 pt, title 26 pt, colour bar 16-18 pt).
    :param output: .png to save; None = don't save.
    :return: the matplotlib figure.
    """
    import matplotlib.pyplot as plt

    mosaic_dir = mosaic_dir or os.path.join(output_root, DATASETS[dataset].format(year=year))
    dates = list_dates(mosaic_dir, year, months)
    if not dates:
        raise FileNotFoundError(f"no Disco_Proba_<date>.tif of {year}, months {months} in {mosaic_dir}")
    if extent is None:
        import rasterio
        with rasterio.open(next(iter(dates.values()))) as src:
            extent = tuple(src.bounds)
    bounds = get_extent(extent, pad)
    shapes = None
    if outline and (isinstance(extent, (str, os.PathLike)) or hasattr(extent, "total_bounds")):
        import geopandas as gpd
        shapes = gpd.read_file(extent) if isinstance(extent, (str, os.PathLike)) else extent
        shapes = shapes.to_crs(CRS) if shapes.crs is not None else shapes
    print(f"{len(dates)} dates from {mosaic_dir}; fetching the basemap ...")
    basemap = fetch_basemap(bounds, basemap_width)

    by_month = {}
    for d in dates:
        by_month.setdefault(int(d[5:7]), []).append(d)
    cols = [m for m in months if m in by_month]
    nrows = max(len(v) for v in by_month.values())
    xmin, ymin, xmax, ymax = bounds
    aspect = (ymax - ymin) / (xmax - xmin)

    fs = font_scale
    fig, axes = plt.subplots(nrows, len(cols), squeeze=False,
                             figsize=(panel_size * len(cols) + 1.2 + 0.8 * fs,
                                      panel_size * aspect * nrows + 1.6 * fs))
    colormap = plt.get_cmap(cmap if cmap is not None else DISCO_CMAP).copy()
    colormap.set_bad(alpha=0)  # NaN (nodata, below min_proba) transparent
    im = None
    for j, m in enumerate(cols):
        axes[0, j].set_title(calendar.month_name[m], fontsize=22 * fs, fontweight="bold", pad=14 * fs)
        for i in range(nrows):
            ax = axes[i, j]
            ax.set_xticks([])
            ax.set_yticks([])
            if i >= len(by_month[m]):
                ax.set_visible(False)
                continue
            d = by_month[m][i]
            p, ext = read_proba(dates[d], bounds, max_pixels)
            if min_proba is not None:
                p = np.where(p >= min_proba, p, np.nan)
            ax.imshow(basemap, extent=(xmin, xmax, ymin, ymax), interpolation="bilinear")
            im = ax.imshow(p, extent=ext, cmap=colormap, vmin=0, vmax=1, alpha=alpha, interpolation="nearest")
            if shapes is not None:
                shapes.boundary.plot(ax=ax, color="black", linewidth=0.6)
            ax.set_xlim(xmin, xmax)
            ax.set_ylim(ymin, ymax)
            ax.text(0.02, 0.98, f"{int(d[8:10])} {calendar.month_abbr[m]}", transform=ax.transAxes,
                    ha="left", va="top", fontsize=18 * fs,
                    bbox=dict(facecolor="white", edgecolor="none", alpha=0.8, pad=2))

    label = {"nrt": "near real time", "retrospective": "retrospective (all data)"}.get(dataset, dataset)
    top = 1 - 1.4 * fs / fig.get_figheight()  # room for the title above the month headers
    fig.suptitle(title or f"Discoloration probability {year}, {label}", fontsize=26 * fs,
                 y=1 - 0.2 / fig.get_figheight(), va="top")
    fig.subplots_adjust(left=0.01, right=0.88, bottom=0.01, top=top, wspace=0.03, hspace=0.03)
    cax = fig.add_axes([0.9, 0.2, 0.02, 0.6])
    # the colours as they look at `alpha` over white, drawn opaque (a transparent colorbar shows seams)
    from matplotlib.cm import ScalarMappable
    from matplotlib.colors import ListedColormap, Normalize
    rgb = colormap(np.linspace(0, 1, 256))[:, :3]
    shown = ListedColormap(alpha * rgb + (1 - alpha))
    cb = fig.colorbar(ScalarMappable(norm=Normalize(0, 1), cmap=shown), cax=cax)
    cb.set_label("Probability" + (f" (shown >= {min_proba:g})" if min_proba is not None else ""),
                 fontsize=18 * fs)
    cb.ax.tick_params(labelsize=16 * fs)
    if output:
        Path(output).parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(output, dpi=200, bbox_inches="tight")
        print(f"Saved {output}")
    if show:
        plt.show()
    return fig


if __name__ == "__main__":
    # Example: canton Jura (ktnr 26) of the Bezirke layer as extent, retrospective run
    import geopandas as gpd

    districts = gpd.read_file(r"B:\bloomc\DiscoCH_2026_08_03\Validation\Shp_Forstkreise_2023_Bezirke"
                              r"\fgeolib_geolib_g1b20_LV95.shp")
    for dataset in ("nrt", "retrospective"):
        plot_date_grid(dataset=dataset,
                       extent=districts[districts["ktnr"] == 26],  # or a shapefile path
                       alpha=0.6, min_proba=0.1,
                       output=OUTPUT_ROOT / "2026" / "date_grids" / f"date_grid_jura_{dataset}.png",
                       show=False)
