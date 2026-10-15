"""
compare.py

Plots the seasonal course of discoloration of several years on one
month/day axis (e.g. 2018 vs 2026): one line per year, or per year and
region when regions of a zone layer (e.g. forest districts) are chosen.

Each dataset is either a folder of mosaics, counted with
disco_statistics.discoloration_stats(), or a table that function (or
disco_statistics.summarize_discoloration()) already returned -- so a year that
was just summarized isn't counted again. Every date is moved onto the
common year 2000 (a leap year, so 29 Feb is safe) to line the seasons up.

  - layout="overlay": everything on one axis; the colour shows the region
    and the line style/marker the year (with one region, colour = year).
  - layout="panels": one panel per region, with the years in the same
    colours in every panel -- easier to read than an overlay of many
    districts.

Replaces plot_discoloration_comparison() of plot_discoloration_timeseries.py.
Differences:
  - any number of years, and regions from a zone layer;
  - larger text (font_scale);
  - the threshold must be the same for every dataset counted here; tables
    passed in keep the threshold they were counted with;
  - the per-date hectare CSV is the long table of disco_statistics.py, written
    only with output_dir.

Usage (notebook):

    from src.disco_ch_operational.compare import plot_year_comparison

    plot_year_comparison({
        "2018": dict(mosaic_dir=mosaics_2018, pattern="Proba_mosaic_*.tif", year=2018),
        "2026": stats_2026,             # summarize_discoloration() result
    }, threshold=4000, output_dir=stats_output)

    # Per district (regions must be in every dataset, so count the 2018
    # mosaics with the same zones):
    plot_year_comparison({...}, zones=districts, zone_field="bzname",
                         regions=["Bezirk Aarau", "Bezirk Baden"], layout="panels")
"""

import os

import numpy as np
import pandas as pd

from src.disco_ch_operational.disco_statistics import TOTAL, discoloration_stats

LINESTYLES = ["-", "--", "-.", ":"]
MARKERS = ["o", "s", "^", "D", "v", "P"]


def rc_params(font_scale=1.4):
    """matplotlib rcParams for larger text (12 pt x font_scale) on white."""
    base = 12 * font_scale
    return {
        "font.size": base, "axes.titlesize": base * 1.35, "axes.labelsize": base * 1.15,
        "xtick.labelsize": base, "ytick.labelsize": base, "legend.fontsize": base,
        "legend.title_fontsize": base, "figure.titlesize": base * 1.5,
        "figure.facecolor": "white", "axes.facecolor": "white", "savefig.facecolor": "white",
    }


def collect_stats(datasets, threshold=4000, zones=None, zone_field="bzname", **stats_kw):
    """One long table (disco_statistics.discoloration_stats() columns plus
    "dataset") from {label: DataFrame | dict(mosaic_dir=..., pattern=...,
    year=..., nodata=...)}; dict entries are counted with `threshold` and
    `zones`, plus stats_kw (chunk_size, max_workers)."""
    tables = []
    for label, src in datasets.items():
        if isinstance(src, pd.DataFrame):
            df = src.copy()
        else:
            src = dict(src) if isinstance(src, dict) else {"mosaic_dir": src}
            df = discoloration_stats(threshold=threshold, zones=zones, zone_field=zone_field,
                                     **src, **stats_kw)
        df["date"] = pd.to_datetime(df["date"])
        df.insert(0, "dataset", str(label))
        tables.append(df)
    return pd.concat(tables, ignore_index=True)


def plot_year_comparison(datasets, threshold=4000, zones=None, zone_field="bzname", regions=None,
                         layout="overlay", ncols=3, font_scale=1.4, value="proportion",
                         title=None, output_dir=None, output_name=None, show=True, **stats_kw):
    """Plots `value` of each dataset against month/day.

    :param datasets: {label: DataFrame or dict(mosaic_dir, pattern, year,
        nodata) or mosaic_dir} -- see collect_stats(). Labels are the
        legend entries, e.g. "2018".
    :param threshold: for the datasets counted here (stored value, 4000 =
        40 %).
    :param zones, zone_field: zone layer for the datasets counted here.
    :param regions: region names to plot (e.g. ["All", "Bezirk Aarau"]);
        None = those present in every dataset (just "All" unless every
        dataset was counted with the same zones).
    :param layout: "overlay" (one axis) or "panels" (one per region).
    :param ncols: panels per row.
    :param font_scale: text size relative to 12 pt.
    :param value: "proportion" (shown as %), "discolored_ha" or a
        statistics column.
    :param output_dir: saves <output_name>.png and the counts as
        <output_name>.csv there; output_name defaults to
        discoloration_comparison_<labels>.
    :return: the long table of every dataset.
    """
    import matplotlib.pyplot as plt
    import matplotlib.dates as mdates
    import matplotlib.ticker as mtick

    stats = collect_stats(datasets, threshold, zones, zone_field, **stats_kw)
    labels = [str(k) for k in datasets]
    if regions is None:
        common = set.intersection(*(set(stats.loc[stats.dataset == lab, "zone"]) for lab in labels))
        regions = [TOTAL] if TOTAL in common else sorted(common)
    else:
        regions = [regions] if isinstance(regions, str) else list(regions)
        missing = {(lab, r) for lab in labels for r in regions
                   if not ((stats.dataset == lab) & (stats.zone == r)).any()}
        if missing:
            print("  Not in the data (no line): " + ", ".join(f"{r} [{lab}]" for lab, r in sorted(missing)))
    if layout not in ("overlay", "panels"):
        raise ValueError(f"layout must be 'overlay' or 'panels', got {layout!r}")

    data = stats[stats.zone.isin(regions)].copy()
    data["season_date"] = data["date"].map(lambda d: d.replace(year=2000))
    is_prop = value == "proportion"
    ylabel = {"proportion": "Beech forest discolored",
              "discolored_ha": "Discolored beech forest (ha)"}.get(value, value)

    colors = plt.rcParams["axes.prop_cycle"].by_key()["color"]
    year_color = {lab: colors[i % len(colors)] for i, lab in enumerate(labels)}
    region_color = {r: colors[i % len(colors)] for i, r in enumerate(regions)}
    year_style = {lab: (LINESTYLES[i % len(LINESTYLES)], MARKERS[i % len(MARKERS)])
                  for i, lab in enumerate(labels)}

    with plt.rc_context(rc_params(font_scale)):
        if layout == "panels" and len(regions) > 1:
            ncols = min(ncols, len(regions))
            nrows = -(-len(regions) // ncols)
            fig, axes = plt.subplots(nrows, ncols, figsize=(6.5 * ncols, 4.8 * nrows),
                                     sharex=True, sharey=True, squeeze=False)
            panels = dict(zip(regions, axes.ravel()))
            for ax in axes.ravel()[len(regions):]:
                ax.set_visible(False)
        else:
            fig, ax = plt.subplots(figsize=(13, 7))
            panels = {r: ax for r in regions}

        for r, ax in panels.items():
            for lab in labels:
                d = data[(data.dataset == lab) & (data.zone == r)].sort_values("season_date")
                if d.empty:
                    continue
                ls, mk = year_style[lab]
                if layout == "panels" or len(regions) == 1:
                    kw = dict(color=year_color[lab], marker="o", linestyle="-", label=lab)
                else:
                    kw = dict(color=region_color[r], marker=mk, linestyle=ls, label=f"{r} ({lab})")
                ax.plot(d.season_date, d[value], linewidth=2, markersize=6, **kw)
            if layout == "panels" and len(regions) > 1:
                ax.set_title(r)

        all_axes = list(dict.fromkeys(panels.values()))
        for ax in all_axes:
            ax.xaxis.set_major_locator(mdates.MonthLocator())
            ax.xaxis.set_minor_locator(mdates.MonthLocator(bymonthday=15))
            ax.xaxis.set_major_formatter(mdates.DateFormatter("%b %d"))
            if is_prop:
                ax.yaxis.set_major_formatter(mtick.PercentFormatter(1.0, decimals=0))
            ax.set_ylim(bottom=0)
            ax.grid(axis="y", linestyle="--", alpha=0.3)
            for lab in ax.get_xticklabels():
                lab.set_rotation(45)
                lab.set_ha("right")

        thr = f"threshold ≥ {threshold / 10000:g}" if is_prop else None
        heading = title or ("Proportion of European beech forest discolored" if is_prop else ylabel)
        if len(all_axes) == 1:
            ax = all_axes[0]
            ax.set_title(heading + (f"\n({thr})" if thr else ""))
            ax.set_xlabel("Date (month/day)")
            ax.set_ylabel(ylabel)
            many = len(regions) > 1 and len(regions) * len(labels) > 10
            ax.legend(loc="upper left", bbox_to_anchor=(1.01, 1) if many else None,
                      frameon=False if many else True)
        else:
            fig.suptitle(heading + (f" ({thr})" if thr else ""))
            fig.supxlabel("Date (month/day)")
            fig.supylabel(ylabel)
            all_axes[0].legend(loc="upper left")
        fig.tight_layout()

        if output_dir:
            os.makedirs(output_dir, exist_ok=True)
            name = output_name or "discoloration_comparison_" + "_vs_".join(
                lab.replace(" ", "") for lab in labels)
            png = os.path.join(output_dir, name + ".png")
            fig.savefig(png, dpi=200, bbox_inches="tight")
            stats.to_csv(os.path.join(output_dir, name + ".csv"), index=False,
                         date_format="%Y-%m-%d", encoding="utf-8-sig")
            print(f"Saved {png} and {name}.csv")
        if show:
            plt.show()
        plt.close(fig)
    return stats
