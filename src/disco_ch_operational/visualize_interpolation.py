"""
visualize_interpolation.py

Single-pixel diagnostic plot of the RBF interpolation in interpolate.py.
Reads only the one pixel from a FORCE *_TSS.tif (all acquisitions), runs
the same despike_series() / rbf_weights() / apply_rbf() the tiles go
through, and plots:

  - raw observations (black dots; same-day bands averaged, as in the run)
  - observations despike_series() removed (orange x), if despike=True
  - each rbf_sigma's own curve (thin dashed), from the near-real-time data
    (component_data="nrt", default) or all data ("retrospective")
  - the retrospective ensemble, from all observations -- what
    interpolate_tiles(as_of=None) writes (bold red)
  - with lag_days set: the near-real-time ensemble, where each output date
    only sees observations up to date + lag_days -- what
    interpolate_tiles(as_of=...) writes (bold blue)
  - optionally the values actually stored in *_TSS_TSI.tif files (written=),
    as a check that the plot matches the run
Output dates follow the run's grid (every int_day days from 1 January) and
values are shown divided by `scale` (10000: FORCE's stored VI units -> VI).

effective_bandwidth_stats() gives, per output date, how much smoothing the
ensemble really achieved with the observations it had: the same rbf_sigma
averages over many observations where data are dense (2026) but nearly
copies one or two where they are sparse (2018).

Replaces force_tsi_plot.py (src/disco_ch). Differences:
  - uses interpolate.py's functions instead of force_tsi.py's (same
    method), and reads one pixel without force_tsi.load_tss_pixel();
  - the "stepwise" simulation (wait_days, re-despiked per date) is replaced
    by lag_days, matching interpolate.py's as_of mode: despiking (if on)
    sees all data up to as_of once, and the kernel drops observations more
    than lag_days after each output date;
  - despike defaults to False, as in the operational run;
  - effective bandwidth is returned as a DataFrame (print_effective_bandwidth
    still prints it), computed from rbf_weights() rather than a copy of the
    kernel code;
  - ax= draws into an existing axis (e.g. 2018 and 2026 panels in one figure);
  - returns 1D series for the pixel instead of the (dates, 1, 1) window.

Usage (notebook):

    from src.disco_ch_operational.visualize_interpolation import plot_pixel

    fig, (row, col), dates, raw, cleaned, removed = plot_pixel(
        r"...\\X0056_Y0061\\20260101-20261231_001-365_HL_TSA_SEN2L_EVI_TSS.tif",
        row=2425, col=2156, rbf_sigma=(10, 20, 30), lag_days=5,
    )
"""

import os
import re
from datetime import date, timedelta

import numpy as np
import pandas as pd
import rasterio
import matplotlib.dates as mdates
import matplotlib.pyplot as plt
from rasterio.windows import Window

from src.disco_ch_operational.compare import rc_params
from src.disco_ch_operational.interpolate import (
    DEFAULT_LAG_DAYS, NODATA, apply_rbf, despike_series, output_dates_from_range,
    rbf_cutoff_radius, rbf_weights,
)
from src.disco_ch_operational.utils import (
    as_date, band_dates, chunk_windows, mean_bands, parse_band_date, read_float,
)

_VI_IN_NAME = re.compile(r"_([A-Za-z]+)_TSS(?:_TSI)?\.tif$")
# per-sigma curves; kept off the ensembles' red and blue
_COMPONENT_COLORS = ["tab:green", "tab:orange", "tab:purple", "tab:brown", "tab:pink", "tab:olive"]


# ---------------------------------------------------------------------
# Reading
# ---------------------------------------------------------------------

def load_pixel(tss_path, row, col):
    """(acquisition dates, float32 values) of one pixel of a TSS file,
    same-day bands averaged and nodata as NaN -- the series
    interpolate_file() feeds the RBF for that pixel."""
    bands_by_date = band_dates(tss_path)
    dates = sorted(bands_by_date)
    with rasterio.open(tss_path) as src:
        if not (0 <= row < src.height and 0 <= col < src.width):
            raise ValueError(f"pixel ({row}, {col}) outside {src.height} x {src.width} {tss_path}")
        raw = read_float(src, None, Window(col, row, 1, 1))[:, 0, 0]
    values = np.array([mean_bands(raw[[b - 1 for b in bands_by_date[d]]]) for d in dates],
                      dtype="float32")
    return dates, values


def load_written(tsi_path, row, col):
    """{date: value} of one pixel of an interpolated *_TSS_TSI.tif (NaN
    where nodata)."""
    with rasterio.open(tsi_path) as src:
        vals = src.read(window=Window(col, row, 1, 1))[:, 0, 0].astype("float32")
        dates = [parse_band_date(d) for d in src.descriptions]
        nodata = src.nodata if src.nodata is not None else NODATA
    vals[vals == nodata] = np.nan
    return {d: v for d, v in zip(dates, vals) if d is not None}


def find_sample_pixel(tss_path, min_obs=10, chunk_size=512):
    """(row, col) of a pixel with at least min_obs valid acquisitions --
    the middle one of all such pixels in reading order -- for a quick look
    when no pixel is given. Reads the whole tile, block by block."""
    candidates = []
    with rasterio.open(tss_path) as src:
        for window in chunk_windows(src.width, src.height, chunk_size):
            count = np.sum(~np.isnan(read_float(src, None, window)), axis=0)
            rows, cols = np.nonzero(count >= min_obs)
            candidates += list(zip(rows + window.row_off, cols + window.col_off))
    if not candidates:
        raise ValueError(f"no pixel with at least {min_obs} valid observations in {tss_path}")
    row, col = sorted(candidates)[len(candidates) // 2]
    return int(row), int(col)


# ---------------------------------------------------------------------
# Interpolation of one series
# ---------------------------------------------------------------------

def interpolate_series(dates, values, output_dates, rbf_sigma, rbf_cutoff=0.95, lag_days=None):
    """RBF ensemble of one series at output_dates (NaN = no observation in
    reach); lag_days drops observations more than that many days after
    each output date, as interpolate_file() does in as_of mode."""
    weights = rbf_weights(dates, output_dates, rbf_sigma, rbf_cutoff, max_ahead_days=lag_days)
    return apply_rbf(values[:, None, None], weights)[:, 0, 0]


def effective_bandwidth_stats(dates, values, output_dates, rbf_sigma, rbf_cutoff=0.95, lag_days=None):
    """Smoothing the ensemble actually achieved at each output date, from
    the combined per-observation weights rbf_weights() gives (summed over
    sigmas, valid observations only):

      n_obs               valid observations within reach
      eff_dof             Kish effective sample size (sum w)^2 / sum w^2:
                          ~1 = the value just copies one observation,
                          higher = a real average over several
      eff_bandwidth_days  weighted RMS distance (days) of those observations
      rbf_value           the fitted (ensemble) value
      local_rmse          weighted RMS of observation - fitted value; in-sample,
                          so optimistic where eff_dof is low

    values should be the series the RBF sees (despiked if despiking).
    The all-but-n_obs columns are NaN where no observation is in reach.
    """
    ordinals = np.array([d.toordinal() for d in dates], dtype="float64")
    valid = ~np.isnan(values)
    rows = []
    for od, (idx, w) in zip(output_dates, rbf_weights(dates, output_dates, rbf_sigma, rbf_cutoff,
                                                         max_ahead_days=lag_days)):
        row = dict(date=od, n_obs=0, eff_dof=np.nan, eff_bandwidth_days=np.nan,
                   rbf_value=np.nan, local_rmse=np.nan)
        if w is not None:
            keep = valid[idx]
            weight = w.sum(axis=0)[keep]
            row["n_obs"] = int(np.count_nonzero(weight > 0))
            if weight.sum() > 0:
                x = values[idx][keep].astype("float64")
                delta = ordinals[idx][keep] - od.toordinal()
                wsum = weight.sum()
                fitted = np.sum(weight * x) / wsum
                row.update(eff_dof=wsum ** 2 / np.sum(weight ** 2),
                           eff_bandwidth_days=np.sqrt(np.sum(weight * delta ** 2) / wsum),
                           rbf_value=fitted,
                           local_rmse=np.sqrt(np.sum(weight * (x - fitted) ** 2) / wsum))
        rows.append(row)
    return pd.DataFrame(rows)


def _print_effective_bandwidth(stats, rbf_sigma, rbf_cutoff):
    print(f"Effective bandwidth per output date (rbf_sigma={rbf_sigma}, rbf_cutoff={rbf_cutoff}):")
    for s in stats.itertuples():
        if np.isnan(s.eff_dof):
            print(f"  {s.date}: n_obs={s.n_obs:3d}  (no data within cutoff radius)")
        else:
            print(f"  {s.date}: n_obs={s.n_obs:3d}  eff_dof={s.eff_dof:5.2f}  "
                  f"eff_bandwidth={s.eff_bandwidth_days:6.1f}d  local_rmse={s.local_rmse:7.1f}")
    ok = stats.dropna(subset=["eff_dof"])
    if ok.empty:
        print("Summary: no output date had any observation within the cutoff radius.")
        return
    print(f"Summary over {len(ok)}/{len(stats)} output dates with data: "
          f"eff_dof median {ok.eff_dof.median():.2f} ({ok.eff_dof.min():.2f}-{ok.eff_dof.max():.2f}), "
          f"eff_bandwidth median {ok.eff_bandwidth_days.median():.1f}d "
          f"({ok.eff_bandwidth_days.min():.1f}-{ok.eff_bandwidth_days.max():.1f}d), "
          f"local_rmse median {ok.local_rmse.median():.1f}, overall {np.sqrt(np.mean(ok.local_rmse ** 2)):.1f}")


# ---------------------------------------------------------------------
# Plot
# ---------------------------------------------------------------------

def plot_pixel(tss_path, row=None, col=None, date_range=None, doy_range=(1, 365), int_day=5,
               rbf_sigma=(10, 20, 30), rbf_cutoff=0.95, lag_days=DEFAULT_LAG_DAYS, as_of=None,
               despike=False, above_noise=3.0, below_noise=1.0, show_components=True,
               component_data="nrt", written=None, scale=10000,
               print_effective_bandwidth=False, title=None, ax=None,
               output_path=None, font_scale=1.2):
    """Plot one pixel's observations and its RBF interpolation(s).

    :param tss_path: FORCE *_TSS.tif (one band per acquisition).
    :param row, col: pixel; None picks one with find_sample_pixel() (reads
        the whole tile -- pass row/col for repeated use).
    :param date_range: (first, last) output date; default 1 January -
        31 December of the first acquisition's year, the run's own grid
        (another start shifts every output date). int_day/doy_range as in
        interpolate_tiles().
    :param rbf_sigma, rbf_cutoff: as in interpolate_tiles().
    :param lag_days: also plot the near-real-time ensemble (each output
        date uses observations up to date + lag_days); None = only the
        retrospective (all-data) ensemble.
    :param as_of: drop acquisitions after this date (like an as_of run on
        that day); None = all of them.
    :param despike, above_noise, below_noise: FORCE outlier filter, run
        once over all (as_of) data, as in interpolate_file().
    :param show_components: one thin dashed curve per sigma.
    :param component_data: what those curves see -- "nrt" (observations up
        to date + lag_days, like the near-real-time ensemble) or
        "retrospective" (all data); always "retrospective" when lag_days is
        None.
    :param written: {label: *_TSS_TSI.tif} files whose stored values for
        this pixel are overlaid as open markers.
    The x axis spans date_range with one tick per month, so panels of
    different years line up.
    :param scale: stored values are divided by this for the y axis (10000
        for FORCE VIs; 1 = file units). The returned arrays stay unscaled.
    :param print_effective_bandwidth: print effective_bandwidth_stats() for
        the retrospective ensemble (in file units).
    :param ax: draw into this axis instead of a new figure.
    :return: (fig, (row, col), dates, raw, cleaned, removed) -- acquisition
        dates and the pixel's raw series, the series the RBF sees, and the
        mask of despiked observations.
    """
    if row is None or col is None:
        row, col = find_sample_pixel(tss_path)
    dates, raw = load_pixel(tss_path, row, col)
    as_of = as_date(as_of)
    if as_of is not None:
        raw = np.where(np.array([d > as_of for d in dates]), np.nan, raw).astype("float32")

    if despike:
        cleaned, removed = despike_series(dates, raw[:, None, None], above_noise, below_noise)
        cleaned, removed = cleaned[:, 0, 0], removed[:, 0, 0]
    else:
        cleaned, removed = raw.copy(), np.zeros(raw.shape, dtype=bool)

    if not np.isfinite(raw).any():
        raise ValueError(f"pixel ({row}, {col}) has no valid observation in {tss_path}")
    if date_range is None:
        year = dates[0].year
        date_range = (date(year, 1, 1), date(year, 12, 31))
    output_dates = output_dates_from_range(tuple(as_date(d) for d in date_range), int_day, doy_range)

    if print_effective_bandwidth:
        _print_effective_bandwidth(
            effective_bandwidth_stats(dates, cleaned, output_dates, rbf_sigma, rbf_cutoff),
            rbf_sigma, rbf_cutoff)

    tile = os.path.basename(os.path.dirname(tss_path))
    m = _VI_IN_NAME.search(os.path.basename(tss_path))
    vi = m.group(1) if m else "VI"

    with plt.rc_context(rc_params(font_scale)):
        if ax is None:
            fig, ax = plt.subplots(figsize=(13, 6))
        else:
            fig = ax.figure
        d_arr = np.array(dates)
        kept = ~np.isnan(raw) & ~removed
        ax.scatter(d_arr[kept], raw[kept] / scale, color="black", s=28, zorder=5,
                   label=f"Observations ({kept.sum()})")
        if removed.any():
            ax.scatter(d_arr[removed], raw[removed] / scale, color="darkorange", marker="x", s=70,
                       linewidths=2, zorder=6, label=f"Removed as outlier ({removed.sum()})")

        def nrt_series(sigmas):
            values = interpolate_series(dates, cleaned, output_dates, sigmas, rbf_cutoff, lag_days)
            if as_of is not None:  # an as_of run only writes dates whose lag has passed
                values[[d + timedelta(days=lag_days) > as_of for d in output_dates]] = np.nan
            return values

        if component_data not in ("nrt", "retrospective", "all"):  # "all" = "retrospective"
            raise ValueError("component_data must be 'nrt' or 'retrospective'")
        components_nrt = component_data == "nrt" and lag_days is not None
        if show_components and len(rbf_sigma) > 1:
            for sigma, color in zip(rbf_sigma, _COMPONENT_COLORS * len(rbf_sigma)):
                curve = (nrt_series((sigma,)) if components_nrt
                         else interpolate_series(dates, cleaned, output_dates, (sigma,), rbf_cutoff))
                kind = "near-real-time" if components_nrt else "retrospective"
                ax.plot(output_dates, curve / scale, ls="--", lw=1.2, alpha=0.7, color=color,
                        label=f"{sigma}-day kernel ({kind})")

        full = interpolate_series(dates, cleaned, output_dates, rbf_sigma, rbf_cutoff)
        ax.plot(output_dates, full / scale, color="tab:red", lw=2.5, marker=".", ms=6,
                label="Ensemble, retrospective")
        if lag_days is not None:
            ax.plot(output_dates, nrt_series(rbf_sigma) / scale, color="tab:blue", lw=2.5, marker=".",
                    ms=6, label="Ensemble, near-real-time")

        for (label, path), marker in zip((written or {}).items(), "osD^v"):
            stored = load_written(path, row, col)
            ax.scatter(list(stored), np.array(list(stored.values())) / scale, marker=marker, s=60,
                       facecolors="none", edgecolors="0.3", linewidths=1.2, zorder=8,
                       label=f"Written: {label}")

        reach = rbf_cutoff_radius(max(rbf_sigma), rbf_cutoff)
        ax.set_title(title or f"{tile} {vi}, pixel (row {row}, col {col}); "
                              f"sigma {', '.join(map(str, rbf_sigma))} d (reach {reach:.0f} d)")
        ax.set_ylabel(vi if scale == 10000 else f"{vi} (file units / {scale:g})")
        # same x extent and a label for every month in every panel, whatever the data cover
        start, end = (as_date(d) for d in date_range)
        ax.set_xlim(start, end)
        ax.xaxis.set_major_locator(mdates.MonthLocator())
        ax.xaxis.set_major_formatter(mdates.DateFormatter("%Y-%m"))
        ax.grid(alpha=0.3)
        ax.legend(fontsize="small", loc="upper left", bbox_to_anchor=(1.01, 1))
        for label in ax.get_xticklabels():  # per axis: autofmt_xdate() would hide the
            label.set_rotation(30)          # dates of all but the bottom panel
            label.set_horizontalalignment("right")
        fig.tight_layout()
        if output_path:
            fig.savefig(output_path, dpi=200, bbox_inches="tight")
            print(f"Saved {output_path}")

    return fig, (row, col), dates, raw, cleaned, removed
