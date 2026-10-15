"""
QAQC.py - Stratified random sampling of predicted-probability rasters
for design-based accuracy assessment (Olofsson et al. 2014; Stehman 2014).

Notebook usage:
    from QAQC import qaqc_sample
    key, sheet, summary = qaqc_sample(["prob_2023-06-01.tif", "prob_2023-07-01.tif"])

    # options
    key, sheet, summary = qaqc_sample(
        glob.glob("prob_*.tif"),
        n_per_raster=200,
        edges=[0, 0.1, 0.3, 0.5, 0.7, 0.9, 1],   # put edges at your reporting thresholds
        method="fixed",                          # or "quantile" (with nstrata=5)
        backups=3,                               # replacements for cloudy / unusable points
        scale=10000,                             # raster value / scale = probability (0-1)
        seed=42,
        outdir="qaqc_sample",                    # None = don't write files
    )

Returns three DataFrames:
    key      full record per point: probability, stratum, N_h, W_h, n_h, inclusion prob
             (keep it hidden until labelling is done)
    sheet    blind labelling sheet: id, date, coordinates, empty label columns
    summary  per raster and stratum: edges, N_h, W_h, n_h

If outdir is set, also writes sample_key.csv, labeling_sheet.csv, strata_summary.csv
and labeling_points.gpkg (one layer per date, if geopandas is installed).

After labelling, qaqc_assess() joins the labels to the key by sample_id and
computes stratified (design-based) estimates, Stehman (2014) eqs. 3-5 and 27-28:
    calibration  per stratum: mean predicted probability vs. labelled share
                 discoloured (+ SE), and the estimated discoloured share/area
                 of all valid pixels
    accuracy     per threshold (map = probability >= t): accuracy, precision
                 (user's accuracy), recall (producer's accuracy), F1 (+ SEs) and the estimated area
                 discoloured vs. the mapped area
The strata do not need to match the threshold: every point carries its own
probability, and each stratum is weighted by its pixel count N_h. Unlabelled
points (label empty / NA) are dropped, which assumes they are missing at
random within their stratum.

    from src.disco_ch_operational.QAQC import qaqc_assess
    cal, acc = qaqc_assess(r"..\\..\\Validation\\Probabalistic_Validation_14_08_2026.shp",
                           "proba_validation_test/sample_key.csv",
                           summary="proba_validation_test/strata_summary.csv",
                           outdir="proba_validation_test")

Requires: numpy, pandas, rasterio (geopandas to read shapefiles, matplotlib for plots).
"""
from pathlib import Path

import numpy as np
import pandas as pd
import rasterio
from rasterio.transform import xy
from rasterio.warp import transform as warp_transform

DEFAULT_EDGES = [0.0, 0.1, 0.3, 0.5, 0.7, 0.9, 1.0]


def _get_edges(vals, method, nstrata, edges):
    if method == "fixed":
        e = np.asarray(sorted(edges), dtype=float)
    elif method == "quantile":
        e = np.unique(np.quantile(vals, np.linspace(0, 1, nstrata + 1)))
        if e.size < 2:  # constant raster
            e = np.array([0.0, 1.0])
    else:
        raise ValueError("method must be 'fixed' or 'quantile'")
    e[0], e[-1] = min(e[0], vals.min()), max(e[-1], vals.max())
    return e


def _allocate_equal(N_h, n):
    """Equal allocation; strata with too few pixels give their share to the others."""
    alloc = np.zeros(len(N_h), dtype=int)
    while alloc.sum() < n:
        open_strata = np.flatnonzero(alloc < N_h)
        if open_strata.size == 0:
            break
        for h in open_strata:
            if alloc.sum() >= n:
                break
            alloc[h] += 1
    return alloc


def _sample_raster(path, n, method, edges, nstrata, backups, scale, rng):
    date = Path(path).stem
    with rasterio.open(path) as src:
        band = src.read(1, masked=True).astype("float32")
        transform, crs, shape = src.transform, src.crs, (src.height, src.width)

    p = band.filled(np.nan).ravel() / scale
    idx_valid = np.flatnonzero(np.isfinite(p))
    if idx_valid.size == 0:
        print(f"[{date}] no valid pixels, skipped")
        return None, None
    vals = p[idx_valid]
    if vals.min() < 0 or vals.max() > 1:
        print(f"[{date}] WARNING: values outside 0-1 ({vals.min():.3g}-{vals.max():.3g}); check scale")

    e = _get_edges(vals, method, nstrata, edges)
    k = len(e) - 1
    # stratum h holds e[h] <= p < e[h+1]; top stratum includes its upper edge
    strata = np.clip(np.digitize(vals, e[1:-1]), 0, k - 1)
    N_h = np.bincount(strata, minlength=k)
    N = N_h.sum()
    alloc = _allocate_equal(N_h, n)

    records, summary = [], []
    for h in range(k):
        members = idx_valid[strata == h]
        n_draw = min(alloc[h] + backups, members.size)
        chosen = rng.choice(members, size=n_draw, replace=False) if n_draw else np.array([], int)
        rows, cols = np.unravel_index(chosen, shape)
        for i, (r, c, flat) in enumerate(zip(rows, cols, chosen)):
            backup = i >= alloc[h]
            records.append(dict(
                date=date, row=int(r), col=int(c), probability=float(p[flat]),
                stratum=h, stratum_lo=e[h], stratum_hi=e[h + 1],
                N_h=int(N_h[h]), W_h=N_h[h] / N, n_h=int(alloc[h]),
                incl_prob=alloc[h] / N_h[h] if N_h[h] else np.nan,
                is_backup=backup, backup_rank=i - alloc[h] + 1 if backup else 0,
            ))
        summary.append(dict(date=date, stratum=h, lo=e[h], hi=e[h + 1],
                            N_h=int(N_h[h]), W_h=N_h[h] / N, n_h=int(alloc[h])))

    df = pd.DataFrame(records)
    if not df.empty:
        xs, ys = xy(transform, df["row"].values, df["col"].values)  # pixel centres
        df["x"], df["y"] = np.asarray(xs), np.asarray(ys)
        if crs is not None:
            lon, lat = warp_transform(crs, "EPSG:4326", df["x"].tolist(), df["y"].tolist())
            df["lon"], df["lat"] = lon, lat
        df["crs"] = crs.to_string() if crs else ""
        df = df.sample(frac=1, random_state=int(rng.integers(1e9))).reset_index(drop=True)
    print(f"[{date}] N={N:,} valid pixels; n_h={alloc.tolist()} (+{backups} backups each); "
          f"edges={np.round(e, 3).tolist()}")
    return df, pd.DataFrame(summary)


def qaqc_sample(rasters, n_per_raster=200, edges=DEFAULT_EDGES, method="fixed", nstrata=5,
                backups=0, scale=10000, seed=42, outdir="qaqc_sample"):
    """Draw a stratified random sample from each probability raster (band 1)."""
    if isinstance(rasters, (str, Path)):
        rasters = [rasters]
    rng = np.random.default_rng(seed)

    samples, summaries = [], []
    for path in rasters:
        df, summ = _sample_raster(path, n_per_raster, method, edges, nstrata, backups, scale, rng)
        if df is not None:
            samples.append(df)
            summaries.append(summ)
    if not samples:
        raise RuntimeError("No samples drawn.")

    key = pd.concat(samples, ignore_index=True)
    key.insert(0, "sample_id", [f"S{i + 1:05d}" for i in range(len(key))])
    summary = pd.concat(summaries, ignore_index=True)

    # Blind labelling sheet: grouped by date (one composite at a time), random order within date
    blind_cols = [c for c in ["sample_id", "date", "x", "y", "lon", "lat", "is_backup", "backup_rank"]
                  if c in key.columns]
    sheet = key[blind_cols].copy()
    sheet["label"] = ""        # 1 = target present, 0 = absent, NA = unusable (cloud etc.)
    sheet["confidence"] = ""   # high / low
    sheet["notes"] = ""
    sheet = sheet.sort_values(["date", "is_backup", "backup_rank"], kind="stable").reset_index(drop=True)

    if outdir is not None:
        out = Path(outdir)
        out.mkdir(parents=True, exist_ok=True)
        key.to_csv(out / "sample_key.csv", index=False)
        sheet.to_csv(out / "labeling_sheet.csv", index=False)
        summary.to_csv(out / "strata_summary.csv", index=False)
        try:
            import geopandas as gpd
            gdf = gpd.GeoDataFrame(sheet, geometry=gpd.points_from_xy(sheet["x"], sheet["y"]),
                                   crs=key["crs"].iloc[0] or None)
            gpkg = out / "labeling_points.gpkg"
            gpkg.unlink(missing_ok=True)
            for date, g in gdf.groupby("date"):
                g.to_file(gpkg, layer=str(date), driver="GPKG")
        except ImportError:
            print("geopandas not installed: skipped GeoPackage (CSVs written).")
        print(f"Wrote {len(key)} points ({(~key['is_backup']).sum()} primary) to {out.resolve()}")

    return key, sheet, summary


# ---------------------------------------------------------------------------
# Assessment of the labelled sample
# ---------------------------------------------------------------------------
def _read_table(obj):
    """DataFrame from a DataFrame or a .csv / .shp / .gpkg path (geometry dropped)."""
    if isinstance(obj, pd.DataFrame):
        return pd.DataFrame(obj.drop(columns="geometry", errors="ignore"))
    path = Path(obj)
    if path.suffix.lower() == ".csv":
        return pd.read_csv(path)
    import geopandas as gpd
    return pd.DataFrame(gpd.read_file(path).drop(columns="geometry"))


def _labelled_sample(labels, key, label_col, summary):
    lab = _read_table(labels)[["sample_id", label_col]].rename(columns={label_col: "label"})
    lab["label"] = pd.to_numeric(lab["label"], errors="coerce")
    bad = lab["label"].notna() & ~lab["label"].isin([0, 1])
    if bad.any():
        raise ValueError(f"labels must be 0, 1 or empty; found {sorted(lab.loc[bad, 'label'].unique())}")
    df = _read_table(key).merge(lab, on="sample_id", how="inner", validate="1:1")
    missing = df["label"].isna()
    print(f"{len(df)} sample points matched to labels; {missing.sum()} unlabelled dropped")
    df = df[~missing].copy()
    df["label"] = df["label"].astype(int)

    # N_h per (date, stratum), from the summary if given so strata with no labelled point show up
    strata = (_read_table(summary)[["date", "stratum", "lo", "hi", "N_h"]] if summary is not None
              else df.groupby(["date", "stratum"], as_index=False)
              .agg(lo=("stratum_lo", "first"), hi=("stratum_hi", "first"), N_h=("N_h", "first")))
    n_lab = df.groupby(["date", "stratum"]).size().rename("n_labelled")
    strata = strata.join(n_lab, on=["date", "stratum"]).fillna({"n_labelled": 0})
    empty = strata[(strata["n_labelled"] < 2) & (strata["N_h"] > 0)]
    if len(empty):
        print(f"WARNING: {len(empty)} stratum/strata with < 2 labelled points; its variance "
              f"cannot be estimated:\n{empty.to_string(index=False)}")
    return df, strata


def _strat_mean(g, y, strata):
    """Stratified estimate of the population mean of y (+ SE), Stehman (2014) eqs. 3-5."""
    N = strata["N_h"].sum()
    est = var = 0.0
    for h, N_h in strata.set_index("stratum")["N_h"].items():
        gh = g[g["stratum"] == h]
        n_h = len(gh)
        if n_h == 0:
            continue
        yh = y[gh.index]
        est += N_h * yh.mean()
        if n_h > 1:
            var += N_h ** 2 * (1 - n_h / N_h) * yh.var(ddof=1) / n_h
    return est / N, np.sqrt(var) / N


def _strat_ratio(g, y, x, strata):
    """Stratified ratio estimate R = Y/X (+ SE), Stehman (2014) eqs. 27-28."""
    Y = X = 0.0
    parts = []
    for h, N_h in strata.set_index("stratum")["N_h"].items():
        gh = g[g["stratum"] == h]
        n_h = len(gh)
        if n_h == 0:
            continue
        yh, xh = y[gh.index], x[gh.index]
        Y += N_h * yh.mean()
        X += N_h * xh.mean()
        parts.append((N_h, n_h, yh, xh))
    if X == 0:
        return np.nan, np.nan
    R = Y / X
    var = sum(N_h ** 2 * (1 - n_h / N_h)
              * (yh.var(ddof=1) + R ** 2 * xh.var(ddof=1) - 2 * R * np.cov(yh, xh, ddof=1)[0, 1]) / n_h
              for N_h, n_h, yh, xh in parts if n_h > 1)
    return R, np.sqrt(max(var, 0.0)) / X


def calibration_table(df, strata, pixel_area_ha=0.01):
    """Per date and stratum: mean predicted probability vs. labelled share
    discoloured (+ binomial SE), plus a row "all" with the stratified estimate
    of the discoloured share and area (ha) of all valid pixels."""
    rows = []
    for date, g in df.groupby("date"):
        s = strata[strata["date"] == date]
        for _, st in s.iterrows():
            gh = g[g["stratum"] == st["stratum"]]
            p = gh["label"].mean() if len(gh) else np.nan
            rows.append(dict(date=date, stratum=st["stratum"], lo=st["lo"], hi=st["hi"],
                             N_h=int(st["N_h"]), n=len(gh),
                             mean_probability=gh["probability"].mean(),
                             share_discolored=p,
                             se=np.sqrt(p * (1 - p) / (len(gh) - 1)) if len(gh) > 1 else np.nan))
        share, se = _strat_mean(g, g["label"].astype(float), s)
        N = s["N_h"].sum()
        mean_p, _ = _strat_mean(g, g["probability"], s)
        rows.append(dict(date=date, stratum="all", lo=s["lo"].min(), hi=s["hi"].max(), N_h=int(N),
                         n=len(g), mean_probability=mean_p, share_discolored=share, se=se,
                         area_ha=share * N * pixel_area_ha, area_ha_ci95=1.96 * se * N * pixel_area_ha))
    return pd.DataFrame(rows)


def accuracy_table(df, strata, thresholds=None, pixel_area_ha=0.01):
    """Per date and threshold t (map discoloured = probability >= t):
    stratified accuracy, precision, recall, F1 and areas, with SEs."""
    thresholds = np.round(np.arange(0.05, 0.96, 0.05), 2) if thresholds is None else thresholds
    rows = []
    for date, g in df.groupby("date"):
        s = strata[strata["date"] == date]
        N = s["N_h"].sum()
        ref = g["label"].astype(float)
        ref_share, ref_se = _strat_mean(g, ref, s)
        for t in thresholds:
            m = (g["probability"] >= t).astype(float)
            oa, oa_se = _strat_mean(g, (m == ref).astype(float), s)
            ua, ua_se = _strat_ratio(g, m * ref, m, s)          # precision = P(ref=1 | map=1)
            pa, pa_se = _strat_ratio(g, m * ref, ref, s)        # recall = P(map=1 | ref=1)
            map_share, _ = _strat_mean(g, m, s)
            rows.append(dict(
                date=date, threshold=t, n=len(g), n_map_discolored=int(m.sum()),
                accuracy=oa, accuracy_se=oa_se,
                precision=ua, precision_se=ua_se,
                recall=pa, recall_se=pa_se,
                f1=2 * ua * pa / (ua + pa) if ua + pa > 0 else np.nan,
                mapped_share=map_share, estimated_share=ref_share, estimated_share_se=ref_se,
                mapped_area_ha=map_share * N * pixel_area_ha,
                estimated_area_ha=ref_share * N * pixel_area_ha))
    return pd.DataFrame(rows)


def plot_assessment(cal, acc, date, output=None, show=True):
    """Calibration (labelled share vs. predicted probability per stratum) and
    accuracy vs. threshold for one date."""
    import matplotlib.pyplot as plt

    c = cal[(cal["date"] == date) & (cal["stratum"] != "all")]
    a = acc[acc["date"] == date]
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(13, 5.5))
    ax1.plot([0, 1], [0, 1], color="0.6", ls="--", lw=1, label="perfect calibration")
    ax1.errorbar(c["mean_probability"], c["share_discolored"], yerr=1.96 * c["se"],
                 fmt="o-", capsize=4, label="labelled share (95% CI)")
    for _, r in c.iterrows():
        ax1.annotate(f"n={r['n']}", (r["mean_probability"], r["share_discolored"]),
                     textcoords="offset points", xytext=(6, -12), fontsize=9)
    ax1.set(xlim=(0, 1), ylim=(-0.05, 1.05), xlabel="Mean predicted probability (stratum)",
            ylabel="Share labelled discoloured", title="Calibration")
    ax1.legend(loc="upper left")

    for col, lab in [("accuracy", "Accuracy"), ("precision", "Precision (user's acc.)"),
                     ("recall", "Recall (producer's acc.)"), ("f1", "F1")]:
        ax2.plot(a["threshold"], a[col], marker=".", label=lab)
        if f"{col}_se" in a:
            ax2.fill_between(a["threshold"], a[col] - 1.96 * a[f"{col}_se"],
                             a[col] + 1.96 * a[f"{col}_se"], alpha=0.15)
    best = a.loc[a["f1"].idxmax()]
    ax2.axvline(best["threshold"], color="0.4", ls=":", lw=1)
    ax2.set(xlim=(0, 1), ylim=(0, 1.02), xlabel="Threshold (probability)", ylabel="Score",
            title=f"Scores vs. threshold (best F1 {best['f1']:.2f} at {best['threshold']:.2f})")
    ax2.legend(loc="lower center")
    fig.suptitle(str(date))
    fig.tight_layout()
    if output:
        fig.savefig(output, dpi=200, bbox_inches="tight")
    if show:
        plt.show()
    else:
        plt.close(fig)
    return fig


def qaqc_assess(labels, key, summary=None, label_col="label", thresholds=None,
                pixel_area_ha=0.01, outdir=None, show=True):
    """Score a labelled qaqc_sample().

    labels   labelled sheet/points (DataFrame, .csv, .shp or .gpkg) with sample_id
             and label_col (1 = discoloured, 0 = not, empty = unusable)
    key      sample_key.csv (or its DataFrame) from qaqc_sample()
    summary  strata_summary.csv (optional; adds strata with no labelled point)
    pixel_area_ha  area of one pixel (10 m pixels = 0.01 ha)

    Returns (calibration, accuracy) DataFrames; with outdir, also writes
    qaqc_calibration.csv, qaqc_accuracy.csv and qaqc_<date>.png.
    """
    df, strata = _labelled_sample(labels, key, label_col, summary)
    cal = calibration_table(df, strata, pixel_area_ha)
    acc = accuracy_table(df, strata, thresholds, pixel_area_ha)

    out = Path(outdir) if outdir is not None else None
    if out is not None:
        out.mkdir(parents=True, exist_ok=True)
        cal.to_csv(out / "qaqc_calibration.csv", index=False)
        acc.to_csv(out / "qaqc_accuracy.csv", index=False)
    for date in cal["date"].unique():
        a = acc[acc["date"] == date]
        best = a.loc[a["f1"].idxmax()]
        tot = cal[(cal["date"] == date) & (cal["stratum"] == "all")].iloc[0]
        print(f"[{date}] estimated discoloured share {tot['share_discolored']:.3f} "
              f"(+/- {1.96 * tot['se']:.3f}) = {tot['area_ha']:,.0f} +/- {tot['area_ha_ci95']:,.0f} ha; "
              f"best F1 {best['f1']:.3f} at threshold {best['threshold']:.2f} "
              f"(accuracy {best['accuracy']:.3f}, precision {best['precision']:.3f}, "
              f"recall {best['recall']:.3f})")
        plot_assessment(cal, acc, date, output=out / f"qaqc_{date}.png" if out else None, show=show)
    return cal, acc


if __name__ == '__main__':
    july_raster = r"B:\bloomc\DiscoCH_2026_08_03\DiscoCH_2026_Operational\output\DiscoCH_rbf_10_20_30\2026\masked\Disco_Proba_2026-07-10.tif"
    august_raster = r"B:\bloomc\DiscoCH_2026_08_03\DiscoCH_2026_Operational\output\DiscoCH_rbf_10_20_30\2026\masked\Disco_Proba_2026-08-14.tif"
    september_raster = r"B:\bloomc\DiscoCH_2026_08_03\DiscoCH_2026_Operational\output\DiscoCH_rbf_10_20_30\2026\masked\Disco_Proba_2026-09-28.tif"

    key, sheet, summary = qaqc_sample([july_raster, august_raster, september_raster], n_per_raster=500,
                                      outdir=r"B:\bloomc\DiscoCH_2026_08_03\DiscoCH_2026_Operational\output\DiscoCH_rbf_10_20_30\2026\proba_validation_test")
    # qa_dir = r"B:\bloomc\DiscoCH_2026_08_03\DiscoCH_2026_Operational\output\DiscoCH_rbf_10_20_30\2026\proba_validation_test"
    # # after labelling the points (here in QGIS, saved as a shapefile):
    # cal, acc = qaqc_assess(r"B:\bloomc\DiscoCH_2026_08_03\Validation\Probabalistic_Validation_14_08_2026.shp",
    #                        qa_dir + r"\sample_key.csv",
    #                        summary=qa_dir + r"\strata_summary.csv",
    #                        outdir=qa_dir)
