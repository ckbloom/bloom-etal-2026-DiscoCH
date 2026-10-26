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

qaqc_report() gives the confusion matrix of one date at one threshold, raw and
area-weighted, with all statistics for all points and per labelling confidence
(1 = low, 2 = moderate, 3 = high):
    from src.disco_ch_operational.QAQC import qaqc_report
    rep = qaqc_report("proba_validation_test/labeling_points.gpkg",
                      "proba_validation_test/sample_key.csv",
                      summary="proba_validation_test/strata_summary.csv",
                      layer="Disco_Proba_2026-08-14", threshold=0.5,
                      outdir="proba_validation_test")
    rep["confusion"], rep["metrics"], rep["by_stratum"], rep["points"]

polygon_recall() gives only the recall of a probability raster against mapped
discoloured polygons (no non-discoloured areas, no sample design), per polygon
and per pixel, at several thresholds:
    from src.disco_ch_operational.QAQC import polygon_recall
    recall, polygons = polygon_recall(r"..\\..\\Validation\\CH30km_Validation.gpkg",
                                      r"..\\output\\...\\masked\\Disco_Proba_2026-08-14.tif",
                                      thresholds=(0.3, 0.5), outdir="recall",
                                      # also per canton (ktnr) of the Bezirke shapefile:
                                      regions=r"..\\..\\Validation\\Shp_Forstkreise_2023_Bezirke\\fgeolib_geolib_g1b20_LV95.shp")

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


def rescore_key(key, raster_dir, outfile=None):
    """The sample key with the probabilities of another map (e.g. a retrained
    model) at the same points, to assess that map with the same labels.

    The design stays that of the sampled map: every point keeps its stratum,
    N_h and inclusion probability, which keeps the stratified estimates of
    accuracy, precision, recall and area unbiased for the new map (the strata
    need not come from the map being assessed; Stehman 2014). Only the
    per-stratum rows of the calibration table refer to the old map's
    probability classes. Points where the new map has no value (nodata,
    e.g. a different mask) get probability NaN and are dropped by qaqc_assess().

    :param key: sample_key.csv (or its DataFrame) from qaqc_sample().
    :param raster_dir: folder with one <date>.tif per date of the key
        (e.g. the new model's masked/Disco_Proba_2026-08-14.tif); band scale applied.
    :param outfile: optional .csv to write the new key to.
    :return: key DataFrame; the old value is kept as probability_sampled.
    """
    key = _read_table(key).copy()
    key["probability_sampled"] = key["probability"]
    key["probability"] = np.nan
    for date, g in key.groupby("date"):
        path = Path(raster_dir) / f"{date}.tif"
        if not path.exists():
            print(f"{date}: {path} missing; its {len(g)} points get no probability")
            continue
        with rasterio.open(path) as src:
            xs, ys = g["x"].to_numpy(), g["y"].to_numpy()
            crs = g["crs"].iloc[0] if "crs" in g else src.crs
            if src.crs and str(src.crs) != str(crs):
                xs, ys = warp_transform(crs, src.crs, xs, ys)
            vals = np.array([v[0] for v in src.sample(zip(xs, ys), masked=True)], dtype=object)
            p = np.array([np.nan if np.ma.is_masked(v) else float(v) for v in vals])
            key.loc[g.index, "probability"] = p * src.scales[0] + src.offsets[0]
        n_nan = int(np.isnan(key.loc[g.index, "probability"]).sum())
        print(f"{date}: {len(g) - n_nan} of {len(g)} points with a probability from {path.name}"
              + (f" ({n_nan} nodata)" if n_nan else ""))
    if outfile is not None:
        Path(outfile).parent.mkdir(parents=True, exist_ok=True)
        key.to_csv(outfile, index=False)
    return key


# ---------------------------------------------------------------------------
# Assessment of the labelled sample
# ---------------------------------------------------------------------------
def _read_table(obj, layer=None):
    """DataFrame from a DataFrame or a .csv / .shp / .gpkg path (geometry dropped).
    layer: GeoPackage layer (e.g. "Disco_Proba_2026-08-14"); None = first layer."""
    if isinstance(obj, pd.DataFrame):
        return pd.DataFrame(obj.drop(columns="geometry", errors="ignore"))
    path = Path(obj)
    if path.suffix.lower() == ".csv":
        return pd.read_csv(path)
    import geopandas as gpd
    return pd.DataFrame(gpd.read_file(path, layer=layer).drop(columns="geometry"))


def _read_labels(labels, layer=None):
    """_read_table(), but layer may also be a list of GeoPackage layers or
    "all" (every layer, e.g. one per date), stacked."""
    if layer == "all" or isinstance(layer, (list, tuple)):
        import pyogrio
        layers = [n for n, _ in pyogrio.list_layers(labels)] if layer == "all" else layer
        return pd.concat([_read_table(labels, lyr) for lyr in layers], ignore_index=True)
    return _read_table(labels, layer)


ALL_DATES = "all_dates"


def _combine_dates(df, strata):
    """Adds the points of all dates once more as date ALL_DATES, with every
    (date, stratum) its own stratum: the stratified estimates then cover all
    pixels of all dates. N_h is divided by the number of dates, so areas are
    the mean area per date."""
    n_dates = strata["date"].nunique()
    key = lambda t: t["date"] + "|" + t["stratum"].astype(str)
    df_all = df.assign(stratum=key(df), date=ALL_DATES)
    strata_all = strata.assign(stratum=key(strata), date=ALL_DATES, N_h=strata["N_h"] / n_dates)
    return pd.concat([df, df_all], ignore_index=True), pd.concat([strata, strata_all], ignore_index=True)


def _pool_calibration(cal):
    """The ALL_DATES per-stratum calibration rows (one per date and stratum)
    pooled per probability class (lo, hi), weighted by N_h."""
    comb = cal["date"] == ALL_DATES
    per = cal[comb & (cal["stratum"] != "all") & (cal["n"] > 0)]
    rows = []
    for (lo, hi), g in per.groupby(["lo", "hi"]):
        w = g["N_h"] / g["N_h"].sum()
        rows.append(dict(date=ALL_DATES, stratum=str(g["stratum"].iloc[0]).split("|")[-1], lo=lo, hi=hi,
                         N_h=int(g["N_h"].sum()), n=int(g["n"].sum()),
                         mean_probability=(w * g["mean_probability"]).sum(),
                         share_discolored=(w * g["share_discolored"]).sum(),
                         se=np.sqrt((w ** 2 * g["se"].fillna(0) ** 2).sum())))
    return pd.concat([cal[~comb], pd.DataFrame(rows), cal[comb & (cal["stratum"] == "all")]], ignore_index=True)


def _labelled_sample(labels, key, label_col, summary, layer=None, conf_col="confidence",
                     min_confidence=None, exclude_notes=False, notes_col="notes"):
    lab = _read_labels(labels, layer)
    cols = ["sample_id", label_col] + [c for c in (conf_col, notes_col) if c in lab.columns]
    lab = lab[cols].rename(columns={label_col: "label", conf_col: "confidence", notes_col: "notes"})
    lab = lab[lab["sample_id"].notna()]  # points added by hand while labelling
    lab["label"] = pd.to_numeric(lab["label"], errors="coerce")
    bad = lab["label"].notna() & ~lab["label"].isin([0, 1])
    if bad.any():
        raise ValueError(f"labels must be 0, 1 or empty; found {sorted(lab.loc[bad, 'label'].unique())}")
    if "confidence" in lab:
        lab["confidence"] = pd.to_numeric(lab["confidence"], errors="coerce")
    df = _read_table(key).merge(lab, on="sample_id", how="inner", validate="1:1")
    missing = df["label"].isna()
    print(f"{len(df)} sample points matched to labels; {missing.sum()} unlabelled dropped")
    df = df[~missing].copy()
    df["label"] = df["label"].astype(int)
    no_p = df["probability"].isna()  # rescore_key(): no value in the new map
    if no_p.any():
        print(f"{no_p.sum()} labelled points without a probability (nodata in the map) dropped")
        df = df[~no_p].copy()
    if min_confidence is not None:
        keep = df["confidence"] >= min_confidence
        print(f"min_confidence={min_confidence}: {(~keep).sum()} points dropped")
        df = df[keep].copy()
    if exclude_notes:
        has_note = df["notes"].fillna("").astype(str).str.strip() != ""
        print(f"exclude_notes: {has_note.sum()} points with notes dropped "
              f"({df.loc[has_note, 'notes'].value_counts().to_dict()})")
        df = df[~has_note].copy()
    print(f"{len(df)} points used")

    # N_h per (date, stratum), from the summary if given so strata with no labelled point show up
    strata = (_read_table(summary)[["date", "stratum", "lo", "hi", "N_h"]] if summary is not None
              else df.groupby(["date", "stratum"], as_index=False)
              .agg(lo=("stratum_lo", "first"), hi=("stratum_hi", "first"), N_h=("N_h", "first")))
    strata = strata[strata["date"].isin(df["date"].unique())]  # dates with labels only
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
    fig.suptitle("All dates combined" if date == ALL_DATES else str(date))
    fig.tight_layout()
    if output:
        fig.savefig(output, dpi=200, bbox_inches="tight")
    if show:
        plt.show()
    else:
        plt.close(fig)
    return fig


def qaqc_assess(labels, key, summary=None, label_col="label", thresholds=None,
                pixel_area_ha=0.01, outdir=None, show=True, layer=None, min_confidence=None,
                threshold=0.5, exclude_notes=False, combine_dates=False):
    """Score a labelled qaqc_sample().

    labels   labelled sheet/points (DataFrame, .csv, .shp or .gpkg) with sample_id
             and label_col (1 = discoloured, 0 = not, empty = unusable)
    key      sample_key.csv (or its DataFrame) from qaqc_sample()
    summary  strata_summary.csv (optional; adds strata with no labelled point)
    pixel_area_ha  area of one pixel (10 m pixels = 0.01 ha)
    layer    GeoPackage layer to read from labels (None = first layer; a list of
             layers or "all" = the labels of several dates)
    combine_dates  True: also the results of all dates together (date "all_dates":
             each date's strata kept as separate strata, so the estimates cover
             all pixels of all dates; areas are the mean per date; the
             calibration pools each probability class over the dates);
             "only": just the combined results. Needs the labels of several
             dates, e.g. layer="all".
    min_confidence  keep only points with confidence >= this (None = all)
    exclude_notes   drop points with anything in the notes column
    threshold  map = probability >= threshold for the confusion matrix plot
               (raw and area-weighted, metrics per confidence); None = skip it

    Returns (calibration, accuracy) DataFrames; with outdir, also writes
    qaqc_calibration_<date>.csv, qaqc_accuracy_<date>.csv, qaqc_<date>.png and
    qaqc_report_<date>_t<threshold>.png. With a filter, the file names get a
    suffix (e.g. _conf2_nonotes) so the unfiltered results are not overwritten.
    """
    df, strata = _labelled_sample(labels, key, label_col, summary, layer,
                                  min_confidence=min_confidence, exclude_notes=exclude_notes)
    tag = (f"_conf{min_confidence}" if min_confidence is not None else "") + \
          ("_nonotes" if exclude_notes else "")
    if combine_dates:
        if df["date"].nunique() < 2:
            print(f"combine_dates: labels of only {df['date'].nunique()} date(s); read several, e.g. layer='all'")
        df, strata = _combine_dates(df, strata)
        if combine_dates == "only":
            df, strata = df[df["date"] == ALL_DATES], strata[strata["date"] == ALL_DATES]
    cal = calibration_table(df, strata, pixel_area_ha)
    if combine_dates:
        cal = _pool_calibration(cal)
    acc = accuracy_table(df, strata, thresholds, pixel_area_ha)
    conf = confusion_by_confidence(df, strata, threshold, pixel_area_ha=pixel_area_ha) \
        if threshold is not None else None

    out = Path(outdir) if outdir is not None else None
    if out is not None:
        out.mkdir(parents=True, exist_ok=True)
        for date in cal["date"].unique():  # one file per date, named like the plots
            cal[cal["date"] == date].to_csv(out / f"qaqc_calibration_{date}{tag}.csv", index=False)
            acc[acc["date"] == date].to_csv(out / f"qaqc_accuracy_{date}{tag}.csv", index=False)
    for date in cal["date"].unique():
        a = acc[acc["date"] == date]
        best = a.loc[a["f1"].idxmax()]
        tot = cal[(cal["date"] == date) & (cal["stratum"] == "all")].iloc[0]
        print(f"[{date}] estimated discoloured share {tot['share_discolored']:.3f} "
              f"(+/- {1.96 * tot['se']:.3f}) = {tot['area_ha']:,.0f} +/- {tot['area_ha_ci95']:,.0f} ha; "
              f"best F1 {best['f1']:.3f} at threshold {best['threshold']:.2f} "
              f"(accuracy {best['accuracy']:.3f}, precision {best['precision']:.3f}, "
              f"recall {best['recall']:.3f})")
        plot_assessment(cal, acc, date, output=out / f"qaqc_{date}{tag}.png" if out else None, show=show)
        if conf is not None:
            plot_confusion(conf["confusion"][conf["confusion"]["date"] == date],
                           conf["metrics"][conf["metrics"]["date"] == date], date, threshold,
                           output=out / f"qaqc_report_{date}_t{threshold}{tag}.png" if out else None,
                           show=show)
    return cal, acc


# ---------------------------------------------------------------------------
# Confusion matrix at one threshold, raw and area-weighted, by label confidence
# ---------------------------------------------------------------------------
def _confidence_subsets(df, levels=(1, 2, 3)):
    """(name, points): all points, each confidence level, and >= each middle level."""
    subsets = [("all", df)]
    if "confidence" not in df or df["confidence"].isna().all():
        return subsets
    subsets += [(f"conf={c}", df[df["confidence"] == c]) for c in levels]
    subsets += [(f"conf>={c}", df[df["confidence"] >= c]) for c in levels[1:-1]]
    return subsets


def _raw_ratio(y, x):
    return y.sum() / x.sum() if x.sum() else np.nan


def confusion_stats(g, s, threshold=0.5, pixel_area_ha=0.01):
    """Confusion matrix (map = probability >= threshold vs. label) and metrics for
    the points g of one date and their strata s, both raw (each point counts once)
    and area-weighted (each stratum weighted by its pixel count N_h, Olofsson et
    al. 2014 eq. 9; SEs from Stehman 2014). Strata without points in g are left
    out; coverage is the share of pixels in the strata that are represented."""
    s_all = s
    s = s[s["stratum"].isin(g["stratum"])]
    N = s["N_h"].sum()
    m = (g["probability"] >= threshold).astype(float)
    ref = g["label"].astype(float)

    cm = []
    for mi in (1, 0):
        for ri in (1, 0):
            ind = ((m == mi) & (ref == ri)).astype(float)
            p, se = _strat_mean(g, ind, s)
            cm.append(dict(map=mi, reference=ri, count=int(ind.sum()), raw_share=ind.mean(),
                           weighted_share=p, weighted_se=se, area_ha=p * N * pixel_area_ha,
                           area_ha_ci95=1.96 * se * N * pixel_area_ha))
    cm = pd.DataFrame(cm)

    hit1, hit0 = m * ref, (1 - m) * (1 - ref)
    defs = {  # metric: (raw estimate, (weighted estimate, se))
        "overall_accuracy": ((m == ref).mean(), _strat_mean(g, (m == ref).astype(float), s)),
        "precision (UA disc.)": (_raw_ratio(hit1, m), _strat_ratio(g, hit1, m, s)),
        "recall (PA disc.)": (_raw_ratio(hit1, ref), _strat_ratio(g, hit1, ref, s)),
        "UA not disc.": (_raw_ratio(hit0, 1 - m), _strat_ratio(g, hit0, 1 - m, s)),
        "PA not disc. (specificity)": (_raw_ratio(hit0, 1 - ref), _strat_ratio(g, hit0, 1 - ref, s)),
        "reference_share_disc.": (ref.mean(), _strat_mean(g, ref, s)),
        "mapped_share_disc.": (m.mean(), _strat_mean(g, m, s)),
    }
    rows = [dict(metric=k, raw=raw, weighted=w, weighted_se=se) for k, (raw, (w, se)) in defs.items()]
    met = pd.DataFrame(rows).set_index("metric")
    for col in ("raw", "weighted"):
        p, r = met.loc["precision (UA disc.)", col], met.loc["recall (PA disc.)", col]
        met.loc["f1", col] = 2 * p * r / (p + r) if p + r > 0 else np.nan
        met.loc["balanced_accuracy", col] = (r + met.loc["PA not disc. (specificity)", col]) / 2
    met["ci95_lo"] = met["weighted"] - 1.96 * met["weighted_se"]
    met["ci95_hi"] = met["weighted"] + 1.96 * met["weighted_se"]
    met.loc["estimated_area_ha", ["weighted", "weighted_se"]] = \
        met.loc["reference_share_disc.", ["weighted", "weighted_se"]].values * N * pixel_area_ha
    met.loc["mapped_area_ha", "weighted"] = met.loc["mapped_share_disc.", "weighted"] * N * pixel_area_ha
    info = dict(n=len(g), n_strata=len(s), coverage=N / s_all["N_h"].sum())
    return cm, met.reset_index(), info


def confusion_by_confidence(df, strata, threshold=0.5, levels=(1, 2, 3), pixel_area_ha=0.01):
    """confusion_stats() per date for all points and each confidence subset;
    returns {"confusion": DataFrame, "metrics": DataFrame}."""
    results = {k: [] for k in ("confusion", "metrics")}
    for date, g in df.groupby("date"):
        s = strata[strata["date"] == date]
        for name, sub in _confidence_subsets(g, levels):
            if len(sub) == 0:
                continue
            cm, met, info = confusion_stats(sub, s, threshold, pixel_area_ha)
            for k, t in (("confusion", cm), ("metrics", met)):
                results[k].append(t.assign(date=date, subset=name, **info))
    return {k: pd.concat(v, ignore_index=True) for k, v in results.items()}


def add_map_columns(gpkg, layer, key, threshold=0.5, label_col="label", backup=True):
    """Add the map side to the labelled points of one GeoPackage layer, for
    styling/filtering in QGIS: probability, stratum, map_<t> (1 = probability
    >= threshold) and outcome_<t> (TP / FP / FN / TN; empty where unlabelled),
    with <t> the threshold in percent (0.2 -> map_20). Existing columns, geometry
    and the other layers are kept; rerunning replaces the added columns.
    backup  first copy the GeoPackage to <name>_backup.gpkg
    Close the layer in QGIS before running (the file is locked while open)."""
    import shutil
    import geopandas as gpd

    gpkg = Path(gpkg)
    if backup:
        shutil.copy2(gpkg, gpkg.with_name(f"{gpkg.stem}_backup.gpkg"))
    gdf = gpd.read_file(gpkg, layer=layer)
    t = f"{round(threshold * 100):02d}"
    k = _read_table(key).set_index("sample_id")
    gdf = gdf.drop(columns=["probability", "stratum", f"map_{t}", f"outcome_{t}"], errors="ignore")
    gdf["probability"] = gdf["sample_id"].map(k["probability"])
    gdf["stratum"] = gdf["sample_id"].map(k["stratum"]).astype("Int64")
    m = gdf["probability"] >= threshold
    has_p = gdf["probability"].notna()
    lab = pd.to_numeric(gdf[label_col], errors="coerce")
    gdf[f"map_{t}"] = m.astype("Int64").where(has_p)
    ok = has_p & lab.notna()
    gdf[f"outcome_{t}"] = pd.Series(np.select([m & (lab == 1), m & (lab == 0), ~m & (lab == 1)],
                                              ["TP", "FP", "FN"], "TN"), index=gdf.index).where(ok)
    gdf.to_file(gpkg, layer=layer, driver="GPKG")
    print(f"[{layer}] added probability, stratum, map_{t}, outcome_{t}: "
          f"{gdf[f'outcome_{t}'].value_counts().to_dict()}")
    return gdf


def stratum_confidence_table(df, strata, threshold=0.5):
    """Raw counts per date, stratum and confidence: labelled share discoloured and
    agreement with the map at the threshold."""
    d = df.assign(map=(df["probability"] >= threshold).astype(int))
    d["agree"] = d["map"] == d["label"]
    if "confidence" not in d:
        d["confidence"] = np.nan
    t = (d.groupby(["date", "stratum", "confidence"], dropna=False)
         .agg(n=("label", "size"), n_discolored=("label", "sum"),
              mean_probability=("probability", "mean"), agreement=("agree", "mean"))
         .reset_index())
    t["share_discolored"] = t["n_discolored"] / t["n"]
    return strata[["date", "stratum", "lo", "hi", "N_h"]].merge(t, on=["date", "stratum"], how="right")


def plot_confusion(cm, met, date, threshold, output=None, show=True):
    """Raw and area-weighted confusion matrix (all points) and weighted metrics
    with 95% CIs per confidence subset."""
    import matplotlib.pyplot as plt

    fig, axes = plt.subplots(1, 3, figsize=(17, 5), gridspec_kw=dict(width_ratios=[1, 1, 2]))
    c = cm[cm["subset"] == "all"].set_index(["map", "reference"])
    labels = ["discoloured", "not disc."]
    for ax, col, title, fmt in [(axes[0], "count", "Raw counts", "{:.0f}"),
                                (axes[1], "weighted_share", "Area-weighted share", "{:.1%}")]:
        mat = np.array([[c.loc[(mi, ri), col] for ri in (1, 0)] for mi in (1, 0)])
        ax.imshow(mat, cmap="Blues", vmin=0)
        for i in range(2):
            for j in range(2):
                ax.text(j, i, fmt.format(mat[i, j]), ha="center", va="center",
                        color="white" if mat[i, j] > mat.max() / 2 else "black", fontsize=12)
        ax.set(xticks=[0, 1], yticks=[0, 1], xticklabels=labels, yticklabels=labels,
               xlabel="Reference (label)", ylabel=f"Map (p >= {threshold})", title=title)

    ax = axes[2]
    names = ["overall_accuracy", "precision (UA disc.)", "recall (PA disc.)",
             "PA not disc. (specificity)", "f1"]
    subsets = list(dict.fromkeys(met["subset"]))
    w = 0.8 / len(subsets)
    for k, sub in enumerate(subsets):
        mm = met[met["subset"] == sub].set_index("metric").loc[names]
        n = int(met.loc[met["subset"] == sub, "n"].iloc[0])
        ax.bar(np.arange(len(names)) + (k - (len(subsets) - 1) / 2) * w, mm["weighted"], w * 0.9,
               yerr=1.96 * mm["weighted_se"].fillna(0), capsize=2, label=f"{sub} (n={n})")
    ax.set_xticks(np.arange(len(names)), [n.replace(" (", "\n(") for n in names])
    ax.set(ylim=(0, 1.05), ylabel="Area-weighted estimate (95% CI)", title="Metrics by label confidence")
    ax.legend(fontsize=8, loc="upper left", bbox_to_anchor=(1.01, 1))
    fig.suptitle(f"{'All dates combined' if date == ALL_DATES else date}  threshold {threshold}")
    fig.tight_layout()
    if output:
        fig.savefig(output, dpi=200, bbox_inches="tight")
    if show:
        plt.show()
    else:
        plt.close(fig)
    return fig


def qaqc_report(labels, key, summary=None, layer=None, threshold=0.5, label_col="label",
                conf_col="confidence", levels=(1, 2, 3), pixel_area_ha=0.01, outdir=None, show=True):
    """Confusion matrix and statistics of one labelled date at one threshold,
    raw and area-weighted, for all points and per labelling confidence
    (1 = low, 2 = moderate, 3 = high; subsets conf=1, conf=2, conf=3, conf>=2).

    Confidence subsets keep the strata weights N_h, which assumes the label share
    within a stratum does not depend on confidence; read them as a sensitivity check.

    Returns a dict of DataFrames:
        confusion   per subset: map x reference counts, weighted shares (+ SE), areas
        metrics     per subset: raw and weighted accuracy, precision/recall (UA/PA)
                    for both classes, F1, balanced accuracy, shares and areas
        by_stratum  raw counts per stratum x confidence: share discoloured, agreement
        points      the labelled points with map class and outcome (TP/FP/FN/TN)
    With outdir, writes qaqc_report_<date>_t<threshold>.xlsx (one sheet each) and .png.
    """
    df, strata = _labelled_sample(labels, key, label_col, summary, layer, conf_col)
    out = confusion_by_confidence(df, strata, threshold, levels, pixel_area_ha)
    out["by_stratum"] = stratum_confidence_table(df, strata, threshold)
    pts = df.assign(map=(df["probability"] >= threshold).astype(int))
    pts["outcome"] = np.select([(pts["map"] == 1) & (pts["label"] == 1), (pts["map"] == 1),
                                (pts["label"] == 1)], ["TP", "FP", "FN"], "TN")
    out["points"] = pts

    pd.set_option("display.width", 200)
    for date in df["date"].unique():
        cm = out["confusion"][out["confusion"]["date"] == date]
        met = out["metrics"][out["metrics"]["date"] == date]
        c = cm[cm["subset"] == "all"]
        print(f"\n=== {date}  (map discoloured = p >= {threshold}) ===")
        for col, title in (("count", "raw counts"), ("weighted_share", "area-weighted share")):
            print(f"\nConfusion matrix, {title} (rows map, columns reference):")
            print(c.pivot(index="map", columns="reference", values=col).loc[[1, 0], [1, 0]]
                  .rename(index={1: "map disc.", 0: "map not"}, columns={1: "ref disc.", 0: "ref not"})
                  .to_string(float_format=lambda v: f"{v:.4f}"))
        print("\nArea-weighted estimates (+/- 1.96 SE) by confidence subset:")
        w = met.assign(v=[f"{a:.3f} +/- {1.96 * b:.3f}" if pd.notna(b) else f"{a:,.3f}"
                          for a, b in zip(met["weighted"], met["weighted_se"])])
        print(w.pivot(index="metric", columns="subset", values="v")
              .loc[met["metric"].unique(), met["subset"].unique()].to_string())
        print("\nRaw (unweighted) estimates by confidence subset:")
        print(met.pivot(index="metric", columns="subset", values="raw")
              .loc[met["metric"].unique(), met["subset"].unique()].dropna(how="all")
              .to_string(float_format=lambda v: f"{v:.3f}"))
        print("\nPoints and pixel coverage per subset:")
        print(met.groupby("subset", sort=False)[["n", "n_strata", "coverage"]].first().to_string())

        stem = f"qaqc_report_{date}_t{threshold}"
        if outdir is not None:
            o = Path(outdir)
            o.mkdir(parents=True, exist_ok=True)
            with pd.ExcelWriter(o / f"{stem}.xlsx") as xl:
                for k, t in out.items():
                    (t[t["date"] == date] if "date" in t else t).to_excel(xl, sheet_name=k, index=False)
        plot_confusion(cm, met, date, threshold,
                       output=Path(outdir) / f"{stem}.png" if outdir is not None else None, show=show)
    return out


# ---------------------------------------------------------------------------
# Recall from mapped discoloured polygons (no sample design)
# ---------------------------------------------------------------------------
# canton abbreviations by BFS canton number (ktnr)
CANTONS = {1: "ZH", 2: "BE", 3: "LU", 4: "UR", 5: "SZ", 6: "OW", 7: "NW", 8: "GL", 9: "ZG", 10: "FR",
           11: "SO", 12: "BS", 13: "BL", 14: "SH", 15: "AR", 16: "AI", 17: "SG", 18: "GR", 19: "AG",
           20: "TG", 21: "TI", 22: "VD", 23: "VS", 24: "NE", 25: "GE", 26: "JU"}


def _recall_rows(ok, thresholds):
    """Recall per threshold of the polygons with valid pixels (see polygon_recall)."""
    out = []
    for t in thresholds:
        n_ge = ok[f"n_ge_{t:g}"]
        some = len(ok) > 0
        out.append(dict(threshold=t, n_polygons=len(ok), n_pixels=int(ok["n_valid"].sum()),
                        polygon_recall_mean=(ok["mean_proba"] >= t).mean() if some else np.nan,
                        polygon_recall_any=(n_ge > 0).mean() if some else np.nan,
                        pixel_recall=n_ge.sum() / ok["n_valid"].sum() if some else np.nan,
                        pixel_recall_poly=(n_ge / ok["n_valid"]).mean() if some else np.nan))
    return out


def polygon_recall(polygons, raster, thresholds=(0.3, 0.5), layer=None, label_col="Discolored",
                   regions=None, region_col="ktnr", assume_crs="EPSG:2056", outdir=None, dataset=None):
    """Recall (producer's accuracy) of a probability raster against mapped
    discoloured polygons, e.g. Validation/CH30km_Validation.gpkg. Without
    mapped non-discoloured areas only recall can be computed, and it is the
    recall of what was mapped (not a design-based estimate for the country).

    Pixels belong to a polygon if their centre lies inside it; nodata pixels
    (masked / no prediction) are left out, and polygons without a valid pixel
    are reported but not counted.

    :param polygons: GeoDataFrame or .gpkg/.shp path; only rows with
        label_col == 1 are used (all rows if label_col is missing).
    :param raster: probability GeoTIFF (band scale applied, e.g. int16 x 0.0001),
        e.g. masked/Disco_Proba_2026-08-14.tif for a 14 Aug mapping.
    :param thresholds: map discoloured = probability >= t.
    :param layer: GeoPackage layer (None = first).
    :param regions: optional GeoDataFrame or path of areas to report recall
        for as well (e.g. the Bezirke shapefile with the canton number ktnr);
        each polygon goes to the region containing its representative point.
    :param region_col: column of `regions` to group by (several areas with
        the same value, e.g. the districts of a canton, are one region; ktnr
        is shown with its canton abbreviation).
    :param assume_crs: CRS of polygons without one (the Planet mapping is LV95).
    :param dataset: label of the map in the file names and the dataset column
        (default: the two folders above the raster's, e.g.
        "DiscoCH_rbf_10_20_30_2026" for .../DiscoCH_rbf_10_20_30/2026/masked/x.tif).
    :param outdir: writes polygon_recall_<dataset>_<raster>.csv, and
        polygon_recall_by_<region_col>_... and polygon_recall_polygons_...;
        every table also has the dataset, raster and validation paths.
    :return: (recall DataFrame per region and threshold -- region "all" =
              overall, per-polygon DataFrame)

    recall columns, for map = p >= t:
        polygon_recall_mean  share of polygons whose mean probability >= t
        polygon_recall_any   share of polygons with at least one pixel >= t
        pixel_recall         share of all mapped pixels >= t (= area recall;
                             large polygons weigh more)
        pixel_recall_poly    mean over polygons of their share of pixels >= t
                             (every polygon weighs the same)
    """
    import geopandas as gpd
    from rasterio.features import geometry_mask
    from rasterio.errors import WindowError
    from rasterio.windows import Window, from_bounds

    gdf = polygons if isinstance(polygons, gpd.GeoDataFrame) else gpd.read_file(polygons, layer=layer)
    if label_col in gdf.columns:
        n_all = len(gdf)
        gdf = gdf[pd.to_numeric(gdf[label_col], errors="coerce") == 1]
        print(f"{len(gdf)} of {n_all} polygons with {label_col} == 1")
    gdf = gdf[gdf.geometry.notna() & ~gdf.geometry.is_empty].reset_index(drop=True)
    thresholds = np.atleast_1d(thresholds).astype(float)
    if gdf.crs is None:
        print(f"polygons have no CRS; assuming {assume_crs}")
        gdf = gdf.set_crs(assume_crs)

    region = None
    if regions is not None:
        reg = regions if isinstance(regions, gpd.GeoDataFrame) else gpd.read_file(regions)
        reg = reg[[region_col, "geometry"]].to_crs(gdf.crs)
        pts = gpd.GeoDataFrame(geometry=gdf.representative_point(), crs=gdf.crs)
        hit = gpd.sjoin(pts, reg, how="left", predicate="within")
        region = hit[~hit.index.duplicated()][region_col].reindex(gdf.index)
        if region_col == "ktnr":
            region = region.map(lambda k: f"{int(k):02d} {CANTONS.get(int(k), '')}".strip() if pd.notna(k) else k)
        region = region.fillna("outside").astype(str)

    rows = []
    with rasterio.open(raster) as src:
        gdf = gdf.to_crs(src.crs)
        scale, offset = src.scales[0], src.offsets[0]
        full = Window(0, 0, src.width, src.height)
        for i, geom in enumerate(gdf.geometry):
            try:  # pixels around the polygon, clipped to the raster
                w = from_bounds(*geom.bounds, transform=src.transform)
                c0, r0 = int(np.floor(w.col_off)), int(np.floor(w.row_off))
                c1, r1 = int(np.ceil(w.col_off + w.width)), int(np.ceil(w.row_off + w.height))
                win = Window(c0, r0, c1 - c0, r1 - r0).intersection(full)
            except WindowError:  # outside the raster
                win = None
            vals = np.array([])
            n_inside = 0
            if win is not None and win.width > 0 and win.height > 0:
                a = src.read(1, window=win, masked=True)
                inside = geometry_mask([geom], out_shape=a.shape, transform=src.window_transform(win), invert=True)
                n_inside = int(inside.sum())
                vals = a[inside].compressed().astype(float) * scale + offset
            r = dict(polygon=i, area_m2=geom.area, n_pixels=n_inside, n_valid=len(vals),
                     mean_proba=vals.mean() if len(vals) else np.nan,
                     max_proba=vals.max() if len(vals) else np.nan)
            for t in thresholds:
                r[f"n_ge_{t:g}"] = int((vals >= t).sum())
            rows.append(r)
    per = pd.concat([gdf.drop(columns="geometry").reset_index(drop=True), pd.DataFrame(rows)], axis=1)
    if region is not None:
        per.insert(0, "region", region.to_numpy())

    ok = per[per["n_valid"] > 0]
    groups = [("all", per)] + (sorted(per.groupby("region"), key=lambda g: g[0]) if region is not None else [])
    recall = pd.DataFrame([dict(region=name, n_mapped=len(g), n_no_valid_pixel=int((g["n_valid"] == 0).sum()), **r)
                           for name, g in groups for r in _recall_rows(g[g["n_valid"] > 0], thresholds)])

    n_none = int((per["n_valid"] == 0).sum())
    print(f"{Path(raster).name}: {len(ok)} polygons / {int(ok['n_valid'].sum()):,} pixels with probabilities; "
          f"{n_none} polygon(s) without a valid pixel left out "
          f"({int((per['n_pixels'] == 0).sum())} smaller than a pixel centre or outside the raster)")
    fmt = lambda v: f"{v:.3f}"
    print(recall[recall["region"] == "all"].drop(columns="region").to_string(index=False, float_format=fmt))
    by_region = None
    if region is not None:
        # one row per region: counts, then polygon (by mean) and pixel recall per threshold; worst first
        r = recall[recall["region"] != "all"]
        by_region = r.pivot(index="region", columns="threshold", values=["polygon_recall_mean", "pixel_recall"])
        by_region.columns = [f"{m.split('_')[0]}_t{t:g}" for m, t in by_region.columns]
        counts = r.groupby("region")[["n_mapped", "n_no_valid_pixel", "n_pixels"]].first()
        by_region = counts.join(by_region).sort_values(f"pixel_t{thresholds[0]:g}", na_position="first")
        print(f"\nBy {region_col}, worst pixel recall at t={thresholds[0]:g} first (polygon = share of polygons "
              f"with mean probability >= t, pixel = share of pixels >= t):")
        print(by_region.to_string(float_format=fmt))
    # which map and validation data the results are from (the raster names are the same for every model)
    rp = Path(raster).resolve()
    dataset = dataset or f"{rp.parent.parent.parent.name}_{rp.parent.parent.name}"
    source = dict(dataset=dataset, raster=str(rp),
                  validation=str(Path(polygons).resolve()) if isinstance(polygons, (str, Path)) else "<GeoDataFrame>")
    recall = recall.assign(**source)
    per = per.assign(**source)
    if outdir is not None:
        o = Path(outdir)
        o.mkdir(parents=True, exist_ok=True)
        stem = f"{dataset}_{rp.stem}"
        recall.to_csv(o / f"polygon_recall_{stem}.csv", index=False)
        if by_region is not None:
            by_region.assign(**source).to_csv(o / f"polygon_recall_by_{region_col}_{stem}.csv")
        per.to_csv(o / f"polygon_recall_polygons_{stem}.csv", index=False)
        print(f"wrote polygon_recall*_{stem}.csv to {o}")
    return recall, per


if __name__ == '__main__':
    # july_raster = r"B:\bloomc\DiscoCH_2026_08_03\DiscoCH_2026_Operational\output\DiscoCH_rbf_10_20_30\2026\masked\Disco_Proba_2026-07-10.tif"
    # august_raster = r"B:\bloomc\DiscoCH_2026_08_03\DiscoCH_2026_Operational\output\DiscoCH_rbf_10_20_30\2026\masked\Disco_Proba_2026-08-14.tif"
    # september_raster = r"B:\bloomc\DiscoCH_2026_08_03\DiscoCH_2026_Operational\output\DiscoCH_rbf_10_20_30\2026\masked\Disco_Proba_2026-09-28.tif"
    #
    # key, sheet, summary = qaqc_sample([july_raster, august_raster, september_raster], n_per_raster=500,
    #                                   outdir=r"B:\bloomc\DiscoCH_2026_08_03\DiscoCH_2026_Operational\output\DiscoCH_rbf_10_20_30\2026\proba_validation_test")


    qa_dir = r"B:\bloomc\DiscoCH_2026_08_03\DiscoCH_2026_Operational\output\DiscoCH_rbf_10_20_30\2026\proba_validation_test"
    # # after labelling the points (here in QGIS, saved as a shapefile):
    # cal, acc = qaqc_assess(qa_dir + r"\labeling_points.gpkg", qa_dir + r"\sample_key.csv",
    #                        summary=qa_dir + r"\strata_summary.csv",
    #                        layer="Disco_Proba_2026-08-14",
    #                        threshold=0.3, min_confidence=1, exclude_notes=True,
    #                        outdir=qa_dir)


    # add_map_columns(qa_dir + r"\labeling_points.gpkg", "Disco_Proba_2026-09-28",
    #                 qa_dir + r"\sample_key.csv", threshold=0.2)


    # # recall against the mapped discoloured polygons (Planet, 14 Aug 2026), one run per model's masked map
    # validation = r"B:\bloomc\DiscoCH_2026_08_03\Validation\CH30km_Validation.gpkg"
    # cantons = r"B:\bloomc\DiscoCH_2026_08_03\Validation\Shp_Forstkreise_2023_Bezirke\fgeolib_geolib_g1b20_LV95.shp"
    # output = r"B:\bloomc\DiscoCH_2026_08_03\DiscoCH_2026_Operational\output"
    # recall_dir = output + r"\DiscoCH_rbf_10_20_30\2026\polygon_recall"  # both datasets, named by dataset
    # for dataset, proba_raster in {
    #     "nrt": output + r"\DiscoCH_rbf_10_20_30\2026\masked\Disco_Proba_2026-08-14.tif",
    #     "retrospective": output + r"\DiscoCH_rbf_10_20_30\2026_full_data\masked\Disco_Proba_2026-08-14.tif",
    #     # "logreg": output + r"\DiscoCH_rbf_10\2026\discoch_logreg_2026_10_21\masked\Disco_Proba_2026-08-14.tif",
    # }.items():
    #     if not Path(proba_raster).exists():
    #         print(f"missing (not masked yet?): {proba_raster}")
    #         continue
    #     recall, polygons = polygon_recall(validation, proba_raster, thresholds=(0.4, ),
    #                                       regions=cantons, region_col="ktnr",
    #                                       dataset=dataset, outdir=recall_dir)
    #
    # # the labelled points of qa_dir scored against another model's maps (same design, its probabilities)
    # qa_dir = r"B:\bloomc\DiscoCH_2026_08_03\DiscoCH_2026_Operational\output\DiscoCH_rbf_10_20_30\2026\proba_validation_test"
    # nrt_dir = r"B:\bloomc\DiscoCH_2026_08_03\DiscoCH_2026_Operational\output\DiscoCH_rbf_10_20_30\2026\masked"
    # key_lr = rescore_key(qa_dir + r"\sample_key.csv", nrt_dir,
    #                      outfile=nrt_dir + r"\proba_validation\sample_key.csv")
    # cal, acc = qaqc_assess(qa_dir + r"\labeling_points.gpkg", key_lr,
    #                        summary=qa_dir + r"\strata_summary.csv",
    #                        layer="all",  # Disco_Proba_2026-07-10
    #                        combine_dates=True,  # also all dates in one pot ("only" = just that)
    #                        threshold=0.4, min_confidence=2, # exclude_notes=True,
    #                        outdir=qa_dir + r"\proba_validation_nrt")
    #
    #
    # the labelled points of qa_dir scored against another model's maps (same design, its probabilities)
    qa_dir = r"B:\bloomc\DiscoCH_2026_08_03\DiscoCH_2026_Operational\output\DiscoCH_rbf_10_20_30\2026\proba_validation_test"
    retrospective_dir = r"B:\bloomc\DiscoCH_2026_08_03\DiscoCH_2026_Operational\output\DiscoCH_rbf_10_20_30\2026_full_data\masked"
    key_lr = rescore_key(qa_dir + r"\sample_key.csv", retrospective_dir,
                         outfile=retrospective_dir + r"\proba_validation\sample_key.csv")
    cal, acc = qaqc_assess(qa_dir + r"\labeling_points.gpkg", key_lr,
                           summary=qa_dir + r"\strata_summary.csv",
                           layer="all", # Disco_Proba_2026-07-10
                           combine_dates=True,  # also all dates in one pot ("only" = just that)
                           threshold=0.4, min_confidence=2, exclude_notes=True,
                           outdir=qa_dir + r"\proba_validation_retrospective")
