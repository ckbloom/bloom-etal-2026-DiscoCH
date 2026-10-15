"""
validate.py

Validates one Disco_Proba_<date>.tif mosaic (or any single-band raster,
e.g. a normalized GRVI mosaic) against ground-truth polygons classified
1 (discolored) / 0 (not discolored), at one threshold or a sweep of them.

  - Scored pixels: inside a validation polygon (pixel centre, as
    rasterio.features.rasterize) and not nodata, on the raster's own grid
    (no resampling). Where polygons overlap, the later polygon's label
    wins. Polygons are reprojected to the raster's CRS if needed, and
    polygons with an empty label are skipped (with a message).
  - Thresholds are on the raster's stored scale, e.g. 4000 on a 0-10000
    int16 Disco_Proba mosaic = 40 % probability. The masked outputs of the
    old mask_and_scale_rasters.py and of mask.py store the same integers,
    so the same thresholds work for both.
  - direction="ge" (default): value >= threshold is predicted discolored
    (Disco_Proba: higher = more likely). "le": value <= threshold, for a
    raster where lower values mean discoloration (normalized GRVI).

Metrics: tn/fp/fn/tp, accuracy, balanced accuracy, precision, recall,
specificity, F1, Cohen's kappa and IoU.

Replaces validate_discoloration_proba.py with the same metrics. Differences:
  - only the pixels under the polygons are read, not the whole mosaic;
  - the sweep counts every threshold in one pass over the sorted values,
    and the metrics come straight from the counts (no sklearn);
  - nothing is written to disk unless output_dir is given (the old code
    saved next to the mosaic by default); the confusion-matrix PNG name
    includes the threshold so different thresholds don't overwrite.

Usage (notebook):

    from src.disco_ch_operational.validate import validate_threshold, validate_thresholds

    validate_threshold(proba_tif, validation_gpkg, 4000, layer="Sentinel_2_13_08_2026")
    df = validate_thresholds(proba_tif, validation_gpkg, layer="Sentinel_2_13_08_2026",
                             output_dir=validation_output)
"""

import os

import numpy as np
import pandas as pd
import rasterio
from rasterio.features import rasterize
from rasterio.windows import Window

DEFAULT_THRESHOLDS = np.arange(0, 10001, 500)   # 0, 500, ..., 10000
PLOT_SCALE = 10000                              # plots show threshold / PLOT_SCALE (0-1 probability)
DIRECTIONS = {"ge": ">=", "le": "<="}


# ---------------------------------------------------------------------
# Pixels under the polygons
# ---------------------------------------------------------------------

def _polygon_window(geom_bounds, src):
    """Pixel window (clipped to the raster) covering a polygon's bounds,
    padded by one pixel; None if the polygon is outside the raster."""
    minx, miny, maxx, maxy = geom_bounds
    inv = ~src.transform
    cols, rows = zip(*(inv * (x, y) for x in (minx, maxx) for y in (miny, maxy)))
    c0, c1 = max(int(np.floor(min(cols))) - 1, 0), min(int(np.ceil(max(cols))) + 1, src.width)
    r0, r1 = max(int(np.floor(min(rows))) - 1, 0), min(int(np.ceil(max(rows))) + 1, src.height)
    if c0 >= c1 or r0 >= r1:
        return None
    return Window(c0, r0, c1 - c0, r1 - r0)


def scored_pixels(proba_tif, validation_gpkg, class_field="Discolored", layer=None):
    """Raw raster values and 0/1 labels of every valid pixel inside a
    validation polygon -- everything needed to score any threshold.

    :return: (values, y_true) 1-D arrays; values on the raster's stored
        scale.
    """
    import geopandas as gpd

    gdf = gpd.read_file(validation_gpkg, layer=layer)
    unlabelled = gdf[class_field].isna()
    if unlabelled.any():
        print(f"Skipping {unlabelled.sum()} of {len(gdf)} polygons with no {class_field!r} value.")
        gdf = gdf[~unlabelled]
        if gdf.empty:
            raise ValueError(f"No polygon in {os.path.basename(validation_gpkg)}"
                             f"{f' [{layer}]' if layer else ''} has a {class_field!r} value.")
    labels = gdf[class_field].astype(int).to_numpy()
    if not set(np.unique(labels)) <= {0, 1}:
        raise ValueError(f"class_field {class_field!r} must be 0/1, found {sorted(set(labels))}")

    idx, lab, val, ok = [], [], [], []
    with rasterio.open(proba_tif) as src:
        if gdf.crs is not None and gdf.crs != src.crs:
            gdf = gdf.to_crs(src.crs)
        for geom, label in zip(gdf.geometry, labels):
            if geom is None or geom.is_empty:
                continue
            win = _polygon_window(geom.bounds, src)
            if win is None:
                continue
            inside = rasterize([(geom, 1)], out_shape=(win.height, win.width),
                               transform=src.window_transform(win), fill=0, dtype="uint8").astype(bool)
            if not inside.any():
                continue
            data = src.read(1, window=win, masked=True)
            rows, cols = np.nonzero(inside)
            idx.append((rows + win.row_off) * src.width + (cols + win.col_off))
            lab.append(np.full(rows.size, label, dtype="uint8"))
            val.append(data.data[rows, cols])
            ok.append(~np.ma.getmaskarray(data)[rows, cols])

    if not idx:
        raise ValueError("No validation polygon covers a pixel of the raster -- check extent/CRS.")
    idx, lab, val, ok = (np.concatenate(a) for a in (idx, lab, val, ok))
    # Each pixel once, keeping the label of the last polygon covering it.
    _, last = np.unique(idx[::-1], return_index=True)
    keep = len(idx) - 1 - last
    lab, val, ok = lab[keep], val[keep], ok[keep]

    print(f"Validation polygons cover {ok.size} pixels of {os.path.basename(proba_tif)}; "
          f"{ok.sum()} are valid and scored ({ok.size - ok.sum()} nodata dropped).")
    if not ok.any():
        raise ValueError("No valid, polygon-covered pixels to score.")
    return val[ok], lab[ok]


# ---------------------------------------------------------------------
# Metrics
# ---------------------------------------------------------------------

def _ratio(a, b):
    with np.errstate(invalid="ignore", divide="ignore"):
        return np.where(b > 0, a / np.where(b > 0, b, 1), np.nan)


def metrics_from_counts(tn, fp, fn, tp):
    """Binary metrics from confusion counts (scalars or arrays). Precision,
    recall and F1 are 0 when undefined (sklearn's zero_division=0);
    specificity, IoU and kappa are NaN."""
    tn, fp, fn, tp = (np.asarray(x, dtype="float64") for x in (tn, fp, fn, tp))
    n = tn + fp + fn + tp
    recall = np.nan_to_num(_ratio(tp, tp + fn))
    specificity = _ratio(tn, tn + fp)
    expected = ((tp + fp) * (tp + fn) + (tn + fn) * (tn + fp)) / n ** 2
    out = {
        "n_pixels": n, "n_positive": tp + fn, "n_negative": tn + fp,
        "tn": tn, "fp": fp, "fn": fn, "tp": tp,
        "accuracy": (tp + tn) / n,
        "balanced_accuracy": (recall + np.nan_to_num(specificity)) / 2,
        "precision": np.nan_to_num(_ratio(tp, tp + fp)),
        "recall": recall,
        "specificity": specificity,
        "f1": np.nan_to_num(_ratio(2 * tp, 2 * tp + fp + fn)),
        "cohen_kappa": _ratio((tp + tn) / n - expected, 1 - expected),
        "iou": _ratio(tp, tp + fp + fn),
    }
    for k in ("n_pixels", "n_positive", "n_negative", "tn", "fp", "fn", "tp"):
        out[k] = out[k].astype("int64")
    return out


def confusion_counts(values, y_true, thresholds, direction="ge"):
    """(tn, fp, fn, tp) arrays, one entry per threshold, from one sort of
    the positive and negative pixels' values."""
    if direction not in DIRECTIONS:
        raise ValueError(f"direction must be 'ge' or 'le', got {direction!r}")
    t = np.asarray(thresholds, dtype="float64")
    pos, neg = np.sort(values[y_true == 1]), np.sort(values[y_true == 0])
    if direction == "ge":
        tp = pos.size - np.searchsorted(pos, t, side="left")
        fp = neg.size - np.searchsorted(neg, t, side="left")
    else:
        tp = np.searchsorted(pos, t, side="right")
        fp = np.searchsorted(neg, t, side="right")
    return neg.size - fp, fp, pos.size - tp, tp


def metrics_table(values, y_true, thresholds=None, direction="ge"):
    """DataFrame of metrics indexed by threshold."""
    thresholds = DEFAULT_THRESHOLDS if thresholds is None else np.asarray(list(thresholds))
    df = pd.DataFrame(metrics_from_counts(*confusion_counts(values, y_true, thresholds, direction)),
                      index=pd.Index(thresholds, name="threshold"))
    return df


# ---------------------------------------------------------------------
# Plots
# ---------------------------------------------------------------------

def plot_confusion_matrix(cm, title, output=None, show=True):
    import matplotlib.pyplot as plt
    from sklearn.metrics import ConfusionMatrixDisplay

    fig, ax = plt.subplots(figsize=(5, 5))
    ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=["not discolored", "discolored"]).plot(
        ax=ax, cmap="Blues", colorbar=False)
    ax.set_title(title)
    fig.tight_layout()
    if output:
        fig.savefig(output, dpi=200)
        print(f"Saved confusion matrix plot to {output}")
    if show:
        plt.show()
    plt.close(fig)


def plot_metrics_vs_threshold(df, output=None, show=True, title=None, scale=PLOT_SCALE):
    """Precision/recall/F1/accuracy vs. threshold, best F1 marked; the
    thresholds are plotted as stored value / scale (0-1 by default)."""
    import matplotlib.pyplot as plt

    best_t = df["f1"].idxmax() / scale
    fig, ax = plt.subplots(figsize=(8, 5))
    for col, style in (("precision", "-o"), ("recall", "-o"), ("f1", "-o"), ("accuracy", "--")):
        ax.plot(df.index / scale, df[col], style, label=col, markersize=4)
    ax.axvline(best_t, color="gray", linestyle=":", linewidth=1, label=f"best F1 @ {best_t:g}")
    ax.set_xlabel("Threshold" if scale == 1 else "Threshold (probability)")
    ax.set_ylabel("Score")
    ax.set_ylim(0, 1.05)
    ax.set_title(title or "Performance vs. threshold")
    ax.legend()
    ax.grid(axis="y", linestyle="--", alpha=0.3)
    fig.tight_layout()
    if output:
        fig.savefig(output, dpi=200)
        print(f"Saved threshold-sweep plot to {output}")
    if show:
        plt.show()
    plt.close(fig)


# ---------------------------------------------------------------------
# Drivers
# ---------------------------------------------------------------------

def validate_threshold(proba_tif, validation_gpkg, threshold, class_field="Discolored", layer=None,
                       direction="ge", output_dir=None, show=True, plot_scale=PLOT_SCALE):
    """Scores proba_tif at one threshold: prints the metrics and plots the
    confusion matrix (saved to output_dir if given).

    :param layer: GeoPackage layer (Mapping_2026.gpkg has one per survey,
        e.g. "Sentinel_2_13_08_2026"; None = the first).
    :param plot_scale: the plot title shows threshold / plot_scale (10000 =
        0-1 probability for Disco_Proba; 1 = the stored value).
    :return: dict of metrics, plus the 2x2 array under "confusion_matrix"
        (rows true 0/1, columns predicted 0/1).
    """
    values, y_true = scored_pixels(proba_tif, validation_gpkg, class_field, layer)
    df = metrics_table(values, y_true, [threshold], direction)
    metrics = {k: (v.item() if hasattr(v, "item") else v) for k, v in df.iloc[0].items()}
    for k in ("n_pixels", "n_positive", "n_negative", "tn", "fp", "fn", "tp"):
        metrics[k] = int(metrics[k])
    cm = np.array([[metrics["tn"], metrics["fp"]], [metrics["fn"], metrics["tp"]]])
    metrics["confusion_matrix"] = cm

    name, op = os.path.basename(proba_tif), DIRECTIONS[direction]
    print(f"\nValidation of {name} against {os.path.basename(validation_gpkg)}"
          f"{f' [{layer}]' if layer else ''} (threshold {op} {threshold})")
    print(f"  confusion matrix (rows true 0/1, cols predicted 0/1):\n{cm}")
    for k, v in metrics.items():
        if k not in ("confusion_matrix", "tn", "fp", "fn", "tp"):
            print(f"  {k:>18s}: {v:.4f}" if isinstance(v, float) else f"  {k:>18s}: {v}")

    output = None
    if output_dir:
        os.makedirs(output_dir, exist_ok=True)
        output = os.path.join(output_dir, f"confusion_matrix_{os.path.splitext(name)[0]}_t{threshold:g}.png")
    if output or show:
        plot_confusion_matrix(cm, f"{name}\nthreshold {op} {threshold / plot_scale:g}", output, show)
    return metrics


def validate_thresholds(proba_tif, validation_gpkg, thresholds=None, class_field="Discolored", layer=None,
                        direction="ge", output_dir=None, show=True, plot_scale=PLOT_SCALE):
    """Sweeps thresholds (default 0, 500, ..., 10000): prints the metrics
    table and best F1, plots precision/recall/F1 vs. threshold / plot_scale
    (10000 = 0-1 probability; 1 = the stored value). With output_dir, saves
    metrics_by_threshold_<raster>.csv and metrics_vs_threshold_<raster>.png
    there.

    :return: DataFrame of metrics indexed by threshold.
    """
    values, y_true = scored_pixels(proba_tif, validation_gpkg, class_field, layer)
    df = metrics_table(values, y_true, thresholds, direction)

    name = os.path.basename(proba_tif)
    print(f"\nThreshold sweep ({DIRECTIONS[direction]}) for {name} against "
          f"{os.path.basename(validation_gpkg)}{f' [{layer}]' if layer else ''}:")
    with pd.option_context("display.width", 200, "display.max_columns", None):
        print(df.round(4).to_string())
    best_t = df["f1"].idxmax()
    print(f"Best F1 = {df.loc[best_t, 'f1']:.4f} at threshold {best_t:g}")

    png = None
    if output_dir:
        os.makedirs(output_dir, exist_ok=True)
        base = os.path.splitext(name)[0]
        csv_path = os.path.join(output_dir, f"metrics_by_threshold_{base}.csv")
        df.to_csv(csv_path)
        print(f"Saved metrics table to {csv_path}")
        png = os.path.join(output_dir, f"metrics_vs_threshold_{base}.png")
    if png or show:
        plot_metrics_vs_threshold(df, png, show, title=f"{name}: performance vs. threshold",
                                  scale=plot_scale)
    return df
