"""T02 · TV4 — Exposure / glare features (dark + glare).

Interface (chốt bởi TV1, xem src/t02/README.md)
----------------------------------------------
``compute(images, root) -> DataFrame``
    Exactly one row per image (all 96, dev + eval), column ``image_id`` (str),
    plus the agreed feature columns.

Calibration with the mini-gold labels
-------------------------------------
``reports/mini_gold_dev.csv`` holds 20 hand-labelled **dev** frames
(``is_defect``, ``defect_type``, ``note``). TV4 owns two defect types:
``dark`` and ``glare``. The labels are used *only* to fit thresholds on the
dev split — never on eval, and never inside the pipeline's own run.

Why the labels are needed at all
---------------------------------
Absolute thresholds cannot work here. Measured on the fixture:

    img_18_0  night  none  frac_dark = 0.630   <- perfectly usable night frame
    img_18_1  night  none  frac_dark = 0.632   <- also usable
    img_02_3  day    dark  frac_dark = 0.752   <- the actual defect

A "too dark" threshold on ``frac_dark`` cannot separate the defect from the
usable night frames — they live in the same range. The only signal that
separates them is *relative*: a frame is too dark **compared to the other
frames of the same light/camera group**. That is what ``luma_rel`` encodes,
and the mini-gold labels are what prove the separation holds.

Public API
----------
``compute(images, root)``                -> feature DataFrame (interface spec)
``calibrate(features, labels)``          -> thresholds fitted on mini-gold dev
``apply_calibration(features, params)``  -> per-image TV4 evidence
``load_mini_gold(path)``                 -> helper to read the label file

This module never reads ``fixtures/*/gt/`` and never writes to ``out/``.
"""
from __future__ import annotations

from pathlib import Path
from typing import Sequence

import numpy as np
import pandas as pd
from scipy import ndimage

from ..common.image_ops import load_rgb, to_gray

__all__ = [
    "compute",
    "calibrate",
    "apply_calibration",
    "load_mini_gold",
    "load_calibration",
    "summarize_by_group",
    "FEATURE_COLUMNS",
    "CORE_COLUMNS",
    "DARK_LEVEL",
    "SAT_LEVEL",
    "DEFAULT_MINI_GOLD",
    "DEFAULT_CALIBRATION",
]

# ---------------------------------------------------------------- constants

DARK_LEVEL = 30.0        # gray < 30  -> "shadow" pixel
SAT_LEVEL = 245.0        # gray > 245 -> "burned" pixel
HOT_TILE_FRAC = 0.5      # a tile is "hot" when >50% of its pixels are burned
TILE_GRID = (4, 6)       # 4 rows x 6 cols of tiles over a 180x320 frame
MIN_GROUP_SIZE = 3       # groups smaller than this fall back to a coarser group

#: The columns agreed in src/t02/README.md. ``luma_rel`` is added by
#: :func:`compute` after within-group normalisation.
CORE_COLUMNS: list[str] = [
    "luma_mean",
    "luma_p05",
    "luma_p95",
    "sat_frac",
    "sat_tile_max",
]

#: Every column this module produces (core + supporting evidence).
FEATURE_COLUMNS: list[str] = CORE_COLUMNS + [
    "luma_rel",       # luma_mean z-scored within its light/camera group
    "luma_std",
    "luma_range",
    "dark_frac",      # fraction of shadow pixels
    "sat_blob_frac",  # largest connected burned blob / total pixels
    "sat_blob_px",
    "dark_tile_min",  # darkest tile's mean luma (a dim region, not a dim frame)
]

_META_COLUMNS = ("camera_id", "light", "split")

#: Default location of the hand-made dev labels.
DEFAULT_MINI_GOLD = "reports/mini_gold_dev.csv"

#: Default location of the frozen thresholds.
DEFAULT_CALIBRATION = "configs/tv4_calibration.yaml"


# ---------------------------------------------------------------- primitives

def _tile_grid(values: np.ndarray, grid: tuple[int, int] = TILE_GRID) -> np.ndarray:
    """Average-pool ``values`` (HxW) into a (rows, cols) tile grid.

    Tiles that do not divide the image evenly are cropped away, so the grid
    stays rectangular.
    """
    rows, cols = grid
    h, w = values.shape
    h2, w2 = (h // rows) * rows, (w // cols) * cols
    if h2 == 0 or w2 == 0:
        return values.reshape(1, 1)
    pooled = values[:h2, :w2].reshape(rows, h2 // rows, cols, w2 // cols)
    return pooled.mean(axis=(1, 3))


def _largest_burned_blob(burned: np.ndarray) -> int:
    """Size (px) of the largest 4-connected cluster of burned pixels."""
    if not burned.any():
        return 0
    labels, n = ndimage.label(burned)
    if n == 0:
        return 0
    sizes = ndimage.sum(burned, labels, index=range(1, n + 1))
    return int(sizes.max()) if len(sizes) else 0


def _robust_z(values: np.ndarray) -> np.ndarray:
    """Median-centred, MAD-scaled z-score. Resistant to the outliers we hunt.

    The scale is floored by a fraction of the standard deviation. Without that
    floor a small group whose values cluster tightly (MAD ~ 0) turns a tiny
    absolute difference into an enormous z-score: on the fixture we measured
    ``img_12_2`` (bright_mean 88.1 vs group median 88.8 — practically identical)
    receiving a z of -780 because its group's MAD collapsed. That is noise
    amplified into signal and would corrupt the ranking.

    Taking ``max(MAD * 1.4826, 0.25 * std)`` keeps the robustness of the MAD
    while refusing to inflate negligible differences.
    """
    med = float(np.median(values))
    mad = float(np.median(np.abs(values - med)))
    std = float(values.std())
    scale = max(mad * 1.4826, 0.25 * std)
    if scale < 1e-9:                       # degenerate spread -> no deviation at all
        scale = 1.0
    return (values - med) / scale


def _within_group_z(df: pd.DataFrame, column: str, groups: Sequence[str]) -> np.ndarray:
    """z-score ``column`` inside each group, with a fallback ladder.

    Finest grouping first; groups smaller than ``MIN_GROUP_SIZE`` are too
    noisy to z-score on their own, so they fall back to a coarser group and
    finally to the whole pool. The result never contains NaN.
    """
    out = np.full(len(df), np.nan)
    group_cols = [g for g in groups if g in df.columns]
    values = df[column].to_numpy(dtype=float)
    if not group_cols:
        return _robust_z(values)

    assigned = np.zeros(len(df), dtype=bool)
    for k in range(len(group_cols), 0, -1):
        cols = group_cols[:k]
        # .indices maps group key -> POSITIONAL indices, which stays aligned with
        # the boolean masks below (groupby().groups would yield labels).
        for _, pos in df.groupby(cols, dropna=False).indices.items():
            pos = np.asarray(pos, dtype=int)
            if len(pos) < MIN_GROUP_SIZE:
                continue      # too noisy -> handled by a coarser group below
            free = pos[~assigned[pos]]
            if len(free) == 0:
                continue
            out[free] = _robust_z(values[free])
            assigned[free] = True

    rest = ~assigned
    if rest.any():
        out[rest] = _robust_z(values[rest])
    return out


# ---------------------------------------------------------------- features

def _image_features(img: np.ndarray) -> dict:
    """Raw per-image measurements (no group context yet)."""
    gray = to_gray(img)
    h, w = gray.shape
    total = float(h * w)

    p05, _, p95 = (float(np.percentile(gray, p)) for p in (5, 50, 95))
    burned = gray > SAT_LEVEL
    blob_px = _largest_burned_blob(burned)
    burned_tiles = _tile_grid(burned.astype(np.float32))

    return {
        "luma_mean": float(gray.mean()),
        "luma_std": float(gray.std()),
        "luma_p05": p05,
        "luma_p95": p95,
        "luma_range": p95 - p05,
        "dark_frac": float((gray < DARK_LEVEL).mean()),
        "sat_frac": float(burned.mean()),
        "sat_blob_frac": float(blob_px / total) if total else 0.0,
        "sat_blob_px": float(blob_px),
        "sat_tile_max": float(burned_tiles.max()),
        "dark_tile_min": float(_tile_grid(gray).min()),
    }


def compute(images: pd.DataFrame, root: str | Path) -> pd.DataFrame:
    """TV4 features for every image.

    Parameters
    ----------
    images : DataFrame
        Must carry ``image_id`` and ``path``; ``camera_id`` / ``light`` /
        ``split`` are copied through when present.
    root : Path
        Folder that ``path`` is relative to (usually ``ctx.input_dir``).

    Returns
    -------
    DataFrame
        One row per image with ``image_id`` plus :data:`FEATURE_COLUMNS`.
        ``luma_rel`` is the within-group (light x camera) z-score of
        ``luma_mean`` — the feature that actually separates a dark defect from
        a perfectly usable night frame.
    """
    root = Path(root)
    rows: list[dict] = []
    for rec in images.to_dict("records"):
        img = load_rgb(root / rec["path"])
        row: dict = {"image_id": str(rec["image_id"])}
        for key in _META_COLUMNS:
            if key in rec and rec[key] is not None:
                row[key] = rec[key]
        row.update(_image_features(img))
        rows.append(row)

    df = pd.DataFrame(rows)
    df["luma_rel"] = _within_group_z(df, "luma_mean", ("light", "camera_id"))
    # Relative glare features: the same reasoning as luma_rel. A saturated
    # pixel count only means something *for its own light/camera group* — a
    # night frame with a few burned pixels is normal, the same count at noon
    # is not.
    df["sat_frac_rel"] = _within_group_z(df, "sat_frac", ("light", "camera_id"))
    df["sat_tile_max_rel"] = _within_group_z(df, "sat_tile_max", ("light", "camera_id"))
    df["sat_blob_frac_rel"] = _within_group_z(df, "sat_blob_frac", ("light", "camera_id"))
    columns = (
        ["image_id"]
        + [c for c in _META_COLUMNS if c in df.columns]
        + FEATURE_COLUMNS
        + ["sat_frac_rel", "sat_tile_max_rel", "sat_blob_frac_rel"]
    )
    return df[columns]


# ---------------------------------------------------------------- calibration

def load_mini_gold(path: str | Path = DEFAULT_MINI_GOLD) -> pd.DataFrame:
    """Read the hand-made dev labels.

    Expects columns ``image_id, split, light, is_defect, defect_type, note``.
    Rows outside the dev split are dropped — thresholds must never be fitted
    on eval data (protocol requirement in docs/tracks/T02.md).
    """
    df = pd.read_csv(path, dtype={"image_id": str})
    if "split" in df.columns:
        df = df[df["split"].astype(str) == "dev"].copy()
    return df.reset_index(drop=True)


def load_calibration(path: str | Path = DEFAULT_CALIBRATION) -> dict:
    """Read the frozen TV4 thresholds from ``configs/tv4_calibration.yaml``.

    The pipeline should load the *frozen* file rather than re-fitting, so a
    run is reproducible and so the thresholds cannot drift between the
    baseline and the PoC. Use :func:`calibrate` only to (re)produce that file.
    """
    import yaml

    with open(path, "r", encoding="utf-8") as fh:
        return yaml.safe_load(fh) or {}


def _best_threshold(
    positive: np.ndarray,
    negative: np.ndarray,
    higher_is_positive: bool = True,
) -> float:
    """Threshold separating positives from negatives, best balanced accuracy.

    When the two sets are cleanly separable the midpoint is returned, which
    is the most conservative choice. Otherwise the threshold maximising
    balanced accuracy is used.
    """
    pos = np.asarray(positive, dtype=float)
    neg = np.asarray(negative, dtype=float)
    pos = pos[np.isfinite(pos)]
    neg = neg[np.isfinite(neg)]
    if len(pos) == 0 or len(neg) == 0:
        return float("nan")

    sign = 1.0 if higher_is_positive else -1.0
    p = sign * pos
    n = sign * neg

    if p.min() > n.max():                       # perfectly separable
        return float((np.median(p) + np.median(n)) / 2 * sign) if sign > 0 else float(
            (np.median(p) + np.median(n)) / 2 / sign
        )

    candidates = np.unique(np.concatenate([p, n]))
    best_t, best_acc = float(np.median(candidates)), -1.0
    for t in candidates:
        tpr = float((p >= t).mean())
        tnr = float((n < t).mean())
        bal = 0.5 * (tpr + tnr)
        if bal > best_acc:
            best_acc, best_t = bal, float(t)
    return best_t * sign if sign > 0 else best_t / sign


def _margin_above_negatives(negative: np.ndarray, factor: float = 1.5) -> float:
    """Threshold derived from the clean frames only.

    Used for the *local* glare view, where the labelled positives are too few
    to pin a threshold down: with a single labelled glare frame the midpoint
    between positives and negatives sits far above the local-blow-out range,
    and a genuine local glare (a burning plate in an otherwise fine frame)
    slips through.

    The rule "a clean frame never looks like this" is calibrated from the
    negatives alone: take the most extreme clean frame and add a margin.
    """
    neg = np.asarray(negative, dtype=float)
    neg = neg[np.isfinite(neg)]
    if len(neg) == 0:
        return float("nan")
    top = float(neg.max())
    if abs(top) < 1e-9:
        return 1.0
    return top * factor if top > 0 else top / factor


def calibrate(features: pd.DataFrame, labels: pd.DataFrame) -> dict:
    """Fit TV4 thresholds from the mini-gold dev labels.

    TV4 owns ``dark`` and ``glare`` only. The labelled ``blur`` and
    ``compression`` frames belong to TV3's slice, so they are excluded from
    the fit — they are neither positives nor negatives for "is this dark or
    blown out?".

    Two tiers are fitted for glare:

    * **global** — ``sat_frac_rel``: whole-frame blow-out. Threshold fitted
      between the labelled glare frame and the clean frames.
    * **local** — ``sat_tile_max_rel`` / ``sat_blob_frac_rel``: a blown-out
      region inside an otherwise fine frame. Fitted from the clean frames only
      (see :func:`_margin_above_negatives`), because a single labelled glare
      frame cannot pin these thresholds down.

    Returns a dict of thresholds plus a short provenance record, ready to be
    dumped into ``configs/lab.yaml`` (freeze before evaluating!).
    """
    lab = labels[["image_id", "is_defect", "defect_type"]].copy()
    lab["image_id"] = lab["image_id"].astype(str)
    merged = features.merge(lab, on="image_id", how="inner")

    tv4_pos = merged[merged["defect_type"].astype(str).isin(["dark", "glare"])]
    negatives = merged[merged["defect_type"].astype(str) == "none"]
    glare_pos = tv4_pos[tv4_pos.defect_type == "glare"]

    # Thresholds are fitted on the *relative* (within-group) features, not on
    # the raw ones. A raw threshold cannot work: usable night frames sit in
    # exactly the same absolute range as the dark defect (frac_dark 0.63 vs
    # 0.75). Relative thresholds separate them cleanly.
    params: dict = {
        # dark: luma_mean below the group median -> luma_rel is negative
        "tv4_dark_luma_rel": _best_threshold(
            tv4_pos[tv4_pos.defect_type == "dark"]["luma_rel"].to_numpy(),
            negatives["luma_rel"].to_numpy(),
            higher_is_positive=False,
        ),
    # --- glare, global view: calibrated against the labelled glare frame ---
    # The *global* view alone is not enough: it also fires on frames whose
    # burned pixels are scattered speckle rather than one blown-out region
    # (img_03_2 / img_06_3 have sat_frac_rel of 6.6 / 7.7 but their largest
    # connected blob is only 60-65 px, while a real glare like img_18_2 has a
    # 321 px contiguous blob and img_17_2 has 13246 px).
    # The blob-size floor below separates the two cases.
    "tv4_glare_sat_frac_rel": _best_threshold(
        glare_pos["sat_frac_rel"].to_numpy(),
        negatives["sat_frac_rel"].to_numpy(),
        higher_is_positive=True,
    ),
    # --- glare, local view: fitted from the clean frames only ---
    "tv4_glare_sat_tile_max_rel": _margin_above_negatives(
        negatives["sat_tile_max_rel"].to_numpy()
    ),
    "tv4_glare_sat_blob_frac_rel": _margin_above_negatives(
        negatives["sat_blob_frac_rel"].to_numpy()
    ),
    # --- a real glare is a CONTIGUOUS blown-out region, not scattered pixels ---
    # Fitted from the labels: the smallest labelled glare blob vs the largest
    # clean blob (321 px vs 15 px on the mini gold).
    "tv4_glare_min_blob_px": _best_threshold(
        glare_pos["sat_blob_px"].to_numpy(),
        negatives["sat_blob_px"].to_numpy(),
        higher_is_positive=True,
    ),
    # absolute floor, so a degenerate calibration can never make every frame
    # a glare; also acts as a sanity bound on the fitted value
    "tv4_glare_blob_px_floor": 40.0,
    }

    params["tv4_fit"] = {
        "n_labelled": int(len(merged)),
        "n_dark": int((merged.defect_type == "dark").sum()),
        "n_glare": int((merged.defect_type == "glare").sum()),
        "n_none": int((merged.defect_type == "none").sum()),
        "split": "dev",
        "source": DEFAULT_MINI_GOLD,
    }
    return params


def apply_calibration(features: pd.DataFrame, params: dict) -> pd.DataFrame:
    """Add TV4 evidence columns using calibrated thresholds.

    Adds
    -----
    ``tv4_dark_evidence``   1.0 when the frame is too dark for its group
    ``tv4_glare_evidence``  1.0 when the frame is blown out (globally or locally)
    ``tv4_evidence``        max of the two, 0 when neither fires

    Glare uses an **OR** over two tiers rather than a plain OR over three
    loose thresholds:

    * the *global* view (``sat_frac_rel``) catches a whole-frame blow-out;
    * the *local* views (``sat_tile_max_rel``, ``sat_blob_frac_rel``) catch a
      burning region inside an otherwise fine frame.

    The local thresholds are fitted from the clean frames only, so they are
    tight enough to keep false alarms out while still reaching a genuine
    local glare such as ``img_05_3`` (sat_frac_rel = 1.16 — barely above the
    clean frames — but sat_tile_max_rel = 6.54).
    """
    out = features.copy()

    dark_thr = params.get("tv4_dark_luma_rel")
    out["tv4_dark_evidence"] = (
        (out["luma_rel"] < float(dark_thr)).astype(float)
        if dark_thr is not None and np.isfinite(dark_thr)
        else 0.0
    )

    def _hit(feature: str, key: str) -> pd.Series:
        thr = params.get(key)
        if thr is None or not np.isfinite(thr):
            return pd.Series(False, index=out.index)
        return out[feature] > float(thr)

    global_hit = _hit("sat_frac_rel", "tv4_glare_sat_frac_rel")
    local_hit = (
        _hit("sat_tile_max_rel", "tv4_glare_sat_tile_max_rel")
        | _hit("sat_blob_frac_rel", "tv4_glare_sat_blob_frac_rel")
    )

    # A real glare is one contiguous blown-out region. Scattered burned pixels
    # (sensor speckle at night) raise the saturation counts but must not be
    # reported as glare.
    blob_thr = params.get("tv4_glare_min_blob_px")
    floor = params.get("tv4_glare_blob_px_floor", 40.0)
    if blob_thr is None or not np.isfinite(blob_thr):
        blob_ok = pd.Series(True, index=out.index)
    else:
        blob_ok = out["sat_blob_px"] >= max(float(blob_thr), float(floor))

    out["tv4_glare_evidence"] = ((global_hit | local_hit) & blob_ok).astype(float)

    out["tv4_evidence"] = out[["tv4_dark_evidence", "tv4_glare_evidence"]].max(axis=1)
    return out


# ---------------------------------------------------------------- diagnostics

def summarize_by_group(features: pd.DataFrame, group: str = "light") -> pd.DataFrame:
    """Median feature values per group — the evidence for the failure-mode write-up."""
    cols = [c for c in FEATURE_COLUMNS if c in features.columns]
    return features.groupby(group)[cols].median().round(4)
