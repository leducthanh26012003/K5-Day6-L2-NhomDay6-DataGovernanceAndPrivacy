"""T02 · TV5 — gộp feature thành điểm review, loại lỗi và đề xuất KEEP / RECAPTURE.

Giao diện (chốt trong src/t02/README.md)
---------------------------------------
``combine(feats, meta, params) -> DataFrame``
    Đúng 1 dòng cho mỗi ảnh (đủ 96), cột ``image_id, score, pred_defect_type,
    pred_action, reason`` + các cột giải thích ``margin_<loại lỗi>``, ``candidate_type``.

Cách chấm điểm
--------------
Mỗi loại lỗi ``d`` có một **biên** (margin) ``m_d`` trên cùng một thang đo:

    m_d = (độ lệch của feature so với ngưỡng, theo hướng "càng lỗi") / scale_d

* ngưỡng: ``m_d = 0``. ``m_d > 0`` nghĩa là loại lỗi này "bật".
* ``scale_d`` = nửa khe hở giữa ảnh lỗi gần ngưỡng nhất và ảnh sạch gần ngưỡng
  nhất trong mini gold dev, nên ``m_d = ±1`` đúng tại hai mẫu có nhãn khó nhất.
  Nhờ vậy biên của bốn loại lỗi so được với nhau và lấy max có nghĩa.

    score            = sigmoid(max_d m_d)          (0..1, càng cao càng cần review)
    pred_defect_type = argmax_d m_d nếu max > 0, ngược lại "none"
    pred_action      = RECAPTURE nếu max > 0 (tức score > 0.5), ngược lại KEEP

Feature cho từng loại (feature do TV3 / TV4 tính, TV5 chỉ ghép):

* ``blur``        log10(lap_norm) thấp. ``lap_norm`` đã chuẩn hoá theo camera
                  (TV3), nên cam_B quang học mềm không bị báo nhầm.
* ``compression`` log10(blockiness) cao (TV3).
* ``dark``        luma_rel thấp, ngưỡng **frozen của TV4** (z-score trong nhóm
                  light × camera, nên ảnh đêm hữu ích không bị loại).
* ``glare``       max của hai đường:
                  - *cháy sáng* (TV4): một tầng sat_* vượt ngưỡng VÀ blob cháy
                    liền khối đủ lớn (fuzzy AND = min của hai biên). Dấu của biên
                    này được ép khớp tuyệt đối với ``tv4_glare_evidence``.
                  - *veiling glare*: luma_rel cao bất thường. Glare trong
                    ``src/common/corruptions.py`` là một vệt sáng Gaussian lớn;
                    trên ảnh đêm nó làm sáng cả vùng nhưng không làm pixel cháy
                    (> 245), nên các feature sat_* không thấy. Ngưỡng là ảnh gương
                    của ngưỡng dark TV4 (+|tv4_dark_luma_rel|): sáng hơn nhóm đúng
                    bằng mức "tối hơn nhóm" mà TV4 đã calibrate.

Protocol
--------
Ngưỡng và scale được fit bằng :func:`fit` trên ``reports/mini_gold_dev.csv``
(chỉ split dev) và **freeze** vào ``configs/tv5_score.yaml``. Pipeline chỉ đọc
file freeze, không đọc nhãn lúc chạy, không đọc ``fixtures/*/gt/``.
Tái tạo file freeze: ``python -m src.t02.score --fit`` (chỉ làm trước khi chấm eval).
"""
from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd

from .feat_exposure import (
    DEFAULT_CALIBRATION,
    DEFAULT_MINI_GOLD,
    apply_calibration,
    load_calibration,
    load_mini_gold,
)

__all__ = [
    "DEFECT_TYPES",
    "DEFAULT_SCORE_CONFIG",
    "REQUIRED_FEATURES",
    "combine",
    "fit",
    "load_params",
    "margins",
]

#: Thứ tự cũng là thứ tự ưu tiên khi hai biên bằng nhau tuyệt đối.
DEFECT_TYPES: tuple[str, ...] = ("dark", "glare", "blur", "compression")

DEFAULT_SCORE_CONFIG = "configs/tv5_score.yaml"

#: Feature TV5 cần từ TV3 (sharpness) và TV4 (exposure).
REQUIRED_FEATURES: tuple[str, ...] = (
    "lap_var", "blockiness",                                     # TV3
    "luma_rel", "sat_frac_rel", "sat_tile_max_rel", "sat_blob_frac_rel", "sat_blob_px",  # TV4
)

_EPS = 1e-6
_REPO_ROOT = Path(__file__).resolve().parents[2]

# (tên ngưỡng TV4, feature) cho ba tầng cháy sáng
_GLARE_TIERS: tuple[tuple[str, str], ...] = (
    ("tv4_glare_sat_frac_rel", "sat_frac_rel"),
    ("tv4_glare_sat_tile_max_rel", "sat_tile_max_rel"),
    ("tv4_glare_sat_blob_frac_rel", "sat_blob_frac_rel"),
)


# ---------------------------------------------------------------- helpers

def _resolve(path: str | Path) -> Path:
    """Đường dẫn tương đối: thử theo thư mục hiện tại, rồi theo gốc repo."""
    p = Path(path)
    if p.is_absolute() or p.exists():
        return p
    return _REPO_ROOT / p


def load_params(path: str | Path = DEFAULT_SCORE_CONFIG) -> dict:
    """Đọc ngưỡng TV5 đã freeze (``configs/tv5_score.yaml``)."""
    import yaml

    p = _resolve(path)
    if not p.is_file():
        raise FileNotFoundError(
            f"không thấy {p}: chạy `python -m src.t02.score --fit` trên mini gold dev để tạo (rồi freeze)"
        )
    with open(p, "r", encoding="utf-8") as fh:
        return yaml.safe_load(fh) or {}


def _sigmoid(x: np.ndarray) -> np.ndarray:
    return 1.0 / (1.0 + np.exp(-np.clip(x, -50, 50)))


def _lap_norm(df: pd.DataFrame) -> pd.Series:
    """``lap_norm`` của TV3; nếu thiếu thì tự chuẩn hoá theo trung vị dev của camera."""
    if "lap_norm" in df.columns:
        return df["lap_norm"].astype(float)
    dev = df[df["split"].astype(str) == "dev"]
    med = dev.groupby("camera_id")["lap_var"].median()
    fallback = float(dev["lap_var"].median())
    return df["lap_var"] / df["camera_id"].map(med).fillna(fallback)


def _gap(pos: np.ndarray, neg: np.ndarray, higher_is_defect: bool) -> tuple[float, float]:
    """Ngưỡng ở giữa khe hở và scale = nửa khe hở (1-D max-margin).

    Nếu hai tập chồng nhau (không tách được) thì dùng giữa hai trung vị và
    nửa khoảng cách hai trung vị, rồi ghi lại trong provenance.
    """
    pos = np.asarray(pos, dtype=float)
    neg = np.asarray(neg, dtype=float)
    pos, neg = pos[np.isfinite(pos)], neg[np.isfinite(neg)]
    if len(pos) == 0 or len(neg) == 0:
        raise ValueError("cần ít nhất 1 ảnh lỗi và 1 ảnh sạch có nhãn để fit")
    sign = 1.0 if higher_is_defect else -1.0
    p, n = sign * pos, sign * neg
    lo, hi = float(n.max()), float(p.min())
    if hi <= lo:  # không tách được -> giữa hai trung vị
        lo, hi = float(np.median(n)), float(np.median(p))
    thr = sign * (lo + hi) / 2.0
    scale = max(abs(hi - lo) / 2.0, _EPS)
    return float(thr), float(scale)


def _check(feats: pd.DataFrame) -> None:
    missing = [c for c in ("image_id", *REQUIRED_FEATURES) if c not in feats.columns]
    if missing:
        raise ValueError(f"score.combine (TV5) thiếu feature {missing} (xem src/t02/README.md)")
    bad = feats[list(REQUIRED_FEATURES)].isna().any(axis=1)
    if bad.any():
        ids = feats.loc[bad, "image_id"].astype(str).tolist()
        raise ValueError(f"score.combine (TV5): feature bị NaN ở {len(ids)} ảnh, ví dụ {ids[:5]}")


# ---------------------------------------------------------------- margins

def _glare_burn_margin(df: pd.DataFrame, tv4: dict) -> pd.Series:
    """Đường cháy sáng của TV4 thành biên liên tục: min(tầng mạnh nhất, blob)."""
    tiers = []
    for key, col in _GLARE_TIERS:
        thr = float(tv4[key])
        tiers.append((df[col].astype(float) - thr) / max(abs(thr), _EPS))   # độ vượt tương đối
    tier = pd.concat(tiers, axis=1).max(axis=1)
    blob_thr = max(float(tv4["tv4_glare_min_blob_px"]), float(tv4.get("tv4_glare_blob_px_floor", 40.0)))
    blob = np.log10(df["sat_blob_px"].astype(float).clip(lower=1.0)) - np.log10(blob_thr)
    return pd.concat([tier, blob], axis=1).min(axis=1), tier, blob


def margins(feats: pd.DataFrame, p: dict, tv4: dict) -> pd.DataFrame:
    """Biên ``margin_<loại>`` cho từng ảnh (> 0 = loại lỗi đó bật)."""
    df = feats
    if "tv4_glare_evidence" not in df.columns or "tv4_dark_evidence" not in df.columns:
        df = apply_calibration(df, tv4)

    out = pd.DataFrame({"image_id": df["image_id"].astype(str)}, index=df.index)

    b = p["blur"]
    out["margin_blur"] = (b["log10_lap_norm_thr"] - np.log10(_lap_norm(df).clip(lower=_EPS))) / b["scale"]

    c = p["compression"]
    out["margin_compression"] = (np.log10(df["blockiness"].astype(float).clip(lower=_EPS)) - c["log10_blockiness_thr"]) / c["scale"]

    d = p["dark"]
    m_dark = (float(tv4["tv4_dark_luma_rel"]) - df["luma_rel"].astype(float)) / d["scale"]
    # ép dấu khớp tuyệt đối với quyết định frozen của TV4 (chỉ khác nhau khi bằng đúng ngưỡng)
    ev = df["tv4_dark_evidence"].astype(float) > 0
    out["margin_dark"] = np.where(ev, np.maximum(m_dark, _EPS), np.minimum(m_dark, 0.0))

    g = p["glare"]
    burn, _, _ = _glare_burn_margin(df, tv4)
    ev = df["tv4_glare_evidence"].astype(float) > 0
    burn = np.where(ev, np.maximum(burn / g["burn_scale"], _EPS), np.minimum(burn / g["burn_scale"], 0.0))
    veil = (df["luma_rel"].astype(float) - g["veil_luma_rel_thr"]) / d["scale"]
    out["margin_glare"] = np.maximum(burn, veil)
    out["glare_route"] = np.where(burn >= veil, "burn", "veil")
    return out


# ---------------------------------------------------------------- reason text

def _describe(t: str, row: pd.Series, p: dict, tv4: dict) -> str:
    """Một câu ngắn (ASCII, để Excel mở CSV không lỗi font) cho reviewer."""
    if t == "blur":
        return f"blur lap_norm={row['lap_norm']:.3f} (thr<{p['blur']['lap_norm_thr']:.3f})"
    if t == "compression":
        return f"compression blockiness={row['blockiness']:.2f} (thr>{p['compression']['blockiness_thr']:.2f})"
    if t == "dark":
        return f"dark luma_rel={row['luma_rel']:.2f} (thr<{float(tv4['tv4_dark_luma_rel']):.2f})"
    if row["glare_route"] == "veil":
        return f"glare(veil) luma_rel={row['luma_rel']:.2f} (thr>{p['glare']['veil_luma_rel_thr']:.2f})"
    return (f"glare(burn) sat_tile_max_rel={row['sat_tile_max_rel']:.2f} "
            f"sat_blob_px={row['sat_blob_px']:.0f}")


# ---------------------------------------------------------------- public API

def combine(feats: pd.DataFrame, meta: pd.DataFrame, params: dict | None = None) -> pd.DataFrame:
    """Gộp feature TV3 + TV4 thành điểm review (xem docstring module)."""
    params = dict(params or {})
    p = load_params(params.get("tv5_score_config", DEFAULT_SCORE_CONFIG))
    tv4 = load_calibration(_resolve(params.get("tv4_calibration", DEFAULT_CALIBRATION)))

    df = feats.copy()
    df["image_id"] = df["image_id"].astype(str)
    if df["image_id"].duplicated().any():
        raise ValueError("score.combine (TV5): feats có image_id bị trùng")
    m = meta.astype({"image_id": str})
    add = [c for c in ("camera_id", "light", "split") if c not in df.columns and c in m.columns]
    if add:
        df = df.merge(m[["image_id", *add]], on="image_id", how="left", validate="one_to_one")
    _check(df)
    df["lap_norm"] = _lap_norm(df)

    mg = margins(df, p, tv4)
    cols = [f"margin_{t}" for t in DEFECT_TYPES]
    mat = mg[cols].to_numpy(dtype=float)
    best = mat.max(axis=1)
    top = np.array(DEFECT_TYPES)[mat.argmax(axis=1)]          # argmax: tie -> thứ tự DEFECT_TYPES

    out = pd.DataFrame({"image_id": df["image_id"].to_numpy()})
    out["score"] = _sigmoid(best)
    out["pred_defect_type"] = np.where(best > 0, top, "none")
    out["pred_action"] = np.where(best > 0, "RECAPTURE", "KEEP")
    out["candidate_type"] = top
    for c in cols:
        out[c] = mat[:, cols.index(c)].round(6)

    info = pd.concat([df.reset_index(drop=True), mg[["glare_route"]].reset_index(drop=True)], axis=1)
    reasons = []
    for i, row in info.iterrows():
        fired = [t for t, v in zip(DEFECT_TYPES, mat[i]) if v > 0]
        if fired:
            order = sorted(fired, key=lambda t: -mat[i][DEFECT_TYPES.index(t)])
            reasons.append(" | ".join(_describe(t, row, p, tv4) for t in order))
        else:
            reasons.append("ok; closest: " + _describe(top[i], row, p, tv4))
    out["reason"] = reasons
    return out


def fit(feats: pd.DataFrame, labels: pd.DataFrame, tv4: dict) -> dict:
    """Fit ngưỡng + scale TV5 trên mini gold **dev**. Không bao giờ gọi trên eval.

    Mỗi loại lỗi: positive = ảnh có nhãn loại đó, negative = ảnh nhãn ``none``.
    Ảnh lỗi loại khác không dùng cho loại đang fit (giống TV4).
    """
    lab = labels.copy()
    lab["image_id"] = lab["image_id"].astype(str)
    if "split" in lab.columns and (lab["split"].astype(str) != "dev").any():
        raise ValueError("fit (TV5): nhãn chỉ được lấy từ split dev")
    df = feats.copy()
    df["image_id"] = df["image_id"].astype(str)
    df["lap_norm"] = _lap_norm(df)
    if "tv4_glare_evidence" not in df.columns:
        df = apply_calibration(df, tv4)
    df = df.merge(lab[["image_id", "defect_type"]], on="image_id", how="inner")
    neg = df[df["defect_type"] == "none"]

    def pos(t: str) -> pd.DataFrame:
        return df[df["defect_type"] == t]

    blur_thr, blur_scale = _gap(np.log10(pos("blur")["lap_norm"]), np.log10(neg["lap_norm"]), higher_is_defect=False)
    comp_thr, comp_scale = _gap(np.log10(pos("compression")["blockiness"]), np.log10(neg["blockiness"]), higher_is_defect=True)

    # dark: ngưỡng là của TV4 (frozen); TV5 chỉ fit scale từ khe hở quanh ngưỡng đó
    dark_thr = float(tv4["tv4_dark_luma_rel"])
    dark_pos = pos("dark")["luma_rel"].to_numpy(float)
    dark_scale = max((float(neg["luma_rel"].min()) - float(dark_pos.max())) / 2.0, _EPS)

    burn, _, _ = _glare_burn_margin(df, tv4)
    burn_pos = burn[df["defect_type"] == "glare"].to_numpy(float)
    burn_neg = burn[df["defect_type"] == "none"].to_numpy(float)
    burn_scale = (float(burn_pos.min()) - float(burn_neg.max())) / 2.0 if len(burn_pos) else 1.0
    burn_scale = burn_scale if burn_scale > _EPS else 1.0

    n = {t: int((df["defect_type"] == t).sum()) for t in (*DEFECT_TYPES, "none")}
    return {
        "blur": {
            "lap_norm_thr": float(10 ** blur_thr), "log10_lap_norm_thr": blur_thr, "scale": blur_scale,
        },
        "compression": {
            "blockiness_thr": float(10 ** comp_thr), "log10_blockiness_thr": comp_thr, "scale": comp_scale,
        },
        "dark": {"luma_rel_thr_from_tv4": dark_thr, "scale": dark_scale},
        "glare": {"burn_scale": float(burn_scale), "veil_luma_rel_thr": abs(dark_thr)},
        "tv5_fit": {
            "source": DEFAULT_MINI_GOLD, "split": "dev", "n_labelled": int(len(df)),
            **{f"n_{k}": v for k, v in n.items()},
            "tv4_calibration": DEFAULT_CALIBRATION,
            "note": ("thr = giua khe ho anh loi / anh sach gan nhat; scale = nua khe ho. "
                     "dark + glare(burn) dung nguong frozen cua TV4; veil = guong cua nguong dark TV4. "
                     "Mau nho (blur 3, compression 2, dark 3, glare 1): nguong la uoc luong tho."),
        },
    }


def _fit_and_freeze(out_path: str | Path, input_dir: str | Path) -> dict:
    """Tính feature TV3 + TV4 trên fixture, fit trên mini gold dev, ghi YAML."""
    import yaml

    from . import feat_exposure, feat_sharpness

    input_dir = Path(input_dir)
    images = pd.read_csv(input_dir / "images.csv", dtype={"image_id": str})
    sharp = feat_sharpness.compute(images, input_dir).astype({"image_id": str})
    expo = feat_exposure.compute(images, input_dir).astype({"image_id": str})
    feats = sharp.merge(expo, on="image_id", validate="one_to_one")
    tv4 = load_calibration(_resolve(DEFAULT_CALIBRATION))
    params = fit(feats, load_mini_gold(_resolve(DEFAULT_MINI_GOLD)), tv4)

    header = (
        "# TV5 · Tham so cham diem (score.py) — FROZEN\n"
        "# Fit bang `python -m src.t02.score --fit` tren reports/mini_gold_dev.csv (chi split dev).\n"
        "# Khong sua tay, khong fit lai sau khi da xem ket qua split eval.\n"
        "# margin = (do lech feature so voi thr, theo huong 'cang loi') / scale; score = sigmoid(max margin).\n\n"
    )
    out_path = _resolve(out_path) if not Path(out_path).is_absolute() else Path(out_path)
    out_path.write_text(header + yaml.safe_dump(params, sort_keys=False, allow_unicode=True), encoding="utf-8")
    return params


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description="TV5 · fit + freeze ngưỡng chấm điểm T02 trên mini gold dev")
    ap.add_argument("--fit", action="store_true", help="fit trên reports/mini_gold_dev.csv và ghi file YAML")
    ap.add_argument("--input", default="fixtures/t02_raw_qc")
    ap.add_argument("--out", default=DEFAULT_SCORE_CONFIG)
    a = ap.parse_args()
    if not a.fit:
        ap.error("dùng --fit để tạo configs/tv5_score.yaml")
    print(_fit_and_freeze(a.out, a.input))
