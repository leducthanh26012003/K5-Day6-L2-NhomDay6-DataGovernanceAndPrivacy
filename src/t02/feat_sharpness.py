"""TV3 · Feature độ nét và lỗi nén cho T02."""
from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

from ..common.image_ops import laplacian_variance, load_rgb, to_gray

GRID_ROWS, GRID_COLS = 3, 4


def _tile_laplacian(img: np.ndarray) -> list[float]:
    # Chia ảnh thành lưới 3 hàng x 4 cột, tính độ nét từng ô.
    return [laplacian_variance(tile)
            for band in np.array_split(img, GRID_ROWS, axis=0)
            for tile in np.array_split(band, GRID_COLS, axis=1)]


def _blockiness(img: np.ndarray, block: int = 8) -> float:
    g = to_gray(img)
    dh = np.abs(np.diff(g, axis=1))   # chênh lệch giữa các cột liền kề
    dv = np.abs(np.diff(g, axis=0))   # chênh lệch giữa các hàng liền kề
    col_edge = np.arange(dh.shape[1]) % block == block - 1   # biên khối theo cột
    row_edge = np.arange(dv.shape[0]) % block == block - 1   # biên khối theo hàng
    eps = 1e-6                        # tránh chia cho 0
    ratio_h = dh[:, col_edge].mean() / (dh[:, ~col_edge].mean() + eps)
    ratio_v = dv[row_edge, :].mean() / (dv[~row_edge, :].mean() + eps)
    return float((ratio_h + ratio_v) / 2)


def compute(images: pd.DataFrame, root: Path) -> pd.DataFrame:
    root = Path(root)
    out = images[["image_id"]].astype(str).copy()

    lap, tile_min, blk = [], [], []
    for path in images["path"]:
        img = load_rgb(root / path)
        lap.append(laplacian_variance(img))
        tile_min.append(min(_tile_laplacian(img)))
        blk.append(_blockiness(img))
    out["lap_var"] = lap
    out["lap_tile_min"] = tile_min
    out["blockiness"] = blk

    # Chuẩn hoá theo camera, chỉ dùng trung vị trên tập dev.
    meta = images[["image_id", "camera_id", "split"]].astype({"image_id": str})
    out = out.merge(meta, on="image_id", how="left")
    dev = out[out["split"] == "dev"]
    med = dev.groupby("camera_id")["lap_var"].median()
    out["lap_norm"] = out["lap_var"] / out["camera_id"].map(med).fillna(dev["lap_var"].median())
    out.attrs["lap_median_by_camera_dev"] = med.to_dict()

    return out[["image_id", "lap_var", "lap_tile_min", "blockiness", "lap_norm"]]