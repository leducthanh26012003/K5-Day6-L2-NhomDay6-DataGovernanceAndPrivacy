"""TV4 — exposure/glare features + mini-gold calibration for T02.

The calibration tests use the hand-made dev labels in
``reports/mini_gold_dev.csv`` (20 frames). No instructor ground truth is
involved: the labels are the team's own dev annotations.
"""
import numpy as np
import pandas as pd
import pytest

from src.common.corruptions import apply
from src.common.image_ops import load_rgb
from src.t02.feat_exposure import (
    CORE_COLUMNS,
    FEATURE_COLUMNS,
    apply_calibration,
    calibrate,
    compute,
    load_mini_gold,
)

FIXTURE = "fixtures/t02_raw_qc"
MINI_GOLD = "reports/mini_gold_dev.csv"


@pytest.fixture(scope="module")
def images() -> pd.DataFrame:
    return pd.read_csv(f"{FIXTURE}/images.csv")


@pytest.fixture(scope="module")
def features(images: pd.DataFrame) -> pd.DataFrame:
    return compute(images, FIXTURE)


@pytest.fixture(scope="module")
def gold() -> pd.DataFrame:
    return load_mini_gold(MINI_GOLD)


@pytest.fixture(scope="module")
def calibrated(features: pd.DataFrame, gold: pd.DataFrame) -> pd.DataFrame:
    return apply_calibration(features, calibrate(features, gold))


# ---------------------------------------------------------------- interface

def test_interface_columns_present(features: pd.DataFrame):
    """The column names agreed in src/t02/README.md must exist."""
    for col in CORE_COLUMNS:
        assert col in features.columns
    assert "luma_rel" in features.columns


def test_one_row_per_image(features: pd.DataFrame, images: pd.DataFrame):
    assert len(features) == 96 == len(images)
    assert features.image_id.is_unique
    assert features.image_id.nunique() == images.image_id.nunique()


def test_features_are_finite(features: pd.DataFrame):
    numeric = features.select_dtypes(include=[np.number])
    assert numeric.isna().sum().sum() == 0
    assert np.isfinite(numeric.to_numpy()).all()


def test_degenerate_images_do_not_crash(tmp_path):
    """An all-black and an all-white frame must not break the extractor."""
    from PIL import Image

    for name, colour in (("black.jpg", (0, 0, 0)), ("white.jpg", (255, 255, 255))):
        Image.new("RGB", (32, 16), colour).save(tmp_path / name)

    out = compute(
        pd.DataFrame({"image_id": ["black", "white"], "path": ["black.jpg", "white.jpg"]}),
        tmp_path,
    )
    assert len(out) == 2
    numeric = out.select_dtypes(include=[np.number])
    assert np.isfinite(numeric.to_numpy()).all()


# ---------------------------------------------------------------- direction

def test_dark_increases_darkness():
    from src.t02.feat_exposure import _image_features
    clean = _image_features(load_rgb(f"{FIXTURE}/images/img_02_0.jpg"))
    dark = _image_features(apply("dark", load_rgb(f"{FIXTURE}/images/img_02_0.jpg"), severity=2))
    assert dark["luma_mean"] < clean["luma_mean"]
    assert dark["dark_frac"] > clean["dark_frac"]
    assert dark["dark_tile_min"] < clean["dark_tile_min"]


def test_glare_raises_local_and_global_measures():
    from src.t02.feat_exposure import _image_features
    clean = _image_features(load_rgb(f"{FIXTURE}/images/img_02_0.jpg"))
    glare = _image_features(apply("glare", load_rgb(f"{FIXTURE}/images/img_02_0.jpg"), severity=1))
    assert glare["sat_frac"] > clean["sat_frac"]
    assert glare["sat_tile_max"] > clean["sat_tile_max"]
    assert glare["sat_blob_px"] > clean["sat_blob_px"]


def test_jpeg_does_not_look_like_an_exposure_defect():
    from src.t02.feat_exposure import _image_features
    clean = _image_features(load_rgb(f"{FIXTURE}/images/img_02_0.jpg"))
    jpeg = _image_features(apply("jpeg", load_rgb(f"{FIXTURE}/images/img_02_0.jpg"), severity=2))
    assert jpeg["luma_mean"] == pytest.approx(clean["luma_mean"], abs=3.0)
    assert jpeg["sat_frac"] == pytest.approx(clean["sat_frac"], abs=0.005)


# ---------------------------------------------------------------- calibration

def test_mini_gold_is_dev_only(gold: pd.DataFrame):
    assert (gold["split"] == "dev").all()


def test_calibration_uses_relative_thresholds(features: pd.DataFrame, gold: pd.DataFrame):
    params = calibrate(features, gold)
    assert "tv4_dark_luma_rel" in params
    assert np.isfinite(params["tv4_dark_luma_rel"])
    # fitted on dev only
    assert params["tv4_fit"]["split"] == "dev"


def test_calibration_recalls_all_tv4_defects(calibrated: pd.DataFrame, gold: pd.DataFrame):
    """Every dark/glare frame in the mini gold must be detected."""
    merged = calibrated.merge(gold[["image_id", "defect_type"]], on="image_id", how="inner")
    tv4 = merged[merged.defect_type.isin(["dark", "glare"])]
    assert len(tv4) == 4
    missed = tv4[tv4.tv4_evidence == 0].image_id.tolist()
    assert missed == [], f"missed TV4 defects: {missed}"


def test_calibration_has_no_false_alarm_on_clean_frames(calibrated: pd.DataFrame, gold: pd.DataFrame):
    merged = calibrated.merge(gold[["image_id", "defect_type"]], on="image_id", how="inner")
    clean = merged[merged.defect_type == "none"]
    fa = clean[clean.tv4_evidence > 0].image_id.tolist()
    assert fa == [], f"false alarms on clean frames: {fa}"


def test_usable_night_frames_are_not_flagged(calibrated: pd.DataFrame, gold: pd.DataFrame):
    """The headline failure mode: a usable night frame must survive."""
    merged = calibrated.merge(
        gold[["image_id", "defect_type", "light"]].rename(columns={"light": "light_gold"}),
        on="image_id",
        how="inner",
    )
    night_ok = merged[(merged.light_gold == "night") & (merged.defect_type == "none")]
    assert len(night_ok) >= 2
    assert (night_ok.tv4_evidence == 0).all()


def test_local_glare_is_detected(calibrated: pd.DataFrame):
    """img_05_3 is a *local* blow-out: a global saturation ratio barely sees it."""
    row = calibrated[calibrated.image_id == "img_05_3"].iloc[0]
    # the global view alone would miss it ...
    assert row.sat_frac_rel < 5.0
    # ... but the local view must not
    assert row.sat_tile_max_rel > 4.0
    assert row.tv4_evidence == 1.0


def test_scattered_speckle_is_not_glare(calibrated: pd.DataFrame):
    """Night frames with scattered burned pixels must not be called glare."""
    for gid in ("img_03_2", "img_06_3"):
        row = calibrated[calibrated.image_id == gid].iloc[0]
        assert row.sat_blob_px < 100        # tiny, fragmented blobs
        assert row.tv4_glare_evidence == 0.0


def test_calibration_does_not_read_ground_truth(monkeypatch):
    """The module must never open anything under fixtures/*/gt/."""
    import src.t02.feat_exposure as fe
    opened = []
    real_open = open

    def spy(file, *a, **k):
        opened.append(str(file))
        return real_open(file, *a, **k)

    monkeypatch.setattr("builtins.open", spy)
    images = pd.read_csv(f"{FIXTURE}/images.csv")
    feats = compute(images, FIXTURE)
    calibrate(feats, load_mini_gold(MINI_GOLD))
    monkeypatch.undo()
    assert not [p for p in opened if "gt" in p.split("\\")[-2:-1] or "/gt/" in p.replace("\\", "/")]
