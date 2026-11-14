"""TV5 — score.combine (gộp feature) + review_step.run (human-in-the-loop) cho T02.

Nhãn dùng ở đây là mini gold dev của nhóm (reports/mini_gold_dev.csv), không phải
ground truth của giảng viên. Các kiểm tra trên mini gold là in-sample (ngưỡng được
fit trên chính tập này): chúng bảo vệ khỏi hồi quy, không phải kết quả đánh giá.
"""
import json
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from src.common.context import RunContext
from src.common.review import AuditLog, make_review_queue
from src.t02 import score
from src.t02.review_step import REVIEWED_FILE, resolve_override, run

FIXTURE = Path("fixtures/t02_raw_qc")


# ---------------------------------------------------------------- score.combine

@pytest.fixture(scope="module")
def images() -> pd.DataFrame:
    return pd.read_csv(FIXTURE / "images.csv", dtype={"image_id": str})


@pytest.fixture(scope="module")
def feats(images: pd.DataFrame) -> pd.DataFrame:
    sharp = pytest.importorskip("src.t02.feat_sharpness", reason="cần feat_sharpness.py của TV3")
    from src.t02 import feat_exposure

    s = sharp.compute(images, FIXTURE).astype({"image_id": str})
    e = feat_exposure.compute(images, FIXTURE).astype({"image_id": str})
    return s.merge(e, on="image_id", validate="one_to_one")


@pytest.fixture(scope="module")
def scored(feats: pd.DataFrame, images: pd.DataFrame) -> pd.DataFrame:
    return score.combine(feats, images, {})


def test_combine_contract(scored: pd.DataFrame, feats: pd.DataFrame, images: pd.DataFrame):
    assert len(scored) == 96 and scored.image_id.is_unique
    for col in ["image_id", "score", "pred_defect_type", "pred_action", "reason"]:
        assert col in scored.columns
    assert scored.score.between(0, 1).all()
    assert set(scored.pred_action) <= {"KEEP", "RECAPTURE"}
    assert set(scored.pred_defect_type) <= {"none", *score.DEFECT_TYPES}
    # KEEP <-> none, RECAPTURE <-> một loại lỗi; RECAPTURE <-> score > 0.5
    assert ((scored.pred_action == "KEEP") == (scored.pred_defect_type == "none")).all()
    assert ((scored.pred_action == "RECAPTURE") == (scored.score > 0.5)).all()
    # tái lập: chạy lại ra y hệt
    pd.testing.assert_frame_equal(scored, score.combine(feats, images, {}))


def test_agrees_with_tv4_frozen_decisions(scored: pd.DataFrame, feats: pd.DataFrame):
    from src.t02.feat_exposure import apply_calibration, load_calibration

    tv4 = apply_calibration(feats, load_calibration()).set_index("image_id")
    m = scored.set_index("image_id")
    assert ((m.margin_dark > 0) == (tv4.tv4_dark_evidence > 0)).all()
    # glare của TV5 = glare(burn) của TV4 HOẶC veiling glare -> mọi glare TV4 đều được giữ
    assert (m.loc[tv4.tv4_glare_evidence > 0, "margin_glare"] > 0).all()


def test_mini_gold_in_sample_and_failure_modes(scored: pd.DataFrame):
    from src.t02.feat_exposure import load_mini_gold

    gold = load_mini_gold().merge(scored, on="image_id")
    assert (gold.pred_defect_type == gold.defect_type).all(), gold[["image_id", "defect_type", "pred_defect_type"]]
    by_id = scored.set_index("image_id")
    # failure mode 1: ảnh đêm hữu ích không bị loại
    assert (by_id.loc[["img_18_0", "img_18_1"], "pred_action"] == "KEEP").all()
    # failure mode 3: cam_B quang học mềm không bị báo blur
    assert (by_id.loc[["img_09_0", "img_12_0"], "pred_action"] == "KEEP").all()
    # failure mode 2: glare cục bộ (vùng biển số) được bắt
    assert by_id.loc["img_18_2", "pred_defect_type"] == "glare"


def test_frozen_config_matches_dev_fit(feats: pd.DataFrame):
    """configs/tv5_score.yaml phải đúng là kết quả fit trên mini gold dev (không sửa tay)."""
    from src.t02.feat_exposure import load_calibration, load_mini_gold

    fitted = score.fit(feats, load_mini_gold(), load_calibration())
    frozen = score.load_params()
    for block in ("blur", "compression", "dark", "glare"):
        for key, value in fitted[block].items():
            assert frozen[block][key] == pytest.approx(value, rel=1e-9), f"{block}.{key}"


def _toy_feats(n: int = 3) -> pd.DataFrame:
    return pd.DataFrame({
        "image_id": [f"x{i}" for i in range(n)], "camera_id": "cam_A", "light": "day", "split": "dev",
        "lap_var": 900.0, "lap_norm": 1.0, "blockiness": 1.1,
        "luma_rel": 0.0, "sat_frac_rel": 0.0, "sat_tile_max_rel": 0.0, "sat_blob_frac_rel": 0.0, "sat_blob_px": 0.0,
    })


def test_combine_fails_clearly_on_missing_or_nan_feature():
    toy = _toy_feats()
    meta = toy[["image_id", "camera_id", "light", "split"]]
    assert (score.combine(toy, meta, {}).pred_action == "KEEP").all()          # ảnh sạch -> KEEP
    with pytest.raises(ValueError, match="thiếu feature"):
        score.combine(toy.drop(columns=["blockiness"]), meta, {})
    bad = toy.copy()
    bad.loc[1, "luma_rel"] = np.nan
    with pytest.raises(ValueError, match="NaN"):
        score.combine(bad, meta, {})


def test_fit_refuses_eval_labels():
    labels = pd.DataFrame({"image_id": ["x0"], "split": ["eval"], "defect_type": ["blur"]})
    with pytest.raises(ValueError, match="dev"):
        score.fit(_toy_feats(), labels, {})


# ---------------------------------------------------------------- review_step.run

def _ctx(tmp_path: Path, budget: int = 3) -> RunContext:
    return RunContext(track="t02", input_dir=FIXTURE, output_dir=tmp_path, config={"params": {"budget": budget}},
                      seed=42, rng=np.random.default_rng(42), run_name="test",
                      audit=AuditLog(tmp_path / "audit_log.jsonl"))


def _toy_scored() -> pd.DataFrame:
    return pd.DataFrame({
        "image_id": ["img_11_2", "img_23_3", "img_19_3", "img_00_1"],
        "score": [0.9, 0.8, 0.7, 0.2],
        "pred_defect_type": ["compression", "dark", "blur", "none"],
        "pred_action": ["RECAPTURE", "RECAPTURE", "RECAPTURE", "KEEP"],
        "reason": ["r"] * 4,
        "candidate_type": ["compression", "dark", "blur", "glare"],
        "path": ["images/img_11_2.jpg", "images/img_23_3.jpg", "images/img_19_3.jpg", "images/img_00_1.jpg"],
        "light": ["day"] * 4,
    })


def _write_review(ctx: RunContext, scored: pd.DataFrame, decisions: dict, reviewer: str = "tv5") -> None:
    q = make_review_queue(scored, id_col="image_id", score_col="score", budget=int(ctx.param("budget")),
                          reason_col="reason", action_col="pred_action", slice_col="light", evidence_col="path")
    for item, (dec, val) in decisions.items():
        q.loc[q.item_id == item, ["reviewer", "reviewer_decision", "override_value", "review_note"]] = [reviewer, dec, val, "note"]
    q.to_csv(ctx.output_dir / REVIEWED_FILE, index=False)


def _audit(tmp_path: Path) -> list[dict]:
    return [json.loads(line) for line in (tmp_path / "audit_log.jsonl").read_text(encoding="utf-8").splitlines()]


def test_review_pending_writes_contact_sheet_and_changes_nothing(tmp_path: Path):
    ctx = _ctx(tmp_path)
    s = _toy_scored()
    final = run(ctx, s)
    assert (final.pred_action == s.pred_action).all() and (final.score == s.score).all()
    assert (tmp_path / "contact_sheet.html").is_file()
    assert ctx.stats["review_done"] is False
    assert _audit(tmp_path)[-1]["action"] == "review:pending"


def test_reject_and_edit_apply_overrides_with_audit_but_keep_score(tmp_path: Path):
    ctx = _ctx(tmp_path)
    s = _toy_scored()
    _write_review(ctx, s, {"img_11_2": ("reject", ""), "img_23_3": ("edit", "glare"), "img_19_3": ("approve", "")})
    final = run(ctx, s).set_index("image_id")

    assert (final.loc["img_11_2", ["pred_action", "pred_defect_type"]] == ["KEEP", "none"]).all()
    assert (final.loc["img_23_3", ["pred_action", "pred_defect_type"]] == ["RECAPTURE", "glare"]).all()
    assert final.loc["img_19_3", "pred_action"] == "RECAPTURE"
    assert final.loc["img_11_2", "model_action"] == "RECAPTURE"                # quyết định gốc vẫn giữ
    assert final.score.tolist() == s.score.tolist()                            # metric không bị người sửa
    assert ctx.stats["review_done"] is True and ctx.stats["n_overrides"] == 2

    events = {(e["action"], e["item_id"]) for e in _audit(tmp_path)}
    assert ("override:reject", "img_11_2") in events
    assert ("override_type:edit", "img_23_3") in events
    assert ("review:approve", "img_19_3") in events
    assert all(e["actor"] == "tv5" for e in _audit(tmp_path))


def test_reject_keep_uses_candidate_type(tmp_path: Path):
    assert resolve_override("reject", "", "KEEP", "none", "glare") == ("RECAPTURE", "glare")
    assert resolve_override("edit", "keep", "RECAPTURE", "blur", "blur") == ("KEEP", "none")
    assert resolve_override("escalate", "", "RECAPTURE", "blur", "blur") == ("RECAPTURE", "blur")


def test_partial_review_is_not_review_done(tmp_path: Path):
    ctx = _ctx(tmp_path)
    s = _toy_scored()
    _write_review(ctx, s, {"img_11_2": ("approve", "")})
    run(ctx, s)
    assert ctx.stats["review_done"] is False and ctx.stats["n_queue_unreviewed"] == 2


@pytest.mark.parametrize("decisions, message", [
    ({"img_11_2": ("edit", "")}, "cần override_value"),
    ({"img_11_2": ("edit", "BLURRY")}, "không hợp lệ"),
])
def test_bad_review_values_fail_clearly(tmp_path: Path, decisions: dict, message: str):
    ctx = _ctx(tmp_path)
    s = _toy_scored()
    _write_review(ctx, s, decisions)
    with pytest.raises(ValueError, match=message):
        run(ctx, s)


def test_review_without_reviewer_name_fails(tmp_path: Path):
    ctx = _ctx(tmp_path)
    s = _toy_scored()
    _write_review(ctx, s, {"img_11_2": ("approve", "")}, reviewer="")
    with pytest.raises(ValueError, match="reviewer"):
        run(ctx, s)


def test_review_of_unknown_image_fails(tmp_path: Path):
    ctx = _ctx(tmp_path)
    s = _toy_scored()
    _write_review(ctx, s, {"img_11_2": ("approve", "")})
    f = pd.read_csv(tmp_path / REVIEWED_FILE, dtype=str)
    f.loc[0, "item_id"] = "img_99_9"
    f.to_csv(tmp_path / REVIEWED_FILE, index=False)
    with pytest.raises(ValueError, match="không có trong lần chạy"):
        run(ctx, s)
