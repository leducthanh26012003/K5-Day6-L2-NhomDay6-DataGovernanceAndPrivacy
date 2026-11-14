"""T02 Raw QC Triage — pipeline của nhóm (TV1 nối dây).

Năm bước của contract, chạy bằng `python -m src.main ...`:
    01 load      đọc images.csv, kiểm hash manifest, kiểm đủ ảnh
    02 process   feature độ nét (TV3) + phơi sáng (TV4) -> chấm điểm (TV5)
    03 rank      top-B ảnh nghi lỗi nhất -> out/review_queue.csv
    04 review    contact sheet + áp quyết định reviewer + audit log (TV5)
    05 decide    GO / HOLD cho cả lô, có reason_code + owner

Giao diện giữa các module: src/t02/README.md. File này chỉ nối dây; thuật toán nằm trong src/t02/.
Không đọc fixtures/*/gt/, mọi ngưỡng chốt trên split dev, không gọi mạng.
"""
from __future__ import annotations

import importlib

import pandas as pd

from .common.context import RunContext
from .common.decision import Decision, write_decision
from .common.manifest import verify_manifest
from .common.review import make_review_queue, write_review_queue
from .metrics.tracks import get_track

IMAGE_COLUMNS = ["image_id", "path", "camera_id", "light", "split"]
SHARPNESS_COLUMNS = ["image_id", "lap_var", "lap_tile_min", "blockiness"]
EXPOSURE_COLUMNS = ["image_id", "luma_mean", "luma_p05", "luma_p95", "sat_frac", "sat_tile_max"]
SCORE_COLUMNS = ["image_id", "score", "pred_defect_type", "pred_action", "reason"]
MODULE_OWNERS = {"feat_sharpness": "TV3", "feat_exposure": "TV4", "score": "TV5", "review_step": "TV5"}


def _module(name: str):
    """Import src/t02/<name>.py. Nếu file chưa có thì báo rõ ai cần push (exit 3)."""
    full_name = f"{__package__}.t02.{name}"
    try:
        return importlib.import_module(full_name)
    except ModuleNotFoundError as exc:
        if exc.name != full_name:
            raise  # lỗi import bên trong module đó: để nguyên thông báo gốc
        raise NotImplementedError(
            f"src/t02/{name}.py chưa có ({MODULE_OWNERS[name]} cần push bản khung). Xem src/t02/README.md"
        ) from exc


def _check_frame(df: pd.DataFrame, who: str, columns: list[str], n_images: int) -> None:
    """Mỗi module phải trả đúng 1 dòng cho mỗi ảnh và đủ cột đã chốt."""
    missing = [c for c in columns if c not in df.columns]
    if missing:
        raise ValueError(f"{who} thiếu cột {missing} (xem src/t02/README.md)")
    if len(df) != n_images or df["image_id"].astype(str).duplicated().any():
        raise ValueError(f"{who} phải trả đúng {n_images} dòng, mỗi image_id một dòng (đang có {len(df)})")


# ---------------------------------------------------------------- 01 load
def load(ctx: RunContext) -> dict:
    manifest = ctx.input_dir / "manifest.csv"
    problems = verify_manifest(manifest) if manifest.exists() else [{"problem": "no_manifest"}]
    if problems:
        ctx.audit.log(actor="pipeline", action="provenance_problems", item_id=str(manifest), after=problems[:20])
        if ctx.config.get("strict_provenance", True):
            raise ValueError(f"provenance check failed ({len(problems)} problems), first: {problems[:3]}")

    images = pd.read_csv(ctx.input_dir / "images.csv", dtype={"image_id": str})
    missing_cols = [c for c in IMAGE_COLUMNS if c not in images.columns]
    if missing_cols:
        raise ValueError(f"images.csv thiếu cột {missing_cols}")
    if images["image_id"].duplicated().any():
        raise ValueError("images.csv có image_id bị trùng")
    missing_files = [p for p in images["path"] if not (ctx.input_dir / p).is_file()]
    if missing_files:
        raise FileNotFoundError(f"thiếu {len(missing_files)} ảnh trong {ctx.input_dir}, ví dụ {missing_files[:3]}")

    ctx.stats["n_dev"] = int((images["split"] == ctx.param("dev_split", "dev")).sum())
    return {"manifest": pd.read_csv(manifest), "images": images}


# ---------------------------------------------------------------- 02 process
def process(ctx: RunContext, data: dict) -> pd.DataFrame:
    images = data["images"]
    n = len(images)

    sharp = _module("feat_sharpness").compute(images, ctx.input_dir)
    _check_frame(sharp, "feat_sharpness.compute (TV3)", SHARPNESS_COLUMNS, n)
    expo = _module("feat_exposure").compute(images, ctx.input_dir)
    _check_frame(expo, "feat_exposure.compute (TV4)", EXPOSURE_COLUMNS, n)
    feats = sharp.astype({"image_id": str}).merge(expo.astype({"image_id": str}), on="image_id", validate="one_to_one")

    meta = images[IMAGE_COLUMNS]
    params = dict(ctx.config.get("params") or {})
    scored = _module("score").combine(feats, meta, params)
    _check_frame(scored, "score.combine (TV5)", SCORE_COLUMNS, n)

    scored = scored.astype({"image_id": str})
    extra_meta = [c for c in IMAGE_COLUMNS if c not in scored.columns or c == "image_id"]
    scored = scored.merge(meta[extra_meta], on="image_id", how="left", validate="one_to_one")
    extra_feats = [c for c in feats.columns if c not in scored.columns or c == "image_id"]
    scored = scored.merge(feats[extra_feats], on="image_id", how="left", validate="one_to_one")

    ctx.stats["n_recapture_before_review"] = int((scored["pred_action"] == "RECAPTURE").sum())
    return scored[SCORE_COLUMNS + [c for c in scored.columns if c not in SCORE_COLUMNS]]


# ---------------------------------------------------------------- 03 rank
def rank(ctx: RunContext, scored: pd.DataFrame) -> pd.DataFrame:
    """Top-B ảnh có score cao nhất vào hàng chờ review (B = params.budget, mặc định 20)."""
    budget = int(ctx.param("budget", 20))
    queue = make_review_queue(
        scored, id_col="image_id", score_col="score", budget=budget,
        reason_col="reason", action_col="pred_action", slice_col="light", evidence_col="path",
    )
    write_review_queue(queue, ctx.out("review_queue.csv"))
    return queue


# ---------------------------------------------------------------- 04 review
def review(ctx: RunContext, scored: pd.DataFrame, queue: pd.DataFrame) -> pd.DataFrame:
    """TV5: contact sheet + đọc out/.../review_queue_reviewed.csv + áp override + audit log."""
    ctx.stats["review_done"] = (ctx.output_dir / "review_queue_reviewed.csv").exists()
    final = _module("review_step").run(ctx, scored)
    _check_frame(final, "review_step.run (TV5)", SCORE_COLUMNS, len(scored))
    ctx.stats["n_recapture_after_review"] = int((final["pred_action"] == "RECAPTURE").sum())
    return final


# ---------------------------------------------------------------- 05 decide
def decide(ctx: RunContext, final: pd.DataFrame) -> Decision:
    """GO chỉ khi đã có người review và tỉ lệ ảnh cần chụp lại không vượt ngưỡng đã chốt trước."""
    spec = get_track(ctx.track)
    owner = ctx.config.get("owner", "TODO-owner")
    max_rate = float(ctx.param("max_recapture_rate", 0.25))
    n = len(final)
    n_recapture = int((final["pred_action"] == "RECAPTURE").sum())
    rate = n_recapture / n if n else 0.0
    evidence = [str(ctx.out(spec.pred_file)), str(ctx.out("review_queue.csv")), str(ctx.output_dir / "audit_log.jsonl")]

    if not ctx.stats.get("review_done"):
        return Decision(status="HOLD", reason_code="PENDING_HUMAN_REVIEW",
                        reason=f"{n_recapture}/{n} ảnh đề xuất RECAPTURE nhưng chưa có review_queue_reviewed.csv",
                        owner=owner, scope=ctx.track, evidence=evidence)
    if rate > max_rate:
        return Decision(status="HOLD", reason_code="HIGH_RECAPTURE_RATE",
                        reason=f"{n_recapture}/{n} ảnh ({rate:.0%}) cần chụp lại, vượt ngưỡng {max_rate:.0%}",
                        owner=owner, scope=ctx.track, evidence=evidence)
    return Decision(status="GO", reason_code="RECAPTURE_RATE_OK",
                    reason=f"{n_recapture}/{n} ảnh ({rate:.0%}) cần chụp lại, trong ngưỡng {max_rate:.0%}; đã review",
                    owner=owner, scope=ctx.track, evidence=evidence)


def run(ctx: RunContext) -> None:
    spec = get_track(ctx.track)
    data = load(ctx)
    scored = process(ctx, data)
    ctx.stats["n_items"] = len(scored)
    queue = rank(ctx, scored)
    final = review(ctx, scored, queue)
    final[spec.pred_columns + [c for c in final.columns if c not in spec.pred_columns]].to_csv(ctx.out(spec.pred_file), index=False)
    write_decision(decide(ctx, final), ctx.out("decision.json"), ctx.out("decision_log.csv"))
