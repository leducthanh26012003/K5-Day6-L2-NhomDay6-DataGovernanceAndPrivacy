"""T02 · TV5 — bước 04 review: contact sheet + quyết định reviewer + audit log.

Giao diện (chốt trong src/t02/README.md)
---------------------------------------
``run(ctx, scored) -> DataFrame``
    Trả lại ``scored`` (đủ 96 dòng) sau khi áp quyết định của reviewer, và ghi:

    * ``<output>/contact_sheet.html``  20 ảnh top-B (ảnh, score, loại lỗi, lý do,
      đề xuất) + dropdown quyết định + nút Export CSV (offline).
    * ``<output>/audit_log.jsonl``     mọi quyết định của reviewer (append-only).

Vòng human-in-the-loop
----------------------
1. ``make run`` lần 1 -> ``review_queue.csv`` + ``contact_sheet.html``; decide() = HOLD.
2. Reviewer mở contact sheet, nhập tên, chọn quyết định, bấm Export CSV.
3. Chép file tải về thành ``<output>/review_queue_reviewed.csv`` rồi ``make run`` lại.

Ý nghĩa quyết định (cột ``reviewer_decision``)
---------------------------------------------
=========  ==========================================================================
approve    đồng ý với đề xuất của model, không đổi gì.
reject     không đồng ý. ``override_value`` trống -> lật đề xuất (RECAPTURE <-> KEEP);
           có giá trị -> dùng giá trị đó (như ``edit``).
edit       bắt buộc có ``override_value``: ``KEEP`` / ``RECAPTURE`` hoặc một loại lỗi
           (``none, blur, glare, dark, compression``).
escalate   chưa quyết được, chuyển người có thẩm quyền; giữ đề xuất, ghi ``n_escalated``.
skip       reviewer chủ động bỏ qua; giữ đề xuất.
=========  ==========================================================================

Luôn giữ ``pred_action`` và ``pred_defect_type`` nhất quán: KEEP <-> ``none``,
RECAPTURE <-> một loại lỗi (nếu reviewer không nói loại nào thì dùng ``candidate_type``
của model).

``score`` **không bao giờ bị sửa**: metric (Recall@B / Precision@B) đo xếp hạng
của thuật toán trước khi người can thiệp (reports/metric_contract.md). Quyết định
gốc của model được giữ ở ``model_action`` / ``model_defect_type`` để diff được.
"""
from __future__ import annotations

import pandas as pd

from ..common.contact_sheet import render_contact_sheet
from ..common.context import RunContext
from ..common.review import AuditLog, apply_overrides, load_review_decisions, make_review_queue
from .score import DEFECT_TYPES

__all__ = ["REVIEWED_FILE", "run", "resolve_override"]

REVIEWED_FILE = "review_queue_reviewed.csv"
ACTIONS = ("KEEP", "RECAPTURE")
TYPES = ("none", *DEFECT_TYPES)
DISPLAY_FIELDS = ("rank", "item_id", "score", "pred_defect_type", "reason", "suggested_action", "slice")


def _flip(action: str) -> str:
    return "KEEP" if action == "RECAPTURE" else "RECAPTURE"


def resolve_override(decision: str, override_value: str, action: str, defect_type: str,
                     candidate_type: str) -> tuple[str, str]:
    """Từ quyết định reviewer -> (pred_action, pred_defect_type) mới.

    Raise ``ValueError`` khi giá trị không hợp lệ, để lỗi của reviewer không bị
    bỏ qua âm thầm.
    """
    decision = decision.strip().lower()
    value = str(override_value).strip()
    if decision in {"approve", "escalate", "skip"}:
        return action, defect_type
    if decision == "edit" and not value:
        raise ValueError("decision 'edit' cần override_value (KEEP / RECAPTURE / " + " / ".join(TYPES) + ")")
    if decision == "reject" and not value:
        value = _flip(action)
    if decision not in {"edit", "reject"}:
        raise ValueError(f"reviewer_decision không hợp lệ: {decision!r}")

    if value.upper() in ACTIONS:
        new_action = value.upper()
        if new_action == "KEEP":
            return "KEEP", "none"
        fallback = candidate_type if candidate_type in DEFECT_TYPES else DEFECT_TYPES[0]
        return "RECAPTURE", defect_type if defect_type in DEFECT_TYPES else fallback
    if value.lower() in TYPES:
        new_type = value.lower()
        return ("KEEP" if new_type == "none" else "RECAPTURE"), new_type
    raise ValueError(f"override_value {value!r} không hợp lệ; dùng {list(ACTIONS)} hoặc {list(TYPES)}")


def _queue(ctx: RunContext, final: pd.DataFrame) -> pd.DataFrame:
    """Hàng chờ đã ghi ở bước 03 rank; nếu chưa có thì dựng lại y hệt (cùng hàm, cùng budget)."""
    path = ctx.output_dir / "review_queue.csv"
    if path.is_file():
        return pd.read_csv(path, dtype=str).fillna("")
    return make_review_queue(
        final, id_col="image_id", score_col="score", budget=int(ctx.param("budget", 20)),
        reason_col="reason", action_col="pred_action", slice_col="light", evidence_col="path",
    )


def _render(ctx: RunContext, queue: pd.DataFrame, final: pd.DataFrame) -> None:
    show = queue.merge(
        final[["image_id", "pred_defect_type"]].rename(columns={"image_id": "item_id"}),
        on="item_id", how="left",
    )
    render_contact_sheet(
        show, ctx.out("contact_sheet.html"), image_root=ctx.input_dir, fields=DISPLAY_FIELDS,
        title=f"T02 Raw QC · review top-{len(show)} ({ctx.run_name})",
    )


def run(ctx: RunContext, scored: pd.DataFrame) -> pd.DataFrame:
    audit = ctx.audit or AuditLog(ctx.output_dir / "audit_log.jsonl")
    final = scored.copy()
    final["image_id"] = final["image_id"].astype(str)
    final["model_action"] = final["pred_action"]
    final["model_defect_type"] = final["pred_defect_type"]
    final["reviewer"] = ""
    final["reviewer_decision"] = ""
    if "candidate_type" not in final.columns:
        final["candidate_type"] = final["pred_defect_type"]

    queue = _queue(ctx, final)
    _render(ctx, queue, final)

    reviewed_path = ctx.output_dir / REVIEWED_FILE
    if not reviewed_path.is_file():
        ctx.stats.update(review_done=False, n_reviewed=0, n_overrides=0, n_escalated=0)
        audit.log(actor="pipeline", action="review:pending", item_id=str(ctx.out("contact_sheet.html")),
                  after={"queue_size": int(len(queue))}, reason=f"chờ reviewer điền {REVIEWED_FILE}")
        return final

    reviewed = load_review_decisions(reviewed_path)        # kiểm decision hợp lệ + có reviewer
    reviewed["item_id"] = reviewed["item_id"].astype(str)
    dup = reviewed["item_id"][reviewed["item_id"].duplicated()].tolist()
    if dup:
        raise ValueError(f"{REVIEWED_FILE}: image_id bị review hai lần {dup[:5]}")
    unknown = sorted(set(reviewed["item_id"]) - set(final["image_id"]))
    if unknown:
        raise ValueError(f"{REVIEWED_FILE}: image_id không có trong lần chạy này {unknown[:5]}")
    in_queue = set(queue["item_id"].astype(str))

    by_id = final.set_index("image_id")
    action_changes, type_changes = [], []
    for r in reviewed.itertuples(index=False):
        row = by_id.loc[r.item_id]
        try:
            new_action, new_type = resolve_override(
                r.reviewer_decision, r.override_value, row["pred_action"], row["pred_defect_type"], row["candidate_type"],
            )
        except ValueError as exc:
            raise ValueError(f"{REVIEWED_FILE}, ảnh {r.item_id}: {exc}") from exc
        if new_action != row["pred_action"]:
            action_changes.append(r._replace(override_value=new_action))
        if new_type != row["pred_defect_type"]:
            type_changes.append((r, row["pred_defect_type"], new_type))
        if new_action == row["pred_action"] and new_type == row["pred_defect_type"]:
            audit.log(actor=r.reviewer, action=f"review:{r.reviewer_decision}", item_id=r.item_id,
                      before=row["pred_action"], after=row["pred_action"], reason=r.review_note,
                      in_queue=r.item_id in in_queue)

    # đổi KEEP/RECAPTURE: dùng helper chung -> audit "override:<decision>"
    if action_changes:
        final = apply_overrides(final, pd.DataFrame(action_changes), id_col="image_id",
                                value_col="pred_action", audit=audit)
    # đổi loại lỗi cho khớp (KEEP <-> none) -> audit "override_type:<decision>"
    for r, before, after in type_changes:
        final.loc[final["image_id"] == r.item_id, "pred_defect_type"] = after
        audit.log(actor=r.reviewer, action=f"override_type:{r.reviewer_decision}", item_id=r.item_id,
                  before=before, after=after, reason=r.review_note)

    lookup = reviewed.set_index("item_id")
    mask = final["image_id"].isin(lookup.index)
    final.loc[mask, "reviewer"] = final.loc[mask, "image_id"].map(lookup["reviewer"])
    final.loc[mask, "reviewer_decision"] = final.loc[mask, "image_id"].map(lookup["reviewer_decision"])

    covered = in_queue <= set(reviewed["item_id"])
    n_overrides = int((final["pred_action"] != final["model_action"]).sum()
                      + ((final["pred_action"] == final["model_action"])
                         & (final["pred_defect_type"] != final["model_defect_type"])).sum())
    ctx.stats.update(
        # "đã review" = mọi ảnh trong top-B đều có quyết định, không chỉ là file tồn tại
        review_done=bool(covered),
        n_reviewed=int(len(reviewed)),
        n_queue_unreviewed=int(len(in_queue - set(reviewed["item_id"]))),
        n_overrides=n_overrides,
        n_escalated=int((reviewed["reviewer_decision"] == "escalate").sum()),
        review_decisions=reviewed["reviewer_decision"].value_counts().to_dict(),
    )
    return final
