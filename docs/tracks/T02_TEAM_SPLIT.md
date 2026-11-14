# T02 · Chia đều công việc cho 6 người

**Track:** Raw QC Triage · **Fixture:** `fixtures/t02_raw_qc/` · **Config:** `configs/tracks/t02.yaml`
**Metric chính:** Defect Recall@B (B=20, split eval) + Precision@B
**Output bắt buộc:** `out/ranked_review.csv` → `image_id, score` (+ tuỳ chọn `pred_defect_type, pred_action`)

---

## Nguyên tắc thiết kế: 6 hạt × 6 người

Công việc được cắt thành **6 hạt (slice) độc lập**, mỗi hạt là **1 file riêng** do **1 người duy nhất** sở hữu:

| Hạt | File sở hữu | Người |
|---|---|---|
| 1 | `src/tracks/t02_loader.py` | **P1** |
| 2 | `src/tracks/t02_baseline.py` | **P2** |
| 3 | `src/tracks/t02_features.py` | **P3** |
| 4 | `src/tracks/t02_scoring.py` | **P4** |
| 5 | `src/tracks/t02_eval.py` | **P5** |
| 6 | `src/tracks/t02_review.py` | **P6** |

**Luật bất di bất dịch:** *Không ai sửa file của người khác.* Mọi thay đổi đi qua PR vào `main`. `src/pipeline.py` là **file glue duy nhất** — do **P1** sở hữu, và P1 chỉ sửa nó **sau khi** cả 6 hạt có API ổn định.

**Không phụ thuộc tuần tự:** P2, P3, P5, P6 có thể làm song song từ phút 0. Chỉ P4 phụ thuộc P3 (1 interface), và P1 phụ thuộc tất cả (1 lần glue).

---

## Interface giữa các hạt (chốt trong 15 phút đầu, không đổi nữa)

```python
# ① src/tracks/t02_loader.py  (P1) → cấp cho mọi người
def load_images(ctx) -> pd.DataFrame:
    """Cột BẮT BUỘC: image_id, path, camera_id, light, split
       + cột hash: sha256_ok (bool) — đã verify manifest
       path là ĐƯỜNG DẪN TUYỆT ĐỐI đã join với ctx.input_dir"""

# ② src/tracks/t02_baseline.py  (P2) → độc lập
def baseline_scores(ctx, images: pd.DataFrame) -> pd.DataFrame:
    """Cột BẮT BUỘC: image_id, score   (score CAO = nghi lỗi hơn)"""

# ③ src/tracks/t02_features.py  (P3) → cấp cho P4
def extract_features(ctx, images: pd.DataFrame) -> pd.DataFrame:
    """image_id + feature_*: sharp, bright_mean, frac_dark, frac_saturated,
       blockiness, ... — giá trị THÔ, chưa chuẩn hoá"""

# ④ src/tracks/t02_scoring.py  (P4) → cấp cho P1
def poc_scores(ctx, images: pd.DataFrame, features: pd.DataFrame) -> pd.DataFrame:
    """Cột BẮT BUỘC: image_id, score
       + tuỳ chọn: pred_defect_type, pred_action"""

# ⑤ src/tracks/t02_eval.py  (P5) → độc lập
def run_eval_report(pred_csv: Path, run_name: str) -> dict:
    """Gọi evaluator, trả metrics + slice table (không tự viết lại metric!)"""

# ⑥ src/tracks/t02_review.py  (P6) → độc lập
def build_review(ctx, scored: pd.DataFrame) -> pd.DataFrame:
    """Trả review_queue (schema REVIEW_COLUMNS) + render contact sheet HTML"""
```

---

## Bảng phân công chi tiết

| Người | Hạt | Nội dung | Deliverable cụ thể | Files được sở hữu | Công |
|---|---|---|---|---|---|
| **P1** | ① Loader + glue | Load fixture, verify manifest/hash, dựng khung `src/pipeline.py` | `src/tracks/t02_loader.py`, `src/pipeline.py` | 2 file | ⭐⭐⭐ |
| **P2** | ② Baseline | Baseline Laplacian 1 ngưỡng + tune ngưỡng trên `dev` | `src/tracks/t02_baseline.py`, `out/baseline/ranked_review.csv` | 2 file | ⭐⭐ |
| **P3** | ③ Features | 5 nhóm feature: sharpness, brightness, glare, blockiness, histogram | `src/tracks/t02_features.py` | 1 file | ⭐⭐⭐ |
| **P4** | ④ Scoring | Ghép feature → score, chuẩn hoá camera/light, xếp hạng | `src/tracks/t02_scoring.py` | 1 file | ⭐⭐⭐ |
| **P5** | ⑤ Evaluation | Metric vs baseline, 3 failure mode, false alarm slice | `src/tracks/t02_eval.py`, `reports/table1_baseline_vs_poc.md`, `reports/failure_modes.md` | 3 file | ⭐⭐⭐ |
| **P6** | ⑥ Review | Contact sheet, review queue, override+audit, decision KEEP/RECAPTURE | `src/tracks/t02_review.py`, `out/poc/contact_sheet.html`, `out/poc/decision.json` | 3 file | ⭐⭐⭐ |

**Cân bằng:** P2 là người nhẹ nhất về mã (baseline cố tình đơn giản) nhưng nhiều việc "âm thầm" (đọc code, hiểu metric, chạy lệnh). Để cân, **P2 nhận thêm** việc viết `reports/team_contributions.csv` + tag `demo-180m`/`final-24h`, và **P1 nhận thêm** việc viết `README`/`docs/tracks/T02_SOLUTION.md` tổng kết.

---

## Chi tiết từng hạt

### ① P1 — Loader + glue

- `load_images(ctx)`: đọc `images.csv` + `manifest.csv`, gọi `verify_manifest()` (`src/common/manifest.py:49`), join path tuyệt đối, cột `sha256_ok`.
- **Không** đọc `gt/` (GT guard → exit code 5).
- `src/pipeline.py`: wire 5 bước. Chỉ P1 sửa file này → **zero merge conflict**.

```46:60:e:\Lab6\K5-Day6-L2-NhomDay6-DataGovernanceAndPrivacy\src\pipeline.py
def process(ctx: RunContext, data: dict) -> pd.DataFrame:
    spec = get_track(ctx.track)
    raise NotImplementedError(
        f"Implement process() for {spec.track} ({spec.title}). It must produce a DataFrame with at least "
        f"{spec.pred_columns} (saved as out/{spec.pred_file}). See docs/tracks/{spec.track.upper()}.md"
    )
```

- Sửa `configs/lab.yaml`: `group_id`, `owner` (bắt buộc cho `Decision`), `params`.
- Viết `docs/tracks/T02_SOLUTION.md` (giải thích thuật toán, để nhóm bảo vệ được khi hỏi).

---

### ② P2 — Baseline (là chuẩn, KHÔNG dùng random)

> §4: *"Baseline: Một ngưỡng Laplacian toàn ảnh duy nhất"*. `src/examples/random_baseline.py` chỉ là ví dụ wiring — dùng nó thì cột so sánh §4 vô nghĩa.

- `score = -laplacian_variance(img)` (điểm cao = mờ hơn = nghi lỗi hơn).
- Chuẩn hoá theo `camera_id` (cam_B quang học mềm → variance thấp → dễ bị báo nhầm).
- Tune ngưỡng **chỉ trên split `dev`**.
- Chạy 2 lệnh, lưu JSON:

```bash
python -m src.main --track t02 --pipeline src.tracks.t02_baseline_entry \
    --output out/baseline --run-name baseline --offline --seed 42
python -m src.evaluate --track t02 --pred out/baseline/ranked_review.csv \
    --run-name baseline --out reports/metrics_baseline.json
```

- **Báo cả kết quả bằng 0 hoặc âm** — baseline tệ là chuyện bình thường.
- Ghi `reports/team_contributions.csv`; đánh dấu tag `demo-180m`, `final-24h`.

---

### ③ P3 — Feature (hạt có giá trị cao nhất, làm kỹ)

Dùng sẵn `src/common/image_ops.py` — **không viết lại** hàm có sẵn:

| Feature | API có sẵn | Bắt lỗi |
|---|---|---|
| Sharpness | `laplacian_variance(img)` | blur |
| Brightness | `brightness_stats(img)` → `mean/std/p05/p95` | dark |
| Dark fraction | `brightness_stats()["frac_dark"]` | dark |
| Saturated fraction | `brightness_stats()["frac_saturated"]` | glare |
| Color histogram | `color_histogram(img)` | bổ sung |
| JPEG blockiness | tự tính (DCT 8×8 trên ảnh gray → năng lượng biên block) | compression |

- **Kiểm chứng feature** bằng `src/common/corruptions.py`: blur radius 2.0/5.0, dark gain 0.45/0.2, glare 0.6/1.0, JPEG q=15/5 → feature phải phản ứng đúng hướng. Đây là bằng chứng cho báo cáo.
- Ảnh chỉ **320×180** → feature phải ổn định ở kích thước nhỏ (cân nhắc tính trên ROI lớn hơn).
- **Quan trọng:** feature **THÔ**, để P4 chuẩn hoá. Nhưng phải kèm cột `camera_id`, `light` để P4 chuẩn hoá theo nhóm.

---

### ④ P4 — Scoring & xếp hạng

- Ghép nhiều feature thành 1 score (theo gợi ý §8: sharpness + histogram + saturated fraction + blockiness).
- **Chuẩn hoá theo `light` (day/dusk/night) VÀ `camera_id`** — dùng percentile/rank *trong nhóm* thay vì toàn cục. Đây là chìa khoá để không báo nhầm cam_B.
- **Score cao = nghi lỗi hơn.**
- Threshold **chỉ tune trên `dev`**; **freeze trong `configs/lab.yaml` trước khi chạy eval** (vi phạm protocol = mất điểm).
- Xuất `image_id, score` + tuỳ chọn `pred_defect_type ∈ {blur,glare,dark,compression,none}`, `pred_action ∈ {KEEP,RECAPTURE}`.
- Cột `pred_defect_type` mở khoá metric **defect-type accuracy**:

```159:161:e:\Lab6\K5-Day6-L2-NhomDay6-DataGovernanceAndPrivacy\src\metrics\tracks.py
    if "pred_defect_type" in pred.columns:
        m = gt[gt.is_defect == 1].merge(pred[["image_id", "pred_defect_type"]], on="image_id", how="left")
        acc = (m.defect_type == m.pred_defect_type).mean() if len(m) else None
```

- Kiểm tra **không NaN** trong `score` (NaN làm sort sai → mất top-B).

---

### ⑤ P5 — Evaluation & failure analysis

**Không tự viết lại metric** — dùng `src/metrics/tracks.py::eval_t02` qua `src/evaluate.py`.

```bash
python -m src.evaluate --track t02 --pred out/baseline/ranked_review.csv --run-name baseline --out reports/metrics_baseline.json
python -m src.evaluate --track t02 --pred out/poc/ranked_review.csv      --run-name poc      --out reports/metrics_poc.json
python -m src.compare reports/metrics_baseline.json reports/metrics_poc.json --out reports/table1_baseline_vs_poc.md
```

**3 failure mode bắt buộc (§5)** — phải tự tìm, minh hoạ bằng ảnh/metadata, nêu **root cause**:

| Failure mode | Root cause dự kiến | Minh hoạ |
|---|---|---|
| Ảnh ban đêm hữu ích bị loại hết | Điểm trừ darkness nặng hơn blur ở `light=night` | Liệt kê ảnh `light=night` bị đẩy cuối bảng |
| Ảnh nét nhưng biển số cháy sáng bị bỏ qua | Glare **cục bộ** — `frac_saturated` toàn ảnh thấp dù vùng lóa cháy | Chỉ ra ảnh có vùng cháy cục bộ bị xếp thấp |
| cam_B bị báo nhầm | Quang học mềm → Laplacian thấp nên bị coi là mờ | So phân bố `sharpness` cam_A vs cam_B |

Ngoài ra: bảng **false alarm theo slice `light`** (`src/metrics/tracks.py:166`), và kiểm **10 ảnh out-of-domain** (deadline +24h).

---

### ⑥ P6 — Human-in-the-loop & quyết định

- `make_review_queue(...)` từ `src/common/review.py:44` → top-B=20, đúng schema `REVIEW_COLUMNS`.
- `render_contact_sheet(...)` từ `src/common/contact_sheet.py:40` → HTML offline, nhúng base64, có nút **Export CSV**.
- Chạy vòng review thật: điền `reviewer` + `reviewer_decision` + `override_value` + `review_note` → export → `load_review_decisions()` → `apply_overrides()` → **ghi `AuditLog`**.
- `decision.json` + `decision_log.csv` qua `write_decision()`: `status` ∈ GO/HOLD/REWORK (alias `KEEP`→GO, `RECAPTURE`→REWORK), `reason_code` (**bắt buộc**), `owner` (**bắt buộc**), `evidence`, `metrics`.

```40:52:e:\Lab6\K5-Day6-L2-NhomDay6-DataGovernanceAndPrivacy\src\common\decision.py
@dataclass
class Decision:
    status: str                 # e.g. GO / HOLD / REWORK / PASS / ACCEPT / RECAPTURE
    reason_code: str            # short machine code, e.g. "PRIMARY_BELOW_THRESHOLD"
    reason: str                 # one human sentence
    owner: str                  # who signs off (name/role) — mandatory
    scope: str = ""             # batch/release/request id this decision is about
    evidence: list[str] = field(default_factory=list)   # paths to files backing the decision
```

> `reason_code`/`owner` rỗng → `ValueError`. `owner` đặt trong `configs/lab.yaml` (P1 làm, P6 dùng).

---

## Mốc thời gian

| Mốc | Việc | Người |
|---|---|---|
| **+15 phút** | Chốt interface ①–⑥, tạo 6 branch, set `configs/lab.yaml` | P1 + cả nhóm |
| **+90 phút** | ① loader ✅ · ② baseline ✅ + `metrics_baseline.json` · ③ feature xong | P1, P2, P3 |
| **+150 phút** | ④ scoring xong → `out/poc/ranked_review.csv` | P4 |
| **+180 phút** | ⭐ **Demo**: `ranked_review.csv` + contact sheet + metric vs baseline + 3 failure mode | Toàn nhóm |
| **+24h** | Threshold theo camera/ROI · 10 ảnh out-of-domain · báo cáo cuối + `docs/tracks/T02_SOLUTION.md` | P4, P5, P1 |

---

## Rủi ro & người canh chừng

| Rủi ro | Canh | Xử lý |
|---|---|---|
| Đọc nhầm `gt/` → exit code 5 | P1 | Không `open()` path trong `fixtures/*/gt/`; chỉ dùng `src/evaluate` |
| Tune trên `eval` → vi phạm protocol | P4, P5 | Threshold ghi vào `configs/lab.yaml` **trước** khi chạy eval |
| Baseline làm sai (random) → so sánh vô nghĩa | P2 | Bắt buộc Laplacian 1 ngưỡng |
| Merge conflict | P1 | 1 file / 1 người; `src/pipeline.py` chỉ P1 sửa |
| B=20 / 96 ảnh → metric nhiễu | P5 | Báo kèm số đếm cụ thể + CI, không chỉ tỉ lệ |
| `owner`/`reason_code` rỗng → `ValueError` | P1, P6 | Set `owner` sớm |
| P3/P4 lệch interface giữa chừng | P1 | Interface đã chốt ở trên, không đổi giữa chừng |
| Quên `pred_defect_type` → mất metric phụ | P4 | Thêm cột nếu đủ confidence |

---

## Definition of Done

- [ ] `python -m src.main --config configs/lab.yaml --track t02 --output out/poc --run-name poc --offline --seed 42` → exit 0
- [ ] `out/poc/ranked_review.csv` có `image_id, score` (96 dòng, không NaN)
- [ ] `out/baseline/ranked_review.csv` tồn tại, dùng Laplacian 1 ngưỡng
- [ ] `reports/metrics_baseline.json` + `reports/metrics_poc.json` + `reports/table1_baseline_vs_poc.md`
- [ ] `out/poc/contact_sheet.html` mở offline, có nút Export CSV, đã review ≥1 vòng có `reviewer`
- [ ] `out/poc/audit_log.jsonl` ghi lại mọi override
- [ ] `out/poc/decision.json` + `decision_log.csv` có `reason_code` + `owner` không rỗng
- [ ] `reports/failure_modes.md` phân tích đủ 3 failure mode, kèm ảnh/metadata + root cause
- [ ] `docs/tracks/T02_SOLUTION.md` giải thích thuật toán
- [ ] `reports/team_contributions.csv` khớp `git shortlog -sne --all`
- [ ] Bảng false alarm theo slice `light` (day/dusk/night)
- [ ] Tag `demo-180m` (mốc 180 phút), tag `final-24h` (deadline)
