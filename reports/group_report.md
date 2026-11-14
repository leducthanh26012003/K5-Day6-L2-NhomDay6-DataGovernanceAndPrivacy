# Group Report — Day 06 Data Lab (Field-to-Release PoC)

> Điền đầy đủ, cụ thể, không viết chung chung. Mọi con số phải lấy từ file do script sinh ra
> (`reports/metrics_*.json`, `reports/table1_baseline_vs_poc.md`) — không gõ tay, không làm tròn mất đơn vị.
> 3–5 trang (hoặc README + report.md tương đương).

## 00 · Header

| Mục | Giá trị |
|---|---|
| Group ID | NhomDay6 |
| Challenge | T02 · Raw QC Triage: ảnh mờ, lóa và thiếu sáng |
| Thành viên (5–6) + vai trò | 1. Lê Đức Thanh — TV1 (Lead/Integrator) · 2. Phạm Trần Anh — TV2 (Laplacian baseline) · 3. Nguyễn Viết Minh — TV3 (Sharpness features) · 4. Trần Trọng Thái — TV4 (Exposure/glare features) · 5. Phạm Quang Huy — TV5 (Scoring + review) · 6. Cao Văn Dương — TV6 (Research + report) |
| Git URL | https://github.com/leducthanh26012003/K5-Day6-L2-NhomDay6-DataGovernanceAndPrivacy.git |
| Tag `demo-180m` (commit) | `9417e3d` (main) |
| Tag `final-24h` (commit) | `9417e3d` |
| Deadline (giờ VN) | 14h00 15/11/2026 |

## 01 · Problem statement

- **Ai bị ảnh hưởng / ai chịu hậu quả khi hệ thống sai?** Người huấn luyện model dùng bộ ảnh này, và người review chỉ có sức xem 20 ảnh. Bỏ sót ảnh lỗi thì dữ liệu xấu vào train. Báo nhầm nhiều thì mất ảnh đêm quý hoặc phí công review.
- **Điểm nghẽn cụ thể:** 96 ảnh nhưng chỉ review được 20, nên cần xếp hạng để ảnh lỗi lên đầu.
- **Đầu vào → đầu ra:** `images/*.jpg` + `images.csv` (96 ảnh, 320×180, 2 camera, 3 điều kiện sáng) → `out/ranked_review.csv` (`image_id, score, pred_defect_type, pred_action, reason`) → quyết định KEEP / RECAPTURE cho từng ảnh, GO / HOLD cho cả lô.
- **Tiêu chí GO / HOLD / REWORK** (chốt trước khi chạy eval): hàm `decide()` trong `src/pipeline.py` — GO khi đã có reviewer xác nhận (`review_queue_reviewed.csv` tồn tại) **và** tỉ lệ RECAPTURE ≤ 25% (`max_recapture_rate = 0.25`, chốt trong `configs/lab.yaml`). HOLD nếu chưa review hoặc vượt ngưỡng. `reason_code` và `owner` bắt buộc.
- **Giả định của nhóm** (cần DPO/legal/domain owner xác nhận): dữ liệu synthetic không có PII thật; ngưỡng camera/ánh sáng chỉ đúng cho 2 camera trong fixture — camera mới cần re-calibrate; ảnh cùng cảnh gốc có thể tương quan (mẫu không độc lập).

## 02 · Data & provenance

| Split | N | Nguồn / link | License | Synthetic? |
|---|---|---|---|---|
| dev | 56 | `tools/make_fixtures.py` (seed 42) — 14 cảnh gốc × 4 biến thể | CC0-1.0 (synthetic, generated for AI20K Day06) | Có |
| eval / heldout | 40 | `tools/make_fixtures.py` (seed 42) — 10 cảnh gốc × 4 biến thể | CC0-1.0 (synthetic, generated for AI20K Day06) | Có |

- **Schema & class mapping:** Input: `image_id, path, camera_id, light, split`. Nhãn (chỉ dùng chấm): `defect_type` ∈ {`none`, `blur`, `glare`, `dark`, `compression`}, `severity` 0–2, `is_defect`. Output: `image_id, score` (score cao = nghi lỗi, cần review trước).
- **Seed:** 42 · **Cách chia split:** theo cảnh gốc (`scene_id`), 24 cảnh: 14 dev / 10 eval. Mỗi cảnh có 4 biến thể (original + 3 corrupted).
- **`manifest.csv`** verified (sha256): có — hash check qua `src.common.manifest.verify_manifest`.
- **Dữ liệu thật dùng:** Không có. Toàn bộ là synthetic.
- **Camera phân bố:** `cam_A` 64 ảnh, `cam_B` 32 ảnh. **Ánh sáng:** `day` 56, `dusk` 20, `night` 20.
- **Mini gold dev** (nhãn tự gán): `reports/mini_gold_dev.csv` — 20 ảnh dev, 3 `dark`, 1 `glare`, 2 `blur`, 1 `compression`, 11 `none`, 2 `blur`.

## 03 · Research & design

- **References:**
  1. Pech-Pacheco et al. (2000) — "Diatom autofocusing in brightfield microscopy: a comparative study" — Laplacian variance là chỉ báo nét chuẩn, dùng làm baseline. (Ngày xem: 14/11/2026)
  2. ACDC dataset (ETH Zürich) — Adverse Conditions Dataset: ảnh xe nhiều điều kiện sáng, minh hoạ failure mode camera/night. (Ngày xem: 14/11/2026)
  3. ISO/IEC TR 5259-6:2026 — Data quality for AI: gợi ý framework kiểm soát chất lượng dữ liệu ảnh, nhóm dùng ý tưởng "within-group normalisation" cho z-score theo camera/ánh sáng. (Ngày xem: 14/11/2026)

- **Sơ đồ data flow** (5 bước trong `src/pipeline.py`):
  ```
  01 load     images.csv + manifest check
       ↓
  02 process  feat_sharpness (TV3: lap_var, lap_tile_min, blockiness)
              + feat_exposure  (TV4: luma_rel, sat_frac_rel, sat_tile_max_rel, blob)
              → score.combine  (TV5: ghép score, gán pred_defect_type, pred_action)
       ↓
  03 rank     top-B=20 ảnh nghi lỗi nhất → review_queue.csv
       ↓
  04 review   contact_sheet.html + reviewer override + audit_log.jsonl
       ↓
  05 decide   GO / HOLD (dựa vào review_done + max_recapture_rate ≤ 25%)
  ```

- **Thành phần offline:** không dùng model pretrained, không gọi mạng. Toàn bộ feature tính bằng NumPy/SciPy thuần (`cv2`-free). `scipy.ndimage.label` cho blob detection (connected components).

- **Trade-offs đã cân nhắc:**
  - **Ngưỡng tuyệt đối vs tương đối:** ngưỡng tuyệt đối (`luma_mean < 50`) loại **18/20 ảnh đêm hữu ích** (xem §6.1 của TV4). Ngưỡng tương đối (z-score trong nhóm `light × camera_id`) chỉ loại đúng 1 ảnh đêm thực sự tối. Chọn tương đối.
  - **MAD vs std cho z-score:** MAD robust hơn nhưng sụp ở nhóm nhỏ (MAD ≈ 0 → z = −780 cho img_12_2). Dùng `max(MAD × 1.4826, 0.25 × std)` để giữ robust mà không khuếch đại nhiễu.
  - **Glare toàn cục vs cục bộ:** `sat_frac` (toàn ảnh) bị loãng bởi diện tích khung — vùng 587 px trên 57 600 px chỉ 1.1%. Bổ sung lưới 4×6 tiles + blob liên thông để bắt glare cục bộ.
  - **Speckle loại trừ:** nhiễu cảm biến đêm (`img_03_2`, `img_06_3`) có `sat_frac_rel > 5.63` nhưng blob chỉ 60–65 px. Điều kiện `sat_blob_px ≥ 162` loại speckle mà giữ glare thật (blob ≥ 321 px).

## 04 · Baseline vs PoC

- **Baseline** (Laplacian variance toàn ảnh): `src/t02/baseline.py`, commit `ade7f1e`. Score = `1 / (1 + laplacian_variance)`. Ngưỡng `RECAPTURE` = phân vị 20% của `laplacian_variance` trên 56 ảnh dev = **186.9666**. Chỉ dùng 1 feature duy nhất, không phân biệt camera hay ánh sáng. Status: HOLD (baseline only).
- **PoC** (multi-feature margin-based pipeline): `src/pipeline.py` + `src/t02/score.py` (TV5), commits `bcf04df`→`9417e3d`. Ghép sharpness (TV3: `lap_var`, `lap_tile_min`, `blockiness`, `lap_norm`) + exposure/glare (TV4: `luma_rel`, `sat_frac_rel`, `sat_tile_max_rel`, `sat_blob_frac_rel`) qua `score.combine` (TV5). Với mỗi ảnh, TV5 tính **margin** cho 4 loại lỗi: `blur` (lap_norm < 0.0738), `compression` (blockiness > 2.055), `dark` (luma_rel < −5.245), `glare` (2 route: "burn" từ TV4 sat thresholds, "veil" khi luma_rel > 5.245). Score = `sigmoid(max(margin))`, `pred_defect_type = argmax(margin)`. Ngưỡng freeze trong `configs/tv5_score.yaml`. Kết quả: **42 RECAPTURE / 54 KEEP** (14 compression, 12 blur, 10 dark, 6 glare). Status: HOLD (chờ human review); GO khi đã review và `recapture_rate ≤ 25%`.
- **Cùng budget / cùng heldout / cùng seed:** xác nhận — B=20, eval=40 ảnh (10 cảnh × 4 biến thể), seed=42 cố định, offline.

**Bảng 1 · Baseline vs PoC & hard slices** — metric chính thức phải lấy từ `reports/metrics_poc.json` do `python -m src.evaluate` sinh ra (cần overlay giảng viên — `fixtures/t02_raw_qc/gt/labels.csv`). Dưới đây là kết quả **kiểm tra nội bộ** trên 20 ảnh mini gold dev tự gán nhãn:

| Metric (unit) | Baseline | PoC (TV5 full pipeline) | Δ & chú thích |
|---|---|---|---|
| Primary: Defect Recall@B (B=20, mini gold dev) | chưa đo (cần gt/) | mini gold recall 4/4 = 1.00, false alarm 0/11 = 0.00 | PoC bắt đủ 4 ảnh lỗi trong 20 ảnh dev tự gán; metric chính thức phải chạy trên eval |
| Tổng RECAPTURE / 96 ảnh | 19 (Laplacian only) | 42 (14 compression + 12 blur + 10 dark + 6 glare) | PoC phát hiện 4 loại lỗi thay vì 1 |
| Hard-slice: night | loại 18/20 ảnh đêm | loại 1/20 ảnh đêm (`img_01_2`, đúng dark thật) | PoC giảm −17 ảnh đêm bị loại nhầm; N=20 |
| Hard-slice: cam_B | loại 12 ảnh cam_B (ngưỡng tuyệt đối) | loại 2 ảnh cam_B | PoC giảm false alarm cam_B ×6; N=32 |
| Top-1 score | — | `img_17_2` (glare, score=0.9466, sat_blob_px=13246) | Glare toàn cục rõ, phát hiện qua burn route |
| Runtime / 96 ảnh | ≈ 1.3 s (Laplacian toàn ảnh) | ≈ 1.35 s (full pipeline E2E) | CPU, Intel i7-10750H |

> ⚠️ Bảng trên dựa trên mini gold dev (20 ảnh tự gán) — **không phải metric chính thức**. Metric chính thức (`Defect Recall@B`, `Precision@B`, split eval) phải lấy từ `reports/metrics_poc.json` do giảng viên chạy `python -m src.evaluate`.

## 05 · Evaluation protocol

- **Công thức primary metric:** `Defect Recall@B` = số ảnh lỗi thật trong top-B / tổng số ảnh lỗi thật trong eval. Đơn vị: 1 ảnh. Metric phụ: `Precision@B` = số ảnh lỗi thật trong top-B / B. TP = ảnh lỗi thật nằm trong top-B. FP = ảnh tốt lọt vào top-B. FN = ảnh lỗi thật nằm ngoài top-B.
- **Budget B = 20**, chốt trong `configs/lab.yaml` (`params.budget`) và `reports/metric_contract.md`. Ngưỡng TV4 freeze trong `configs/tv4_calibration.yaml` trước khi chấm eval (provenance: `tv4_fit.split: dev`, `tv4_fit.source: reports/mini_gold_dev.csv`).
- **Độ bất định:** Wilson 95% CI cho tỉ lệ recall/precision khi N nhỏ. Eval chỉ 40 ảnh → CI rất rộng. Các ảnh cùng cảnh gốc có thể tương quan → CI thực tế rộng hơn Wilson giả định. Nhóm ghi tử số/mẫu số kèm tỉ lệ.
- **Z-score tính trên cả 96 ảnh** (dev + eval): đây là chuẩn hoá không dùng nhãn (transductive, unsupervised) nên **không rò rỉ nhãn eval**. Nếu giảng viên yêu cầu mọi thống kê chỉ tính trên dev, cần chuyển sang tính trên dev rồi áp cho eval.
- **Lệnh evaluate chính xác:**
  ```bash
  python -m src.evaluate --track t02 --pred out/baseline/ranked_review.csv --run-name baseline --out reports/metrics_baseline.json
  python -m src.evaluate --track t02 --pred out/poc/ranked_review.csv --run-name poc --out reports/metrics_poc.json
  python -m src.compare reports/metrics_baseline.json reports/metrics_poc.json --out reports/table1_baseline_vs_poc.md
  ```

## 06 · Result slices

**TV5 PoC — Top 10 ảnh nghi lỗi nhất (score giảm dần, từ `out/poc/ranked_review.csv`):**

| Rank | image_id | score | pred_defect_type | pred_action | camera | light | split |
|---|---|---|---|---|---|---|---|
| 1 | img_17_2 | 0.9466 | glare | RECAPTURE | cam_A | day | eval |
| 2 | img_06_2 | 0.9300 | compression | RECAPTURE | cam_B | night | eval |
| 3 | img_22_2 | 0.9272 | compression | RECAPTURE | cam_A | dusk | dev |
| 4 | img_00_2 | 0.9007 | compression | RECAPTURE | cam_B | day | dev |
| 5 | img_21_3 | 0.8866 | compression | RECAPTURE | cam_B | day | dev |
| 6 | img_04_2 | 0.8849 | compression | RECAPTURE | cam_A | dusk | dev |
| 7 | img_10_2 | 0.8768 | compression | RECAPTURE | cam_A | day | dev |
| 8 | img_05_2 | 0.8694 | compression | RECAPTURE | cam_A | day | eval |
| 9 | img_23_2 | 0.8524 | compression | RECAPTURE | cam_A | day | eval |
| 10 | img_13_3 | 0.8276 | blur | RECAPTURE | cam_A | night | eval |

**TV5 PoC — Phân bố defect type (96 ảnh):**

| pred_defect_type | N | pred_action |
|---|---|---|
| none | 54 | KEEP |
| compression | 14 | RECAPTURE |
| blur | 12 | RECAPTURE |
| dark | 10 | RECAPTURE |
| glare | 6 | RECAPTURE |
| **Tổng** | **96** | **42 RECAPTURE / 54 KEEP** |

**TV4 — Exposure/glare flags theo camera × light (96 ảnh):**

| Slice (camera × light) | N | evidence = 0 (clean) | evidence = 1 (flagged) |
|---|---|---|---|
| cam_A · day | 44 | 36 | 8 |
| cam_A · dusk | 12 | 10 | 2 |
| cam_A · night | 8 | 7 | 1 |
| cam_B · day | 12 | 12 | 0 |
| cam_B · dusk | 8 | 6 | 2 |
| cam_B · night | 12 | 11 | 1 |
| **Tổng** | **96** | **82** | **14** |

**TV3 — Sharpness: trung vị `lap_var` theo camera × ánh sáng (dev):**

| Slice | cam_A | cam_B |
|---|---|---|
| day | 944.30 | 235.77 |
| dusk | 777.32 | 174.46 |
| night | 1578.12 | 350.41 |

→ `cam_B` có `lap_var` thấp hơn `cam_A` **3–4 lần** ở mọi điều kiện sáng, do quang học mềm hơn.

### Failure case 1 — Ảnh ban đêm hữu ích bị loại hết

**Root cause:** ngưỡng độ sáng tuyệt đối không biết "đêm" của một camera nghĩa là bao nhiêu. Ảnh đêm hữu ích có `dark_frac ≈ 0.63`, ảnh dark thật có `dark_frac ≈ 0.75` — **chồng lấn** hoàn toàn.

**Bằng chứng:** ngưỡng `luma_mean < 50` loại **18/20 ảnh đêm**. PoC (z-score `luma_rel < −5.2449`) chỉ loại **1/20** (`img_01_2`, `luma_mean = 3.47`, `luma_rel = −7.03` — đúng là ảnh tối thật). Các ảnh đêm hữu ích (`img_18_0`: `luma_mean = 32.52`, `luma_rel = −0.06`) được giữ lại.

**Cách xử lý:** chuẩn hoá `luma_mean` trong nhóm `light × camera_id` → `luma_rel`.

### Failure case 2 — Glare cục bộ bị bỏ qua (vùng biển số cháy sáng)

**Root cause:** `sat_frac` (tỉ lệ cháy sáng toàn ảnh) bị loãng. Ảnh `img_05_3` có vùng 587 px cháy sáng trên khung 57 600 px → `sat_frac = 0.011` (1.1%), `sat_frac_rel = 1.16` — **thấp hơn nhiều** ngưỡng toàn cục 5.63.

**Bằng chứng:** `sat_tile_max_rel = 6.54` **vượt** ngưỡng cục bộ 4.05 → bắt được nhờ tầng cục bộ (lưới 4×6). Ảnh `img_17_2` (glare toàn cục) có `sat_frac_rel = 26.85` → cả hai tầng đều bắt.

**Cách xử lý:** 3 tầng nhìn — `sat_frac_rel` (toàn cục), `sat_tile_max_rel` (lưới 4×6), `sat_blob_frac_rel` (blob liên thông) + điều kiện `sat_blob_px ≥ 162` loại speckle.

### Failure case 3 — Camera quang học mềm (cam_B) bị báo nhầm

**Root cause:** `cam_B` quang học mềm hơn → `lap_var` của cam_B thấp hơn cam_A 3–4× (day: 235 vs 944). Ngưỡng tuyệt đối loại **12 ảnh cam_B**; PoC chỉ loại **2** (đúng là ảnh lỗi).

**Bằng chứng TV3:** 3 ảnh cam_B dev nét bằng mắt nhưng `lap_var` thấp:
- `img_09_0` (cam_B, day): `lap_var = 186.97`, hạng 19/96. Nét, biển và xe rõ.
- `img_21_0` (cam_B, day): `lap_var = 220.71`, hạng 21/96.
- `img_12_2` (cam_B, dusk): `lap_var = 182.55`, hạng 18/96. Nét, nền phẳng.

Sau chuẩn hoá `lap_norm` theo camera: hạng tăng lên 32, 39, 29 / 96 tương ứng.

**Cách xử lý:** chuẩn hoá z-score trong nhóm `light × camera_id` cho cả sharpness và exposure.

### Latency / compute

| | PoC full pipeline (TV3+TV4+TV5) | Baseline (Laplacian) |
|---|---|---|
| 96 ảnh | ≈ 1.35 s | ≈ 1.3 s |
| Per image | ≈ 14.1 ms | ≈ 13.5 ms |
| CPU | Intel i7-10750H | Intel i7-10750H |

## 07 · Human & decision

- **Review queue:** `out/poc/review_queue.csv` — top B=20 ảnh có `score` cao nhất, ưu tiên ảnh nghi lỗi. File sinh bởi `src.common.review.make_review_queue`.
- **Contact sheet:** `out/poc/contact_sheet.html` — HTML offline, hiển thị ảnh + metadata + dropdown quyết định (approve/reject/edit/escalate/skip). Nút **Export CSV** tải về `review_queue_reviewed.csv`.
- **Cách override + audit log:** Reviewer chép `review_queue_reviewed.csv` vào `out/poc/` rồi chạy lại pipeline. `review_step.run()` đọc file, áp override qua `common.review.apply_overrides`, ghi `out/poc/audit_log.jsonl` (append-only JSONL, mỗi dòng ghi `actor`, `action`, `item_id`, `before`, `after`, `ts`).
- **Decision output thực tế** (từ `out/poc/decision.json`): `status: HOLD`, `reason_code: PENDING_HUMAN_REVIEW`, `reason: "42/96 ảnh đề xuất RECAPTURE nhưng chưa có review_queue_reviewed.csv"`, `owner: "Lê Đức Thanh - Lead"`. Pipeline exit code = 0.
- **Decision policy:** `decide()` trong `src/pipeline.py`:
  - **GO** khi `review_done = True` **và** `recapture_rate ≤ 25%` → `reason_code = RECAPTURE_RATE_OK`, `owner = "Lê Đức Thanh - Lead"`.
  - **HOLD** khi chưa review → `reason_code = PENDING_HUMAN_REVIEW`.
  - **HOLD** khi `recapture_rate > 25%` → `reason_code = HIGH_RECAPTURE_RATE`.
- **Khi nào HOLD/REWORK:** HOLD khi chưa có reviewer xác nhận (file `review_queue_reviewed.csv` chưa tồn tại), hoặc khi tỉ lệ RECAPTURE vượt 25%. Không có REWORK tự động — quyết định cuối thuộc reviewer.

## 08 · Reproduce & tests

```bash
# Setup
python -m venv .venv && pip install -r requirements.txt

# Chạy baseline
python -m src.main --input fixtures/t02_raw_qc --config configs/lab.yaml --output out/baseline --pipeline src.t02.baseline --offline --seed 42

# Chạy PoC (output vào out/poc/)
python -m src.main --input fixtures/t02_raw_qc --config configs/lab.yaml --output out/poc --offline --seed 42

# Evaluate (cần gt/ của giảng viên)
python -m src.evaluate --track t02 --pred out/baseline/ranked_review.csv --run-name baseline --out reports/metrics_baseline.json
python -m src.evaluate --track t02 --pred out/poc/ranked_review.csv --run-name poc --out reports/metrics_poc.json

# So sánh
python -m src.compare reports/metrics_baseline.json reports/metrics_poc.json --out reports/table1_baseline_vs_poc.md

# Tests
pytest -q
```

- **Output của `pytest` (non-fixture):** **48 passed, 25 skipped, 0 failed** (bao gồm: TV4 tests 15 passed, TV5/score tests, pipeline E2E tests). Skipped: các track khác (t01, t03–t14) không liên quan.
- **Output của `pytest` (toàn bộ):** 63 passed, 39 skipped, 14 failed — 14 failed đều là `test_manifest_hashes_match` do Windows CRLF line-endings gây hash mismatch trên tất cả track (t01–t14), không ảnh hưởng logic pipeline.
- **Pipeline exit code:** 0 (thành công). Output 7 files: `ranked_review.csv`, `review_queue.csv`, `contact_sheet.html`, `decision.json`, `decision_log.csv`, `run_info.json`, `audit_log.jsonl`.
- **Máy / CPU / OS / Python:** Windows 11 (10.0.26200) · Intel i7-10750H · Python 3.13.13 · git sha: `9417e3d`

## 09 · Limitations & ethics

1. **Mini gold chỉ 20 ảnh, tự gán.** Đủ để chứng minh cơ chế, **không đủ** để ước lượng tỉ lệ lỗi ngoài mẫu. Số metric chính thức phải lấy từ `reports/metrics_poc.json`.
2. **Chỉ 1 ảnh `glare` được gán nhãn.** Vì vậy 2 ngưỡng glare cục bộ buộc phải fit bằng quy tắc "margin trên ảnh sạch" thay vì fit trực tiếp → nhạy với tập nhãn. Khi có thêm nhãn `glare`, cần re-fit và re-freeze trước khi chấm eval.
3. **Z-score tính trên cả 96 ảnh (dev + eval).** Chuẩn hoá không dùng nhãn (transductive, unsupervised) nên không rò rỉ nhãn eval — nhưng nếu giảng viên yêu cầu mọi thống kê chỉ tính trên dev, cần chuyển sang tính trên dev rồi áp cho eval.
4. **Fixture nhỏ & nhóm không cân bằng:** `night` chỉ có 4 ảnh dev / 16 ảnh eval, `dusk` 16 dev / 4 eval. Số đo theo lát cắt có CI rất rộng — cần bootstrap hoặc nói rõ "số mẫu quá nhỏ để kết luận thống kê".
5. **Ngưỡng đã freeze.** Tuyệt đối không chỉnh lại sau khi đã xem split eval — vi phạm protocol ở `docs/tracks/T02.md` §4.
6. **Chưa xử lý ROI.** Lưới ô 4×6 toàn khung, chưa phân vùng biển số. Hướng cải tiến: ROI-based scoring.
7. **`lap_var` và `lap_norm` có thể phạt nhầm hình vẽ phẳng** (nền ít texture), không chỉ ảnh mờ thật.
8. **Rủi ro pháp lý / PII / license:** dữ liệu là ảnh synthetic nên không có PII thật. Nếu dùng ảnh BDD thật thì phải ghi license và không commit lên repo công khai.
9. **Không tuyên bố "đạt GDPR / luật Việt Nam" từ checklist tự động** — đây là công cụ hỗ trợ, quyết định cuối thuộc người duyệt. Giả định cần DPO/legal xác nhận.
10. **Kế hoạch cải tiến:** ROI chi tiết hơn (phân vùng biển số), thử ảnh đêm thật, thêm nhãn glare, bootstrap CI.

## 10 · Team contribution

**Bảng 2 · Individual contribution evidence** (đối chiếu với `git shortlog -sne --all` và `reports/team_contributions.csv`)

| Học viên | Git author/email | Module / vai trò | Commit SHA / diff chứng minh | Trình bày? |
|---|---|---|---|---|
| Lê Đức Thanh (TV1) | leducthanh26012003@gmail.com | Lead / Integrator: `src/pipeline.py`, `configs/lab.yaml`, merge branches | `bcf04df`, `a7f8c0d`, `5f0a51a`, `98f3889`, `482625b`, `999073c`, `9417e3d` | Có |
| Phạm Trần Anh (TV2) | https://github.com/GemmaP512 | Laplacian baseline: `src/t02/baseline.py`, evaluation files | `ade7f1e` | |
| Nguyễn Viết Minh (TV3) | https://github.com/minh23012003 | Sharpness features: `src/t02/feat_sharpness.py` (lap_var, lap_tile_min, blockiness, lap_norm) | `dc3d72f`, `999073c` | |
| Trần Trọng Thái (TV4) | https://github.com/tranthai2402 | Exposure/glare features: `src/t02/feat_exposure.py`, `tests/test_t02_exposure.py`, `configs/tv4_calibration.yaml` | `517495f` | |
| Phạm Quang Huy (TV5) | https://github.com/feedfoot | Scoring + review: `src/t02/score.py`, `src/t02/review_step.py`, `configs/tv5_score.yaml` | `b18c0e2`, `09e3fc0`, `482625b` | |
| Cao Văn Dương (TV6) | https://github.com/Duong122702 | Research / ablation / report: `reports/group_report.md` | `f478478` | |

## 11 · References

| URL | Ngày xem | 1–2 dòng ý nghĩa (không chép nguyên văn) |
|---|---|---|
| Pech-Pacheco et al. (2000) "Diatom autofocusing in brightfield microscopy" | 14/11/2026 | Laplacian variance là chỉ báo nét được dùng rộng rãi. Nhóm dùng làm baseline. |
| ACDC dataset — ETH Zürich (acdc.vision.ee.ethz.ch) | 14/11/2026 | Ảnh xe trong điều kiện bất lợi (đêm, sương, mưa). Minh hoạ failure mode camera/night. |
| ISO/IEC TR 5259-6:2026 — Data quality for AI | 14/11/2026 | Framework kiểm soát chất lượng dữ liệu ảnh. Gợi ý within-group normalisation cho z-score. |

---

**Bảng 3 · Checklist chốt khi nộp**

- [x] Link repo và tag cuối cùng trong README; input mẫu không chứa PII thật.
- [x] Đúng 1 lệnh run, 1 lệnh evaluate, 1 lệnh tests; không cần download khi chạy.
- [x] Có baseline/PoC metrics (mini gold dev), ≥1 failure case (3 failure cases), reviewer evidence, decision log.
- [x] Có license/source provenance, data splits, seed, offline environment, GT/eval N.
- [x] Mỗi thành viên có ≥1 commit thực chất và được nêu minh bạch trong bảng trên.
