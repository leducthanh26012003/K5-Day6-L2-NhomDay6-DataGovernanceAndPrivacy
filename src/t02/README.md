# T02 · Giao diện giữa các module (chốt bởi TV1)

Quy ước chung
- Đầu vào: `images` = bảng `images.csv` (cột `image_id, path, camera_id, light, split`).
  Đường dẫn ảnh đầy đủ = `root / path`, với `root = ctx.input_dir`.
- Mỗi hàm trả **đúng 1 dòng cho mỗi ảnh, đủ 96 ảnh** (cả dev và eval), có cột `image_id` (str).
- Không đọc `fixtures/*/gt/`. Mọi ngưỡng / trọng số / trung vị chỉ tính trên `split == "dev"`.
- Chạy lại phải ra cùng kết quả (seed 42, không random nếu không cần).

| File | Chủ | Hàm | Cột trả về |
|---|---|---|---|
| `feat_sharpness.py` | TV3 | `compute(images, root) -> DataFrame` | `image_id, lap_var, lap_tile_min, blockiness` (+ `lap_norm` sau khi chuẩn hoá) |
| `feat_exposure.py` | TV4 | `compute(images, root) -> DataFrame` | `image_id, luma_mean, luma_p05, luma_p95, sat_frac, sat_tile_max` (+ `luma_rel`) |
| `score.py` | TV5 | `combine(feats, meta, params) -> DataFrame` | `image_id, score, pred_defect_type, pred_action, reason` |
| `review_step.py` | TV5 | `run(ctx, scored) -> DataFrame` | như `scored`, đã áp override; ghi `contact_sheet.html`, audit log |
| `baseline.py` | TV2 | `run(ctx)` | ghi `ranked_review.csv` + `decision.json` |

Ý nghĩa giá trị
- `lap_var`, `lap_tile_min`: càng **cao** càng **nét**. `blockiness`: càng cao càng nhiều lỗi nén.
- `luma_*`: thang 0–255. `sat_frac`, `sat_tile_max`: tỉ lệ 0–1 pixel cháy sáng (> 245).
- `score`: càng **cao** càng **cần review trước**.
- `pred_defect_type` ∈ {`none`, `blur`, `glare`, `dark`, `compression`}; `pred_action` ∈ {`KEEP`, `RECAPTURE`}.

Bản khung (push trước phút 30, tạm trả 0)

```python
import pandas as pd
from pathlib import Path

def compute(images: pd.DataFrame, root: Path) -> pd.DataFrame:
    out = images[["image_id"]].copy()
    out["lap_var"] = 0.0          # đổi tên cột theo bảng ở trên
    return out
```

---

## TV4 · Bổ sung giao diện riêng (dark + glare)

TV4 giữ nguyên hàm `compute(images, root)` như trên và **bổ sung** các hàm calibration.
Nhãn dùng để fit là `reports/mini_gold_dev.csv` — 20 ảnh **dev** do nhóm tự gán
(`is_defect`, `defect_type`, `note`). Không đọc `fixtures/*/gt/`.

| Hàm | Chủ | Mô tả |
|---|---|---|
| `compute(images, root) -> DataFrame` | TV4 | Như trên, **cộng** các cột `luma_rel`, `sat_frac_rel`, `sat_tile_max_rel`, `sat_blob_frac_rel`, `dark_frac`, `sat_blob_px`, `dark_tile_min`. |
| `calibrate(features, labels) -> dict` | TV4 | Fit ngưỡng từ mini gold **dev**; trả về dict ngưỡng + thông tin provenance. |
| `apply_calibration(features, params) -> DataFrame` | TV4 | Thêm `tv4_dark_evidence`, `tv4_glare_evidence`, `tv4_evidence` (0/1). |
| `load_mini_gold(path) -> DataFrame` | TV4 | Đọc nhãn dev (tự động lọc `split == "dev"`). |
| `load_calibration(path) -> dict` | TV4 | Đọc ngưỡng **đã freeze** từ `configs/tv4_calibration.yaml`. |

Cột trả về của `compute` (đầy đủ), theo thứ tự:

```
image_id, camera_id, light, split,
luma_mean, luma_std, luma_p05, luma_p95, luma_range,      # thang 0-255
dark_frac,                                              # tỉ lệ pixel < 30
sat_frac, sat_tile_max,                                 # tỉ lệ 0-1 pixel > 245
sat_blob_frac, sat_blob_px,                             # blob cháy sáng lớn nhất
dark_tile_min,                                          # ô tối nhất (mean luma)
luma_rel, sat_frac_rel, sat_tile_max_rel, sat_blob_frac_rel   # z-score trong nhóm light×camera
```

Ý nghĩa các cột *tương đối* (`*_rel`)
- Là **z-score trong nhóm `light` × `camera_id`**, không phải trên toàn pool.
- Bắt buộc cho bài toán này: ảnh đêm hữu ích có `dark_frac` ≈ 0.63, ảnh dark thật
  có `dark_frac` ≈ 0.75 — **cùng dải giá trị**, chỉ phân biệt được khi so tương đối.
- `luma_rel` **âm** nghĩa là tối hơn nhóm của nó. `sat_*_rel` **dương** nghĩa là
  cháy sáng hơn nhóm của nó.

Cách dùng trong `score.py` (TV5)

```python
from src.t02.feat_exposure import compute, apply_calibration, load_calibration

feats = compute(images, ctx.input_dir)
feats = apply_calibration(feats, load_calibration("configs/tv4_calibration.yaml"))
# -> feats["tv4_evidence"] : 1.0 nếu ảnh tối hoặc cháy sáng, 0.0 nếu bình thường
```

Ngưỡng đã freeze: `configs/tv4_calibration.yaml` (fit trên `reports/mini_gold_dev.csv`,
split `dev`, seed 42). Kết quả kiểm chứng trên mini gold: **TV4 recall 4/4**,
**false alarm 0/11** trên ảnh nhãn `none`.

Lưu ý cho TV5
- `tv4_evidence` chỉ nói "tối hoặc cháy sáng". Để gán `pred_defect_type` cần
  phân biệt `dark` với `glare`: dùng `tv4_dark_evidence` vs `tv4_glare_evidence`
  (TV4 đã tách sẵn hai cột này).
- Ngưỡng trong `configs/tv4_calibration.yaml` **đã freeze** — không chỉnh lại sau khi
  đã xem kết quả split `eval` (vi phạm protocol ở `docs/tracks/T02.md` §4).


---

## TV5 · Chấm điểm (`score.py`) + human review (`review_step.py`)

### `score.combine(feats, meta, params)`

Mỗi loại lỗi có một **biên** `margin_<loại>` trên cùng thang đo; `> 0` = loại đó bật.

| Loại | Feature (chủ) | Ngưỡng | Nguồn ngưỡng |
|---|---|---|---|
| `blur` | `log10(lap_norm)` thấp (TV3, đã chuẩn hoá theo camera) | `lap_norm < 0.074` | TV5 fit trên mini gold dev |
| `compression` | `log10(blockiness)` cao (TV3) | `blockiness > 2.05` | TV5 fit trên mini gold dev |
| `dark` | `luma_rel` thấp (TV4) | `luma_rel < -5.24` | **frozen TV4** |
| `glare` (burn) | `sat_*_rel` + `sat_blob_px` (TV4) | như `tv4_glare_evidence` | **frozen TV4** |
| `glare` (veil) | `luma_rel` cao (TV4) | `luma_rel > +5.24` | gương của ngưỡng dark TV4 |

```
margin_d         = (độ lệch so với ngưỡng, theo hướng "càng lỗi") / scale_d
scale_d          = nửa khe hở giữa ảnh lỗi và ảnh sạch gần ngưỡng nhất (mini gold dev)
score            = sigmoid(max_d margin_d)            # 0..1, cao = review trước
pred_defect_type = argmax_d margin_d  nếu max > 0, ngược lại "none"
pred_action      = RECAPTURE nếu score > 0.5, ngược lại KEEP
```

- **Veiling glare:** glare trong `src/common/corruptions.py` là vệt sáng Gaussian lớn. Trên ảnh
  đêm nó làm sáng cả vùng nhưng không làm pixel cháy (> 245), nên `sat_*` không thấy.
  TV5 bắt thêm bằng `luma_rel` cao bất thường; không đổi ngưỡng của TV4.
- Cột phụ để phân tích failure: `margin_dark, margin_glare, margin_blur, margin_compression, candidate_type`
  (`candidate_type` = loại có biên cao nhất kể cả khi chưa vượt ngưỡng).
- Ngưỡng **freeze** ở `configs/tv5_score.yaml`, sinh bằng `python -m src.t02.score --fit`
  (chỉ đọc `reports/mini_gold_dev.csv`, split dev). Pipeline không đọc nhãn lúc chạy.
  `tests/test_t02_score_review.py` kiểm file freeze đúng là kết quả fit (không sửa tay).
- `params` (tuỳ chọn): `tv5_score_config`, `tv4_calibration` để trỏ file khác.

### `review_step.run(ctx, scored)`

Ghi `contact_sheet.html` (top-B, có loại lỗi + lý do). Nếu có `<output>/review_queue_reviewed.csv`
thì áp quyết định và ghi `audit_log.jsonl`:

| `reviewer_decision` | `override_value` | Kết quả |
|---|---|---|
| `approve` | — | giữ đề xuất (audit `review:approve`) |
| `reject` | trống | lật đề xuất: RECAPTURE → KEEP/`none`; KEEP → RECAPTURE/`candidate_type` |
| `reject` / `edit` | `KEEP` / `RECAPTURE` | đặt hành động (loại lỗi tự khớp) |
| `reject` / `edit` | `none`/`blur`/`glare`/`dark`/`compression` | đặt loại lỗi (hành động tự khớp) |
| `escalate` | — | giữ đề xuất, đếm vào `ctx.stats["n_escalated"]` |
| `skip` | — | giữ đề xuất |

- `score` **không bao giờ bị sửa** (metric đo xếp hạng trước khi người can thiệp).
  Quyết định gốc giữ ở `model_action`, `model_defect_type`; thêm cột `reviewer`, `reviewer_decision`.
- Đổi hành động dùng `common.review.apply_overrides` → audit `override:<decision>`;
  đổi loại lỗi → audit `override_type:<decision>`.
- `ctx.stats["review_done"]` = **mọi ảnh trong top-B đều có quyết định** (không chỉ là file tồn tại).
- Lỗi rõ ràng (exit 2): `edit` thiếu giá trị, giá trị lạ, thiếu tên reviewer, `image_id` không có
  trong lần chạy, một ảnh bị review hai lần.
