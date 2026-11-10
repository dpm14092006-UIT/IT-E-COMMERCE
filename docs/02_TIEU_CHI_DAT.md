# 02 · Thế nào là "đạt"

"Đạt" ở đây **không** có nghĩa là mô hình có điểm cao. Đạt nghĩa là mỗi con số được đo đúng giao thức trong Section 4 và có thể tái lập. Một kết quả âm (ví dụ GT không thêm giá trị) vẫn đạt nếu đúng giao thức.

Có ba tầng:

- **Tầng A – Khung sườn đạt:** cấu trúc, cấu hình, truy vết và kiểm thử đã sẵn sàng.
- **Tầng B – Từng giai đoạn đạt:** cổng G0 → G10. Chưa qua cổng trước thì không mở cổng sau.
- **Tầng C – Kết luận đạt:** đủ bằng chứng để trả lời RQ1–RQ3.

## Tầng A – Khung sườn (trạng thái hiện tại)

| # | Tiêu chí | Trạng thái |
| --- | --- | --- |
| A1 | Mỗi mục 4.x có module/script tương ứng ([01 §2](01_BOC_TACH_PHUONG_PHAP.md)) | Đạt |
| A2 | Mỗi phương trình (1)–(70) truy vết được tới hàm ([01 §3](01_BOC_TACH_PHUONG_PHAP.md)) | Đạt cho bộ model hiện hành; (17)(18) tắt; TFT (47)(48) được lưu trong kết quả lịch sử nhưng đã loại theo D19 |
| A3 | Mọi tham số nằm trong `configs/*.yaml`, không hard-code trong script | Đạt |
| A4 | Raw data chỉ đọc; output tách theo giai đoạn trong `outputs/` | Đạt |
| A5 | Kiểm thử tự động cho các ràng buộc chống leakage | Lần ghi nhận trước khi thu gọn model: `python -m pytest tests` → 21 passed; chưa chạy lại sau D19 |
| A6 | Pipeline chạy hết vòng trên dữ liệu thật | Đạt với `--grouping attribute`, stage validation (00→09, 12, 13, TCN) |
| A7 | Mỗi giai đoạn ghi manifest (config hash, phiên bản thư viện, sha256 output) | Đạt: `outputs/manifests/` |
| A8 | Test không thể chạy nhầm | Đạt: `07_walk_forward --stage test` từ chối nếu thiếu `--frozen` |
| A9 | Phân công role rõ ràng ([03](03_PHAN_CONG_ROLE.md)) | Đạt |

## Tầng B – Cổng theo giai đoạn

Mỗi cổng gồm điều kiện đo được và artifact làm bằng chứng. Ngưỡng số trong `configs/default.yaml` là **đề xuất đăng ký trước**: được chỉnh, nhưng phải chỉnh *trước* khi xem kết quả liên quan và ghi vào `decision_log.md`.

### G0 · Kiểm toán dữ liệu (P1) — `00_audit_data.py`
- [x] Không có FAIL trong `outputs/reports/G0_data_audit.md`.
- [x] Mọi dòng DECIDE có quyết định tương ứng (D01–D08, D14) trong `decision_log.md`.
- Bằng chứng: `G0_data_audit.{md,json}`, `G0_active_products_by_week.csv`.

### G1 · Embedding (P2 FashionCLIP, P4 ResNet-50) — `02_extract_embeddings.py`
- [x] Số vector = số sản phẩm có ảnh giải mã được (5.354; sản phẩm 4442 loại theo D14).
- [x] Số chiều: FashionCLIP 512, ResNet-50 2048. Không NaN, không vector norm 0.
- [x] Sau chuẩn hoá: norm = 1 ± 1e-6 (đo được 0,9999999–1,0000001).
- [x] Determinism: cosine ≥ 0,9999 khi encode lại 32 ảnh đầu (đo được 0,9999999).
- [x] Ghi lại checkpoint id, weights, tiền xử lý (ảnh RGBA ghép nền trắng) trong manifest.
- Bằng chứng: `embeddings/{enc}_G1.json`, manifest.

### G2 · Visual styles (P2, P4) — `03_cluster_styles.py`
- [x] Centroid chỉ fit trên development catalogue (3.549 sản phẩm; `assert_fit_scope` không báo lỗi).
- [x] `grid.csv` đủ mọi K × encoder: Silhouette, DBI, ARI mean/min/sd (≥ 10 seed), size min/median/max, eligible dev origins tối thiểu mỗi style.
- [x] Ứng viên qua gate `pass_all`: size min ≥ 30, ARI mean ≥ 0,70, eligible dev origins ≥ 20. Kết quả: chỉ ResNet-50 K=10 và K=20; FashionCLIP không qua stability ở K nào.
- [~] Contact sheets và `overview_k*.png` đã xem sơ bộ; P2 + P4 cần xem lại.
- [~] Nhãn nháp: `outputs/clusters/resnet50/style_names_k20_draft.csv`.
- [x] NMI/ARI với category/color/fabric được báo cáo (`reports/G2_encoder_comparison.md`).
- [x] (encoder, K) chốt trong `decision_log.md` (D09 = ResNet-50, K=20) **trước** khi chạy 05.
- Không đạt nếu chọn K dựa trên Macro-F1 forecasting.

### G3 · Product–week và style–week panel (P1, P2/P4) — `01`, `04`
- [x] Tổng sales bảo toàn chính xác qua sales.csv → item–shop–week → product–week → style–week (1.599.100 ở mọi tầng).
- [x] Panel là lưới tuần đầy đủ; A = 0 thì Sales/SPAP = NaN (test `test_style_panel_grid_*`).
- [x] Calendar mapping ở cấp item–shop (test `test_item_shop_alignment_*`).
- [x] Báo cáo: số style, tỉ lệ style–week không active, số style–week sales âm (`panels/*_G3.json`).

### G4 · Features (P3) — `05_build_features_targets.py`
- [x] Future-perturbation test: đổi mọi giá trị sau t thì không feature nào tại ≤ t thay đổi (`test_no_feature_uses_the_future`).
- [x] Lag = lag lịch (không phải lag dòng) khi có tuần trống.
- [x] Window thiếu quan sát → NaN, không tự điền.
- [x] Không cột target nào lọt vào feature (`assert_no_target_columns`).

### G5 · Target (P3) — `05_build_features_targets.py`
- [x] (36)–(40) khớp ví dụ tính tay (`test_market_response_hand_example`).
- [x] θ chỉ dùng nhãn có s + H ≤ calibration_end (`test_label_rule_and_threshold`).
- [x] `target_{grouping}.json` báo cáo: θ, quy tắc dùng (median/fallback), n calibration, phân bố lớp ở calibration/validation/test, coverage (42).
- [x] Tỉ lệ mỗi lớp ≥ 10% trong calibration (Declining 24%, Stable 51%, Emerging 26%).

### G6 · Giao thức validation (P3) — `06_tune_models.py`
- [x] Mọi tập train thoả s + H ≤ t (`assert_label_maturity` chạy ở mọi lần fit).
- [x] Validation origin có t + H ≤ development_end (embargo).
- [x] Mọi mô hình hiện hành có cùng số cấu hình search (`validation.n_iter` = 10). Class weighting: ba tabular thử none/balanced; TCN chỉ none.
- [x] Tham số tốt nhất lưu ở `tuning/*_best.json` rồi đóng băng.
- [x] D10–D12 đã chốt trong `decision_log.md` trước `07 --stage test --frozen`.

### G7 · Mô hình (P3: ba tabular, P4: TCN) — `07_walk_forward.py`
- [x] Xác suất luôn 3 cột theo thứ tự cố định, tổng = 1, kể cả khi train thiếu lớp.
- [x] Baseline majority + persistence chạy trên cùng origin.
- [x] TCN: receptive field 29 ≥ lookback 8 (51); causal (`test_tcn_is_causal`); 5 seed, 0,638 ± 0,004.
- [x] TCN refit mỗi 4 origin, tabular mỗi origin. Đối chiếu công bằng: LightGBM/CatBoost refit mỗi 4 origin (robustness, tham chiếu) đạt 0,629/0,639 so với TCN 0,649.

### G8 · Đánh giá (P3) — `08_evaluate.py`
- [x] Mọi so sánh trên **matched sample** (cùng style × origin, nhãn quan sát được).
- [x] Báo cáo Macro-F1 (chính), BA, accuracy, P/R/F1 từng lớp, confusion, coverage, số dự báo theo mô hình.
- [x] CI 95% block bootstrap (block = 4) cho chênh lệch với persistence và giữa các mô hình chính (mọi cặp chứa 0).
- [x] Macro-F1 theo quý (temporal stability).
- [x] Lớp vắng mặt được báo cáo riêng (không có lớp vắng mặt trên test).

### G9 · Thí nghiệm bổ sung
- [x] Ablation B1 → B2 → B3 → B4a → B4b + M, cùng origin và nhãn, trên test (refit mỗi 4 origin).
- [x] Common endpoint product–week 5 cấu hình (thêm ResNet-50 fine-tuned) trên cùng nhãn.
- [x] Robustness: K, θ (4 phân vị), lookback 4/8/12, target `raw_sales` / `shop_exposure` / `age_adjusted` (thay `incumbent_cohort`, vì target cohort bị lão hoá chi phối), weighting.
- [x] Explainability: permutation importance theo feature và theo block trên test, ghi rõ "liên hệ dự báo, không nhân quả". SHAP chưa cài.

### G10 · Lớp quyết định (P4) — `13_decision_signals.py`
- [x] Điểm vận hành (min_prob, min_margin) chọn trên validation, áp lên test không chỉnh lại.
- [x] Báo cáo: số cảnh báo, precision/recall cảnh báo Emerging và Declining, số Emerging bị bỏ sót thành Stable, số cảnh báo sai.
- [x] Không đầu ra nào là số lượng nhập hay lệnh bổ sung hàng (4.10).

### G11 · Tái lập (tất cả)
- [x] Chạy lại walk-forward với cùng config cho ra predictions giống hệt (CatBoost + RF, 2 lần chạy và lần đã lưu).
- [x] `decision_log.md` đủ mọi dòng [CHỐT].

## Tầng C – Đủ để kết luận

| RQ | Kết luận được phép khi | Kết luận KHÔNG được phép |
| --- | --- | --- |
| RQ1 | G1, G2 đạt; có ARI ổn định, contact sheets nhất quán, NMI với attribute không quá cao (không chỉ tìm lại category) | "FashionCLIP tốt hơn vì Silhouette cao hơn" |
| RQ2 | G5–G8 đạt trên final test; CI so với persistence không chứa 0 | Kết luận từ validation; từ một quý; từ accuracy |
| RQ3 | Common endpoint (G9) cho chênh lệch có CI; ablation trên cùng nhãn | So Macro-F1 style-level visual với attribute rồi kết luận "visual tốt hơn" |

## Những việc làm hỏng toàn bộ kết quả (tự động không đạt)
1. Fit K-Means, scaler, θ hay imputation trên dữ liệu có test.
2. Chọn K, θ, feature set hay hyperparameter sau khi nhìn test.
3. Random K-fold trên style–week.
4. Zero-fill tuần ngoài cửa sổ quan sát; gán Declining cho style biến mất.
5. Dùng mức GT chuẩn hoá toàn cửa sổ (chưa expanding-max).
6. Đưa `pmr_future4`, `delta`, `y` hay `label_ready_t` vào feature.
7. Diễn giải SHAP/importance thành nhân quả, hoặc biến xác suất thành số lượng nhập hàng.
