# 01 · Bóc tách Methodology (Section 4) → khung sườn thực thi

> Tài liệu này dịch từng câu "phải / không được" trong Section 4 thành **một hàm, một artifact và một kiểm tra**.
> Đi kèm: [02_TIEU_CHI_DAT.md](02_TIEU_CHI_DAT.md) (thế nào là đạt), [03_PHAN_CONG_ROLE.md](03_PHAN_CONG_ROLE.md) (ai làm gì), [decision_log.md](decision_log.md) (các quyết định phải chốt).

## 0. Bài toán trong một đoạn

- **Đơn vị phân tích:** visual style × calendar week (không phải SKU).
- **Tại forecast origin t** chỉ được dùng thông tin có tại hoặc trước t.
- **Đại lượng thị trường:** SPAP có trọng số exposure trên 4 tuần.
  - Quá khứ: P_past(t) = ΣSales(t−3..t) / ΣA(t−3..t).
  - Tương lai: P_future(t) = ΣSales(t+1..t+4) / ΣA(t+1..t+4).
- **Mục tiêu:** Δ(t) = log(1+P_future) − log(1+P_past) được chia thành 3 lớp bằng ngưỡng θ = median|Δ|. Ngưỡng θ chỉ calibrate trên nhãn đã trưởng thành trong development.
  - Δ > θ → **Emerging**.
  - |Δ| ≤ θ → **Stable**.
  - Δ < −θ → **Declining**.
- **Hai giai đoạn:**
  1. Ảnh → embedding (FashionCLIP, ResNet-50) → K-Means → style.
  2. Style–week panel → features → 4 mô hình hiện hành (3 tabular + TCN) + 2 baseline, đánh giá bằng walk-forward.

## 1. Pipeline, artifact, gate, người phụ trách

```
Data/extracted/visuelle2 (raw, chỉ đọc)
│
├─ 00_audit_data ............ G0  (P1)  outputs/reports/G0_data_audit.{md,json}
├─ 01_build_product_week .... G3a (P1)  interim/{item_shop_week,product_week,products}.parquet   (12)(13)
│   └─ 01b_market_insights ..     (P1)  reports/P1_*.csv  (attribute, lifecycle, seasonality, GT lead-lag)
├─ 02_extract_embeddings .... G1  (P2 FashionCLIP | P4 ResNet-50)  embeddings/{enc}.npy          (1)-(5)
├─ 03_cluster_styles ........ G2  (P2 | P4)  clusters/{enc}/grid.csv, assignments_k*.parquet, sheets/  (6)-(11)(60)
│        └──► CHỐT (encoder, K) trong decision_log + configs/default.yaml: visual.selected
├─ 04_build_style_panel ..... G3b (P2 | P4 | P1 attribute)  panels/style_week_{grouping}.parquet  (14)-(21)
├─ 05_build_features_targets  G4,G5 (P3)  features/features_{grouping}.parquet, target_{grouping}.json  (23)-(42)
├─ 06_tune_models ........... G6  (P3)  tuning/{grouping}_{model}_best.json                         (59)
├─ 07_walk_forward .......... G7  (P3 tabular | P4 TCN)  predictions/{grouping}_{stage}_{tag}.parquet (43)-(58)(66)
├─ 08_evaluate .............. G8  (P3)  metrics/{grouping}_{stage}_{tag}.json                        (61)-(65)(69)(70)
├─ 09_ablation .............. G9  (P3)  metrics/*_ablation.csv                                      (67)
├─ 10_common_endpoint ....... G9  (P3)  [spec]                                                     (68)
├─ 11_robustness ............ G9  (P4)  [spec]                                                     4.9
├─ 12_interpret ............. G9  (P4)  metrics/*_importance*.csv                                   4.10
└─ 13_decision_signals ...... G10 (P4)  reports/*_signals.csv, *_decision_scenarios.csv             4.10
```

## 2. Bóc tách từng mục

| Mục | Phải làm | Ràng buộc bắt buộc (không làm = không đạt) | Hiện thực |
| --- | --- | --- | --- |
| 4 (mở đầu) | Kiểm tra cấu trúc thời gian trước khi forecast | Không mặc định dữ liệu là balanced calendar panel | `data/audit.py`, `00_audit_data.py` |
| 4.1.1 | FashionCLIP làm extractor cố định | Không fine-tune; tiền xử lý theo checkpoint; L2-normalize (2) | `visual/encoders.py::encode_fashionclip` |
| 4.1.2 | ResNet-50 bỏ fc, lấy GAP 2048-d | Frozen; hai embedding **không** ghép với nhau | `encode_resnet50` |
| 4.1.3 | K-Means trên K ∈ {10..30}, nhiều init | Silhouette + DBI + nhìn ảnh + stability + coverage; K chọn trong development | `visual/clustering.py` |
| 4.1.4 | Stability ARI, ảnh đại diện gần centroid **và** ngẫu nhiên | Centroid chỉ fit trên **development catalogue**; sản phẩm mới gán theo (11) | `assert_fit_scope`, `assign`, `inspection.py` |
| 4.2.1 | Release-relative → calendar week, cộng shop → product–week | Không zero-fill ngoài cửa sổ quan sát; flow (sales) cộng, stock (inventory) không cộng lặp | `data/calendar.py`, `product_week.py` |
| 4.2.2 | Sales, A, SPAP theo style–week | A không định nghĩa bằng sales > 0; A = 0 → SPAP **không xác định** | `panel/style_week.py` |
| 4.2.3 | Inventory, availability, price, discount (19) | Chỉ xây khi mẫu số đáng tin; giữ price/discount riêng; giữ A làm feature | `panel/style_week.py`, `data/inventory.py` (tắt) |
| 4.2.4 | Google Trends ở cấp style (21) | Không dùng GT tương lai; không chuẩn hoá bằng dữ liệu tương lai; chỉ gộp khi cùng thang đo | `data/gtrends.py` (expanding-max) |
| 4.2.5 | Phân biệt missing / zero / stock-out / non-availability | Impute chỉ bằng thông tin quá khứ/training | grid NaN + mask trong `features/sequences.py` |
| 4.3.1–4.3.4 | Lag {1,2,4,8}, MA {4,8}, growth, trend, vol, calendar sin/cos | Mọi biến đổi chỉ nhìn về quá khứ; window phải đủ quan sát | `features/historical.py` |
| 4.3.5 | Block B1–B4 | Nested, cùng origin/target | `features/blocks.py` |
| 4.3.6 | Tabular (32) và TCN sequence (33) | Scaler fit trên train; mask missing; **cùng information cutoff** | `features/sequences.py`, `SequenceClassifier` |
| 4.4.1 | P_past, P_future, Δ (36)–(38) | Mẫu số > 0 và tuần liên quan hợp lệ | `target/labels.py` |
| 4.4.2–4.4.3 | Nhãn (40), θ (41) + fallback | θ từ calibration sample đã trưởng thành; cố định cho test | `target/threshold.py` |
| 4.4.4 | Eligibility tại t, coverage (42) | Không dùng "style còn sống trong tương lai" để chọn mẫu | `target/eligibility.py` |
| 4.5 | 4 mô hình hiện hành xuất xác suất 3 lớp (43)–(44): Random Forest, LightGBM, CatBoost, TCN | Cùng target, cùng cutoff; class weight chỉ từ train | `models/tabular.py`, `models/sequence.py` |
| 4.6.1–4.6.3 | Chia theo thời gian, walk-forward mở rộng | **s + H ≤ t** cho mọi mẫu train | `validation/walk_forward.py`, `assert_label_maturity` |
| 4.6.4 | Tuning bằng Macro-F1 trên validation | Không dùng test; ngân sách search như nhau | `validation/tuning.py` |
| 4.6.5 | Chống leakage ở 6 tầng | Không random CV | `utils/leakage.py` + `tests/` |
| 4.7 | Macro-F1 (chính), BA, P/R/F1 từng lớp, confusion, coverage | So sánh trên **matched sample** | `evaluation/metrics.py`, `08_evaluate.py` |
| 4.8 | Baseline, so sánh encoder, ablation, common endpoint, so sánh kiến trúc | Không so Macro-F1 giữa các bài toán có nhãn khác nhau | `09`, `10`, `07` |
| 4.9 | Sensitivity K, θ, lookback, target thay thế; block bootstrap | Không đổi θ chính sau khi thấy test | `11_robustness.py`, `evaluation/bootstrap.py` |
| 4.10 | Importance / SHAP, phân tích lỗi E↔S, false alerts | Liên hệ dự báo, **không** phải nhân quả; không tự động ra số lượng nhập | `interpret/explain.py`, `decision/rules.py` |
| 4.11 | Lưu checkpoint, seed, config, assignments, xác suất từng quan sát | Tái lập được từ raw | `utils/repro.py` → `outputs/manifests/*.json` |

## 3. Ma trận truy vết phương trình → code → test

| Pt | Nội dung | Hàm | Test |
| --- | --- | --- | --- |
| (1)(4) | Embedding FashionCLIP / ResNet-50 | `encode_fashionclip`, `encode_resnet50` | G1 json (dim, NaN, determinism) |
| (2)(5) | L2 normalize | `l2_normalize` | G1 |
| (6)(11) | K-Means, gán centroid gần nhất | `fit_kmeans`, `assign` | — |
| (7) | Lưới K | `visual.kmeans.k_grid` | — |
| (8)(9)(10) | Silhouette, DBI | `evaluate_grid` | — |
| (12) | T(r_{i,r}, h) | `to_calendar_week`, `to_item_shop_week` | `test_calendar_mapping_*`, `test_item_shop_alignment_*` |
| (13) | Cộng shop → product–week | `build_product_week` | bảo toàn tổng sales |
| (14)(15)(16) | Sales, A, SPAP | `build_style_week` | `test_style_panel_grid_*` |
| (17)(18) | Inventory, availability | `data/inventory.py` (tắt chờ G0) | — |
| (19)(21) | Trung bình có trọng số price/discount/GT | `build_style_week(weighting=...)` | — |
| (20) | G_{i,t} theo category/color/fabric | `gtrends.attach_to_products` | `test_gtrends_renormalisation_*` |
| (23)–(31) | Lag, MA, growth, trend, vol, calendar | `add_history_features` | `test_lag_is_calendar_lag`, `test_no_feature_uses_the_future` |
| (32)(33)(34) | Tabular / sequence, lookback | `select_features`, `build_sequences` | `test_sequences_pad_*` |
| (35)–(38) | H = 4, P_past, P_future, Δ | `add_market_response` | `test_market_response_hand_example`, `test_labels_do_use_the_future` |
| (39)(40)(41) | 3 lớp, θ, fallback | `label_from_delta`, `threshold.calibrate` | `test_label_rule_and_threshold` |
| (42) | Coverage | `eligibility.coverage` | — |
| (43)–(46) | Xác suất 3 lớp, RF, boosting | `AlignedClassifier`, `make_model` | `test_aligned_classifier_with_missing_class` |
| (47)(48) | TFT + softmax head | Đã đánh giá trong báo cáo lịch sử; loại khỏi bộ model hiện hành theo D19 | — |
| (49)(50)(51) | TCN, receptive field | `build_tcn`, `receptive_field` | `test_tcn_is_causal`, `test_receptive_field_*` |
| (52)(53) | CE / weighted CE | `SequenceClassifier`, `compute_sample_weight` | — |
| (54)–(58) | Walk-forward, s + H ≤ t | `run_walk_forward`, `assert_label_maturity` | `test_walk_forward_respects_label_maturity`, `test_guards` |
| (59) | argmax Macro-F1 validation | `tuning.search` | — |
| (60) | ARI | `clustering.stability` | — |
| (61)–(65) | P, R, F1, Macro-F1, BA | `metrics.summary` | `test_metrics_and_bootstrap` |
| (66) | Majority baseline | `MajorityClass` | — |
| (67) | ΔMacro-F1 ablation | `09_ablation.py` | — |
| (68) | Context leave-one-out | `leave_one_out_context` | `test_leave_one_out_context` |
| (69)(70) | Block bootstrap CI | `paired_macro_f1_difference` | `test_metrics_and_bootstrap` |

## 4. RQ → thí nghiệm → bằng chứng tối thiểu

| RQ | Thí nghiệm | Bằng chứng phải có | Ghi chú |
| --- | --- | --- | --- |
| RQ1 | 02 + 03 cho cả hai encoder, cùng dev catalogue | Bảng K × encoder (Silhouette, DBI, ARI mean±sd, size min, coverage); NMI/ARI với category/color/fabric; contact sheets; lý do chọn | Silhouette giữa 2 không gian embedding không so trực tiếp được |
| RQ2 | 05–08 trên grouping đã chốt, final test | Macro-F1 + BA + P/R/F1 từng lớp + confusion; so với majority và persistence; CI bootstrap; coverage; theo quý | Kết quả âm vẫn là câu trả lời hợp lệ |
| RQ3 | 09 ablation, 10 common endpoint, 05–08 với `--grouping attribute` | ΔMacro-F1 theo block; endpoint chung cấp product–week cho 4 cấu hình | Visual vs attribute ở cấp style chỉ là sensitivity (nhãn khác nhau); bằng chứng chính là 10 |

## 5. Phát hiện từ dữ liệu thật (G0 đã chạy)

| # | Sự thật đo được | Hệ quả cho phương pháp |
| --- | --- | --- |
| D1 | 100% `release_date` là thứ Hai | Không có partial week; T(r,h) = r + 7h |
| D2 | 91,9% sản phẩm có `release_date` khác nhau giữa các shop (trung vị lệch 4 tuần, tối đa 66) | (12) phải áp dụng ở cấp item–shop trước khi cộng (13); r_i := min r_{i,r}. Sửa văn bản 4.2.1 |
| D3 | Tuần 0 chỉ 0,25% ô bằng 0, tuần 1 là 26,8% | `release_date` gần như là tuần bán đầu tiên (left truncation): item–shop vào panel đúng tuần có bán, nên A và Sales tương lai gắn cơ học với rollout |
| D4 | Mỗi item–shop chỉ có 12 tuần | Sau tuần 11 là không quan sát (censoring), không phải ngừng bán; không phân biệt được discontinuation |
| D5 | 3.386 ô sales âm (min −3), dồn ở tuần 4, 5, 9, 10 | Net sales có hàng trả lại: giữ khi cộng, chặn dưới 0 trong log (27)(38). Cần chốt |
| D6 | Giá: 1 giá chuẩn hoá tĩnh mỗi item–shop; discount theo 12 tuần | Price ở cấp style chỉ đổi do thành phần assortment |
| D7 | Không có stock snapshot; `restock` ≈ 1,4× sales 12 tuần; `restocks.csv` có 379.562 item–shop, năm 2016–2022 | (17)(18) tắt cho tới khi chốt ngữ nghĩa; stock-out là giới hạn |
| D8 | GT là panel tuần liên tục 2015-10-05 → 2019-12-16, 96 term khớp đủ category/color/fabric | Dùng nhánh calendar (21), không phải pre-release window |
| D9 | 80/96 chuỗi GT chạm 100; 660 ô −1 | Chuẩn hoá toàn cửa sổ = leak thang đo. Sửa: g_t / max(g_s, s ≤ t); −1 → NaN |
| D10 | Panel bán hàng tới 2020-03-16, GT dừng 2019-12-16 | ~5% product–week cuối thiếu GT → missing + mask, không ffill |
| D11 | 16,5% style–week (attribute) có A = 0 | Mùa SS/AW tạo khoảng trống theo style → gate coverage khi chọn K |
| D12 | 1 ảnh (sản phẩm 4442) hỏng ngay trong zip gốc; đọc kiểu chấp nhận cụt thì mất khoảng 26% phần dưới | Loại khỏi mẫu phân tích của mọi grouping (D14), cờ `image_ok` trong `products.parquet` |

## 6. Lỗ hổng trong văn bản Methodology và cách khung sườn xử lý

Mức: **Cao** = có thể làm sai kết luận; **TB** = ảnh hưởng diễn giải; **Thấp** = làm rõ.

1. **[Cao] Ablation B3→B4 gộp Google Trends với calendar** (4.8.3 nói B3 vs B4 kiểm tra GT). Khung sườn tách B4a (+calendar) và B4b (+GT); giá trị của GT = B4a→B4b.
2. **[Cao] Target gắn cơ học với rollout và vòng đời (D3, D4).** P_future gồm cả sản phẩm/shop mới vào trong t+1..t+4, và item–shop mới luôn có sales ≥ 1 ở tuần vào. Smoke run (attribute, validation): block B3 tăng Macro-F1 +0,07 → +0,11, feature mạnh nhất là `shop_exposure__growth1`. Xử lý: giữ `mean_age`, `share_new`, `shop_exposure` trong B3; robustness bắt buộc với target `incumbent_cohort` và `shop_exposure`; diễn giải "trend" thận trọng.
3. **[Cao] Rò rỉ thang đo Google Trends (D9).** Văn bản chỉ cấm dùng quan sát tương lai, chưa nói chuẩn hoá max toàn cửa sổ đã chứa tương lai. Sửa bằng expanding-max (có test).
4. **[Cao] Ranh giới validation/test cần embargo.** Validation origin t phải có t + H ≤ development_end, nếu không nhãn validation dùng tuần của test. `origin_sets` cắt validation tại dev_end − H.
5. **[TB] Mốc calibration θ chưa rõ.** Khung sườn calibrate θ trên nhãn trưởng thành trước `validation_start`, nên cả validation lẫn test đều không ảnh hưởng θ.
6. **[TB] Yêu cầu lịch sử khác nhau giữa mô hình.** Trend_8 cần 16 tuần, lookback TCN ≤ 12. Một eligibility chung (`min_history_weeks: 16`) tạo matched sample cho mọi mô hình hiện hành.
7. **[TB] Persistence baseline chưa định nghĩa đủ.** Dùng Δ_hist(t) = log(1+P_past(t)) − log(1+P_past(t−4)), phân lớp bằng cùng θ.
8. **[TB] θ = median|Δ| ngầm tạo khoảng 50% Stable**, nên majority baseline cho Macro-F1 ≈ 0,2 (smoke run: 0,205). Đây là hệ quả cấu trúc.
9. **[TB] Common endpoint (4.8.4) bị giới hạn bởi cửa sổ 12 tuần.** Origin product–week chỉ ở tuổi 3–7 (≤ 5 origin mỗi sản phẩm), Declining sẽ áp đảo do vòng đời; θ product-level calibrate riêng.
10. **[TB] Phân bố lớp dịch chuyển giữa validation và test** (attribute: Declining 29% → 39%). Báo cáo theo quý, không chỉ tổng.
11. **[Thấp] "Active product" chỉ là proxy cửa sổ quan sát**; không có listing status (4.2.2 yêu cầu nêu rõ).
12. **[Thấp] So FashionCLIP với ResNet-50 bằng Silhouette** không hợp lệ giữa hai không gian khác chiều. Thêm NMI/ARI với thuộc tính catalogue.
13. **[Thấp] Ảnh là RGBA.** Phải ghép nền trắng trước khi encode, nếu không vùng trong suốt thành đen.
14. **[Thấp] Mục 3.1 ghi dữ liệu bán tới 12/2019**, thực tế tuần quan sát cuối là 2020-03-16. Sửa văn bản hoặc chốt `data.panel_end`.
