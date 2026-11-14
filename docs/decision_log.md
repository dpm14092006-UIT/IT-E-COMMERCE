# Decision log — chốt TRƯỚC khi chạy final test

Mỗi dòng: quyết định, lựa chọn, bằng chứng (artifact), người chốt, ngày. Đổi một quyết định đã chốt thì thêm dòng mới, không xoá dòng cũ.
Cột "Đề xuất" là mặc định hiện có trong `configs/default.yaml`. Trạng thái **Mở** = chưa chốt.

**2026-10-12:** theo yêu cầu "làm full toàn bộ", mọi quyết định được chốt theo đề xuất trước khi chạy final test. Tuning dùng `validation.n_iter = 10` cho mọi mô hình. Nhóm có thể xem lại; nếu đổi một quyết định thì final test phải chạy lại và ghi rõ đó là lần chạy thứ hai.

| ID | Quyết định | Đề xuất | Bằng chứng | Chủ | Trạng thái | Ngày |
| --- | --- | --- | --- | --- | --- | --- |
| D01 | Calendar mapping khi release khác nhau giữa shop | T(r_{i,r}, h) ở cấp item–shop; r_i = min theo shop | G0: 91,9% sản phẩm nhiều ngày release | P1 | Chốt theo đề xuất | 2026-10-12 |
| D02 | Định nghĩa active product | Cửa sổ quan sát 12 tuần của ≥ 1 shop; ghi giới hạn left truncation | G0: tuần 0 chỉ 0,25% bằng 0 | P1 | Chốt theo đề xuất | 2026-10-12 |
| D03 | Sales âm (3.386 ô) | Giữ net sales khi cộng; chặn dưới 0 trong log (27)(38) | G0 | P1 | Chốt theo đề xuất | 2026-10-12 |
| D04 | `panel_end` | Giữ tới 2020-03-16 (tuần quan sát cuối) | G0 | P1 | Chốt theo đề xuất | 2026-10-12 |
| D05 | Google Trends: thang đo | Expanding-max renormalisation; −1 → NaN; không ffill sau 2019-12-16 | G0 + test | P1 | Chốt theo đề xuất | 2026-10-12 |
| D06 | Google Trends: trọng số gộp (21) | Không trọng số (`panel.weighting: none`); `n_shops` là robustness | — | P1 | Chốt theo đề xuất | 2026-10-12 |
| D07 | Inventory / restock | Tắt cho tới khi rõ nghĩa `restock` và `restocks.csv` (tuần ISO?) | G0 | P1, P4 | Chốt theo đề xuất | 2026-10-12 |
| D08 | Phân vùng thời gian | calibration_end = validation_start = 2018-07-02; development_end = 2018-12-31; test từ 2019-01-07 | `target_*.json` | P1, P3 | Chốt theo đề xuất | 2026-10-12 |
| D09 | Encoder và K chính | **ResNet-50, K = 20** (primary); FashionCLIP K = 20 làm cấu hình so sánh. Lý do: chỉ ResNet K=10/20 qua gate; K=10 quá thô khi xem ảnh; FashionCLIP ARI 0,58–0,67 < 0,70 ở mọi K. Chọn trước khi chạy forecast cho grouping visual | `outputs/reports/G2_encoder_comparison.md/.png`, `outputs/clusters/resnet50/overview_k20.png`, `style_names_k20_draft.csv` | P2, P4 | Chốt theo đề xuất | 2026-10-12 |
| D10 | θ và quy tắc fallback | median abs Δ trên nhãn trưởng thành trước calibration_end | `target_*.json` | P3 | Chốt theo đề xuất | 2026-10-12 |
| D11 | Feature set chính và eligibility | Block B1–B4b; min_history_weeks = 16 | `09` trên validation | P3 | Chốt theo đề xuất | 2026-10-12 |
| D12 | Hyperparameter và class weighting | `tuning/*_best.json` | `06` | P3, P4 | Chốt theo đề xuất | 2026-10-12 |
| D13 | Điểm vận hành cảnh báo | Chọn trên validation bằng F1 cảnh báo Emerging | `13` | P4 | Chốt theo đề xuất | 2026-10-12 |
| D14 | Ảnh không giải mã được (sản phẩm 4442, AI19/04442.png, hỏng ngay trong zip gốc) | Loại khỏi mọi grouping, kể cả attribute, để các grouping dùng chung mẫu (4.2.5) | G0 `products_undecodable_image` | P1, P2 | Chốt theo đề xuất | 2026-10-12 |

## Lần chạy 2 (mở rộng, đăng ký 2026-10-14 TRƯỚC khi xem kết quả validation)

Mục tiêu:
- Tăng khả năng bắt Emerging.
- Kiểm tra lại RQ3 với cách dùng ảnh chi tiết hơn cluster cứng.

Kết quả lần 1 giữ nguyên, không bị thay thế. Mọi lựa chọn dưới đây dựa **chỉ** trên validation.

| ID | Quyết định | Luật đăng ký trước | Trạng thái |
| --- | --- | --- | --- |
| D15 | Thêm block B5 (visual novelty, dispersion, kNN context) vào cấu hình chính lần 2 | Thêm nếu Macro-F1 validation trung bình của 5 mô hình (4 tabular + TCN) tăng ≥ 0,01 so với không B5; nếu không, báo cáo B5 như phân tích bổ sung | **Không thêm**. Trung bình validation 0,6634 (không B5) so với 0,6634 (có B5), Δ = 0,000. B5 chạy trên test như phân tích bổ sung |
| D16 | Common endpoint mở rộng (kNN thị giác, kNN + novelty, cho ResNet-50 và FashionCLIP) | Không chọn: mọi cấu hình đều chạy trên test và báo cáo kèm CI | Đã chốt |
| D17 | Class weighting cho TCN/TFT | Dùng `balanced` cho mô hình nào có Macro-F1 validation cao hơn `none` | **Giữ `none`**. Validation: TCN 0,664 (none) so với 0,604 (balanced), TFT 0,631 so với 0,575. Balanced tăng recall Emerging lên 0,79/0,76 nhưng giảm Macro-F1 |
| D18 | Luật quyết định (điểm vận hành) | Hệ số nhân xác suất chọn trên validation: "macro" (max Macro-F1) và "recall70" (max Macro-F1 với recall Emerging ≥ 0,70); áp nguyên lên test, không huấn luyện lại | Đã chốt. Test: CatBoost "macro" 0,672 (+0,024 [+0,007; +0,044] so với argmax) |

## Cập nhật 2026-11-10 — bộ model hiện hành

| ID | Quyết định | Lựa chọn | Bằng chứng | Chủ | Trạng thái | Ngày |
| --- | --- | --- | --- | --- | --- | --- |
| D19 | Thu gọn model trong code/config | **Giữ Random Forest, LightGBM, CatBoost và TCN; loại XGBoost và TFT khỏi luồng chạy.** Kết quả final test sáu model cũ vẫn được lưu như hồ sơ lịch sử. | Macro-F1 test: TCN 0,649; CatBoost 0,648; LightGBM 0,642; RF 0,637. Chênh lệch giữa các model không có ý nghĩa thống kê; RF được giữ thay cho XGBoost đang gần như đồng điểm để giảm trùng lặp boosting và giữ lựa chọn bagging. | User | Chốt | 2026-11-10 |

## Cập nhật 2026-11-14 — tích hợp Myntra

| ID | Quyết định | Lựa chọn | Bằng chứng |
| --- | --- | --- | --- |
| D20 | Encoder chính sau khi tích hợp Myntra | **FashionCLIP fine-tuned, K = 20**. Luồng trích embedding hỗ trợ huấn luyện cả hai encoder trên Myntra. | Tại K=20, FashionCLIP fine-tuned qua các tiêu chí về kích thước, độ ổn định và coverage (ARI 0,816). ResNet-50 fine-tuned đạt ARI 0,684. Walk-forward validation dùng siêu tham số mặc định. Final test của D09 thuộc cấu hình ResNet-50 pretrained. |
