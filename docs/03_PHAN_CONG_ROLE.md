# 03 · Phân công theo role

Methodology Section 4 là văn bản gốc. Bảng role được ánh xạ vào nó; chỗ nào bảng role yêu cầu khác phương pháp thì ghi ở cột "Lệch so với phương pháp" để nhóm quyết định.

Theo yêu cầu: **Người 4 sở hữu cả hai thành phần CNN** của phương pháp:
- **ResNet-50** (mục 4.1.2): CNN cho ảnh, cấu hình phân nhóm thứ hai.
- **TCN** (mục 4.5.6): CNN theo thời gian để dự báo xu hướng.

**Bộ model hiện hành theo D19:** Random Forest, LightGBM, CatBoost và TCN. XGBoost và TFT là model thử nghiệm đã được đánh giá trước đó, nay không còn trong cấu hình hoặc code chạy.

## Tổng quan

| | Người 1 · Data & Market Intelligence | Người 2 · Visual Trend Intelligence | Người 3 · Forecasting & ML | Người 4 · Business Decision & Analytics Lead (+CNN) |
| --- | --- | --- | --- | --- |
| Mục Section 4 | 3, 4 (mở đầu), 4.2.1, 4.2.4, 4.2.5, nhóm attribute ở 4.8.2 | 4.1.1, 4.1.3, 4.1.4, 4.2.2 (panel visual) | 4.3, 4.4, 4.5.1–4.5.4 (ba tabular hiện hành), 4.6, 4.7, 4.8.1, 4.8.3–4.8.5 | **4.1.2 ResNet-50**, **4.5.6 TCN**, 4.9, 4.10 |
| Script | 00, 01, 01b, 04 `--grouping attribute` | 02/03/04 `fashionclip` | 05, 06, 07 (ba tabular), 08, 09, 10 | 02/03/04 `resnet50`, 07 `--models tcn`, 11, 12, 13 |
| Module | `data/*` | `visual/*` (FashionCLIP), `panel/*` | `features/*`, `target/*`, `validation/*`, `models/tabular.py`, `models/baselines.py`, `evaluation/*`, `experiments/*` | `visual/encoders.py::encode_resnet50`, `models/sequence.py` (TCN), `interpret/*`, `decision/*` |
| Cổng | G0, G3a | G1, G2, G3b (FashionCLIP) | G4–G8, G9 (ablation, endpoint) | G1, G2 (ResNet), G7 (TCN), G9 (robustness, explain), G10 |
| Câu hỏi business | Thị trường và hành vi bán có đặc điểm gì? Yếu tố nào liên quan đến demand? | Phong cách thị giác nào đang nổi lên hay suy giảm? | Dự báo đúng hướng đến mức nào? Ảnh và tín hiệu thị trường giúp được bao nhiêu? | Retailer nên dùng dự báo cho quyết định nào? |
| Đầu ra | Báo cáo G0 + quyết định D01–D08; bảng hiệu suất theo category/color/fabric/mùa; vòng đời; mùa vụ; GT lead-lag | Cluster, ảnh đại diện, nhãn style, ổn định ARI, PCA/UMAP (chỉ để trực quan), Visual Trend Radar | Bảng so sánh mô hình (matched, CI), ablation, common endpoint, coverage | So sánh CNN (ResNet) với FashionCLIP; TCN so với tree models; importance; robustness; cảnh báo + kịch bản + dashboard |

## Người 1 · Data & Market Intelligence Analyst
**Đã chạy:** `00_audit_data.py` (0 FAIL, 7 DECIDE), `01_build_product_week.py`, `01b_market_insights.py`.

Việc tiếp theo:
1. Chốt D01–D08 trong `decision_log.md`: inventory, sales âm, panel_end, GT.
2. Viết insight từ `outputs/reports/P1_*.csv`. Kết quả sơ bộ:
   - Doanh số tuần 0 chiếm 12,2% tổng 12 tuần (vòng đời giảm dần).
   - GT tăng trước sales khoảng **2 tuần**, nhưng tương quan trung bình chỉ khoảng **0,11**: tín hiệu yếu, chỉ là liên hệ.
3. Nối với `EDA_Visuelle2.ipynb` và `figures/` đã có, không nhân đôi.
4. Giao `interim/products.parquet` cho P2/P4 (danh sách ảnh) và `interim/product_week.parquet` cho P3.

```bash
python scripts/00_audit_data.py
```
```bash
python scripts/01b_market_insights.py
```

## Người 2 · Visual Trend Intelligence Engineer (FashionCLIP)
1. G1: `python scripts/02_extract_embeddings.py --encoder fashionclip`. Cần tải checkpoint `patrickjohncyh/fashion-clip` từ Hugging Face.
2. G2: `python scripts/03_cluster_styles.py --encoder fashionclip`, xem `outputs/clusters/fashionclip/sheets/`, đặt tên từng style.
3. Cùng P4 lập bảng RQ1 (cùng K, cùng dev catalogue) và đề xuất D09.
4. G3b: `python scripts/04_build_style_panel.py --grouping fashionclip`.
5. Visual Trend Radar: lấy `outputs/reports/{grouping}_{stage}_signals.csv` do P4 sinh (xác suất E/S/D + `spap__growth4`) và trình bày theo style.

PCA/UMAP chỉ dùng để trực quan hoá. K-Means vẫn chạy trên embedding gốc đã chuẩn hoá (4.1.3).

## Người 3 · Demand Forecasting & Multimodal ML Engineer
**Kết quả validation lịch sử** trên `--grouping attribute`, tham số mặc định (trước khi chốt bộ model hiện hành theo D19):

| Mô hình | Macro-F1 | Ghi chú |
| --- | --- | --- |
| CatBoost | 0,670 | Cao nhất |
| LightGBM | 0,658 | |
| XGBoost | 0,650 | |
| RF | 0,624 | |
| Persistence | 0,410 | Baseline |
| Majority | 0,205 | Baseline |

Đây là kết quả validation, chưa tuning và chưa phải visual style. XGBoost là model thử nghiệm đã bị loại khỏi code/config theo D19; số liệu giữ lại để lưu dấu kết quả cũ.

Việc tiếp theo:
1. `06_tune_models.py` cho grouping đã chốt.
2. Làm `10_common_endpoint.py`: đây là bằng chứng chính của RQ3.

| Bảng role yêu cầu | Phương pháp quy định | Xử lý đề xuất |
| --- | --- | --- |
| Dự báo demand 6 tuần (số lượng) | Phân lớp xu hướng 4 tuần E/S/D | Theo phương pháp. Muốn có hồi quy 6 tuần thì phải bổ sung vào Section 4 như phần mở rộng riêng |
| MLP/LSTM/Transformer | Bộ model trước đây: RF, XGB, LGBM, CatBoost, TFT, TCN | Hiện giữ RF, LGBM, CatBoost, TCN; XGBoost/TFT chỉ còn trong kết quả và quyết định lịch sử |
| Gợi ý tồn kho ban đầu | 4.10: không tự động ra số lượng nhập | Không làm |

## Người 4 · Business Decision & Product Analytics Lead (+ ResNet-50 CNN + TCN)
1. **ResNet-50 (4.1.2):**
   - `python scripts/02_extract_embeddings.py --encoder resnet50` (weights `IMAGENET1K_V2` của torchvision).
   - Sau đó chạy `03 --encoder resnet50`.
   - So với FashionCLIP trên cùng tiêu chí G2: ARI, coverage, NMI với attribute, contact sheets.
2. **TCN (4.5.6):** `python scripts/07_walk_forward.py --grouping <g> --models tcn`.
   - `build_tcn` dùng causal dilated conv + residual, có test tính causal.
   - Đối chiếu receptive field với lookback.
   - Chạy ≥ 5 seed.
   - Đã chạy thử trên attribute/validation: Macro-F1 0,663, khoảng tin cậy so với persistence là [+0,116; +0,346].
3. **Robustness (4.9):** tạo `configs/robustness/*.yaml` theo `11_robustness.py`. Bắt buộc có target `incumbent_cohort` (xem rủi ro 2 trong tài liệu 01).
4. **Explainability (4.10):** `12_interpret.py`. Sơ bộ: block B3 (shop_exposure, discount, share_new) chi phối; GT gần như không đóng góp.
5. **Decision layer và dashboard:** `13_decision_signals.py`.
   - Điểm vận hành chọn trên validation. Sơ bộ: precision cảnh báo Emerging 0,71, recall 0,62.
   - Bảng kịch bản `*_decision_scenarios.csv` cho thấy đánh đổi giữa số cảnh báo và độ chính xác.

| Bảng role yêu cầu | Phương pháp quy định | Xử lý đề xuất |
| --- | --- | --- |
| Inventory planning, overstock/understock | Inventory tắt chờ D07; 4.10 không cho ra quyết định số lượng | Sau khi chốt D07: phân tích mô tả "cảnh báo Declining × tồn kho tái dựng" như tín hiệu rà soát |
| Product selection | Output là trend intelligence | Danh sách style cần rà soát (Emerging / Declining), không phải danh sách mua |

## Thứ tự bàn giao

| Giai đoạn | P1 | P2 | P3 | P4 |
| --- | --- | --- | --- | --- |
| 1. Nền | Chốt D01–D08 | Embedding FashionCLIP (G1) | Hoàn thiện pipeline trên attribute | Embedding ResNet-50 (G1), TCN chạy trên attribute |
| 2. Style | Duyệt panel | K-Means + sheets (G2) | Common endpoint | K-Means ResNet (G2) |
| 3. Chốt | Cùng duyệt D09–D12 | Đề xuất D09 cùng P4 | Tuning (G6), validation eval | Chọn điểm vận hành trên validation |
| 4. Test (1 lần) | Diễn giải thị trường | Visual Trend Radar | `07 --stage test --frozen`, 08, 09, 10 | 11, 12, 13, dashboard, khuyến nghị quản trị |

## Cập nhật 2026-10-12 · G1/G2 và validation

- **G1** (P2, P4): 5.354 embedding mỗi encoder, đạt mọi kiểm tra.
- **G2** (P2, P4): FashionCLIP ARI 0,58–0,67 ở mọi K, không qua gate 0,70. ResNet-50 qua ở K=10 (ARI 0,87, nhưng quá thô khi xem ảnh) và K=20 (ARI 0,72). Đề xuất D09 = ResNet-50 K=20, có tên style nháp trong `outputs/clusters/resnet50/style_names_k20_draft.csv`.
- **Validation** (tham số mặc định, chưa tuning; Macro-F1 trên matched sample):

| Grouping | Tốt nhất | TCN | Persistence | Majority |
| --- | --- | --- | --- | --- |
| ResNet-50 K=20 | XGBoost 0,681 | 0,667 | 0,370 | 0,231 |
| FashionCLIP K=20 | CatBoost 0,684 | 0,646 | 0,361 | 0,235 |
| Attribute (category) | CatBoost 0,670 | 0,663 | 0,410 | 0,205 |

Các dòng có nhãn khác nhau nên **không** so trực tiếp được giữa grouping (xem tài liệu 01 §4). Không được đổi D09 dựa trên bảng này.
