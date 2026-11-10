# 04 · Kết quả và diễn giải (final test, 2026-10-12)

> **Bộ model hiện hành (D19, 2026-11-10):** Random Forest, LightGBM, CatBoost và TCN. Bảng final test dưới đây là hồ sơ lịch sử của lần đánh giá sáu model; các dòng XGBoost và TFT được giữ làm bằng chứng, nhưng hai model này đã bị loại khỏi cấu hình và code chạy.

Số liệu đầy đủ: [outputs/reports/FINAL_RESULTS.md](../outputs/reports/FINAL_RESULTS.md) (sinh bởi `scripts/14_final_report.py`).
Mọi quyết định được chốt trong [decision_log.md](decision_log.md) **trước** final test. Final test chạy một lần cho cấu hình chính.

**Phạm vi test:**
- 59 forecast origin, từ 2019-01-07 đến 2020-02-17 (mùa SS19 + AW19).
- 20 visual style × tuần; 1.159 dòng matched.
- Mọi mô hình đều so trên cùng các dòng.

## Kết luận ngắn

| RQ | Trả lời | Mức chắc chắn |
| --- | --- | --- |
| RQ1 | **Có**, ở mức vừa phải. ResNet-50 K=20 tạo 20 style ổn định (ARI 0,72) và nhìn nhất quán. FashionCLIP nhìn nhất quán nhưng không ổn định giữa các seed (ARI 0,58–0,67). Silhouette của mọi cấu hình thấp (0,06–0,10): ranh giới giữa các style mờ, không có cụm tách bạch | Trung bình |
| RQ2 | **Dự báo được hướng 4 tuần tốt hơn rõ ràng so với baseline.** Macro-F1 0,61–0,65 so với persistence 0,44 và majority 0,22; CI không chứa 0. Nhưng khoảng 2/3 khả năng dự báo đến từ cơ chế rollout/vòng đời sản phẩm, và hiệu năng dao động mạnh theo quý | Cao (vượt baseline), trung bình (bản chất "xu hướng") |
| RQ3 | **Không có bằng chứng visual grouping thêm giá trị dự báo.** Ở endpoint chung cấp sản phẩm, context từ visual style hay category đều không cải thiện so với lịch sử của chính sản phẩm | Cao |

## RQ2 chi tiết – cấu hình chính ResNet-50 K=20

| Mô hình | Macro-F1 | Balanced acc. | vs persistence [95% CI] |
| --- | --- | --- | --- |
| TCN (5 seed, Người 4) | **0,649** | 0,633 | +0,205 [+0,125; +0,295] |
| CatBoost | 0,648 | 0,624 | +0,204 [+0,119; +0,289] |
| LightGBM | 0,642 | 0,617 | +0,198 [+0,107; +0,286] |
| Random Forest | 0,637 | 0,610 | +0,193 [+0,080; +0,299] |
| XGBoost | 0,636 | 0,610 | +0,192 [+0,079; +0,293] |
| TFT (5 seed) | 0,613 | 0,595 | +0,169 [+0,092; +0,272] |
| Persistence | 0,444 | 0,442 | — |
| Majority | 0,222 | 0,333 | — |

- **Kiến trúc (4.8.5):** TCN, CatBoost, LightGBM, RF và XGBoost nằm trong khoảng 0,636–0,649, khác biệt nhỏ hơn nhiều so với độ rộng CI. TFT kém nhất. Không có bằng chứng mô hình chuỗi phức tạp hơn tốt hơn mô hình cây trên dữ liệu ngắn và nhỏ này.
- **So sánh từng cặp (block bootstrap, 95%), mọi CI đều chứa 0:**
  - TCN − CatBoost: +0,001 [−0,029; +0,049]
  - CatBoost − LightGBM: +0,005 [−0,019; +0,031]
  - CatBoost − XGBoost: +0,012 [−0,026; +0,057]
  - CatBoost − RF: +0,011 [−0,025; +0,047]
  - TCN − TFT: +0,036 [−0,008; +0,061]
  - CatBoost − TFT: +0,034 [−0,023; +0,064]

  Không mô hình nào tốt hơn mô hình khác có ý nghĩa thống kê.
- **Tần suất refit:** TCN/TFT refit mỗi 4 origin, tabular mỗi origin. Khi tabular cũng refit mỗi 4 origin (lượt tham chiếu robustness): LightGBM 0,629, CatBoost 0,639.
- **Tái lập:** chạy lại walk-forward (CatBoost, RF) cho dự báo giống hệt từng xác suất.
- **Độ ổn định seed:** TCN 0,638 ± 0,004 (rất ổn định), TFT 0,603 ± 0,017.
- **Theo lớp:** precision Emerging khoảng 0,65–0,75 nhưng recall chỉ khoảng 0,43–0,51. Mô hình thận trọng: khi báo Emerging thường đúng, nhưng bỏ sót khoảng một nửa.
- **Theo thời gian (4.9.5): không ổn định.**
  - Q1 và Q3 2019: Macro-F1 khoảng 0,65–0,76.
  - Q2 và Q4 2019: khoảng 0,47–0,62.
  - Q1 2020: khoảng 0,33–0,41. Đây là cuối panel: chỉ còn sản phẩm phát hành tới 12/2019 đang "chạy hết" vòng đời 12 tuần, nên rollout không còn diễn ra như bình thường.

## Điểm mấu chốt: bao nhiêu là "xu hướng thật"?

Ba bằng chứng độc lập cho thấy phần lớn khả năng dự báo đến từ **cơ chế rollout và vòng đời của assortment**:

1. **Ablation trên test:** chỉ dùng 18 feature rollout/vòng đời/lịch (cấu hình M, không có doanh số, giá hay search) đã đạt Macro-F1 0,56–0,59, so với 0,63–0,67 khi đủ feature.
2. **Giải thích mô hình (permutation importance trên test):** block B3 đứng đầu (0,091), feature mạnh nhất là `shop_exposure__growth1` (tốc độ thêm shop). Tiếp theo là B1 lịch sử doanh số (0,063) và B2 động lực doanh số (0,040).
3. **Target đã khử vòng đời (`age_adjusted`)** đo doanh số so với mức kỳ vọng theo tuổi item–shop:
   - Macro-F1 giảm xuống 0,51–0,54.
   - Nhưng vẫn vượt persistence (0,305) rõ ràng, CI [+0,14; +0,32].
   - Tức **có tín hiệu xu hướng thật ngoài cơ chế**, chỉ yếu hơn khoảng 0,1 Macro-F1.

→ Khi viết bài: báo cáo cả target chính lẫn `age_adjusted`, và diễn giải "Emerging/Declining" là hướng phản ứng thị trường quan sát được của assortment, chưa phải nhu cầu tiềm ẩn.

## RQ3 chi tiết – common endpoint cấp sản phẩm (4.8.4)

Cùng 9.714 product–week và cùng nhãn cho mọi cấu hình. Cấu hình chỉ khác nhau ở context leave-one-out (68) thêm vào. θ product = 0,205; lớp: Declining 49%, Stable 46%, Emerging 5%.

| Mô hình | Chỉ lịch sử sản phẩm | + category | + FashionCLIP | + ResNet-50 | + ResNet-50 fine-tuned |
| --- | --- | --- | --- | --- | --- |
| CatBoost | 0,643 | 0,645 | 0,646 | 0,641 | 0,641 |
| LightGBM | 0,639 | 0,627 | 0,639 | 0,638 | 0,635 |
| XGBoost | 0,645 | 0,634 | 0,643 | 0,641 | 0,635 |
| Random Forest | 0,621 | 0,610 | 0,613 | 0,616 | 0,617 |

- Mọi chênh lệch so với chỉ lịch sử sản phẩm nằm trong khoảng −0,011 đến +0,003, gần như mọi CI chứa 0.
- Visual context không hơn category context có ý nghĩa thống kê.
- Fine-tune ResNet trên Myntra cũng không tạo khác biệt.

→ Ở VISUELLE 2.0, nhóm sản phẩm theo đặc điểm thị giác (dù bằng FashionCLIP, ResNet-50 hay ResNet-50 fine-tuned) **không** cung cấp thông tin dự báo bổ sung cho hướng 4 tuần. Đây là kết quả âm, cần báo cáo đúng như vậy.

Lưu ý quy trình: một lượt chạy trước bị lẫn cột `ctx_resnet50ft` vào cấu hình `resnet50_context` do lỗi lọc cột. Lỗi đã được sửa và chạy lại; bảng trên là số đúng (khớp lượt chạy đầu chưa có cấu hình fine-tuned).

## Google Trends
- **Lead-lag (P1):** tương quan trung bình chỉ khoảng 0,11, đỉnh khi GT đi trước 2 tuần.
- **Ablation B4a → B4b trên test:** thêm GT làm Macro-F1 *giảm* 0,005–0,029 ở 3/4 mô hình.
- **Importance:** block B4b đóng góp 0,006.
- **Kết luận:** GT (đã chuẩn hoá expanding-max để tránh leak) không đóng góp giá trị dự báo bổ sung.

## Robustness (4.9; LightGBM/CatBoost, refit mỗi 4 origin)

| Biến thể | Macro-F1 (LGBM / CatBoost) | Persistence | Nhận xét |
| --- | --- | --- | --- |
| Tham chiếu K=20 | 0,629 / 0,639 | 0,444 | |
| K = 10 / 15 / 25 / 30 | 0,631–0,654 / 0,647–0,668 | 0,43–0,51 | Ổn định theo K. Với K=10, CI của LightGBM chạm 0 (mẫu nhỏ: 590 dòng) |
| θ = phân vị 33 / 40 / 60 / 67 | 0,560–0,653 | 0,43–0,46 | θ càng lớn, Emerging càng hiếm và Macro-F1 giảm. Ở q67, CI của LightGBM chạm 0 |
| Target `raw_sales` | 0,70 / 0,72 | 0,52 | Dễ hơn vì gồm cả hiệu ứng số lượng sản phẩm |
| Target `shop_exposure` | 0,51 / 0,51 | 0,31 | |
| Target `age_adjusted` | 0,51 / 0,54 | 0,31 | Khử vòng đời: vẫn vượt baseline |
| Trọng số `n_shops` | 0,64 / 0,63 | 0,44 | Không đổi |
| TCN lookback 4 / 8 / 12 | 0,652 / 0,649 / 0,638 | 0,44 | Lookback ngắn đủ dùng |

Kết luận vượt baseline ổn định qua mọi biến thể; mức tuyệt đối phụ thuộc nhiều vào định nghĩa target.

## RQ1 chi tiết – ba biểu diễn ảnh (K = 20)

| Encoder | ARI ổn định | Silhouette (trong không gian riêng) | NMI category / color / fabric | Qua gate |
| --- | --- | --- | --- | --- |
| FashionCLIP | 0,636 | 0,065 | 0,41 / 0,13 / 0,37 | Không (ARI) |
| ResNet-50 (ImageNet, frozen) | 0,723 | 0,066 | 0,35 / 0,10 / 0,33 | Có (chọn) |
| ResNet-50 fine-tuned Myntra (Người 4, bổ sung sau chốt) | 0,705 | 0,095 | 0,24 / 0,21 / 0,25 | Có |

Fine-tune trên 8.850 ảnh đồ nữ Myntra (3 epoch, chỉ `layer4`; validation: đúng 76,5% loại sản phẩm, 66,2% màu) làm style nghiêng sang **màu** (NMI color tăng gấp đôi) và ít trùng category hơn. Ở cấp style, final test của cấu hình này cho Macro-F1 0,632–0,647, tương đương các cấu hình khác.

Lưu ý: cấu hình fine-tuned được thêm **sau** khi đã chốt quyết định và chạy final test chính, theo dataset bổ sung. Đây là phân tích bổ sung, không thay cấu hình chính.

## Lớp quyết định (Người 4)
- Điểm vận hành chọn trên validation (min_prob = 0,5), áp lên test.
- **Emerging:** 138 cảnh báo, precision 0,78, recall 0,45. Có 30 cảnh báo sai và 122 style Emerging bị đoán thành Stable.
- **Declining:** 265 cảnh báo, precision 0,72, recall 0,56.
- **Dùng trong thực tế:** phù hợp làm danh sách "style cần xem lại" có độ chính xác khá; không phù hợp làm hệ thống phát hiện đầy đủ, và không dùng để ra số lượng nhập (4.10).

## Giới hạn cần viết vào bài
1. Dữ liệu chỉ có 12 tuần mỗi item–shop và bắt đầu từ tuần có bán đầu tiên, nên target gắn với rollout (đã định lượng ở trên).
2. Không có tồn kho thật, nên không phân biệt được stock-out.
3. Một retailer, 2 năm test, 20 style: CI rộng (khoảng ±0,1).
4. Tuning với ngân sách nhỏ (10 cấu hình mỗi mô hình) do giới hạn CPU.
5. Hiệu năng dao động mạnh theo quý; Q1 2020 là giai đoạn biên của panel.

## Lần chạy 2 (2026-10-14) · mở rộng, chọn trên validation, test chạy lần 2

Luật chọn đăng ký trước trong `decision_log.md` (D15–D18).

**1. Điểm vận hành (D18):** nhân xác suất, hệ số chọn trên validation, không huấn luyện lại. Chi tiết: `outputs/metrics/resnet50_operating_points.csv`.
- **CatBoost, luật "macro"** (×1,3 cho Declining và Emerging):
  - Macro-F1 test 0,648 → **0,672**, Δ +0,024 [+0,007; +0,044], có ý nghĩa thống kê.
  - Recall Emerging 0,46 → 0,55; recall Declining 0,60 → 0,67.
- **CatBoost, luật "recall70":** recall Emerging 0,61 với precision 0,64; Macro-F1 0,671.
- **Mô hình khác:** thay đổi nhỏ, CI chứa 0.

**2. Class weighting cho TCN/TFT (D17):** không dùng. Trọng số tăng recall Emerging lên khoảng 0,79 nhưng giảm Macro-F1 validation khoảng 0,06.

**3. Block B5 thị giác ở cấp style (D15):** không vào cấu hình chính (Δ validation 0,000). Trên test, Macro-F1 với B5:
- TCN 0,648, LightGBM 0,643, CatBoost 0,641, RF 0,637, XGBoost 0,636.
- Gần như trùng cấu hình không B5 (TCN 0,649, CatBoost 0,648, LightGBM 0,642).

**4. Common endpoint mở rộng (D16), test:** kNN thị giác và kNN + novelty (ResNet-50, FashionCLIP) so với chỉ lịch sử sản phẩm cho Δ từ −0,010 đến +0,007. Chỉ 1/16 so sánh dương có CI không chứa 0 (LightGBM kNN ResNet-50: +0,007 [+0,001; +0,013]), với độ lớn không đáng kể.

**Kết luận lần 2:**
- Tính khả thi dự báo **tăng** nhờ điểm vận hành (CatBoost 0,672, bắt được 55–61% Emerging).
- RQ3 **vẫn âm**, kể cả khi dùng ảnh ở mức chi tiết (láng giềng thị giác, độ mới lạ, độ phân tán). Kết luận này giờ được kiểm chứng chắc chắn hơn.
