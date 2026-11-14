# Visual-Style Trend Forecasting · VISUELLE 2.0

Khung sườn thực thi cho **Section 4 – Methodology**: ảnh sản phẩm → visual style → dự báo hướng 4 tuần (Emerging / Stable / Declining) của từng style.

| Đọc trước | Nội dung |
| --- | --- |
| [docs/01_BOC_TACH_PHUONG_PHAP.md](docs/01_BOC_TACH_PHUONG_PHAP.md) | Bóc tách từng mục, truy vết (1)–(70), phát hiện từ dữ liệu thật, 14 lỗ hổng của văn bản |
| [docs/02_TIEU_CHI_DAT.md](docs/02_TIEU_CHI_DAT.md) | Thế nào là đạt: tầng A (khung sườn), cổng G0–G11, tầng C (kết luận) |
| [docs/03_PHAN_CONG_ROLE.md](docs/03_PHAN_CONG_ROLE.md) | 4 role. Người 4 giữ ResNet-50 và TCN (hai thành phần CNN) |
| [docs/decision_log.md](docs/decision_log.md) | D01–D19, gồm lựa chọn bộ model hiện hành |
| [docs/04_KET_QUA.md](docs/04_KET_QUA.md) | Kết quả và diễn giải RQ1–RQ3 |
| [outputs/reports/visual_trend_radar.html](outputs/reports/visual_trend_radar.html) | Dashboard Visual Trend Radar (mở bằng trình duyệt; sinh bởi `scripts/15_trend_radar.py`) |
| [web/](web/) | **Style Scanner**: upload 1 ảnh (out-of-sample) → ResNet-50 và FashionCLIP → style K-Means hoặc style lạ (zero-shot) → dự báo TCN (`scripts/17_build_web_assets.py` + `web/app.py`) |

## Cấu trúc thư mục

Dataset bổ sung cho nhận diện style: [hướng dẫn tải DeepFashion, Fashionpedia và DeepFashion2](docs/05_DATASETS_BO_SUNG.md). Lưu file tải vào từng thư mục tương ứng trong `Data/downloads/`; các nguồn mới chưa được tích hợp vào pipeline huấn luyện.

```
configs/            default.yaml (mọi tham số, có số mục), models.yaml (search space), experiments.yaml
docs/               bóc tách, tiêu chí đạt, phân công, decision log
src/vstyle/
  data/             load, audit (G0), calendar (12), product_week (13), gtrends (20)(21), inventory (tắt)
  visual/           encoders (1)-(5), clustering (6)-(11)(60), inspection (contact sheets)
  panel/            style_week (14)-(19)
  features/         historical (23)-(31), blocks B1-B4b, sequences (33)
  target/           labels (36)-(40), threshold (41), eligibility (42)
  validation/       splits, walk_forward (54)-(58), tuning (59)
  models/           baselines (66), tabular: Random Forest, LightGBM, CatBoost; sequence: TCN
  evaluation/       metrics (61)-(65), bootstrap (69)(70)
  experiments/      common_endpoint (68)
  interpret/        permutation importance, SHAP
  decision/         cảnh báo, chất lượng cảnh báo, kịch bản (Người 4)
  utils/            leakage guards, manifests, seeds
scripts/            00 → 13: mỗi script là một giai đoạn, ghi rõ cổng và người phụ trách
tests/              kiểm tra calendar, panel, chống leakage, target, walk-forward, metrics, GT, TCN causal
outputs/            báo cáo, metrics, manifests, dữ liệu tối thiểu cho manager và assets triển khai web
                    embeddings, clusters, features, panels, predictions lớn được sinh lại bằng scripts và Git bỏ qua
Data/extracted/     dữ liệu gốc (chỉ đọc)
EDA_Visuelle2.ipynb, Journal_Figures.ipynb, figures/   EDA và hình đã có (giữ nguyên)
```

## Chạy

```bash
python -m pytest tests -q
```
```bash
python scripts/00_audit_data.py
```
```bash
python scripts/01_build_product_week.py
```
```bash
python scripts/04_build_style_panel.py --grouping attribute
```
```bash
python scripts/05_build_features_targets.py --grouping attribute
```
```bash
python scripts/07_walk_forward.py --grouping attribute --stage validation
```
```bash
python scripts/08_evaluate.py --grouping attribute --stage validation
```

Nhánh visual cần tải trọng số pretrained (FashionCLIP từ Hugging Face, ResNet-50 từ torchvision):

```bash
python scripts/02_extract_embeddings.py --encoder all
```

`--encoder all` huấn luyện cả hai nhánh Myntra (`resnet50ft` và `fashionclipft`) rồi trích
embedding VISUELLE cho hai encoder pretrained để đối chiếu. Có thể chạy riêng từng
nhánh bằng `--encoder resnet50ft` hoặc `--encoder fashionclipft`. Dùng tên nhánh tương ứng
cho bước 03 (`--encoder`) và các bước 04–08 (`--grouping`). Checkpoint, lịch sử validation
Myntra và embedding VISUELLE được lưu trong `outputs/embeddings/`.

Cấu hình chạy chính sau lượt validation Myntra là `fashionclipft`, K=20 (D20).
Bước 04 dùng encoder này nếu bỏ `--grouping`. Báo cáo final test lịch sử dùng cấu hình D09.
```bash
python scripts/03_cluster_styles.py --encoder fashionclip
```

Sau G2, đặt `visual.selected` trong `configs/default.yaml`, rồi chạy 04 → 08 với `--grouping fashionclip` (hoặc `resnet50`). Final test chỉ chạy một lần, khi decision log đã đủ:

```bash
python scripts/07_walk_forward.py --grouping fashionclip --stage test --frozen
```

## Trạng thái

| Phần | Trạng thái |
| --- | --- |
| G0 audit, product–week, insight P1 | Đã chạy trên dữ liệu thật: 0 FAIL, 7 quyết định cần chốt |
| Pipeline forecasting trên benchmark attribute (validation) | Đã chạy 05 → 09, 12, 13 |
| Embedding + K-Means (G1, G2) | Đã chạy cả hai encoder. G1 đạt. G2: chỉ ResNet-50 K=10/20 qua gate; đề xuất D09 = ResNet-50 K=20 (chờ duyệt). Xem `outputs/reports/G2_encoder_comparison.md` |
| Final test (toàn bộ Section 4) | Đã chạy: tuning, final test 4 grouping (resnet50 chính, fashionclip, attribute, resnet50ft), ablation, common endpoint, robustness, explainability, decision layer. Kết quả: `outputs/reports/FINAL_RESULTS.md`; diễn giải: `docs/04_KET_QUA.md` |
| TCN (Người 4) | Đã chạy walk-forward trên attribute/validation: Macro-F1 0,663 (1 seed, refit mỗi 4 origin, ~43 giây) |
| Bộ model hiện hành | Random Forest, LightGBM, CatBoost, TCN. XGBoost và TFT đã được loại khỏi cấu hình/code; kết quả cũ được giữ trong báo cáo lịch sử. |
| Inventory | Tắt cho tới khi chốt D07 |

## Web Style Scanner

Upload một ảnh thời trang bất kỳ (out-of-sample). Web chạy song song hai encoder và cho xem ở chế độ **so sánh** hoặc **chi tiết từng bước** của mỗi encoder:

0. **Tách trang phục khỏi nền** (`web/segment.py`): SegFormer clothes parsing (`mattmdjaga/segformer_b2_clothes`, chỉ nạp `model.safetensors`) phân vùng ảnh thành áo, quần, váy, mặt, tay, tóc, nền…, giữ món đồ chính rồi đặt lên nền trắng và cắt sát như ảnh catalogue. Chế độ tự động bỏ qua ảnh đã giống catalogue: mặt, tóc, tay, chân chiếm dưới 4% và ít nhất 70% vùng nền gần như trắng. Trên mẫu thử, 0/60 ảnh VISUELLE bị tách và 50/60 ảnh Myntra (phần lớn có người mẫu) được tách. Web hiện cả bảng so sánh style trước và sau khi tách
1. Tiền xử lý giống lúc train: RGBA → nền trắng, sau đó transforms riêng của từng encoder
2. ResNet-50 (frozen ImageNet, 2048-d) và FashionCLIP (ViT-B/32, 512-d) → L2, kèm bản đồ vùng ảnh quyết định style. Với ResNet-50 bản đồ này chính xác, với FashionCLIP là xấp xỉ qua patch token
3. Gán style theo centroid K-Means K = 20 đã đóng băng (11). **Style lạ**: ảnh xa centroid gần nhất hơn 99% sản phẩm của chính style đó thì không bị ép vào style ấy
4. Dự báo Emerging / Stable / Declining 4 tuần của style bằng TCN (chính) và CatBoost. Style lạ không có lịch sử bán (cold start), nên web chỉ hiện ước lượng tham khảo từ 3 style gần nhất, có ghi rõ là không phải dự báo

Nhận diện món đồ (tab **Tổng quan**):
- **Loại sản phẩm, họa tiết, chi tiết, phong cách, chất liệu**: FashionCLIP so ảnh với mô tả bằng chữ, mỗi nhãn chấm bằng nhiều câu mẫu rồi lấy trung bình (prompt ensembling, `web/zero_shot.py`)
- **Màu chủ đạo và sọc** đo trực tiếp trên điểm ảnh của món đồ (`web/colors.py`): K-Means trong không gian CIELAB rồi gán tên màu tiếng Việt; sọc dọc/ngang phát hiện từ hồ sơ sắc độ a*/b* theo cột/hàng (bóng đổ gần như không ảnh hưởng). Trên 60 ảnh VISUELLE chỉ 1 ảnh được gọi là sọc, và đó đúng là váy sọc dọc

Giao diện có 3 tab: **Tổng quan** (dành cho người dùng), **So sánh hai mô hình** và **Quy trình xử lý** (chi tiết kỹ thuật từng bước).

Giao diện có ảnh mẫu từ catalogue để thử nhanh, hướng dẫn trong trang và tiến trình **Chọn ảnh → Phân tích → Khám phá kết quả**. Có thể bỏ ảnh đã chọn để bắt đầu lại; trong khi xử lý, bộ đếm hiển thị thời gian chờ thực tế. Nút **Tải tóm tắt** xuất file `.txt` tiếng Việt gồm đặc điểm nhận diện, kết quả của hai mô hình và dự báo theo tuần đang chọn. Trên điện thoại, phần phong cách đã lưu nằm sau kết quả để ưu tiên thao tác quét và đọc phân tích.

Bấm ảnh sản phẩm tương tự để xem ảnh lớn. Bấm một phong cách đã lưu để xem thư viện tối đa 6 ảnh gần nhất; dùng nút mũi tên, ảnh thu nhỏ hoặc phím **← / →** để chuyển ảnh, **Esc** để đóng. Các nút đổi mô hình và đổi ảnh gốc/tách nền giữ vị trí bàn phím sau khi cập nhật kết quả.

Nhận diện style lạ gồm:
- **Mô tả zero-shot** bằng text encoder FashionCLIP (`web/zero_shot.py`): loại sản phẩm, họa tiết, màu, chất liệu, ngữ cảnh ảnh. Từ vựng gồm thuộc tính VISUELLE cộng từ vựng mở, có đánh dấu "mới"
- **Ghi nhận style mới** (N1, N2…) vào `outputs/web/novel_styles.json`. Ảnh sau nằm trong bán kính điển hình (trung vị khoảng cách sản phẩm → centroid, không gian FashionCLIP) sẽ được nhận ra là cùng style mới
- Cờ ảnh trùng catalogue (cosine ≥ 0,95) và cảnh báo khi ảnh không phải ảnh sản phẩm nền trơn

Bước train (một lần cho mỗi encoder, ~30 giây CPU mỗi lần): fit TCN (3 seed) + CatBoost trên mọi nhãn đã trưởng thành, đóng gói centroid, embedding catalogue, ngưỡng style lạ và lịch sử style vào `outputs/web/{encoder}/`. FashionCLIP dùng hyperparameter đã tune của ResNet-50 (`--params-from resnet50`):

```bash
python scripts/17_build_web_assets.py --grouping resnet50
```
```bash
python scripts/17_build_web_assets.py --grouping fashionclip
```
```bash
python -m uvicorn web.app:app --port 8000
```

Lần đầu cần tải trọng số SegFormer (~110 MB, chỉ file safetensors):

```bash
python -c "from huggingface_hub import snapshot_download; snapshot_download('mattmdjaga/segformer_b2_clothes', allow_patterns=['config.json', 'preprocessor_config.json', 'model.safetensors'])"
```

Mở http://localhost:8000 để vào **Bảng điều hành** cho nhà quản lý; Style Scanner kỹ thuật nằm ở http://localhost:8000/scanner (server khởi động mất khoảng 20–30 giây vì phải nạp FashionCLIP và SegFormer). Tuần ≤ 2020-02-17 dùng dự báo walk-forward, có kết quả thật để đối chiếu; các tuần sau dùng mô hình triển khai. Tuần mặc định là 2019-12-30, vì từ tháng 1/2020 dữ liệu bị cắt cuối kỳ (không còn sản phẩm mới) nên hầu hết style trông như đang giảm.


### Bảng điều hành cho nhà quản lý

`web/manager.py` + `web/static/manager.{html,css,js}`, API dưới `/api/m/*`. Chỉ đọc dữ liệu và dự báo đã có, không train lại. Một ô **Tuần điều hành** trên thanh đầu trang áp dụng cho mọi trang:

- **Tổng quan**: KPI toàn chuỗi, doanh số theo tuần, triển vọng 4 tuần của 20 nhóm style (đối chiếu với thực tế khi đã có), loại sản phẩm bán chạy, doanh số theo mùa, độ tin cậy của mô hình
- **Quét ảnh & dự báo**: upload ảnh, nhận diện, nhóm style, dự báo 4 tuần, khuyến nghị, sản phẩm tương tự và doanh số tham chiếu. Ảnh và kết quả được lưu ở `outputs/web/scans/`
- **Loại sản phẩm**: dashboard cho từng loại trong 27 loại (doanh số theo tuần, màu, mùa, chất liệu, nhóm style, sản phẩm bán chạy) kèm dự báo của loại đó
- **Kho dữ liệu**: toàn bộ 5.355 sản phẩm (lọc, tìm kiếm, chi tiết và dự báo từng sản phẩm), 13 tệp gốc VISUELLE 2.0 cùng các bảng đã xử lý (xem trước 50 dòng, tải về), và lịch sử ảnh đã quét

Tuần mặc định của bảng điều hành là 2019-10-14 (giữa mùa Thu Đông 2019, có số liệu thực tế để đối chiếu). Từ 2019-12-02 trở đi, mọi trang hiện cảnh báo **dữ liệu cuối kỳ**, vì sau đó gần như không còn sản phẩm mới (lần ra mắt cuối 2019-12-30) nên style nào cũng trông như đang giảm.

Dự báo của một sản phẩm hay một loại là dự báo của nhóm style tương ứng; với một loại, đó là trung bình theo số sản phẩm đang bán. Khuyến nghị dùng CatBoost và ngưỡng đã chọn trên validation (`outputs/reports/resnet50_decision.json`). Đây là tín hiệu để rà soát, không phải số lượng đặt hàng.

### Kiểm tra kết nối web và backend

Frontend và API chạy cùng origin trên `http://127.0.0.1:8000/`; không cần một server frontend riêng.
`GET /api/health` xác nhận ResNet-50, FashionCLIP và SegFormer đã được nạp. Nút quét chỉ hoạt động khi backend và ảnh đầu vào sẵn sàng. Khi máy chủ mất kết nối, dùng nút **Kết nối lại** trên thanh đầu trang.

Sau khi khởi động uvicorn, kiểm tra bằng ảnh thật trong dữ liệu dự án:

```bash
python scripts/18_check_web.py --check-saved-styles
```

Script kiểm tra phân tích ở bốn chế độ tách nền, metadata/dự báo, ảnh catalogue, trường hợp lỗi, khả năng phản hồi trong lúc xử lý ảnh và lưu/xoá một phong cách thử. Phong cách thử được dọn sau kiểm tra; các phong cách đã có được giữ nguyên. Báo cáo nằm ở `outputs/web/ui-review/backend-check.json`.

```bash
python -m pytest tests -q
```

Dự báo sử dụng lịch sử VISUELLE 2.0 và tuần được chọn trong giao diện; không phải dự báo theo ngày hiện tại. Trạng thái xử lý đợi phản hồi API thật, không mô phỏng phần trăm hoàn tất.
