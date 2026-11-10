# FINAL RESULTS - Visual-style trend forecasting (VISUELLE 2.0)

Sinh tự động bởi `scripts/14_final_report.py`. Diễn giải: `docs/04_KET_QUA.md`.

## 1. Dữ liệu

- Sản phẩm: 5355 (dùng 5354, loại ảnh hỏng [4442]); shop: 110; item–shop: 106,850.
- Panel tuần lịch: 2016-11-28 → 2020-03-16 (173 tuần).

## 2. RQ1 - Visual style identification

| encoder     |   k |   silhouette |   davies_bouldin |   ari_mean |   ari_sd |   size_min |   min_eligible_dev_origins |   nmi_category |   nmi_color |   nmi_fabric | pass_all   |
|:------------|----:|-------------:|-----------------:|-----------:|---------:|-----------:|---------------------------:|---------------:|------------:|-------------:|:-----------|
| fashionclip |  10 |        0.065 |            3.21  |      0.671 |    0.076 |        160 |                         89 |          0.399 |       0.109 |        0.321 | False      |
| fashionclip |  15 |        0.061 |            3.223 |      0.625 |    0.08  |         93 |                         63 |          0.383 |       0.131 |        0.357 | False      |
| fashionclip |  20 |        0.065 |            3.041 |      0.636 |    0.062 |         93 |                         63 |          0.414 |       0.125 |        0.367 | False      |
| fashionclip |  25 |        0.063 |            2.983 |      0.589 |    0.048 |         81 |                         50 |          0.435 |       0.121 |        0.394 | False      |
| fashionclip |  30 |        0.063 |            2.901 |      0.58  |    0.047 |         69 |                         58 |          0.418 |       0.122 |        0.395 | False      |
| resnet50    |  10 |        0.08  |            2.865 |      0.869 |    0.074 |        135 |                         89 |          0.325 |       0.067 |        0.266 | True       |
| resnet50    |  15 |        0.069 |            2.843 |      0.699 |    0.073 |        116 |                         87 |          0.327 |       0.088 |        0.292 | False      |
| resnet50    |  20 |        0.066 |            2.826 |      0.723 |    0.05  |        111 |                         70 |          0.349 |       0.099 |        0.33  | True       |
| resnet50    |  25 |        0.062 |            2.753 |      0.669 |    0.057 |         71 |                         70 |          0.36  |       0.113 |        0.336 | False      |
| resnet50    |  30 |        0.06  |            2.871 |      0.601 |    0.039 |         64 |                         70 |          0.357 |       0.124 |        0.343 | False      |
| resnet50ft  |  10 |        0.095 |            2.333 |      0.854 |    0.141 |        114 |                         88 |          0.218 |       0.172 |        0.21  | True       |
| resnet50ft  |  15 |        0.095 |            2.496 |      0.826 |    0.066 |        113 |                         88 |          0.231 |       0.205 |        0.232 | True       |
| resnet50ft  |  20 |        0.095 |            2.5   |      0.705 |    0.047 |         40 |                         85 |          0.244 |       0.21  |        0.254 | True       |
| resnet50ft  |  25 |        0.086 |            2.427 |      0.669 |    0.043 |         39 |                         72 |          0.261 |       0.217 |        0.269 | False      |
| resnet50ft  |  30 |        0.08  |            2.463 |      0.629 |    0.036 |         38 |                         81 |          0.26  |       0.237 |        0.274 | False      |

Cấu hình chính (D09): **resnet50, K = 20**. Ảnh: `outputs/clusters/*/overview_k*.png`.

| style                                             | draft_name                                   | evidence                                             |
|:--------------------------------------------------|:---------------------------------------------|:-----------------------------------------------------|
| Váy/áo họa tiết hoa (sát nách                     | dáng dài)                                    | sleeveless 18% long dress 17%; ảnh: hoa nền sáng/tối |
| Áo len trơn dài tay                               | long sleeve 62%                              | nan                                                  |
| Đồ đen dáng rộng (quần ống rộng/jumpsuit)         | black 76%                                    | nan                                                  |
| Quần dáng ôm (tối màu/kẻ)                         | culottes 94%                                 | nan                                                  |
| Blouse sáng màu/pastel                            | white 35% violet 18%                         | nan                                                  |
| Áo sát nách                                       | sleeveless 46%                               | nan                                                  |
| Áo len dệt kim mélange/họa tiết                   | long sleeve 39% kimono dress 29%             | nan                                                  |
| Jumpsuit/chân váy/short (hỗn hợp - kém nhất quán) | long sleeve 18% long dress 15% miniskirt 12% | nan                                                  |
| Blazer và áo khoác tailored                       | long sleeve 54% long coat 13%                | nan                                                  |
| Quần đen ống côn                                  | culottes 99% black 72%                       | nan                                                  |
| Áo dài tay họa tiết hoa                           | patterned top 24%                            | nan                                                  |
| Váy suông họa tiết (kẻ/ren)                       | trapeze dress 17%                            | nan                                                  |
| Áo phao/áo khoác mùa đông                         | coats 77%; chỉ mùa AW                        | nan                                                  |
| Váy dài hoa nền tối                               | doll dress 25% black 62%                     | nan                                                  |
| Áo ngắn tay ren/chất liệu nổi                     | sleeveless 41%                               | nan                                                  |
| Áo khoác da/biker                                 | short coat 23% black 62%                     | nan                                                  |
| Váy xoè màu (doll dress)                          | doll dress 57%                               | nan                                                  |
| Áo dài tay đen/cardigan                           | black 55%                                    | nan                                                  |
| Áo thun in hình (graphic)                         | printed shirt 57%                            | nan                                                  |
| Jeans                                             | culottes 91% blue 96%                        | nan                                                  |

## 3. RQ2 - Final test, ResNet-50 K=20 (matched sample)

- θ = 0.337 (median_abs, n calibration = 1484).
- Test origins [110, 168]; dự báo 1170, chấm được 1159, coverage 0.991.
- Phân bố lớp test: Declining 29%, Stable 50%, Emerging 21%
- Matched rows: 1159.

| model         |   Macro-F1 |   Balanced acc. |   Accuracy |   P(E) |   R(E) |   P(D) |   R(D) | vs persistence [95% CI]   |
|:--------------|-----------:|----------------:|-----------:|-------:|-------:|-------:|-------:|:--------------------------|
| tcn           |      0.649 |           0.633 |      0.67  |  0.688 |  0.506 |  0.687 |  0.636 | +0.205 [+0.125, +0.295]   |
| catboost      |      0.648 |           0.624 |      0.677 |  0.75  |  0.464 |  0.705 |  0.595 | +0.204 [+0.119, +0.289]   |
| lightgbm      |      0.642 |           0.617 |      0.674 |  0.739 |  0.473 |  0.728 |  0.548 | +0.198 [+0.107, +0.286]   |
| random_forest |      0.637 |           0.61  |      0.676 |  0.752 |  0.431 |  0.752 |  0.543 | +0.193 [+0.080, +0.299]   |
| xgboost       |      0.636 |           0.61  |      0.671 |  0.752 |  0.469 |  0.734 |  0.51  | +0.192 [+0.079, +0.293]   |
| tft           |      0.613 |           0.595 |      0.639 |  0.649 |  0.473 |  0.679 |  0.551 | +0.169 [+0.092, +0.272]   |
| persistence   |      0.444 |           0.442 |      0.471 |  0.368 |  0.372 |  0.449 |  0.402 |                           |
| majority      |      0.222 |           0.333 |      0.5   |  0     |  0     |  0     |  0     | -0.222 [-0.289, -0.128]   |

Seed của mô hình chuỗi (Macro-F1 từng seed; bảng trên dùng trung bình xác suất):

| model   |   mean |   std |   count |
|:--------|-------:|------:|--------:|
| tcn     |  0.638 | 0.004 |       5 |
| tft     |  0.603 | 0.017 |       5 |

Macro-F1 theo quý:

| quarter   |   majority |   persistence |   random_forest |   xgboost |   lightgbm |   catboost |   tcn |   tft |
|:----------|-----------:|--------------:|----------------:|----------:|-----------:|-----------:|------:|------:|
| 2019Q1    |      0.213 |         0.406 |           0.759 |     0.755 |      0.708 |      0.733 | 0.715 | 0.592 |
| 2019Q2    |      0.262 |         0.424 |           0.473 |     0.527 |      0.525 |      0.527 | 0.617 | 0.59  |
| 2019Q3    |      0.212 |         0.383 |           0.696 |     0.68  |      0.671 |      0.682 | 0.651 | 0.678 |
| 2019Q4    |      0.257 |         0.359 |           0.484 |     0.512 |      0.55  |      0.534 | 0.499 | 0.475 |
| 2020Q1    |      0.059 |         0.34  |           0.347 |     0.33  |      0.41  |      0.383 | 0.387 | 0.333 |

Confusion matrix (tcn): `FINAL_confusion.png`; theo quý: `FINAL_by_quarter.png`.

## 4. RQ3 - Giá trị của visual grouping

### 4a. Style-level theo từng grouping (nhãn khác nhau → chỉ là sensitivity, không so trực tiếp)

| grouping    |   n_styles |   theta |   n_matched | best model   |   best Macro-F1 |   persistence |   majority |
|:------------|-----------:|--------:|------------:|:-------------|----------------:|--------------:|-----------:|
| resnet50    |         20 |   0.337 |        1159 | tcn          |           0.649 |         0.444 |      0.222 |
| resnet50ft  |         20 |   0.302 |        1148 | tcn          |           0.647 |         0.425 |      0.213 |
| fashionclip |         20 |   0.355 |        1132 | tcn          |           0.674 |         0.431 |      0.222 |
| attribute   |         27 |   0.355 |        1279 | lightgbm     |           0.657 |         0.438 |      0.193 |

### 4b. Common endpoint product–week (cùng hàng, cùng nhãn → so sánh hợp lệ)

θ product = 0.205; product-week chấm được: 9714; lớp: Declining 49%, Stable 46%, Emerging 5%

| model         |   own_history |   attribute_context |   fashionclip_context |   resnet50_context |   resnet50ft_context |
|:--------------|--------------:|--------------------:|----------------------:|-------------------:|---------------------:|
| catboost      |         0.643 |               0.645 |                 0.646 |              0.641 |                0.641 |
| lightgbm      |         0.639 |               0.627 |                 0.639 |              0.638 |                0.635 |
| random_forest |         0.621 |               0.61  |                 0.613 |              0.616 |                0.617 |
| xgboost       |         0.645 |               0.634 |                 0.643 |              0.641 |                0.635 |

| so sánh                                                |   ΔMacro-F1 |   CI low |   CI high |
|:-------------------------------------------------------|------------:|---------:|----------:|
| catboost: resnet50ft_context - own_history             |      -0.003 |   -0.012 |     0.006 |
| catboost: resnet50ft_context - resnet50_context        |      -0.001 |   -0.008 |     0.007 |
| catboost: resnet50_context - own_history               |      -0.002 |   -0.01  |     0.005 |
| catboost: fashionclip_context - own_history            |       0.003 |   -0.009 |     0.012 |
| catboost: attribute_context - own_history              |       0.002 |   -0.009 |     0.013 |
| catboost: resnet50_context - attribute_context         |      -0.004 |   -0.011 |     0.002 |
| catboost: fashionclip_context - attribute_context      |       0.001 |   -0.007 |     0.006 |
| lightgbm: resnet50ft_context - own_history             |      -0.003 |   -0.014 |     0.01  |
| lightgbm: resnet50ft_context - resnet50_context        |      -0.003 |   -0.015 |     0.015 |
| lightgbm: resnet50_context - own_history               |      -0     |   -0.014 |     0.01  |
| lightgbm: fashionclip_context - own_history            |       0     |   -0.008 |     0.009 |
| lightgbm: attribute_context - own_history              |      -0.011 |   -0.026 |     0.004 |
| lightgbm: resnet50_context - attribute_context         |       0.011 |   -0.006 |     0.025 |
| lightgbm: fashionclip_context - attribute_context      |       0.011 |   -0.002 |     0.024 |
| random_forest: resnet50ft_context - own_history        |      -0.004 |   -0.011 |     0.003 |
| random_forest: resnet50ft_context - resnet50_context   |       0.001 |   -0.007 |     0.009 |
| random_forest: resnet50_context - own_history          |      -0.005 |   -0.011 |     0.003 |
| random_forest: fashionclip_context - own_history       |      -0.008 |   -0.016 |     0.001 |
| random_forest: attribute_context - own_history         |      -0.011 |   -0.021 |    -0     |
| random_forest: resnet50_context - attribute_context    |       0.006 |   -0.002 |     0.015 |
| random_forest: fashionclip_context - attribute_context |       0.003 |   -0.003 |     0.008 |
| xgboost: resnet50ft_context - own_history              |      -0.01  |   -0.02  |    -0     |
| xgboost: resnet50ft_context - resnet50_context         |      -0.006 |   -0.014 |     0.006 |
| xgboost: resnet50_context - own_history                |      -0.004 |   -0.015 |     0.003 |
| xgboost: fashionclip_context - own_history             |      -0.002 |   -0.012 |     0.006 |
| xgboost: attribute_context - own_history               |      -0.011 |   -0.022 |    -0     |
| xgboost: resnet50_context - attribute_context          |       0.007 |   -0.006 |     0.019 |
| xgboost: fashionclip_context - attribute_context       |       0.009 |   -0.005 |     0.021 |

## 5. Feature ablation (test, ResNet-50 K=20)

B1 sales level → B2 dynamics → B3 operations → B4a calendar → B4b Google Trends; M = chỉ rollout/vòng đời/calendar.

| model         |    B1 |    B2 |    B3 |   B4a |   B4b |     M |   delta_B2 |   delta_B3 |   delta_B4a |   delta_B4b |   delta_M |
|:--------------|------:|------:|------:|------:|------:|------:|-----------:|-----------:|------------:|------------:|----------:|
| catboost      | 0.595 | 0.619 | 0.652 | 0.668 | 0.639 | 0.594 |      0.024 |      0.032 |       0.017 |      -0.029 |    -0.046 |
| lightgbm      | 0.606 | 0.615 | 0.651 | 0.65  | 0.629 | 0.589 |      0.009 |      0.037 |      -0.001 |      -0.021 |    -0.04  |
| random_forest | 0.573 | 0.622 | 0.62  | 0.629 | 0.637 | 0.56  |      0.049 |     -0.002 |       0.009 |       0.009 |    -0.078 |
| xgboost       | 0.585 | 0.625 | 0.651 | 0.639 | 0.634 | 0.583 |      0.04  |      0.027 |      -0.012 |      -0.005 |    -0.051 |

## 6. Robustness (test; LightGBM + CatBoost, refit mỗi 4 origin)

| variant                |   theta |   n_matched |   share_emerging |   share_declining |   lightgbm |   catboost |   persistence |   majority | lightgbm_vs_persistence_ci   | catboost_vs_persistence_ci   |     tcn |
|:-----------------------|--------:|------------:|-----------------:|------------------:|-----------:|-----------:|--------------:|-----------:|:-----------------------------|:-----------------------------|--------:|
| reference              |   0.337 |        1159 |            0.206 |             0.294 |      0.629 |      0.639 |         0.444 |      0.222 | [+0.086, +0.300]             | [+0.105, +0.296]             | nan     |
| K=10                   |   0.268 |         590 |            0.195 |             0.317 |      0.631 |      0.648 |         0.508 |      0.219 | [-0.003, +0.271]             | [+0.015, +0.288]             | nan     |
| K=15                   |   0.316 |         880 |            0.185 |             0.293 |      0.64  |      0.65  |         0.49  |      0.229 | [+0.037, +0.296]             | [+0.048, +0.301]             | nan     |
| K=25                   |   0.336 |        1456 |            0.222 |             0.321 |      0.654 |      0.668 |         0.436 |      0.209 | [+0.141, +0.303]             | [+0.164, +0.311]             | nan     |
| K=30                   |   0.337 |        1735 |            0.214 |             0.324 |      0.639 |      0.647 |         0.434 |      0.211 | [+0.137, +0.275]             | [+0.151, +0.281]             | nan     |
| theta = q33 |Delta|    |   0.21  |        1159 |            0.289 |             0.388 |      0.626 |      0.626 |         0.426 |      0.163 | [+0.136, +0.265]             | [+0.135, +0.270]             | nan     |
| theta = q40 |Delta|    |   0.261 |        1159 |            0.246 |             0.347 |      0.653 |      0.653 |         0.457 |      0.193 | [+0.123, +0.277]             | [+0.130, +0.278]             | nan     |
| theta = q60 |Delta|    |   0.414 |        1159 |            0.162 |             0.256 |      0.605 |      0.607 |         0.457 |      0.245 | [+0.047, +0.256]             | [+0.044, +0.267]             | nan     |
| theta = q67 |Delta|    |   0.481 |        1159 |            0.126 |             0.228 |      0.56  |      0.598 |         0.453 |      0.262 | [-0.003, +0.247]             | [+0.047, +0.263]             | nan     |
| target = raw_sales     |   0.47  |        1159 |            0.175 |             0.304 |      0.701 |      0.72  |         0.522 |      0.228 | [+0.106, +0.261]             | [+0.140, +0.267]             | nan     |
| target = shop_exposure |   0.101 |        1159 |            0.204 |             0.302 |      0.506 |      0.511 |         0.305 |      0.221 | [+0.139, +0.282]             | [+0.142, +0.300]             | nan     |
| target = age_adjusted  |   0.075 |        1159 |            0.226 |             0.27  |      0.51  |      0.541 |         0.305 |      0.223 | [+0.141, +0.286]             | [+0.167, +0.322]             | nan     |
| weighting = n_shops    |   0.337 |        1159 |            0.206 |             0.294 |      0.64  |      0.634 |         0.444 |      0.222 | [+0.102, +0.292]             | [+0.098, +0.298]             | nan     |
| TCN lookback = 4       | nan     |        1159 |          nan     |           nan     |    nan     |    nan     |         0.444 |      0.222 | nan                          | nan                          |   0.652 |
| TCN lookback = 8       | nan     |        1159 |          nan     |           nan     |    nan     |    nan     |         0.444 |      0.222 | nan                          | nan                          |   0.649 |
| TCN lookback = 12      | nan     |        1159 |          nan     |           nan     |    nan     |    nan     |         0.444 |      0.222 | nan                          | nan                          |   0.638 |

## 7. Explainability (CatBoost, permutation importance trên Macro-F1 test; liên hệ, không nhân quả)

| feature                |   importance_mean |   importance_sd | block   |
|:-----------------------|------------------:|----------------:|:--------|
| shop_exposure__growth1 |             0.058 |           0.006 | B3      |
| spap__lag4             |             0.039 |           0.005 | B1      |
| spap__growth4          |             0.015 |           0.008 | B2      |
| sales__growth1         |             0.015 |           0.004 | B2      |
| shop_exposure__growth4 |             0.014 |           0.004 | B3      |
| spap__ma8              |             0.012 |           0.006 | B1      |
| spap__growth1          |             0.012 |           0.002 | B2      |
| gt_color__trend8       |             0.012 |           0.004 | B4b     |
| cal__cos               |             0.011 |           0.003 | B4a     |
| mean_age               |             0.011 |           0.002 | B3      |
| sales__growth4         |             0.009 |           0.002 | B2      |
| spap__lag2             |             0.007 |           0.005 | B1      |
| gt_color__trend4       |             0.007 |           0.002 | B4b     |
| share_new              |             0.007 |           0.003 | B3      |
| gt_category__trend8    |             0.007 |           0.003 | B4b     |

| block   |   importance_mean |
|:--------|------------------:|
| B3      |             0.09  |
| B1      |             0.062 |
| B2      |             0.04  |
| B4a     |             0.012 |
| B4b     |             0.006 |

## 8. Decision layer (Người 4)

Điểm vận hành chọn trên validation: {'min_prob': 0.5, 'min_margin': 0.0, 'chosen_on': 'validation'}; áp lên test:

|    n |   emerging_alerts |   emerging_alert_precision |   emerging_recall |   false_emerging_alerts |   missed_emerging_as_stable |   declining_alerts |   declining_alert_precision |   declining_recall |
|-----:|------------------:|---------------------------:|------------------:|------------------------:|----------------------------:|-------------------:|----------------------------:|-------------------:|
| 1159 |               138 |                      0.783 |             0.452 |                      30 |                         122 |                265 |                       0.717 |              0.557 |
