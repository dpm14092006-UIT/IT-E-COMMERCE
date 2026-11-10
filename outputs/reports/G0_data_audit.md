# G0 - Kiem toan du lieu VISUELLE 2.0

Sinh tu `scripts/00_audit_data.py`. PASS = dat; FAIL = phai sua truoc khi di tiep; DECIDE = phai ghi quyet dinh vao `docs/decision_log.md`.

| Trang thai | Kiem tra |
| --- | --- |
| PASS | Khoa item-shop duy nhat trong sales.csv |
| PASS | Khoa item-shop duy nhat trong price_discount_series.csv |
| PASS | Moi san pham co anh (5355) |
| DECIDE | Anh khong giai ma duoc: [4442] -> loai khoi moi grouping (D14) |
| PASS | release_date deu la thu Hai -> khong co partial week |
| DECIDE | release_date khac nhau giua cac shop (91.9% san pham) -> (12) ap dung o cap item-shop |
| DECIDE | Ti le tuan 0 bang 0 = 0.25% (tuan 1: 26.83%) -> release_date gan nhu la tuan ban dau tien (left truncation) |
| PASS | Khong co o sales bi thieu |
| DECIDE | 3386 o sales am (net sales / hang tra lai?) tap trung o tuan {4: 893, 5: 961, 9: 804, 10: 728} |
| PASS | Google Trends la chuoi tuan (thu Hai) lien tuc |
| DECIDE | Google Trends co 660 o gia tri -1 (missing) |
| DECIDE | 80/96 chuoi GT dat 100 -> chuan hoa tren toan cua so (leak thang do) |
| PASS | Moi category/color/fabric co chuoi GT |
| DECIDE | GT ket thuc 2019-12-16, panel ban hang ket thuc 2020-03-16 |
| DECIDE | restocks.csv: 379562 item-shop (106839 co trong sales), nam 2016-2022 -> inventory tat cho toi khi chot ngu nghia |
| PASS | Tuan khong co san pham active: 0 |

## So lieu chinh

```json
{
  "sales": {
    "rows_item_shop": 106850,
    "n_products": 5355,
    "n_shops": 110,
    "duplicate_item_shop": 0,
    "products_per_season": {
      "AW17": 852,
      "AW18": 944,
      "AW19": 1054,
      "SS17": 901,
      "SS18": 758,
      "SS19": 846
    },
    "release_weekday_share_monday": 1.0,
    "release_first": "2016-11-28 00:00:00",
    "release_last": "2019-12-30 00:00:00",
    "last_observed_week": "2020-03-16 00:00:00",
    "share_products_multi_release_date": 0.919140989729225,
    "release_spread_weeks": {
      "count": 5355.0,
      "mean": 5.3,
      "std": 5.67,
      "min": 0.0,
      "25%": 2.0,
      "50%": 4.0,
      "75%": 7.0,
      "max": 66.0
    },
    "zero_share_by_relative_week": [
      0.0025,
      0.2683,
      0.2871,
      0.2918,
      0.2976,
      0.3111,
      0.3518,
      0.4035,
      0.4656,
      0.5279,
      0.5988,
      0.6644
    ],
    "negative_sales_cells": 3386,
    "negative_cells_by_relative_week": {
      "4": 893,
      "5": 961,
      "9": 804,
      "10": 728
    },
    "min_sales_cell": -3.0,
    "missing_sales_cells": 0,
    "attribute_cardinality": {
      "category": 27,
      "color": 10,
      "fabric": 59
    },
    "restock_to_12w_sales_ratio": {
      "count": 106850.0,
      "mean": 1.526,
      "std": 0.459,
      "min": 0.071,
      "25%": 1.222,
      "50%": 1.417,
      "75%": 1.714,
      "max": 9.0
    }
  },
  "panel": {
    "first_week": "2016-11-28 00:00:00",
    "last_week": "2020-03-16 00:00:00",
    "n_weeks": 173,
    "weeks_with_no_active_product": 0,
    "active_products_per_week": {
      "count": 173.0,
      "mean": 522.9,
      "std": 158.1,
      "min": 4.0,
      "25%": 510.0,
      "50%": 559.0,
      "75%": 611.0,
      "max": 755.0
    }
  }
}
```
