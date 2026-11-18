"""Read-only view of the raw and processed dataset files shown in the console's data-storage tab."""
from functools import lru_cache
from pathlib import Path

import pandas as pd
from fastapi import HTTPException

PREVIEW_ROWS = 50
PROCESSED = ("products.parquet", "product_week.parquet", "item_shop_week.parquet", "calendar.json")
DESCRIBE = {
    "sales.csv": "Doanh số 12 tuần đầu của từng sản phẩm tại từng cửa hàng, kèm thuộc tính sản phẩm",
    "customer_data.csv": "Giao dịch theo khách hàng (mã khách, cửa hàng, sản phẩm, thời điểm, số lượng)",
    "price_discount_series.csv": "Chuỗi chiết khấu 12 tuần và giá (đã chuẩn hoá) theo sản phẩm × cửa hàng",
    "restocks.csv": "Lượt bổ sung hàng theo sản phẩm × cửa hàng × tuần",
    "vis2_gtrends_data.csv": "Google Trends theo tuần cho loại sản phẩm, màu, chất liệu",
    "vis2_weather_data.csv": "Thời tiết theo ngày tại các địa phương có cửa hàng",
    "stfore_train.csv": "Tập train chuẩn của benchmark VISUELLE (doanh số đã chuẩn hoá)",
    "stfore_test.csv": "Tập test chuẩn của benchmark VISUELLE (doanh số đã chuẩn hoá)",
    "category_labels.pt": "Từ điển nhãn loại sản phẩm (PyTorch)", "color_labels.pt": "Từ điển nhãn màu (PyTorch)",
    "fabric_labels.pt": "Từ điển nhãn chất liệu (PyTorch)",
    "shop_weather_pairs.pt": "Ghép cửa hàng với trạm thời tiết (PyTorch)",
    "stfore_sales_norm_scalar.npy": "Hệ số chuẩn hoá doanh số của benchmark",
    "products.parquet": "Đã xử lý: một dòng mỗi sản phẩm (thuộc tính, ngày ra mắt, số cửa hàng)",
    "product_week.parquet": "Đã xử lý: doanh số theo sản phẩm × tuần lịch (giá, chiết khấu, số cửa hàng)",
    "item_shop_week.parquet": "Đã xử lý: doanh số theo sản phẩm × cửa hàng × tuần lịch",
    "calendar.json": "Đã xử lý: lịch tuần và kiểm tra bảo toàn doanh số",
}


def count_rows(path: Path):
    """Data rows of a CSV (header excluded), counted on raw bytes."""
    n = 0
    with open(path, "rb") as f:
        while chunk := f.read(1 << 22):
            n += chunk.count(b"\n")
    return max(n - 1, 0)


class DatasetFiles:
    def __init__(self, raw_dir: Path, interim_dir: Path, images_dir: Path):
        self.images_dir = images_dir
        out = [{"key": f"raw/{f.name}", "name": f.name, "group": "raw", "path": f}
               for f in sorted(raw_dir.iterdir()) if f.is_file()]
        out += [{"key": f"interim/{name}", "name": name, "group": "interim", "path": interim_dir / name}
                for name in PROCESSED if (interim_dir / name).exists()]
        self.files = {f["key"]: f for f in out}
        self._table_info = lru_cache(maxsize=64)(self._read_table_info)

    def _require(self, key):
        if key not in self.files:
            raise HTTPException(404, "Không có tệp này trong kho dữ liệu.")
        return self.files[key]

    def _read_table_info(self, key):
        f = self.files[key]["path"]
        if f.suffix == ".csv":
            return count_rows(f), pd.read_csv(f, nrows=PREVIEW_ROWS)
        if f.suffix == ".parquet":
            import pyarrow.parquet as pq
            parquet = pq.ParquetFile(f)
            return parquet.metadata.num_rows, next(parquet.iter_batches(batch_size=PREVIEW_ROWS)).to_pandas()
        return None, None

    def listing(self):
        out = []
        for key, f in self.files.items():
            rows, head = self._table_info(key) if f["path"].suffix in (".csv", ".parquet") else (None, None)
            out.append({"key": key, "name": f["name"], "group": f["group"], "bytes": f["path"].stat().st_size,
                        "kind": f["path"].suffix.lstrip("."), "rows": rows,
                        "columns": len(head.columns) if head is not None else None,
                        "description": DESCRIBE.get(f["name"], ""), "previewable": head is not None})
        folders = sorted(d for d in self.images_dir.iterdir() if d.is_dir())
        images = [{"folder": d.name, "count": sum(1 for _ in d.glob("*.png")),
                   "season": ("AW" if d.name.startswith("AI") else "SS") + d.name[2:]} for d in folders]
        return {"files": out, "images": images, "image_total": sum(i["count"] for i in images),
                "raw_bytes": sum(f["bytes"] for f in out if f["group"] == "raw")}

    def preview(self, key):
        self._require(key)
        rows, head = self._table_info(key)
        if head is None:
            raise HTTPException(415, "Tệp nhị phân, chỉ có thể tải về.")
        head = head.loc[:, [c for c in head.columns if not str(c).startswith("Unnamed")]]
        values = head.astype(object).where(head.notna(), None)
        cells = [[v.isoformat() if hasattr(v, "isoformat") else v for v in r] for r in values.itertuples(index=False)]
        return {"key": key, "rows": rows, "columns": [str(c) for c in head.columns],
                "dtypes": [str(d) for d in head.dtypes], "data": cells}

    def download(self, key):
        f = self._require(key)
        return f["path"], f["name"]
