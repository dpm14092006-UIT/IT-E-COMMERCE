"""Manager console - executive view over the VISUELLE 2.0 catalogue, its sales and the style forecasts.

Everything here reads data that already exists; nothing is re-trained:
  * catalogue + weekly sales      outputs/interim/products.parquet, product_week.parquet (scripts/01)
  * product -> style              outputs/web/resnet50/assets.npz (frozen K-Means, scripts/17)
  * style forecasts               outputs/web/{encoder}/meta.json - walk-forward test predictions, then deployment
  * review signals                outputs/reports/resnet50_decision.json - operating point chosen on VALIDATION (4.10)
A product's forecast is the forecast of its style: the models never forecast single products.
Scans uploaded from the console are kept in outputs/web/scans/ ({id}.json + {id}.jpg) so they can be audited.
"""
import io
import json
import os
import re
import tempfile
import uuid
from datetime import datetime
from functools import lru_cache
from pathlib import Path

import numpy as np
import pandas as pd
from fastapi import APIRouter, File, Form, HTTPException, UploadFile
from fastapi.responses import FileResponse, Response
from .uploads import decode_upload
from .zero_shot import CATALOGUE_VI

PRIMARY = "resnet50"            # primary grouping of the study (configs/default.yaml)
MODEL = "catboost"              # the model the decision operating point was chosen for
TREND_VI = ["Đang giảm", "Ổn định", "Đang lên"]
SIGNAL_VI = {"emerging_alert": "Ưu tiên mở rộng", "declining_alert": "Rà soát nhập hàng",
             "hold": "Duy trì, theo dõi"}
SIGNAL_ACTION = {"emerging_alert": "Ưu tiên theo dõi; cân nhắc mở rộng assortment của style.",
                 "declining_alert": "Rà soát nhập mới; cân nhắc markdown hoặc giảm độ phủ.",
                 "hold": "Duy trì; theo dõi định kỳ."}
DEFAULT_WEEK = "2019-10-14"     # mid AW19 season, labels observed; chosen by season, not by accuracy
LATE_FROM = "2019-12-02"        # launches dry up (last release 2019-12-30): every style drifts down after this
LIFECYCLE = 12                 # VISUELLE records 12 weeks of sales per product
SCAN_ID = re.compile(r"^[0-9a-f]{32}$")
PREVIEW_ROWS = 50


def _vi(value):
    return CATALOGUE_VI.get(str(value), str(value))


def _atomic_write(path: Path, data: bytes):
    path.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile(dir=path.parent, prefix=".tmp-", delete=False) as stream:
        stream.write(data)
        stream.flush()
        os.fsync(stream.fileno())
    os.replace(stream.name, path)


def _count_rows(path: Path):
    """Data rows of a CSV (header excluded), counted on raw bytes."""
    n = 0
    with open(path, "rb") as f:
        while chunk := f.read(1 << 22):
            n += chunk.count(b"\n")
    return max(n - 1, 0)


class Catalogue:
    """In-memory product, category and company views; built once at start-up (a few seconds)."""

    def __init__(self, root: Path, cfg, spaces):
        self.root, self.cfg, self.spaces = root, cfg, spaces
        interim = root / cfg["paths"]["interim"]
        products = pd.read_parquet(interim / "products.parquet")
        pw = pd.read_parquet(interim / "product_week.parquet",
                             columns=["external_code", "t_idx", "sales", "price", "discount", "n_shops"])
        meta = spaces[PRIMARY].meta
        self.meta, self.t_max, self.week0 = meta, int(meta["t_max"]), pd.Timestamp(meta["week0"])
        sp = spaces[PRIMARY]
        style_of = dict(zip(sp.ids.tolist(), sp.style_of.tolist()))
        stats = pw.groupby("external_code").agg(units=("sales", "sum"), first_t=("t_idx", "min"),
                                                last_t=("t_idx", "max"), weeks=("t_idx", "nunique"),
                                                price=("price", "mean"), discount=("discount", "mean"),
                                                peak_shops=("n_shops", "max"))
        p = products.set_index("external_code").join(stats, how="left")
        p["units"] = p["units"].fillna(0)
        p["style"] = [style_of.get(int(c), -1) for c in p.index]
        p["release"] = p["release_first"].dt.strftime("%Y-%m-%d")
        self.products = p.sort_index()
        self.pw = pw.sort_values(["external_code", "t_idx"])
        self.shops = int(pd.read_parquet(interim / "item_shop_week.parquet", columns=["retail"])["retail"].nunique())
        self.weekly = self._series(pw)
        self.active = pw.groupby("t_idx")["external_code"].nunique().reindex(range(self.t_max + 1), fill_value=0)
        self.categories = sorted(p["category"].unique(), key=lambda c: -p.loc[p["category"] == c, "units"].sum())
        self.cat_weekly = {c: self._series(pw[pw["external_code"].isin(p.index[p["category"] == c])])
                           for c in self.categories}
        op = root / cfg["paths"]["reports"] / f"{PRIMARY}_decision.json"
        decision = json.loads(op.read_text(encoding="utf-8")) if op.exists() else {}
        self.op = decision.get("operating_point", {"min_prob": 0.5, "min_margin": 0.0})
        self.decision_quality = decision.get("quality")
        to_t = lambda d: int((pd.Timestamp(d) - self.week0).days // 7)  # noqa: E731
        default = to_t(DEFAULT_WEEK)
        self.default_t = default if default in meta["origins"] else int(meta["default_origin"])
        self.late_t = to_t(LATE_FROM)

    def _series(self, frame):
        return frame.groupby("t_idx")["sales"].sum().reindex(range(self.t_max + 1), fill_value=0).round(1).tolist()

    def week(self, t):
        return (self.week0 + pd.Timedelta(weeks=int(t))).strftime("%Y-%m-%d")

    def origin(self, t):
        origins = self.meta["origins"]
        if t is None:
            return self.default_t
        if int(t) not in origins:
            raise HTTPException(400, f"Thời điểm dự báo phải nằm trong tuần {origins[0]}–{origins[-1]}")
        return int(t)

    # ---------- forecasts ----------
    def signal(self, p):
        p = np.asarray(p, dtype=float)
        top, srt = int(p.argmax()), np.sort(p)
        confident = p.max() >= self.op["min_prob"] and srt[-1] - srt[-2] >= self.op["min_margin"]
        return {2: "emerging_alert", 0: "declining_alert"}.get(top, "hold") if confident else "hold"

    def _pack(self, p, extra=None):
        p = [round(float(x), 4) for x in p]
        k, s = int(np.argmax(p)), self.signal(p)
        return {"p": p, "trend": k, "trend_vi": TREND_VI[k], "confidence": max(p),
                "signal": s, "signal_vi": SIGNAL_VI[s], "action": SIGNAL_ACTION[s], **(extra or {})}

    def style_outlook(self, style, t, enc=PRIMARY):
        f = self.spaces[enc].meta["forecasts"].get(str(t), {}).get(str(style))
        if not f or MODEL not in f:
            return None
        m = f[MODEL]
        return self._pack(m["p"], {"style": int(style), "tcn_p": f.get("tcn", {}).get("p"),
                                   "y_true": m.get("y_true"), "source": m.get("source"),
                                   "actual_vi": TREND_VI[m["y_true"]] if m.get("y_true") is not None else None})

    def proxy_outlook(self, ranking, t, enc):
        """A new style has no history: inverse-distance mix of its 3 nearest styles, labelled as a proxy."""
        near = [r for r in ranking if self.style_outlook(r["id"], t, enc)][:3]
        if not near:
            return None
        w = np.array([1 / max(r["distance"], 1e-9) for r in near])
        w /= w.sum()
        p = sum(wi * np.asarray(self.style_outlook(r["id"], t, enc)["p"]) for wi, r in zip(w, near))
        return self._pack(p, {"proxy": True, "parts": [{"style": r["id"], "weight": round(float(wi), 3)}
                                                       for wi, r in zip(w, near)]})

    def is_active(self, codes, t):
        p = self.products.loc[codes]
        return (p["first_t"] <= t) & (p["last_t"] >= t)

    def group_outlook(self, codes, t):
        """Products weighted by their style's forecast; products on sale at t, otherwise all of them."""
        p = self.products.loc[codes]
        p = p[p["style"] >= 0]
        live = p[(p["first_t"] <= t) & (p["last_t"] >= t)]
        base = live if len(live) else p
        weights = base["style"].value_counts()
        rows = [(self.style_outlook(s, t), n) for s, n in weights.items()]
        rows = [(o, n) for o, n in rows if o]
        if not rows:
            return None
        total = sum(n for _, n in rows)
        mix = sum(np.asarray(o["p"]) * n for o, n in rows) / total
        share = {sig: round(sum(n for o, n in rows if o["signal"] == sig) / total, 3) for sig in SIGNAL_VI}
        return self._pack(mix, {"weighted_by": "active" if len(live) else "all", "n_weighted": int(total),
                                "signal_share": share})

    # ---------- views ----------
    def product_row(self, code, t=None):
        r = self.products.loc[code]
        row = {"code": int(code), "category": r["category"], "category_vi": _vi(r["category"]),
               "color": r["color"], "color_vi": _vi(r["color"]), "fabric": r["fabric"], "season": r["season"],
               "release": r["release"], "units": round(float(r["units"]), 0), "shops": int(r["n_shops"]),
               "style": int(r["style"]), "has_image": bool(r["image_ok"] and r["style"] >= 0),
               "price": None if pd.isna(r["price"]) else round(float(r["price"]), 4),
               "discount": None if pd.isna(r["discount"]) else round(float(r["discount"]), 3)}
        if t is not None:
            row["status"] = ("upcoming" if r["first_t"] > t else "ended" if r["last_t"] < t else "active") \
                if not pd.isna(r["first_t"]) else "no_sales"
            row["outlook"] = self.style_outlook(r["style"], t) if r["style"] >= 0 else None
        return row

    def style_name(self, style, enc=PRIMARY):
        return self.spaces[enc].meta["styles"][int(style)]["name"]

    def breakdown(self, frame, col, top=None, translate=True):
        g = frame.groupby(col)["units"].agg(["sum", "count"]).sort_values("sum", ascending=False)
        total = g["sum"].sum() or 1
        rows = [{"key": k, "label": _vi(k) if translate else k, "units": round(float(v["sum"])),
                 "products": int(v["count"]), "share": round(float(v["sum"] / total), 4)} for k, v in g.iterrows()]
        return rows[:top] if top else rows


def build_router(root: Path, cfg, spaces, analyze_image):
    """analyze_image(raw_bytes, segment) -> the /api/analyze payload (web/app.py)."""
    cat = Catalogue(root, cfg, spaces)
    scans_dir = root / "outputs" / "web" / "scans"
    raw_dir = root / cfg["paths"]["raw"]
    images_dir = root / cfg["paths"]["images"]
    router = APIRouter(prefix="/api/m")

    # ---------- company overview ----------
    @router.get("/summary")
    def summary(t: int | None = None):
        t = cat.origin(t)
        p = cat.products
        styles = []
        for s in spaces[PRIMARY].meta["styles"]:
            codes = p.index[p["style"] == s["id"]]
            styles.append({"id": s["id"], "name": s["name"], "n_products": int(len(codes)),
                           "active": int(cat.is_active(codes, t).sum()), "outlook": cat.style_outlook(s["id"], t),
                           "examples": s["examples"][:3]})
        seasons = cat.breakdown(p.reset_index(), "season", translate=False)
        order = {f"{h}{y}": i for i, (y, h) in enumerate((y, h) for y in (17, 18, 19, 20) for h in ("SS", "AW"))}
        seasons.sort(key=lambda r: order.get(r["key"], 99))
        metrics = spaces[PRIMARY].meta["metrics"]
        return {"t": t, "week": cat.week(t), "t_last_label": int(cat.meta["t_last_label"]), "late": t >= cat.late_t,
                "kpi": {"products": int(len(p)), "units": round(float(p["units"].sum())),
                        "categories": len(cat.categories), "colors": int(p["color"].nunique()),
                        "shops": cat.shops,
                        "weeks": cat.t_max + 1, "first_week": cat.week(0), "last_week": cat.week(cat.t_max),
                        "active_now": int(cat.active[t]), "styles": len(styles)},
                "weekly": cat.weekly, "active": cat.active.tolist(), "seasons": seasons,
                "categories": cat.breakdown(p.reset_index(), "category", top=10),
                "styles": styles, "metrics": metrics, "decision": cat.decision_quality, "operating_point": cat.op,
                "scans": len(list(scans_dir.glob("*.json"))) if scans_dir.exists() else 0}

    @router.get("/origins")
    def origins():
        return {"origins": [{"t": t, "week": cat.week(t)} for t in cat.meta["origins"]],
                "default": cat.default_t, "late_from": cat.late_t, "last_release": cat.week(cat.products["first_t"].max()),
                "t_last_label": int(cat.meta["t_last_label"]),
                "horizon": int(cat.meta["horizon"]), "week0": cat.meta["week0"], "t_max": cat.t_max}

    # ---------- categories ----------
    @router.get("/categories")
    def categories(t: int | None = None):
        t = cat.origin(t)
        p = cat.products
        out = []
        for c in cat.categories:
            g = p[p["category"] == c]
            out.append({"key": c, "label": _vi(c), "products": int(len(g)), "units": round(float(g["units"].sum())),
                        "active": int(cat.is_active(g.index, t).sum()), "outlook": cat.group_outlook(g.index, t),
                        "example": int(g.sort_values("units", ascending=False).index[0])})
        return {"t": t, "week": cat.week(t), "categories": out}

    @router.get("/categories/{name}")
    def category(name: str, t: int | None = None):
        t = cat.origin(t)
        if name not in cat.categories:
            raise HTTPException(404, "Không có loại sản phẩm này trong dữ liệu.")
        p = cat.products
        g = p[p["category"] == name]
        flat = g.reset_index()
        styles = []
        for s, n in g[g["style"] >= 0]["style"].value_counts().items():
            styles.append({"id": int(s), "name": cat.style_name(s), "products": int(n),
                           "outlook": cat.style_outlook(s, t)})
        top = g.sort_values("units", ascending=False).index[:12]
        return {"t": t, "week": cat.week(t), "key": name, "label": _vi(name),
                "kpi": {"products": int(len(g)), "units": round(float(g["units"].sum())),
                        "units_per_product": round(float(g["units"].mean()), 1),
                        "share": round(float(g["units"].sum() / p["units"].sum()), 4),
                        "active": int(cat.is_active(g.index, t).sum()),
                        "price": round(float(g["price"].mean()), 4), "discount": round(float(g["discount"].mean()), 3)},
                "weekly": cat.cat_weekly[name], "outlook": cat.group_outlook(g.index, t),
                "colors": cat.breakdown(flat, "color"), "seasons": cat.breakdown(flat, "season", translate=False),
                "fabrics": cat.breakdown(flat, "fabric", top=8, translate=False), "styles": styles,
                "top_products": [cat.product_row(c, t) for c in top]}

    # ---------- storage: catalogue ----------
    @router.get("/products")
    def products(q: str = "", category: str = "", color: str = "", season: str = "", fabric: str = "",
                 sort: str = "units", page: int = 1, size: int = 24, t: int | None = None):
        t = cat.origin(t)
        p = cat.products
        mask = np.ones(len(p), bool)
        for col, val in (("category", category), ("color", color), ("season", season), ("fabric", fabric)):
            if val:
                mask &= (p[col] == val).to_numpy()
        if q.strip():
            q = q.strip().lower()
            text = (p.index.astype(str) + " " + p["category"] + " " + p["category"].map(_vi) + " " + p["color"]
                    + " " + p["color"].map(_vi) + " " + p["fabric"] + " " + p["season"]).str.lower()
            mask &= text.str.contains(re.escape(q)).to_numpy()
        hit = p[mask]
        keys = {"units": ("units", False), "units_asc": ("units", True), "code": (None, True),
                "release": ("release_first", False), "shops": ("n_shops", False)}
        col, asc = keys.get(sort, keys["units"])
        hit = hit.sort_index(ascending=asc) if col is None else hit.sort_values(col, ascending=asc, kind="stable")
        size = min(max(size, 6), 96)
        pages = max(1, -(-len(hit) // size))
        page = min(max(page, 1), pages)
        codes = hit.index[(page - 1) * size: page * size]
        return {"total": int(len(hit)), "page": page, "pages": pages, "size": size,
                "items": [cat.product_row(c, t) for c in codes],
                "facets": {k: sorted(p[k].unique().tolist()) for k in ("category", "color", "season", "fabric")}}

    @router.get("/products/{code}")
    def product(code: int, t: int | None = None):
        t = cat.origin(t)
        if code not in cat.products.index:
            raise HTTPException(404, "Không có sản phẩm này trong dữ liệu.")
        row = cat.product_row(code, t)
        w = cat.pw[cat.pw["external_code"] == code]
        row["weekly"] = [{"t": int(r.t_idx), "week": cat.week(r.t_idx), "units": float(r.sales), "shops": int(r.n_shops),
                          "price": round(float(r.price), 4), "discount": round(float(r.discount), 3)}
                         for r in w.itertuples()]
        if row["style"] >= 0:
            s = row["style"]
            peers = cat.products[cat.products["style"] == s]
            row["style_info"] = {"id": s, "name": cat.style_name(s), "products": int(len(peers)),
                                 "units_per_product": round(float(peers["units"].mean()), 1),
                                 "history": spaces[PRIMARY].meta["history"][str(s)]["spap"]}
            # forecast made at the product's own launch, when that week is a forecast origin
            launch = int(w["t_idx"].min()) if len(w) else None
            row["launch_outlook"] = cat.style_outlook(s, launch) if launch in cat.meta["origins"] else None
            row["launch_t"] = launch
        same = cat.products[cat.products["category"] == row["category"]]["units"]
        row["category_rank"] = {"rank": int((same > row["units"]).sum() + 1), "of": int(len(same)),
                                "median_units": round(float(same.median()), 1)}
        return row

    # ---------- storage: dataset files ----------
    def _files():
        out = []
        for f in sorted(raw_dir.iterdir()):
            if f.is_file():
                out.append({"key": f"raw/{f.name}", "name": f.name, "group": "raw", "path": f})
        interim = root / cfg["paths"]["interim"]
        for name in ("products.parquet", "product_week.parquet", "item_shop_week.parquet", "calendar.json"):
            if (interim / name).exists():
                out.append({"key": f"interim/{name}", "name": name, "group": "interim", "path": interim / name})
        return {f["key"]: f for f in out}

    FILES = _files()
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

    @lru_cache(maxsize=64)
    def _table_info(key):
        f = FILES[key]["path"]
        if f.suffix == ".csv":
            head = pd.read_csv(f, nrows=PREVIEW_ROWS)
            return _count_rows(f), head
        if f.suffix == ".parquet":
            import pyarrow.parquet as pq
            meta = pq.ParquetFile(f).metadata
            head = next(pq.ParquetFile(f).iter_batches(batch_size=PREVIEW_ROWS)).to_pandas()
            return meta.num_rows, head
        return None, None

    @router.get("/files")
    def files():
        out = []
        for key, f in FILES.items():
            rows, head = _table_info(key) if f["path"].suffix in (".csv", ".parquet") else (None, None)
            out.append({"key": key, "name": f["name"], "group": f["group"], "bytes": f["path"].stat().st_size,
                        "kind": f["path"].suffix.lstrip("."), "rows": rows,
                        "columns": len(head.columns) if head is not None else None,
                        "description": DESCRIBE.get(f["name"], ""), "previewable": head is not None})
        folders = sorted(d for d in images_dir.iterdir() if d.is_dir())
        images = [{"folder": d.name, "count": sum(1 for _ in d.glob("*.png")),
                   "season": ("AW" if d.name.startswith("AI") else "SS") + d.name[2:]} for d in folders]
        return {"files": out, "images": images, "image_total": sum(i["count"] for i in images),
                "raw_bytes": sum(f["bytes"] for f in out if f["group"] == "raw")}

    @router.get("/files/preview")
    def file_preview(key: str):
        if key not in FILES:
            raise HTTPException(404, "Không có tệp này trong kho dữ liệu.")
        rows, head = _table_info(key)
        if head is None:
            raise HTTPException(415, "Tệp nhị phân, chỉ có thể tải về.")
        head = head.loc[:, [c for c in head.columns if not str(c).startswith("Unnamed")]]
        values = head.astype(object).where(head.notna(), None)
        cells = [[v.isoformat() if hasattr(v, "isoformat") else v for v in r] for r in values.itertuples(index=False)]
        return {"key": key, "rows": rows, "columns": [str(c) for c in head.columns],
                "dtypes": [str(d) for d in head.dtypes], "data": cells}

    @router.get("/files/download")
    def file_download(key: str):
        if key not in FILES:
            raise HTTPException(404, "Không có tệp này trong kho dữ liệu.")
        return FileResponse(FILES[key]["path"], filename=FILES[key]["name"])

    # ---------- scans: upload + forecast, kept for audit ----------
    def _scan_summary(a, t, name):
        E = a["encoders"]
        enc = next((e for e in (PRIMARY, "fashionclip") if not E[e]["style"]["novel"]), None)
        if enc:
            s = E[enc]["style"]
            outlook = cat.style_outlook(s["id"], t, enc)
            style = {"encoder": enc, "id": s["id"], "name": cat.style_name(s["id"], enc),
                     "typicality_pct": s["typicality_pct"], "novel": False}
        else:
            s = E[PRIMARY]["style"]
            outlook = cat.proxy_outlook(E[PRIMARY]["ranking"], t, PRIMARY)
            style = {"encoder": PRIMARY, "id": s["id"], "name": cat.style_name(s["id"]), "novel": True,
                     "typicality_pct": s["typicality_pct"]}
        neighbors = []
        for n in E[PRIMARY]["neighbors"][:6]:
            r = cat.products.loc[n["code"]]
            neighbors.append({"code": n["code"], "similarity": n["similarity"], "category_vi": n["category_vi"],
                              "color_vi": n["color_vi"], "season": n["season"], "units": round(float(r["units"]))})
        cats = pd.Series([n["category"] for n in E[PRIMARY]["neighbors"]]).value_counts()
        zs = a["zero_shot"]
        category = zs["groups"][0]["items"][0]
        return {"id": a["token"], "name": name, "created": datetime.now().isoformat(timespec="seconds"),
                "t": t, "week": cat.week(t), "description": zs["short"],
                "category": {"vi": category["vi"], "p": category["p"], "known": category["known"],
                             "catalogue": cats.index[0], "catalogue_vi": _vi(cats.index[0])},
                "colors": [{"vi": c["vi"], "hex": c.get("hex"), "share": c.get("share")} for c in zs["colors"][:3]],
                "style": style, "outlook": outlook,
                "near_duplicate": E[PRIMARY]["near_duplicate"], "novel_both": not enc,
                "neighbors": neighbors,
                "reference_units": round(float(np.mean([n["units"] for n in neighbors])), 1) if neighbors else None,
                "agreement": a["agreement"], "segmentation": a["segmentation"].get("applied", False)}

    @router.post("/scans")
    async def scan(file: UploadFile = File(...), segment: str = Form("auto"), t: int | None = Form(None)):
        t = cat.origin(t)
        try:
            raw = await file.read(15 * 1024 * 1024 + 1)
        finally:
            await file.close()
        if len(raw) > 15 * 1024 * 1024:
            raise HTTPException(413, "Ảnh quá lớn (tối đa 15 MB)")
        analysis = await analyze_image(raw, segment)
        summary = _scan_summary(analysis, t, (file.filename or "upload")[:120])
        thumb, _, _ = decode_upload(raw)
        thumb.thumbnail((640, 640))
        buf = io.BytesIO()
        thumb.save(buf, format="JPEG", quality=85)
        _atomic_write(scans_dir / f"{summary['id']}.jpg", buf.getvalue())
        _atomic_write(scans_dir / f"{summary['id']}.json", json.dumps(summary, ensure_ascii=False).encode("utf-8"))
        return {"scan": summary, "analysis": {"segmentation": analysis["segmentation"],
                                              "zero_shot": analysis["zero_shot"],
                                              "evidence": analysis["encoders"][PRIMARY]["evidence"]}}

    def _scan_path(scan_id, ext):
        if not SCAN_ID.match(scan_id):
            raise HTTPException(404, "Không có lượt quét này.")
        f = scans_dir / f"{scan_id}.{ext}"
        if not f.exists():
            raise HTTPException(404, "Không có lượt quét này.")
        return f

    @router.get("/scans")
    def scans():
        if not scans_dir.exists():
            return []
        items = []
        for f in scans_dir.glob("*.json"):
            try:
                items.append(json.loads(f.read_text(encoding="utf-8")))
            except (OSError, ValueError):
                continue
        return sorted(items, key=lambda s: s["created"], reverse=True)

    @router.get("/scans/{scan_id}")
    def scan_detail(scan_id: str):
        return json.loads(_scan_path(scan_id, "json").read_text(encoding="utf-8"))

    @router.get("/scans/{scan_id}/image")
    def scan_image(scan_id: str):
        return Response(_scan_path(scan_id, "jpg").read_bytes(), media_type="image/jpeg",
                        headers={"Cache-Control": "max-age=86400"})

    @router.delete("/scans/{scan_id}")
    def scan_delete(scan_id: str):
        _scan_path(scan_id, "json").unlink()
        (scans_dir / f"{scan_id}.jpg").unlink(missing_ok=True)
        return {"ok": True}

    return router
