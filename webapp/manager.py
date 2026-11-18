"""Manager console API (/api/m/*): an executive view over the catalogue, its sales and the style forecasts.

Data views live in webapp/catalogue.py, dataset files in datafiles.py, saved scans in scan_store.py;
this module only maps HTTP routes onto them.
"""
import io
import re
from pathlib import Path

import numpy as np
from fastapi import APIRouter, File, Form, HTTPException, UploadFile
from fastapi.responses import FileResponse, Response
from starlette.concurrency import run_in_threadpool

from .catalogue import PRIMARY, Catalogue, vi
from .datafiles import DatasetFiles
from .scan_forecasts import ScanForecaster
from .scan_store import ScanStore, ScanSummarizer
from .uploads import decode_upload

MAX_UPLOAD = 15 * 1024 * 1024


class ManagerAPI:
    """Builds the /api/m router. analyze_image(raw_bytes, segment) -> the /api/analyze payload."""

    def __init__(self, root: Path, cfg, spaces, analyze_image):
        self.spaces, self.analyze_image = spaces, analyze_image
        self.cat = Catalogue(root, cfg, spaces)
        self.forecaster = ScanForecaster(root, cfg, spaces)
        self.scans = ScanStore(root / "outputs" / "web" / "scans")
        self.summarizer = ScanSummarizer(self.cat)
        self.datasets = DatasetFiles(root / cfg["paths"]["raw"], root / cfg["paths"]["interim"],
                                     root / cfg["paths"]["images"])
        self.router = APIRouter(prefix="/api/m")
        self._register()

    def _register(self):
        r = self.router
        r.add_api_route("/summary", self.summary, methods=["GET"])
        r.add_api_route("/origins", self.origins, methods=["GET"])
        r.add_api_route("/categories", self.categories, methods=["GET"])
        r.add_api_route("/categories/{name}", self.category, methods=["GET"])
        r.add_api_route("/products", self.products, methods=["GET"])
        r.add_api_route("/products/{code}", self.product, methods=["GET"])
        r.add_api_route("/files", self.files, methods=["GET"])
        r.add_api_route("/files/preview", self.file_preview, methods=["GET"])
        r.add_api_route("/files/download", self.file_download, methods=["GET"])
        r.add_api_route("/scans", self.scan, methods=["POST"])
        r.add_api_route("/scans", self.list_scans, methods=["GET"])
        r.add_api_route("/scans/{scan_id}", self.scan_detail, methods=["GET"])
        r.add_api_route("/scans/{scan_id}/image", self.scan_image, methods=["GET"])
        r.add_api_route("/scans/{scan_id}", self.scan_delete, methods=["DELETE"])

    # ---------- company overview ----------
    def summary(self, t: int | None = None):
        cat = self.cat
        t = cat.origin(t)
        p = cat.products
        styles = []
        for s in self.spaces[PRIMARY].meta["styles"]:
            codes = p.index[p["style"] == s["id"]]
            styles.append({"id": s["id"], "name": s["name"], "n_products": int(len(codes)),
                           "active": int(cat.is_active(codes, t).sum()), "outlook": cat.style_outlook(s["id"], t),
                           "examples": s["examples"][:3]})
        seasons = cat.breakdown(p.reset_index(), "season", translate=False)
        order = {f"{h}{y}": i for i, (y, h) in enumerate((y, h) for y in (17, 18, 19, 20) for h in ("SS", "AW"))}
        seasons.sort(key=lambda r: order.get(r["key"], 99))
        metrics = self.spaces[PRIMARY].meta["metrics"]
        return {"t": t, "week": cat.week(t), "t_last_label": int(cat.meta["t_last_label"]), "late": t >= cat.late_t,
                "kpi": {"products": int(len(p)), "units": round(float(p["units"].sum())),
                        "categories": len(cat.categories), "colors": int(p["color"].nunique()),
                        "shops": cat.shops,
                        "weeks": cat.t_max + 1, "first_week": cat.week(0), "last_week": cat.week(cat.t_max),
                        "active_now": int(cat.active[t]), "styles": len(styles)},
                "weekly": cat.weekly, "active": cat.active.tolist(), "seasons": seasons,
                "categories": cat.breakdown(p.reset_index(), "category", top=10),
                "styles": styles, "metrics": metrics, "decision": cat.decision_quality, "operating_point": cat.op,
                "scans": self.scans.count()}

    def origins(self):
        cat = self.cat
        return {"origins": [{"t": t, "week": cat.week(t)} for t in cat.meta["origins"]],
                "default": cat.default_t, "late_from": cat.late_t,
                "last_release": cat.week(cat.products["first_t"].max()),
                "t_last_label": int(cat.meta["t_last_label"]),
                "horizon": int(cat.meta["horizon"]), "week0": cat.meta["week0"], "t_max": cat.t_max}

    # ---------- categories ----------
    def categories(self, t: int | None = None):
        cat = self.cat
        t = cat.origin(t)
        p = cat.products
        out = []
        for c in cat.categories:
            g = p[p["category"] == c]
            out.append({"key": c, "label": vi(c), "products": int(len(g)), "units": round(float(g["units"].sum())),
                        "active": int(cat.is_active(g.index, t).sum()), "outlook": cat.group_outlook(g.index, t),
                        "example": int(g.sort_values("units", ascending=False).index[0])})
        return {"t": t, "week": cat.week(t), "categories": out}

    def category(self, name: str, t: int | None = None):
        cat = self.cat
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
        return {"t": t, "week": cat.week(t), "key": name, "label": vi(name),
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
    def products(self, q: str = "", category: str = "", color: str = "", season: str = "", fabric: str = "",
                 sort: str = "units", page: int = 1, size: int = 24, t: int | None = None):
        cat = self.cat
        t = cat.origin(t)
        p = cat.products
        mask = np.ones(len(p), bool)
        for col, val in (("category", category), ("color", color), ("season", season), ("fabric", fabric)):
            if val:
                mask &= (p[col] == val).to_numpy()
        if q.strip():
            q = q.strip().lower()
            text = (p.index.astype(str) + " " + p["category"] + " " + p["category"].map(vi) + " " + p["color"]
                    + " " + p["color"].map(vi) + " " + p["fabric"] + " " + p["season"]).str.lower()
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

    def product(self, code: int, t: int | None = None):
        cat = self.cat
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
                                 "history": self.spaces[PRIMARY].meta["history"][str(s)]["spap"]}
            # forecast made at the product's own launch, when that week is a forecast origin
            launch = int(w["t_idx"].min()) if len(w) else None
            row["launch_outlook"] = cat.style_outlook(s, launch) if launch in cat.meta["origins"] else None
            row["launch_t"] = launch
        same = cat.products[cat.products["category"] == row["category"]]["units"]
        row["category_rank"] = {"rank": int((same > row["units"]).sum() + 1), "of": int(len(same)),
                                "median_units": round(float(same.median()), 1)}
        return row

    # ---------- storage: dataset files ----------
    def files(self):
        return self.datasets.listing()

    def file_preview(self, key: str):
        return self.datasets.preview(key)

    def file_download(self, key: str):
        path, name = self.datasets.download(key)
        return FileResponse(path, filename=name)

    # ---------- scans: upload + forecast, kept for audit ----------
    async def scan(self, file: UploadFile = File(...), segment: str = Form("auto"), t: int | None = Form(None)):
        cat = self.cat
        t = cat.origin(t)
        try:
            raw = await file.read(MAX_UPLOAD + 1)
        finally:
            await file.close()
        if len(raw) > MAX_UPLOAD:
            raise HTTPException(413, "Ảnh quá lớn (tối đa 15 MB)")
        analysis = await self.analyze_image(raw, segment)
        summary = self.summarizer.summarize(analysis, t, (file.filename or "upload")[:120])
        # Keep the actual evidence with the saved scan so replay works after a restart.
        summary["process"] = {
            "segmentation": analysis["segmentation"],
            "encoders": {
                e: {**data, "style_name": cat.style_name(data["style"]["id"], e),
                    "forecasts": await run_in_threadpool(self.forecaster.for_scan, e, t, data),
                    "ranking": [{**r, "name": cat.style_name(r["id"], e)} for r in data["ranking"]]}
                for e, data in analysis["encoders"].items()
            },
        }
        thumb, _, _ = decode_upload(raw)
        thumb.thumbnail((640, 640))
        buf = io.BytesIO()
        thumb.save(buf, format="JPEG", quality=85)
        self.scans.save(summary, buf.getvalue())
        return {"scan": summary, "analysis": {"segmentation": analysis["segmentation"],
                                              "zero_shot": analysis["zero_shot"],
                                              "evidence": analysis["encoders"][PRIMARY]["evidence"]}}

    def list_scans(self):
        return self.scans.list()

    def scan_detail(self, scan_id: str):
        return self.scans.get(scan_id)

    def scan_image(self, scan_id: str):
        return Response(self.scans.image(scan_id), media_type="image/jpeg",
                        headers={"Cache-Control": "max-age=86400"})

    def scan_delete(self, scan_id: str):
        self.scans.delete(scan_id)
        return {"ok": True}


def build_router(root: Path, cfg, spaces, analyze_image):
    """Compatibility helper: the APIRouter of a ManagerAPI."""
    return ManagerAPI(root, cfg, spaces, analyze_image).router
