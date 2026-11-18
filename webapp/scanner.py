"""Style Scanner HTTP routes: pages, health/meta, catalogue thumbnails, /api/analyze and saved new styles."""
import io
from functools import lru_cache

import numpy as np
from fastapi import APIRouter, Body, File, Form, HTTPException, UploadFile
from fastapi.responses import FileResponse, JSONResponse, Response

from vstyle.visual.encoders import load_rgb

from .analyzer import ScanAnalyzer
from .novel import NovelStyleRegistry
from .settings import WebSettings
from .spaces import StyleSpace


class ScannerAPI:
    def __init__(self, settings: WebSettings, spaces: dict[str, StyleSpace], analyzer: ScanAnalyzer,
                 registry: NovelStyleRegistry, segmentation_label="SegFormer"):
        self.settings, self.spaces, self.analyzer, self.registry = settings, spaces, analyzer, registry
        self.segmentation_label = segmentation_label
        self._thumb = lru_cache(maxsize=512)(self._render_thumb)
        self.router = APIRouter()
        r = self.router
        r.add_api_route("/", self.manager_console, methods=["GET"])
        r.add_api_route("/scanner", self.index, methods=["GET"])
        r.add_api_route("/api/health", self.health, methods=["GET"])
        r.add_api_route("/api/meta", self.meta, methods=["GET"])
        r.add_api_route("/catalog/{code}.jpg", self.catalog_image, methods=["GET"])
        r.add_api_route("/api/analyze", self.analyze, methods=["POST"])
        r.add_api_route("/api/novel", self.novel_list, methods=["GET"])
        r.add_api_route("/api/novel", self.novel_register, methods=["POST"])
        r.add_api_route("/api/novel/{style_id}", self.novel_delete, methods=["DELETE"])

    # ---------- pages ----------
    def _page(self, name: str, lang: str):
        static = self.settings.static_dir
        target = (static / "en" / name) if lang == "en" else (static / name)
        return FileResponse(target, headers={"Cache-Control": "no-store",
                                             "Content-Language": "en" if lang == "en" else "vi"})

    def manager_console(self, lang: str = "vi"):
        return self._page("manager.html", lang)

    def index(self, lang: str = "vi"):
        return self._page("index.html", lang)

    # ---------- metadata ----------
    async def health(self):
        return {"status": "ready", "encoders": list(self.settings.encoders), "segmentation": self.segmentation_label,
                "max_upload_bytes": self.settings.max_bytes,
                "groups_per_encoder": {e: len(sp.C) for e, sp in self.spaces.items()}}

    def meta(self):
        out = {}
        for e, sp in self.spaces.items():
            styles = [{k: v for k, v in s.items() if k != "dist_sorted"} for s in sp.meta["styles"]]
            out[e] = {**sp.meta, "styles": styles, "label": self.settings.labels[e]}
        return JSONResponse(out)

    def _render_thumb(self, code, size):
        sp = self.spaces["resnet50"]
        where = np.where(sp.ids == code)[0]
        if not len(where):
            raise HTTPException(404)
        im = load_rgb(self.settings.images_dir / str(sp.image_path[where[0]]), self.settings.background)
        im.thumbnail((size, size))
        buf = io.BytesIO()
        im.save(buf, format="JPEG", quality=85)
        return buf.getvalue()

    def catalog_image(self, code: int, size: int = 240):
        return Response(self._thumb(code, min(max(size, 64), 480)), media_type="image/jpeg",
                        headers={"Cache-Control": "max-age=86400"})

    # ---------- analysis ----------
    async def analyze(self, file: UploadFile = File(...), segment: str = Form("auto")):
        self.analyzer.check_mode(segment)
        limit = self.settings.max_bytes
        try:
            raw = await file.read(limit + 1)
        finally:
            await file.close()
        if len(raw) > limit:
            raise HTTPException(413, "Ảnh quá lớn (tối đa 15 MB)")
        return await self.analyzer.analyze_bytes(raw, segment)

    # ---------- saved new styles ----------
    def novel_list(self):
        return self.registry.list()

    def novel_register(self, token: str = Body(...), name: str | None = Body(None)):
        return self.registry.register(token, name)

    def novel_delete(self, style_id: str):
        return self.registry.delete(style_id)
