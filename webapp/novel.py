"""Saved "new styles": recent-upload tokens plus the on-disk registry of styles the user chose to keep."""
import logging
import threading
from collections import OrderedDict
from datetime import datetime

import numpy as np
from fastapi import HTTPException

from .settings import WebSettings
from .storage import read_registry, write_registry

LOGGER = logging.getLogger(__name__)


class NovelStyleRegistry:
    def __init__(self, settings: WebSettings, clip_space):
        self.settings, self.clip_space = settings, clip_space
        self.tokens = OrderedDict()               # recent uploads (embeddings only)
        self.lock = threading.RLock()

    # ---------- persistence ----------
    def _read(self):
        with self.lock:
            try:
                return read_registry(self.settings.registry_path)
            except (OSError, ValueError) as error:
                LOGGER.exception("Could not read saved styles")
                raise HTTPException(503, "Không đọc được phong cách đã lưu. Hãy thử lại sau.") from error

    def _write(self, registry):
        try:
            write_registry(self.settings.registry_path, registry)
        except (OSError, ValueError) as error:
            LOGGER.exception("Could not persist saved styles")
            raise HTTPException(503, "Không ghi được phong cách đã lưu. Hãy thử lại sau.") from error

    @staticmethod
    def _public(style):
        return {k: v for k, v in style.items() if k != "proto"}

    # ---------- uploads ----------
    def remember(self, token: str, item: dict):
        with self.lock:
            self.tokens[token] = item
            while len(self.tokens) > self.settings.token_limit:
                self.tokens.popitem(last=False)

    def match(self, z_clip):
        """Nearest registered new style in FashionCLIP space, within the typical member-to-centroid distance."""
        best = None
        for s in self._read()["styles"]:
            p = np.asarray(s["proto"]["fashionclip"], dtype=np.float32)
            d = float(np.linalg.norm(z_clip - p / np.linalg.norm(p)))
            if best is None or d < best["distance"]:
                best = {"id": s["id"], "name": s["name"], "n": s["n"], "distance": round(d, 4)}
        radius = self.clip_space.meta["novel_radius"]
        return (best | {"radius": radius, "match": best["distance"] <= radius}) if best else None

    # ---------- API operations ----------
    def list(self):
        return [self._public(s) for s in self._read()["styles"]]

    def register(self, token: str, name: str | None):
        """Add the upload to the matching new style (running-mean prototype) or open a new one."""
        with self.lock:
            item = self.tokens.get(token)
            if item is None:
                raise HTTPException(404, "Kết quả quét đã hết hạn, hãy quét lại ảnh")
            reg = self._read()
            if item.get("saved_id"):
                saved = next((s for s in reg["styles"] if s["id"] == item["saved_id"]), None)
                if saved is None:
                    raise HTTPException(410, "Phong cách đã được xoá. Hãy quét lại ảnh trước khi lưu.")
                return self._public(saved)
            z = np.asarray(item["z"]["fashionclip"], dtype=np.float32)
            match = self.match(z)
            if match and match["match"]:
                s = next(x for x in reg["styles"] if x["id"] == match["id"])
                for e in self.settings.encoders:
                    s["proto"][e] = ((np.asarray(s["proto"][e]) * s["n"] + np.asarray(item["z"][e]))
                                     / (s["n"] + 1)).tolist()
                s["n"] += 1
                s["thumbs"] = (s["thumbs"] + [item["thumb"]])[-6:]
            else:
                n = max(reg.get("next_id", 1), 1 + max([int(x["id"][1:]) for x in reg["styles"]] or [0]))
                reg["next_id"] = n + 1
                s = {"id": f"N{n}", "name": ((name or "").strip() or item["description"])[:80], "n": 1,
                     "created": datetime.now().isoformat(timespec="seconds"),
                     "proto": dict(item["z"]), "thumbs": [item["thumb"]]}
                reg["styles"].append(s)
            self._write(reg)
            item["saved_id"] = s["id"]
        return self._public(s)

    def delete(self, style_id: str):
        with self.lock:
            reg = self._read()
            if not any(s["id"] == style_id for s in reg["styles"]):
                raise HTTPException(404, "Phong cách này không còn trong danh sách đã lưu.")
            reg["styles"] = [s for s in reg["styles"] if s["id"] != style_id]
            self._write(reg)
        return {"ok": True}
