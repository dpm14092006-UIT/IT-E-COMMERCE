"""Image analysis service: segmentation, both encoders, placement in the style spaces, zero-shot description."""
import logging
import threading
import uuid

import numpy as np
from fastapi import HTTPException
from starlette.concurrency import run_in_threadpool

from .colors import analyse
from .encoders import ExplainableEncoder
from .novel import NovelStyleRegistry
from .rendering import png_b64
from .segment import MODES, GarmentSegmenter
from .settings import WebSettings
from .spaces import StyleSpace
from .uploads import decode_upload
from .zero_shot import describe

LOGGER = logging.getLogger(__name__)


class ScanAnalyzer:
    def __init__(self, settings: WebSettings, spaces: dict[str, StyleSpace], encoders: dict[str, ExplainableEncoder],
                 segmenter: GarmentSegmenter, text_bank, registry: NovelStyleRegistry):
        self.settings, self.spaces, self.encoders = settings, spaces, encoders
        self.segmenter, self.text_bank, self.registry = segmenter, text_bank, registry
        self.clip = encoders["fashionclip"]
        self._inference_lock = threading.Lock()      # CPU inference is serialised across requests

    def check_mode(self, segment):
        if segment not in MODES:
            raise HTTPException(400, f"segment phải là một trong {MODES}")

    async def analyze_bytes(self, raw, segment):
        self.check_mode(segment)
        try:
            return await run_in_threadpool(self._analyze_image, raw, segment)
        except HTTPException:
            raise
        except Exception as error:
            LOGGER.exception("Image inference failed")
            raise HTTPException(503, "Máy chủ không xử lý được ảnh. Hãy thử lại hoặc đổi chế độ tách nền.") from error

    def _analyze_image(self, raw, segment):
        try:
            rgb, size, mode = decode_upload(raw, self.settings.background)
        except ValueError as error:
            raise HTTPException(400, str(error)) from error
        # CPU inference must not block the API event loop; serialize access to the shared models.
        with self._inference_lock:
            return self._analyze_rgb(rgb, size, mode, segment)

    def _encode_all(self, rgb):
        """Every encoder on one image: per-encoder placement in its frozen style space."""
        result, zs = {}, {}
        for e in self.settings.encoders:
            z_raw, crop, evidence, info = self.encoders[e](rgb)
            sp = self.spaces[e]
            z, style, ranking, neighbors, dup = sp.place(z_raw)
            zs[e] = z
            result[e] = {"crop": png_b64(crop), "info": info | {"dim": int(len(z_raw))},
                         "preview": np.round(z[:64], 4).tolist(),
                         "evidence": png_b64(evidence(sp.C[style["id"]] - sp.C.mean(0))),
                         "style": style, "ranking": ranking, "neighbors": neighbors, "near_duplicate": dup}
        return result, zs

    def _analyze_rgb(self, rgb, size, mode, segment):
        background = self.settings.background
        isolated, seg = self.segmenter(rgb, segment, background)
        mask = seg.pop("mask", None)
        pixels = analyse(rgb, mask) if mask is not None else analyse(isolated)
        result, zs = self._encode_all(isolated)
        if seg["applied"]:
            before, _ = self._encode_all(rgb)                           # same image without isolation, for contrast
            seg["before"] = {e: {k: before[e]["style"][k] for k in ("id", "distance", "q99", "novel")}
                             for e in self.settings.encoders}
            seg["overlay"] = png_b64(np.asarray(seg["overlay"]))
            shown = isolated.copy()
            shown.thumbnail((480, 480))
            seg["isolated"] = png_b64(np.asarray(shown), "JPEG")
        zero_shot = describe(zs["fashionclip"], self.text_bank, self.clip.scale, pixels=pixels)
        token = uuid.uuid4().hex
        small = isolated.copy()
        small.thumbnail((160, 160))
        self.registry.remember(token, {"z": {e: z.tolist() for e, z in zs.items()}, "description": zero_shot["short"],
                                       "thumb": png_b64(np.asarray(small), "JPEG")})
        a, b = ({n["code"] for n in result[e]["neighbors"]} for e in self.settings.encoders)
        return {"token": token,
                "preprocess": {"original_size": list(size), "original_mode": mode,
                               "had_alpha": mode in ("RGBA", "LA", "P")},
                "segmentation": seg, "encoders": result, "zero_shot": zero_shot,
                "registry_match": self.registry.match(zs["fashionclip"]),
                "agreement": {"neighbor_overlap": len(a & b), "n": len(a)}}
