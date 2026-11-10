"""Style Scanner - upload one out-of-sample product image; compare ResNet-50 and FashionCLIP style spaces.

Inference only; everything trained lives in outputs/web/{encoder}/ (scripts/17_build_web_assets.py):
  1. preprocessing   same as training: RGBA -> white background. A photo with a person or a backdrop first goes
                     through garment isolation (web/segment.py, SegFormer clothes parsing) so that it is framed
                     like a catalogue image; then each encoder's own transforms
  2. encoders        ResNet-50 (frozen ImageNet, GAP of layer4, 2048-d) and FashionCLIP (ViT-B/32, 512-d),
                     both L2-normalised (2)(4)(5)
  3. style           nearest frozen K-Means centroid (11). An image is reported as a NEW style only when it is
                     farther from that centroid than 99% of the style's own catalogue products AND its K_NN closest
                     catalogue products are less similar than for 99% of catalogue products (leave-one-out). Far from
                     the centroid alone only means "atypical": the image still has look-alikes in the catalogue;
                     FashionCLIP's text encoder describes it zero-shot, and new styles can be registered so that
                     later uploads are matched against them (outputs/web/novel_styles.json)
  4. trend           TCN / CatBoost forecast of the style at the chosen origin (served from meta.json);
                     a new style has no sales history, so only a labelled proxy from its nearest styles is shown

Pages: / is the manager console (web/manager.py, /api/m/*), /scanner is this technical scanner.
Run:  python -m uvicorn web.app:app --port 8000
"""
import base64
import io
import json
import logging
import sys
import threading
import uuid
from collections import OrderedDict
from datetime import datetime
from functools import lru_cache
from pathlib import Path

import torch  # noqa: F401  - load torch DLLs before numpy/scipy on Windows (see scripts/_common.py)
import numpy as np
from fastapi import Body, FastAPI, File, Form, HTTPException, UploadFile
from fastapi.responses import FileResponse, JSONResponse, Response
from fastapi.staticfiles import StaticFiles
from starlette.concurrency import run_in_threadpool
from PIL import Image

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from vstyle.config import load_config  # noqa: E402
from vstyle.visual.encoders import load_rgb  # noqa: E402

from .colors import analyse  # noqa: E402
from .manager import build_router  # noqa: E402
from .segment import MODES, GarmentSegmenter  # noqa: E402
from .zero_shot import CATALOGUE_VI, describe, text_bank  # noqa: E402
from .storage import read_registry, write_registry  # noqa: E402
from .uploads import decode_upload  # noqa: E402

CFG = load_config()
WEB = ROOT / "outputs" / "web"
IMAGES = ROOT / CFG["paths"]["images"]
STATIC = Path(__file__).parent / "static"
REGISTRY = WEB / "novel_styles.json"
ENCODERS = ("resnet50", "fashionclip")
LABEL = {"resnet50": "ResNet-50", "fashionclip": "FashionCLIP"}
MAX_BYTES = 15 * 1024 * 1024
NEAR_DUPLICATE = 0.95      # cosine above which the upload is treated as a catalogue product, not out-of-sample
K_NN = 5                   # neighbours averaged for the look-alike (kNN) novelty test
KNN_QUANTILE = 0.01        # catalogue leave-one-out quantile below which an upload has no look-alike


def _png_b64(arr, fmt="PNG"):
    buf = io.BytesIO()
    Image.fromarray(arr).save(buf, format=fmt)
    return f"data:image/{fmt.lower()};base64," + base64.b64encode(buf.getvalue()).decode()


def _denormalize(x, mean, std):
    """Model input tensor -> the 224x224 crop the network actually sees."""
    mean, std = np.asarray(mean)[:, None, None], np.asarray(std)[:, None, None]
    return (np.clip(x * std + mean, 0, 1).transpose(1, 2, 0) * 255).astype(np.uint8)


def _overlay(contrib, crop):
    """Positive part of a per-cell contribution map, upsampled and blended over the crop."""
    contrib = np.maximum(contrib - np.median(contrib), 0)
    contrib = contrib / (contrib.max() + 1e-9)
    heat = np.asarray(Image.fromarray((contrib * 255).astype(np.uint8)).resize(crop.shape[:2][::-1],
                                                                                Image.BICUBIC)) / 255.0
    color = np.stack([np.ones_like(heat), 0.25 + 0.6 * heat, 0.1 * np.ones_like(heat)], -1)   # orange -> yellow
    alpha = (0.7 * heat ** 0.8)[..., None]
    return (crop * (1 - alpha) + color * 255 * alpha).astype(np.uint8)


class ResNetEncoder:
    """ResNet-50 exactly as in encode_resnet50, but keeping the layer4 map for the evidence heatmap."""
    MEAN, STD = (0.485, 0.456, 0.406), (0.229, 0.224, 0.225)

    def __init__(self, weights):
        from torchvision.models import ResNet50_Weights, resnet50
        w = getattr(ResNet50_Weights, weights)
        net = resnet50(weights=w).eval()
        self.trunk = torch.nn.Sequential(*list(net.children())[:-2])     # up to layer4: [B, 2048, 7, 7]
        self.preprocess = w.transforms()

    @torch.no_grad()
    def __call__(self, rgb):
        x = self.preprocess(rgb).unsqueeze(0)
        fmap = self.trunk(x)[0]                                          # [2048, h, w]
        z = fmap.mean((1, 2)).numpy()
        crop = _denormalize(x[0].numpy(), self.MEAN, self.STD)
        # GAP is linear, so z.v = mean_ij F[:, i, j].v: each cell's exact share of z.v
        evidence = lambda v: _overlay(np.tensordot(v, fmap.numpy(), axes=(0, 0)), crop)  # noqa: E731
        info = {"input_shape": list(x.shape[1:]), "fmap_shape": list(fmap.shape),
                "norm": round(float(np.linalg.norm(z)), 3), "active_share": round(float((z > 0).mean()), 3),
                "evidence_note": "Màu cam là vùng ảnh góp phần nhiều nhất vào việc xếp món đồ vào nhóm gần nhất, tính chính xác trên lưới 7×7 của ResNet-50."}
        return z, crop, evidence, info


class ClipEncoder:
    """FashionCLIP as in encode_fashionclip; also exposes the text tower for zero-shot description."""

    def __init__(self, checkpoint):
        from transformers import CLIPModel, CLIPProcessor
        self.model = CLIPModel.from_pretrained(checkpoint).eval()
        self.processor = CLIPProcessor.from_pretrained(checkpoint)
        ip = self.processor.image_processor
        self.mean, self.std = ip.image_mean, ip.image_std
        self.scale = float(self.model.logit_scale.detach().exp())

    @staticmethod
    def _features(out):
        if torch.is_tensor(out):
            return out
        return out.image_embeds if getattr(out, "image_embeds", None) is not None else out.pooler_output

    @torch.no_grad()
    def text(self, prompts):
        t = self.processor(text=list(prompts), return_tensors="pt", padding=True)
        f = self.model.get_text_features(**t)
        f = (f if torch.is_tensor(f) else f.pooler_output).numpy()
        return f / np.linalg.norm(f, axis=1, keepdims=True)

    @torch.no_grad()
    def __call__(self, rgb):
        px = self.processor(images=[rgb], return_tensors="pt")["pixel_values"]
        z = self._features(self.model.get_image_features(pixel_values=px))[0].numpy()
        vm = self.model.vision_model
        hidden = vm(pixel_values=px).last_hidden_state                   # [1, 1 + 49, 768]
        patches = self.model.visual_projection(vm.post_layernorm(hidden[0, 1:])).numpy()
        side = int(round(len(patches) ** 0.5))
        crop = _denormalize(px[0].numpy(), self.mean, self.std)
        evidence = lambda v: _overlay((patches @ v).reshape(side, side), crop)  # noqa: E731
        info = {"input_shape": list(px.shape[1:]), "fmap_shape": [len(patches), patches.shape[1]],
                "norm": round(float(np.linalg.norm(z)), 3), "active_share": round(float((z > 0).mean()), 3),
                "evidence_note": f"Màu cam là vùng ảnh liên quan nhất tới nhóm gần nhất, ước lượng trên {len(patches)} ô ảnh của FashionCLIP nên chỉ mang tính gợi ý."}
        return z, crop, evidence, info


class Space:
    """One encoder's frozen style space: centroids, catalogue embeddings, forecasts."""

    def __init__(self, name):
        d = WEB / name
        if not (d / "meta.json").exists():
            raise SystemExit(f"{d}/meta.json missing - run: python scripts/17_build_web_assets.py --grouping {name}")
        self.name = name
        self.meta = json.loads((d / "meta.json").read_text(encoding="utf-8"))
        a = np.load(d / "assets.npz")
        self.C = a["centroids"].astype(np.float32)
        self.Z = a["Z"].astype(np.float32)
        self.ids, self.style_of = a["ids"], a["style"]
        self.category, self.color, self.season = a["category"], a["color"], a["season"]
        self.image_path = a["image_path"]
        self.q99 = np.array([s["q99"] for s in self.meta["styles"]])
        # mean cosine to the K_NN nearest other catalogue products; its 1% quantile is the "no look-alike" bar
        S = self.Z @ self.Z.T
        np.fill_diagonal(S, -np.inf)
        loo = np.partition(S, -K_NN, axis=1)[:, -K_NN:].mean(1)
        self.knn_threshold = float(np.quantile(loo, KNN_QUANTILE))

    def place(self, z):
        z = z / max(np.linalg.norm(z), 1e-12)
        d = np.sqrt(np.maximum(((z[None] - self.C) ** 2).sum(1), 0))
        order = np.argsort(d)
        s = int(order[0])
        members = np.array(self.meta["styles"][s]["dist_sorted"])
        sims = self.Z @ z
        nn = np.argsort(-sims)[:8]
        far = bool(d[s] > self.q99[s])
        knn_sim = float(sims[nn[:K_NN]].mean())
        isolated = knn_sim < self.knn_threshold
        style = {"id": s, "distance": round(float(d[s]), 4), "q99": round(float(self.q99[s]), 4),
                 "typicality_pct": round(float((members >= d[s]).mean() * 100), 1),
                 "far_from_centroid": far, "knn_sim": round(knn_sim, 4), "knn_k": K_NN,
                 "knn_threshold": round(self.knn_threshold, 4), "isolated": bool(isolated),
                 "novel": far and bool(isolated),
                 "ood_global": bool(d[s] > self.meta["ood_threshold"]), "ood_threshold": self.meta["ood_threshold"],
                 "margin": round(float(d[order[1]] - d[s]), 4)}
        neighbors = [{"code": int(self.ids[i]), "style": int(self.style_of[i]), "similarity": round(float(sims[i]), 4),
                      "category": str(self.category[i]), "color": str(self.color[i]), "season": str(self.season[i]),
                      "category_vi": CATALOGUE_VI.get(str(self.category[i]), str(self.category[i])),
                      "color_vi": CATALOGUE_VI.get(str(self.color[i]), str(self.color[i]))}
                     for i in nn]
        ranking = [{"id": int(j), "distance": round(float(d[j]), 4)} for j in order]
        dup = int(self.ids[nn[0]]) if sims[nn[0]] >= NEAR_DUPLICATE else None
        return z, style, ranking, neighbors, dup


SPACES = {e: Space(e) for e in ENCODERS}
MODELS = {"resnet50": ResNetEncoder(CFG["visual"]["encoders"]["resnet50"]["weights"]),
          "fashionclip": ClipEncoder(CFG["visual"]["encoders"]["fashionclip"]["checkpoint"])}
CLIP = MODELS["fashionclip"]
SEGMENTER = GarmentSegmenter()
TEXT = text_bank(CLIP.text)


def _name_clip_styles():
    """FashionCLIP clusters have no hand-written names. Category and colour come from the catalogue profile
    (observed data); the pattern, which VISUELLE does not record, comes zero-shot from the centroid."""
    first = lambda text: text.split(";")[0].rsplit(" ", 1)[0]  # noqa: E731  "long sleeve 44%; ..." -> "long sleeve"
    share = lambda text: int(text.split(";")[0].rsplit(" ", 1)[1].rstrip("%"))  # noqa: E731  -> 44
    for s, c in zip(SPACES["fashionclip"].meta["styles"], SPACES["fashionclip"].C):
        pattern = describe(c / np.linalg.norm(c), TEXT, CLIP.scale, known_only=True)["groups"][1]["items"][0]
        cat, col = first(s["top_category"]), first(s["top_color"])
        # the name is only the plurality category/colour; flag clusters where neither is a majority
        s["name_shares"] = {"category": share(s["top_category"]), "color": share(s["top_color"])}
        s["mixed"] = min(s["name_shares"].values()) < 50
        s["name_note"] = (f"{CATALOGUE_VI.get(cat, cat)} {s['name_shares']['category']}%, "
                          f"{CATALOGUE_VI.get(col, col)} {s['name_shares']['color']}% sản phẩm")
        name = CATALOGUE_VI.get(cat, cat)
        parts = [name] + ([pattern["vi"]] if pattern["term"] != "solid plain color" and pattern["vi"] not in name else [])
        s["name"] = (" ".join(parts) + " · " + CATALOGUE_VI.get(col, col)).capitalize()


_name_clip_styles()
TOKENS = OrderedDict()                  # recent uploads (embeddings only) so a new style can be registered
LOCK = threading.RLock()
INFERENCE_LOCK = threading.Lock()
LOGGER = logging.getLogger(__name__)
app = FastAPI(title="Style Scanner")
app.mount("/static", StaticFiles(directory=STATIC), name="static")


def _registry():
    with LOCK:
        try:
            return read_registry(REGISTRY)
        except (OSError, ValueError) as error:
            LOGGER.exception("Could not read saved styles")
            raise HTTPException(503, "Không đọc được phong cách đã lưu. Hãy thử lại sau.") from error


def _persist_registry(registry):
    try:
        write_registry(REGISTRY, registry)
    except (OSError, ValueError) as error:
        LOGGER.exception("Could not persist saved styles")
        raise HTTPException(503, "Không ghi được phong cách đã lưu. Hãy thử lại sau.") from error


def _match_registry(z_clip):
    """Nearest registered new style in FashionCLIP space, within the typical member-to-centroid distance."""
    best = None
    for s in _registry()["styles"]:
        p = np.asarray(s["proto"]["fashionclip"], dtype=np.float32)
        d = float(np.linalg.norm(z_clip - p / np.linalg.norm(p)))
        if best is None or d < best["distance"]:
            best = {"id": s["id"], "name": s["name"], "n": s["n"], "distance": round(d, 4)}
    radius = SPACES["fashionclip"].meta["novel_radius"]
    return (best | {"radius": radius, "match": best["distance"] <= radius}) if best else None


@app.get("/")
def manager_console():
    return FileResponse(STATIC / "manager.html", headers={"Cache-Control": "no-store"})


@app.get("/scanner")
def index():
    return FileResponse(STATIC / "index.html", headers={"Cache-Control": "no-store"})


@app.get("/api/health")
async def health():
    return {"status": "ready", "encoders": list(MODELS), "segmentation": "SegFormer",
            "max_upload_bytes": MAX_BYTES, "groups_per_encoder": {e: len(sp.C) for e, sp in SPACES.items()}}


@app.get("/api/meta")
def meta():
    out = {}
    for e, sp in SPACES.items():
        styles = [{k: v for k, v in s.items() if k != "dist_sorted"} for s in sp.meta["styles"]]
        out[e] = {**sp.meta, "styles": styles, "label": LABEL[e]}
    return JSONResponse(out)


@lru_cache(maxsize=512)
def _thumb(code, size):
    sp = SPACES["resnet50"]
    where = np.where(sp.ids == code)[0]
    if not len(where):
        raise HTTPException(404)
    im = load_rgb(IMAGES / str(sp.image_path[where[0]]), CFG["visual"]["alpha_background"])
    im.thumbnail((size, size))
    buf = io.BytesIO()
    im.save(buf, format="JPEG", quality=85)
    return buf.getvalue()


@app.get("/catalog/{code}.jpg")
def catalog_image(code: int, size: int = 240):
    return Response(_thumb(code, min(max(size, 64), 480)), media_type="image/jpeg",
                    headers={"Cache-Control": "max-age=86400"})


def _encode_all(rgb):
    """Both encoders on one image: per-encoder placement in its frozen style space."""
    result, zs = {}, {}
    for e in ENCODERS:
        z_raw, crop, evidence, info = MODELS[e](rgb)
        sp = SPACES[e]
        z, style, ranking, neighbors, dup = sp.place(z_raw)
        zs[e] = z
        result[e] = {"crop": _png_b64(crop), "info": info | {"dim": int(len(z_raw))},
                     "preview": np.round(z[:64], 4).tolist(),
                     "evidence": _png_b64(evidence(sp.C[style["id"]] - sp.C.mean(0))),
                     "style": style, "ranking": ranking, "neighbors": neighbors, "near_duplicate": dup}
    return result, zs


@app.post("/api/analyze")
async def analyze(file: UploadFile = File(...), segment: str = Form("auto")):
    if segment not in MODES:
        raise HTTPException(400, f"segment phải là một trong {MODES}")
    try:
        raw = await file.read(MAX_BYTES + 1)
    finally:
        await file.close()
    if len(raw) > MAX_BYTES:
        raise HTTPException(413, "Ảnh quá lớn (tối đa 15 MB)")
    return await analyze_bytes(raw, segment)


async def analyze_bytes(raw, segment):
    if segment not in MODES:
        raise HTTPException(400, f"segment phải là một trong {MODES}")
    try:
        return await run_in_threadpool(_analyze_image, raw, segment)
    except HTTPException:
        raise
    except Exception as error:
        LOGGER.exception("Image inference failed")
        raise HTTPException(503, "Máy chủ không xử lý được ảnh. Hãy thử lại hoặc đổi chế độ tách nền.") from error


def _analyze_image(raw, segment):
    try:
        rgb, size, mode = decode_upload(raw, CFG["visual"]["alpha_background"])
    except ValueError as error:
        raise HTTPException(400, str(error)) from error
    # CPU inference must not block the API event loop; serialize access to the shared models.
    with INFERENCE_LOCK:
        return _analyze_rgb(rgb, size, mode, segment)


def _analyze_rgb(rgb, size, mode, segment):
    isolated, seg = SEGMENTER(rgb, segment, CFG["visual"]["alpha_background"])
    mask = seg.pop("mask", None)
    pixels = analyse(rgb, mask) if mask is not None else analyse(isolated)
    result, zs = _encode_all(isolated)
    if seg["applied"]:
        before, _ = _encode_all(rgb)                                    # same image without isolation, for contrast
        seg["before"] = {e: {k: before[e]["style"][k] for k in ("id", "distance", "q99", "novel")} for e in ENCODERS}
        seg["overlay"] = _png_b64(np.asarray(seg["overlay"]))
        shown = isolated.copy()
        shown.thumbnail((480, 480))
        seg["isolated"] = _png_b64(np.asarray(shown), "JPEG")
    zero_shot = describe(zs["fashionclip"], TEXT, CLIP.scale, pixels=pixels)
    token = uuid.uuid4().hex
    small = isolated.copy()
    small.thumbnail((160, 160))
    with LOCK:
        TOKENS[token] = {"z": {e: z.tolist() for e, z in zs.items()}, "description": zero_shot["short"],
                         "thumb": _png_b64(np.asarray(small), "JPEG")}
        while len(TOKENS) > 50:
            TOKENS.popitem(last=False)
    a, b = ({n["code"] for n in result[e]["neighbors"]} for e in ENCODERS)
    return {"token": token,
            "preprocess": {"original_size": list(size), "original_mode": mode, "had_alpha": mode in ("RGBA", "LA", "P")},
            "segmentation": seg, "encoders": result, "zero_shot": zero_shot,
            "registry_match": _match_registry(zs["fashionclip"]),
            "agreement": {"neighbor_overlap": len(a & b), "n": len(a)}}


@app.get("/api/novel")
def novel_list():
    return [{k: v for k, v in s.items() if k != "proto"} for s in _registry()["styles"]]


@app.post("/api/novel")
def novel_register(token: str = Body(...), name: str | None = Body(None)):
    """Add the upload to the matching new style (running-mean prototype) or open a new one."""
    with LOCK:
        item = TOKENS.get(token)
        if item is None:
            raise HTTPException(404, "Kết quả quét đã hết hạn, hãy quét lại ảnh")
        reg = _registry()
        if item.get("saved_id"):
            saved = next((s for s in reg["styles"] if s["id"] == item["saved_id"]), None)
            if saved is None:
                raise HTTPException(410, "Phong cách đã được xoá. Hãy quét lại ảnh trước khi lưu.")
            return {k: v for k, v in saved.items() if k != "proto"}
        z = np.asarray(item["z"]["fashionclip"], dtype=np.float32)
        match = _match_registry(z)
        if match and match["match"]:
            s = next(x for x in reg["styles"] if x["id"] == match["id"])
            for e in ENCODERS:
                s["proto"][e] = ((np.asarray(s["proto"][e]) * s["n"] + np.asarray(item["z"][e])) / (s["n"] + 1)).tolist()
            s["n"] += 1
            s["thumbs"] = (s["thumbs"] + [item["thumb"]])[-6:]
        else:
            n = max(reg.get("next_id", 1), 1 + max([int(x["id"][1:]) for x in reg["styles"]] or [0]))
            reg["next_id"] = n + 1
            s = {"id": f"N{n}", "name": ((name or "").strip() or item["description"])[:80], "n": 1,
                 "created": datetime.now().isoformat(timespec="seconds"),
                 "proto": dict(item["z"]), "thumbs": [item["thumb"]]}
            reg["styles"].append(s)
        _persist_registry(reg)
        item["saved_id"] = s["id"]
    return {k: v for k, v in s.items() if k != "proto"}


@app.delete("/api/novel/{style_id}")
def novel_delete(style_id: str):
    with LOCK:
        reg = _registry()
        if not any(s["id"] == style_id for s in reg["styles"]):
            raise HTTPException(404, "Phong cách này không còn trong danh sách đã lưu.")
        reg["styles"] = [s for s in reg["styles"] if s["id"] != style_id]
        _persist_registry(reg)
    return {"ok": True}


app.include_router(build_router(ROOT, CFG, SPACES, analyze_bytes))
