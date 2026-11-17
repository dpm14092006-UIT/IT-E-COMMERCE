"""Garment isolation before encoding: clothes parsing with SegFormer (mattmdjaga/segformer_b2_clothes, ATR labels).

VISUELLE 2.0 images are single garments cut out on a plain background, so the encoders and the K-Means styles
only know that framing. A photo of someone wearing the garment adds face, skin, hair and backdrop, which pushes
the embedding away from every style. This step keeps the garment pixels, puts them on white and crops them
the way the catalogue does. Photos that already look like catalogue shots are left untouched in auto mode.
"""
import numpy as np
from PIL import Image, ImageFilter
from scipy import ndimage

LABELS_VI = {"Background": "nền", "Hat": "mũ", "Hair": "tóc", "Sunglasses": "kính", "Upper-clothes": "áo",
             "Skirt": "chân váy", "Pants": "quần", "Dress": "váy liền", "Belt": "thắt lưng", "Left-shoe": "giày",
             "Right-shoe": "giày", "Face": "mặt", "Left-leg": "chân", "Right-leg": "chân", "Left-arm": "tay",
             "Right-arm": "tay", "Bag": "túi", "Scarf": "khăn"}
GARMENTS = ("Upper-clothes", "Skirt", "Pants", "Dress", "Scarf", "Belt")
MODES = ("auto", "main", "all", "off")
MIN_SHARE = 0.02        # garment must cover at least 2% of the photo, otherwise keep the original
MARGIN = 0.06           # white margin around the garment, close to the catalogue framing
PERSON_MAX = 0.04       # auto mode keeps the photo as is below this share of face/hair/arm/leg pixels...
WHITE_MIN = 0.7         # ...when this share of its background pixels is near-white (catalogue-like shot)


PERSON = ("Face", "Hair", "Hat", "Sunglasses", "Left-arm", "Right-arm", "Left-leg", "Right-leg")


def white_share(rgb, where, tol=235):
    """Share of the selected pixels that are near-white."""
    a = np.asarray(rgb)[where]
    return float(((a >= tol).all(1)).mean()) if len(a) else 1.0


def _palette(n):
    rng = np.random.default_rng(7)
    return (rng.uniform(60, 230, (n, 3))).astype(np.uint8)


class GarmentSegmenter:
    def __init__(self, checkpoint="mattmdjaga/segformer_b2_clothes"):
        from transformers import AutoModelForSemanticSegmentation, SegformerImageProcessor
        self.processor = SegformerImageProcessor.from_pretrained(checkpoint)
        self.model = AutoModelForSemanticSegmentation.from_pretrained(checkpoint, use_safetensors=True).eval()
        self.id2label = {int(k): v for k, v in self.model.config.id2label.items()}
        self.garment_ids = [i for i, n in self.id2label.items() if n in GARMENTS]
        self.colors = _palette(len(self.id2label))

    def labels(self, rgb):
        import torch
        with torch.no_grad():
            logits = self.model(**self.processor(images=rgb, return_tensors="pt")).logits
            up = torch.nn.functional.interpolate(logits, size=rgb.size[::-1], mode="bilinear", align_corners=False)
        return up.argmax(1)[0].numpy()

    def overlay(self, rgb, lab, keep, size=320):
        """Every parsed class tinted; kept garment pixels drawn strongly, everything else faded."""
        a = np.asarray(rgb).astype(float)
        col = self.colors[lab].astype(float)
        out = np.where(keep[..., None], 0.45 * a + 0.55 * col, 0.25 * a + 0.75 * 255)
        im = Image.fromarray(out.astype(np.uint8))
        im.thumbnail((size, size))
        return im

    def __call__(self, rgb, mode="auto", background="white"):
        """Returns (image for the encoders, info dict). mode: auto | main | all | off."""
        info = {"mode": mode, "applied": False}
        if mode == "off":
            return rgb, info | {"reason": "Bạn đã tắt bước tách nền, ảnh được phân tích nguyên trạng"}
        lab = self.labels(rgb)
        total = lab.size
        areas = {i: int((lab == i).sum()) for i in self.id2label}
        merged = {}                                                     # left/right arm, leg, shoe -> one entry
        for i, a in areas.items():
            vi = LABELS_VI.get(self.id2label[i], self.id2label[i])
            e = merged.setdefault(vi, {"name": self.id2label[i], "vi": vi, "share": 0.0, "garment": i in self.garment_ids})
            e["share"] += a / total
        classes = [e | {"share": round(e["share"], 4)} for e in sorted(merged.values(), key=lambda e: -e["share"])
                   if e["share"] >= 0.005]
        person = sum(areas[i] for i, n in self.id2label.items() if n in PERSON) / total
        bg_white = white_share(rgb, lab == 0)
        info |= {"classes": classes, "person_share": round(person, 4), "background_white": round(bg_white, 3)}
        present = [i for i in self.garment_ids if areas[i] / total >= MIN_SHARE / 2]
        if not present or sum(areas[i] for i in present) / total < MIN_SHARE:
            return rgb, info | {"reason": "Không tìm thấy món đồ đủ lớn trong ảnh, nên ảnh được phân tích nguyên trạng"}
        info["mask"] = lab == max(present, key=areas.get)              # main garment, for colour analysis
        if mode == "auto" and person < PERSON_MAX and bg_white >= WHITE_MIN:
            return rgb, info | {"reason": "Ảnh đã là ảnh sản phẩm trên nền trắng, giống ảnh trong bộ sưu tập, nên không cần tách nền"}
        kept = [max(present, key=areas.get)] if mode in ("auto", "main") else present
        mask = np.isin(lab, kept)
        comp, n = ndimage.label(mask)                                   # drop specks away from the garment
        if n > 1:
            sizes = ndimage.sum(mask, comp, range(1, n + 1))
            mask = np.isin(comp, 1 + np.where(sizes >= 0.05 * sizes.max())[0])
        mask = ndimage.binary_fill_holes(ndimage.binary_closing(mask, iterations=2))
        alpha = np.asarray(Image.fromarray((mask * 255).astype(np.uint8)).filter(ImageFilter.GaussianBlur(1.2))) / 255.0
        bg = np.asarray(Image.new("RGB", rgb.size, background)).astype(float)
        comp_img = (np.asarray(rgb) * alpha[..., None] + bg * (1 - alpha[..., None])).astype(np.uint8)
        ys, xs = np.where(mask)
        y0, y1, x0, x1 = ys.min(), ys.max() + 1, xs.min(), xs.max() + 1
        side = int(max(y1 - y0, x1 - x0) * (1 + 2 * MARGIN))
        canvas = Image.new("RGB", (side, side), background)
        canvas.paste(Image.fromarray(comp_img[y0:y1, x0:x1]), ((side - (x1 - x0)) // 2, (side - (y1 - y0)) // 2))
        info |= {"applied": True, "mask": mask, "kept": [self.id2label[i] for i in kept],
                 "kept_vi": [LABELS_VI[self.id2label[i]] for i in kept],
                 "garment_share": round(float(mask.mean()), 4),
                 "reason": "Đã giữ lại " + ", ".join(LABELS_VI[self.id2label[i]] for i in kept)
                           + ", bỏ người và phông nền, rồi đặt món đồ lên nền trắng giống ảnh trong bộ sưu tập",
                 "overlay": self.overlay(rgb, lab, mask)}
        return canvas, info
