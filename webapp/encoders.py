"""Single-image encoders that also return the evidence behind a placement (heatmap, crop, diagnostics)."""
from abc import ABC, abstractmethod
from typing import NamedTuple

import numpy as np
import torch

from .rendering import denormalize, overlay


class Encoding(NamedTuple):
    z: np.ndarray            # raw embedding
    crop: np.ndarray         # the pixels the network actually sees
    evidence: object         # callable: direction vector -> heatmap over the crop
    info: dict


class ExplainableEncoder(ABC):
    """An encoder whose embedding can be attributed back to image regions."""

    name: str

    @abstractmethod
    def encode(self, rgb) -> Encoding:
        ...

    def __call__(self, rgb) -> Encoding:
        return self.encode(rgb)

    @staticmethod
    def _stats(z, input_shape, fmap_shape, note):
        return {"input_shape": list(input_shape), "fmap_shape": list(fmap_shape),
                "norm": round(float(np.linalg.norm(z)), 3), "active_share": round(float((z > 0).mean()), 3),
                "evidence_note": note}


class ResNetEncoder(ExplainableEncoder):
    """ResNet-50 exactly as in vstyle.visual.encoders, but keeping the layer4 map for the evidence heatmap."""
    name = "resnet50"
    MEAN, STD = (0.485, 0.456, 0.406), (0.229, 0.224, 0.225)

    def __init__(self, weights):
        from torchvision.models import ResNet50_Weights, resnet50
        w = getattr(ResNet50_Weights, weights)
        net = resnet50(weights=w).eval()
        self.trunk = torch.nn.Sequential(*list(net.children())[:-2])     # up to layer4: [B, 2048, 7, 7]
        self.preprocess = w.transforms()

    @torch.no_grad()
    def encode(self, rgb):
        x = self.preprocess(rgb).unsqueeze(0)
        fmap = self.trunk(x)[0]                                          # [2048, h, w]
        z = fmap.mean((1, 2)).numpy()
        crop = denormalize(x[0].numpy(), self.MEAN, self.STD)
        # GAP is linear, so z.v = mean_ij F[:, i, j].v: each cell's exact share of z.v
        fm = fmap.numpy()
        info = self._stats(z, x.shape[1:], fmap.shape,
                           "Màu cam là vùng ảnh góp phần nhiều nhất vào việc xếp món đồ vào nhóm gần nhất, "
                           "tính chính xác trên lưới 7×7 của ResNet-50.")
        return Encoding(z, crop, lambda v: overlay(np.tensordot(v, fm, axes=(0, 0)), crop), info)


class ClipEncoder(ExplainableEncoder):
    """FashionCLIP as in vstyle.visual.encoders; also exposes the text tower for zero-shot description."""
    name = "fashionclip"

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
    def encode(self, rgb):
        px = self.processor(images=[rgb], return_tensors="pt")["pixel_values"]
        z = self._features(self.model.get_image_features(pixel_values=px))[0].numpy()
        vm = self.model.vision_model
        hidden = vm(pixel_values=px).last_hidden_state                   # [1, 1 + 49, 768]
        patches = self.model.visual_projection(vm.post_layernorm(hidden[0, 1:])).numpy()
        side = int(round(len(patches) ** 0.5))
        crop = denormalize(px[0].numpy(), self.mean, self.std)
        info = self._stats(z, px.shape[1:], (len(patches), patches.shape[1]),
                           f"Màu cam là vùng ảnh liên quan nhất tới nhóm gần nhất, ước lượng trên {len(patches)} "
                           "ô ảnh của FashionCLIP nên chỉ mang tính gợi ý.")
        return Encoding(z, crop, lambda v: overlay((patches @ v).reshape(side, side), crop), info)
