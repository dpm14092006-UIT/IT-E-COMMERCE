"""4.1.1-4.1.2 - frozen pretrained image encoders. Owners: FashionCLIP = Nguoi 2, ResNet-50 = Nguoi 4.

Both encoders run in eval mode under no_grad; weights are never updated (primary analysis).
"""
from abc import ABC, abstractmethod

import numpy as np
from PIL import Image

EXPECTED_DIM = {"fashionclip": 512, "fashionclipft": 512, "resnet50": 2048, "resnet50ft": 2048}


def _open(path, retries=3):
    import io
    import time
    for attempt in range(retries):                  # transient Windows I/O errors under heavy disk load
        try:
            return Image.open(io.BytesIO(open(path, "rb").read()))
        except OSError:
            if attempt == retries - 1:
                raise
            time.sleep(1)


def load_rgb(path, background="white"):
    """VISUELLE images are RGBA: composite onto a flat background instead of dropping alpha."""
    im = _open(path)
    if im.mode in ("RGBA", "LA", "P"):
        im = im.convert("RGBA")
        bg = Image.new("RGBA", im.size, background)
        im = Image.alpha_composite(bg, im)
    return im.convert("RGB")


def l2_normalize(z, eps=1e-12):
    """(2), (5)."""
    return z / np.maximum(np.linalg.norm(z, axis=1, keepdims=True), eps)


def _device(name):
    import torch
    if name == "auto":
        return "cuda" if torch.cuda.is_available() else "cpu"
    return name


def _batches(items, size):
    for i in range(0, len(items), size):
        yield items[i:i + size]


def fashionclip_features(model, **inputs):
    feats = model.get_image_features(**inputs)
    if hasattr(feats, "image_embeds") and feats.image_embeds is not None:
        return feats.image_embeds
    return feats if hasattr(feats, "shape") else feats.pooler_output


def encode_fashionclip(image_paths, checkpoint, batch_size=64, device="auto", background="white",
                       state_dict=None):
    """(1) z^F_i = f_FashionCLIP(I_i). Resize/crop/normalise come from the checkpoint processor."""
    import torch
    from transformers import CLIPModel, CLIPProcessor
    dev = _device(device)
    model = CLIPModel.from_pretrained(checkpoint).to(dev).eval()
    if state_dict is not None:
        model.vision_model.load_state_dict(torch.load(state_dict, map_location="cpu", weights_only=True))
    processor = CLIPProcessor.from_pretrained(checkpoint)
    out = []
    with torch.no_grad():
        for batch in _batches(list(image_paths), batch_size):
            inputs = processor(images=[load_rgb(p, background) for p in batch], return_tensors="pt").to(dev)
            feats = fashionclip_features(model, **inputs)
            out.append(feats.cpu().numpy())
    return np.concatenate(out).astype(np.float32)


def encode_resnet50(image_paths, weights="IMAGENET1K_V2", batch_size=64, device="auto", background="white",
                    state_dict=None):
    """(4) z^C_i = f_ResNet50(I_i): global-average-pooling output, final fc layer removed."""
    import torch
    from torchvision.models import ResNet50_Weights, resnet50
    dev = _device(device)
    w = getattr(ResNet50_Weights, weights)
    model = resnet50(weights=w)
    model.fc = torch.nn.Identity()
    if state_dict is not None:                      # fine-tuned backbone (visual/finetune.py)
        model.load_state_dict(torch.load(state_dict, map_location="cpu"))
    model = model.to(dev).eval()
    preprocess = w.transforms()
    out = []
    with torch.no_grad():
        for batch in _batches(list(image_paths), batch_size):
            x = torch.stack([preprocess(load_rgb(p, background)) for p in batch]).to(dev)
            out.append(model(x).cpu().numpy())
    return np.concatenate(out).astype(np.float32)


def check_embeddings(z, expected_dim):
    """G1 checks on raw embeddings before normalisation."""
    norms = np.linalg.norm(z, axis=1)
    return {"n": int(z.shape[0]), "dim": int(z.shape[1]), "dim_ok": z.shape[1] == expected_dim,
            "nan": int(np.isnan(z).sum()), "zero_norm": int((norms == 0).sum()),
            "norm_min": float(norms.min()), "norm_max": float(norms.max())}


class ImageEncoder(ABC):
    """A frozen image encoder producing one raw embedding per image path."""

    name: str
    dim: int

    def __init__(self, batch_size=64, device="auto", background="white", state_dict=None):
        self.batch_size, self.device, self.background, self.state_dict = batch_size, device, background, state_dict

    @abstractmethod
    def encode(self, image_paths) -> np.ndarray:
        """float32 array of shape (n_images, dim), not yet L2-normalised."""

    def check(self, z):
        return check_embeddings(z, self.dim)


class FashionClipEncoder(ImageEncoder):
    name, dim = "fashionclip", 512

    def __init__(self, checkpoint, **kwargs):
        super().__init__(**kwargs)
        self.checkpoint = checkpoint

    def encode(self, image_paths):
        return encode_fashionclip(image_paths, self.checkpoint, self.batch_size, self.device,
                                  self.background, self.state_dict)


class ResNet50Encoder(ImageEncoder):
    name, dim = "resnet50", 2048

    def __init__(self, weights="IMAGENET1K_V2", **kwargs):
        super().__init__(**kwargs)
        self.weights = weights

    def encode(self, image_paths):
        return encode_resnet50(image_paths, self.weights, self.batch_size, self.device,
                               self.background, self.state_dict)
