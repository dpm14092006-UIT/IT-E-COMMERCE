"""Image helpers shared by the encoders and the analysis service (evidence overlays, data URIs)."""
import base64
import io

import numpy as np
from PIL import Image


def png_b64(arr, fmt="PNG"):
    buf = io.BytesIO()
    Image.fromarray(arr).save(buf, format=fmt)
    return f"data:image/{fmt.lower()};base64," + base64.b64encode(buf.getvalue()).decode()


def denormalize(x, mean, std):
    """Model input tensor -> the 224x224 crop the network actually sees."""
    mean, std = np.asarray(mean)[:, None, None], np.asarray(std)[:, None, None]
    return (np.clip(x * std + mean, 0, 1).transpose(1, 2, 0) * 255).astype(np.uint8)


def overlay(contrib, crop):
    """Positive part of a per-cell contribution map, upsampled and blended over the crop."""
    contrib = np.maximum(contrib - np.median(contrib), 0)
    contrib = contrib / (contrib.max() + 1e-9)
    heat = np.asarray(Image.fromarray((contrib * 255).astype(np.uint8)).resize(crop.shape[:2][::-1],
                                                                                Image.BICUBIC)) / 255.0
    color = np.stack([np.ones_like(heat), 0.25 + 0.6 * heat, 0.1 * np.ones_like(heat)], -1)   # orange -> yellow
    alpha = (0.7 * heat ** 0.8)[..., None]
    return (crop * (1 - alpha) + color * 255 * alpha).astype(np.uint8)
