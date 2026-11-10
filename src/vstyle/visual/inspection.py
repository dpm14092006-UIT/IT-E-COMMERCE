"""4.1.4 - representative images: near-centroid AND random members of each cluster."""
import numpy as np
from PIL import Image

from .encoders import load_rgb


def representatives(X, labels, centroids, n_near=12, n_random=12, seed=0):
    rng = np.random.default_rng(seed)
    out = {}
    for k in range(len(centroids)):
        idx = np.flatnonzero(labels == k)
        d = ((X[idx] - centroids[k]) ** 2).sum(1)
        near = idx[np.argsort(d)[:n_near]]
        rest = np.setdiff1d(idx, near)
        rand = rng.choice(rest, size=min(n_random, len(rest)), replace=False) if len(rest) else rest
        out[k] = {"near": near.tolist(), "random": rand.tolist()}
    return out


def contact_sheet(image_paths, out_path, thumb=128, cols=6, background="white"):
    n = len(image_paths)
    rows = max(1, int(np.ceil(n / cols)))
    sheet = Image.new("RGB", (cols * thumb, rows * thumb), background)
    for i, p in enumerate(image_paths):
        im = load_rgb(p, background)
        im.thumbnail((thumb, thumb))
        x, y = (i % cols) * thumb, (i // cols) * thumb
        sheet.paste(im, (x + (thumb - im.width) // 2, y + (thumb - im.height) // 2))
    sheet.save(out_path)
    return out_path


def overview_sheet(rows, out_path, thumb=96, background="white"):
    """One row per style: near-centroid images first, then random members (4.1.4)."""
    cols = max(len(r) for r in rows)
    sheet = Image.new("RGB", (cols * thumb, len(rows) * thumb), background)
    for y, paths in enumerate(rows):
        for x, p in enumerate(paths):
            im = load_rgb(p, background)
            im.thumbnail((thumb, thumb))
            sheet.paste(im, (x * thumb + (thumb - im.width) // 2, y * thumb + (thumb - im.height) // 2))
    sheet.save(out_path)
    return out_path
