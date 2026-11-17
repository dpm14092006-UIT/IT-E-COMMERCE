"""Pixel-level garment analysis: dominant colours and stripe direction.

CLIP is weak at both: it names colours loosely and confuses bold stripes with colour blocking. Measuring the
garment pixels directly (the mask comes from the clothes parser) is more reliable for these two attributes.
"""
import numpy as np
from sklearn.cluster import KMeans

# Named colours (sRGB) with Vietnamese name, English name and colour family; matched nearest in CIELAB.
NAMED = [
    ("trắng", "white", (245, 245, 243), "trắng"), ("kem", "cream", (238, 228, 205), "trắng"),
    ("be", "beige", (205, 185, 150), "be"), ("xám nhạt", "light grey", (190, 190, 190), "xám"),
    ("xám", "grey", (128, 128, 128), "xám"), ("xám đậm", "charcoal", (64, 64, 66), "xám"),
    ("đen", "black", (20, 20, 22), "đen"), ("nâu", "brown", (115, 75, 45), "nâu"),
    ("nâu nhạt", "camel", (170, 125, 80), "nâu"), ("đỏ", "red", (200, 30, 40), "đỏ"),
    ("đỏ đô", "burgundy", (120, 20, 40), "đỏ"), ("đỏ đô sẫm", "dark burgundy", (62, 14, 30), "đỏ"),
    ("hồng", "pink", (235, 150, 175), "hồng"), ("hồng đậm", "fuchsia", (200, 50, 130), "hồng"),
    ("hồng phấn", "dusty pink", (215, 175, 170), "hồng"), ("đỏ gạch", "terracotta", (165, 90, 70), "cam"),
    ("cam", "orange", (235, 120, 40), "cam"), ("vàng", "yellow", (240, 210, 50), "vàng"),
    ("vàng mù tạt", "mustard", (200, 160, 50), "vàng"), ("xanh lá", "green", (50, 140, 70), "xanh lá"),
    ("xanh rêu", "olive", (110, 115, 60), "xanh lá"), ("xanh ngọc", "teal", (30, 130, 135), "xanh ngọc"),
    ("xanh da trời", "sky blue", (130, 180, 230), "xanh dương"), ("xanh dương", "blue", (35, 80, 170), "xanh dương"),
    ("xanh navy", "navy", (25, 35, 75), "xanh dương"), ("xanh than", "slate blue", (60, 62, 88), "xanh dương"), ("tím", "purple", (120, 60, 150), "tím"),
    ("tím nhạt", "lavender", (185, 160, 215), "tím"),
]


def _lab(rgb):
    """sRGB (0-255) -> CIELAB (D65)."""
    c = np.asarray(rgb, dtype=float) / 255.0
    c = np.where(c > 0.04045, ((c + 0.055) / 1.055) ** 2.4, c / 12.92)
    xyz = c @ np.array([[0.4124, 0.3576, 0.1805], [0.2126, 0.7152, 0.0722], [0.0193, 0.1192, 0.9505]]).T
    xyz /= np.array([0.95047, 1.0, 1.08883])
    f = np.where(xyz > 0.008856, np.cbrt(xyz), 7.787 * xyz + 16 / 116)
    return np.stack([116 * f[..., 1] - 16, 500 * (f[..., 0] - f[..., 1]), 200 * (f[..., 1] - f[..., 2])], -1)


_NAMED_LAB = _lab([c for _, _, c, _ in NAMED])


def analyse(rgb, mask=None, k=6, size=128, seed=0):
    """Dominant colours of the garment (mask=True pixels, else all non-white pixels) and stripe direction."""
    im = rgb.resize((size, size))
    a = np.asarray(im).astype(float)
    if mask is not None:
        from PIL import Image
        m = np.asarray(Image.fromarray(mask.astype(np.uint8) * 255).resize((size, size))) > 127
    else:
        m = ~(a >= 235).all(-1)
    if m.sum() < 50:
        m = np.ones(a.shape[:2], bool)
    px = _lab(a[m])
    km = KMeans(n_clusters=min(k, len(px)), n_init=4, random_state=seed).fit(px)
    share = np.bincount(km.labels_, minlength=km.n_clusters) / len(px)
    names = np.argmin(((km.cluster_centers_[:, None] - _NAMED_LAB[None]) ** 2).sum(-1), 1)
    merged = {}
    for c, (s, n) in enumerate(zip(share, names)):
        rgb_c = a[m][km.labels_ == c].mean(0)
        e = merged.setdefault(int(n), {"share": 0.0, "rgb": np.zeros(3)})
        e["rgb"] = (e["rgb"] * e["share"] + rgb_c * s) / (e["share"] + s)
        e["share"] += s
    colors = [{"vi": NAMED[n][0], "en": NAMED[n][1], "family": NAMED[n][3], "share": round(float(e["share"]), 3),
               "hex": "#%02x%02x%02x" % tuple(int(v) for v in e["rgb"])}
              for n, e in sorted(merged.items(), key=lambda x: -x[1]["share"]) if e["share"] >= 0.05]
    families = {}
    for c in colors:
        families[c["family"]] = families.get(c["family"], 0) + c["share"]

    stripes, scores = _stripes(a, m)
    return {"colors": colors[:4], "stripes": stripes, "stripe_scores": scores,
            "families": [f for f, _ in sorted(families.items(), key=lambda x: -x[1])]}


def _stripes(a, m, min_share=0.3, ratio=3.0, min_bands=3):
    """Stripe direction from chroma (a*, b*), which shading barely touches. Vertical stripes: column means
    explain most of the chroma variance and the profile changes sign at least `min_bands` times
    (4+ alternating bands, so a two-tone split is not called stripes)."""
    ab = _lab(a)[..., 1:]
    total = ab[m].var(0).sum()
    if total < 20:                                         # nearly one colour: nothing to call stripes
        return None, {"col": 0.0, "row": 0.0}

    def explained(axis):
        cnt = m.sum(axis)
        keep = cnt >= 0.3 * cnt.max()
        means = np.where(keep[:, None], (ab * m[..., None]).sum(axis) / np.maximum(cnt, 1)[:, None], np.nan)
        prof = means[keep]
        w = cnt[keep] / cnt[keep].sum()
        mu = (prof * w[:, None]).sum(0)
        between = ((prof - mu) ** 2 * w[:, None]).sum()
        proj = (prof - mu) @ np.linalg.svd(prof - mu, full_matrices=False)[2][0] if len(prof) > 1 else np.zeros(1)
        smooth = np.convolve(proj, np.ones(3) / 3, mode="same")
        bands = int((np.diff(np.sign(smooth[np.abs(smooth) > 0.25 * np.abs(smooth).max()])) != 0).sum())
        return between / total, bands

    col, col_bands = explained(0)
    row, row_bands = explained(1)
    scores = {"col": round(float(col), 3), "row": round(float(row), 3), "col_bands": col_bands, "row_bands": row_bands}
    if col >= min_share and col >= ratio * row and col_bands >= min_bands:
        return "vertical", scores
    if row >= min_share and row >= ratio * col and row_bands >= min_bands:
        return "horizontal", scores
    return None, scores
