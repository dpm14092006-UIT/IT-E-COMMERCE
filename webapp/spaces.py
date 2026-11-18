"""One encoder's frozen style space, and the naming of FashionCLIP clusters."""
import json

import numpy as np

from .settings import WebSettings
from .zero_shot import CATALOGUE_VI, describe


class StyleSpace:
    """Centroids, catalogue embeddings and forecasts of one encoder; places an embedding in the space."""

    def __init__(self, name: str, settings: WebSettings):
        d = settings.web_dir / name
        if not (d / "meta.json").exists():
            raise SystemExit(f"{d}/meta.json missing - run: python scripts/17_build_web_assets.py --grouping {name}")
        self.name, self.settings = name, settings
        self.meta = json.loads((d / "meta.json").read_text(encoding="utf-8"))
        a = np.load(d / "assets.npz")
        self.C = a["centroids"].astype(np.float32)
        self.Z = a["Z"].astype(np.float32)
        self.ids, self.style_of = a["ids"], a["style"]
        self.category, self.color, self.season = a["category"], a["color"], a["season"]
        self.image_path = a["image_path"]
        self.q99 = np.array([s["q99"] for s in self.meta["styles"]])
        # mean cosine to the k nearest other catalogue products; its low quantile is the "no look-alike" bar
        S = self.Z @ self.Z.T
        np.fill_diagonal(S, -np.inf)
        k = settings.k_nn
        loo = np.partition(S, -k, axis=1)[:, -k:].mean(1)
        self.knn_threshold = float(np.quantile(loo, settings.knn_quantile))

    def place(self, z):
        k_nn = self.settings.k_nn
        z = z / max(np.linalg.norm(z), 1e-12)
        d = np.sqrt(np.maximum(((z[None] - self.C) ** 2).sum(1), 0))
        order = np.argsort(d)
        s = int(order[0])
        members = np.array(self.meta["styles"][s]["dist_sorted"])
        sims = self.Z @ z
        nn = np.argsort(-sims)[:8]
        far = bool(d[s] > self.q99[s])
        knn_sim = float(sims[nn[:k_nn]].mean())
        isolated = knn_sim < self.knn_threshold
        style = {"id": s, "distance": round(float(d[s]), 4), "q99": round(float(self.q99[s]), 4),
                 "typicality_pct": round(float((members >= d[s]).mean() * 100), 1),
                 "far_from_centroid": far, "knn_sim": round(knn_sim, 4), "knn_k": k_nn,
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
        dup = int(self.ids[nn[0]]) if sims[nn[0]] >= self.settings.near_duplicate else None
        return z, style, ranking, neighbors, dup


class ClipStyleNamer:
    """FashionCLIP clusters have no hand-written names. Category and colour come from the catalogue profile
    (observed data); the pattern, which VISUELLE does not record, comes zero-shot from the centroid."""

    def __init__(self, space: StyleSpace, clip, text_bank):
        self.space, self.clip, self.text_bank = space, clip, text_bank

    @staticmethod
    def _first(text):
        return text.split(";")[0].rsplit(" ", 1)[0]          # "long sleeve 44%; ..." -> "long sleeve"

    @staticmethod
    def _share(text):
        return int(text.split(";")[0].rsplit(" ", 1)[1].rstrip("%"))   # -> 44

    def apply(self):
        for s, c in zip(self.space.meta["styles"], self.space.C):
            pattern = describe(c / np.linalg.norm(c), self.text_bank, self.clip.scale,
                               known_only=True)["groups"][1]["items"][0]
            cat, col = self._first(s["top_category"]), self._first(s["top_color"])
            # the name is only the plurality category/colour; flag clusters where neither is a majority
            s["name_shares"] = {"category": self._share(s["top_category"]), "color": self._share(s["top_color"])}
            s["mixed"] = min(s["name_shares"].values()) < 50
            s["name_note"] = (f"{CATALOGUE_VI.get(cat, cat)} {s['name_shares']['category']}%, "
                              f"{CATALOGUE_VI.get(col, col)} {s['name_shares']['color']}% sản phẩm")
            name = CATALOGUE_VI.get(cat, cat)
            parts = [name] + ([pattern["vi"]] if pattern["term"] != "solid plain color"
                              and pattern["vi"] not in name else [])
            s["name"] = (" ".join(parts) + " · " + CATALOGUE_VI.get(col, col)).capitalize()
