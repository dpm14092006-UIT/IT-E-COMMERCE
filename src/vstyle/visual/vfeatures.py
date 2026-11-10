"""Visual features beyond hard clusters (extension, run 2). All use only information available at t.

novelty(i)      = 1 - max cosine(z_i, z_j) over products j first released strictly before i
dispersion(i)   = 1 - cosine(z_i, centroid of its style)      (centroids fitted on development only)
knn context     = similarity-weighted mean SPAP at week t of the k visually closest OTHER products
                  that are active at t (any style); its growth uses only weeks <= t.
"""
import numpy as np
import pandas as pd

PRODUCT, TIME = "external_code", "t_idx"


def novelty(Z, release):
    """Z rows aligned with `release` (Timestamp per product). NaN for the earliest release cohort."""
    rel = pd.to_datetime(pd.Series(release)).to_numpy()
    S = Z @ Z.T
    prior = rel[None, :] < rel[:, None]                     # j released strictly before i
    S = np.where(prior, S, -np.inf)
    best = S.max(1)
    return np.where(np.isfinite(best), 1 - best, np.nan)


def centroid_distance(Z, labels, centroids):
    C = centroids / np.linalg.norm(centroids, axis=1, keepdims=True)
    return 1 - (Z * C[labels]).sum(1)


def knn_context(pp, Z_of, k=20, value="spap"):
    """pp: product-week grid with external_code, t_idx, n_shops, value. Z_of: dict code -> unit vector."""
    out = pd.Series(np.nan, index=pp.index)
    active = pp[(pp["n_shops"] > 0) & pp[value].notna() & pp[PRODUCT].isin(Z_of.keys())]
    for t, g in active.groupby(TIME):
        if len(g) < 2:
            continue
        Z = np.stack([Z_of[c] for c in g[PRODUCT]])
        x = g[value].to_numpy(float)
        S = Z @ Z.T
        np.fill_diagonal(S, -np.inf)
        kk = min(k, len(g) - 1)
        idx = np.argpartition(-S, kk - 1, axis=1)[:, :kk]
        w = np.clip(np.take_along_axis(S, idx, 1), 0, None)
        num, den = (w * x[idx]).sum(1), w.sum(1)
        out.loc[g.index] = np.where(den > 0, num / np.where(den > 0, den, 1), np.nan)
    return out


def add_knn_features(pp, Z_of, name, k=20):
    key = pp[PRODUCT]
    ctx = knn_context(pp, Z_of, k)
    lc = np.log1p(ctx.clip(lower=0))
    pp[f"knn_{name}"] = ctx
    pp[f"knn_{name}__lag1"] = ctx.groupby(key).shift(1)
    pp[f"knn_{name}__growth3"] = lc - lc.groupby(key).shift(3)
    return pp


def style_visual_panel(pp, style_of, nov, disp, name, new_age=3):
    """Aggregate product visual features to style-week over products active at t (B5 block)."""
    a = pp[pp["n_shops"] > 0].copy()
    a["style"] = a[PRODUCT].map(style_of)
    a = a[a["style"].notna()]
    a["nov"], a["disp"] = a[PRODUCT].map(nov), a[PRODUCT].map(disp)
    a["nov_new"] = a["nov"].where(a["own_age"] <= new_age)
    g = a.groupby(["style", TIME])
    return pd.DataFrame({"vis_novelty": g["nov"].mean(), "vis_novelty_new": g["nov_new"].mean(),
                         "vis_dispersion": g["disp"].mean(), "vis_knn": g[f"knn_{name}"].mean(),
                         "vis_knn_growth3": g[f"knn_{name}__growth3"].mean()}).reset_index()
