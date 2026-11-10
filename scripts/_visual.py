"""Cached product-level visual features shared by 05 (style B5 block) and 10 (product endpoint)."""
import numpy as np
import pandas as pd

from vstyle.config import path
from vstyle.experiments import common_endpoint as ce
from vstyle.io import load_table, save_table
from vstyle.visual import vfeatures as vf


def product_grid(cfg):
    interim = path(cfg, "interim")
    products = load_table(interim / "products.parquet")
    usable = set(products.loc[products["image_ok"], "external_code"])
    pw = load_table(interim / "product_week.parquet")
    return ce.add_own_features(ce.add_targets(ce.product_panel(pw[pw["external_code"].isin(usable)]))), products


def visual_product_features(cfg, enc, pp, products, k_neighbors=20):
    """Adds knn_{enc}* columns to pp and returns (pp, novelty Series, dispersion Series, style_of)."""
    cache = path(cfg, "interim") / f"visual_{enc}.parquet"
    ids = load_table(path(cfg, "embeddings") / f"{enc}_ids.parquet")["external_code"].to_numpy()
    Z = np.load(path(cfg, "embeddings") / f"{enc}.npy")
    k = cfg["visual"]["selected"]["k"]
    a = load_table(path(cfg, "clusters") / enc / f"assignments_k{k}.parquet").set_index("external_code")
    style_of = a["style"]
    if cache.exists():
        c = load_table(cache)
        pp = pp.merge(c, on=["external_code", "t_idx"], how="left")
    else:
        pp = vf.add_knn_features(pp, dict(zip(ids, Z)), enc, k_neighbors)
        cols = [x for x in pp.columns if x.startswith(f"knn_{enc}")]
        save_table(pp[["external_code", "t_idx", *cols]], cache)
    release = products.set_index("external_code").loc[ids, "release_first"]
    nov = pd.Series(vf.novelty(Z, release.to_numpy()), index=ids)
    disp = pd.Series(vf.centroid_distance(Z, style_of.loc[ids].to_numpy(),
                                          np.load(path(cfg, "clusters") / enc / f"centroids_k{k}.npy")), index=ids)
    return pp, nov, disp, style_of
