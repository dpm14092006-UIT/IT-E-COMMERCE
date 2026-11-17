"""Raw VISUELLE 2.0 readers. Nothing here changes values; see data/audit.py for checks."""
import pandas as pd
from PIL import Image

from ..config import path

REL_WEEKS = [str(h) for h in range(12)]
ATTRIBUTES = ["category", "color", "fabric"]


def load_sales(cfg):
    return pd.read_csv(path(cfg, "raw") / "sales.csv", index_col=0, parse_dates=["release_date"])


def load_price_discount(cfg):
    return pd.read_csv(path(cfg, "raw") / "price_discount_series.csv")


def load_restocks(cfg):
    return pd.read_csv(path(cfg, "raw") / "restocks.csv")


def load_gtrends(cfg):
    return pd.read_csv(path(cfg, "raw") / "vis2_gtrends_data.csv", parse_dates=["date"])


def product_table(sales):
    """One row per product. release_first = earliest shop release (r_i := min_r r_{i,r})."""
    g = sales.groupby("external_code")
    products = g[["season", *ATTRIBUTES, "image_path"]].first()
    products["release_first"] = g["release_date"].min()
    products["release_last"] = g["release_date"].max()
    products["n_shops"] = g["retail"].nunique()
    return products.reset_index()


def image_decodes(path):
    """True if the file decodes completely (one VISUELLE 2.0 image is corrupt in the original zip)."""
    try:
        with Image.open(path) as im:
            im.load()
        return True
    except (OSError, SyntaxError):
        return False
