"""4.3.5 / 4.8.3 - feature blocks. B4 is split so that Google Trends can be isolated:

B1  historical sales level      sales, spap, pmr_past4 (+ lags, moving averages)
B2  market dynamics             growth / trend / volatility of the above, pmr_change4
B3  retailer operations         active_products, shop_exposure, price, discount, mean_age, share_new, inventory
B4a calendar                    cal__sin, cal__cos
B4b external signals            gt_* (+ transforms)
"""
import re

SALES_BASE = {"sales", "spap", "pmr_past4"}
OPS_BASE = {"active_products", "shop_exposure", "price", "discount", "mean_age", "share_new",
            "inventory", "availability"}
BLOCKS = ("B1", "B2", "B3", "B4a", "B4b", "B5")


def _parts(col):
    base, _, transform = col.partition("__")
    return base, re.sub(r"\d+$", "", transform) or "cur"


def block_of(col):
    base, transform = _parts(col)
    if base in SALES_BASE:
        return "B1" if transform in ("cur", "lag", "ma") else "B2"
    if base == "pmr_change4":
        return "B2"
    if base in OPS_BASE:
        return "B3"
    if base == "cal":
        return "B4a"
    if base.startswith("gt_"):
        return "B4b"
    if base.startswith("vis_"):
        return "B5"                     # visual features beyond hard clusters (extension, run 2)
    return None


def select_features(columns, blocks):
    return [c for c in columns if block_of(c) in set(blocks)]


MECHANICS_BASE = {"active_products", "shop_exposure", "mean_age", "share_new", "cal"}


def mechanics_features(columns):
    """Rollout / lifecycle / calendar only - no sales, price, discount or search information.
    If this set alone matches the full model, predictability is largely mechanical (see 01, risk 2)."""
    return [c for c in columns if _parts(c)[0] in MECHANICS_BASE]
