"""4.2.2-4.2.4 - product-week -> visual style-week panel on a complete calendar grid.

Grid rule: every style gets every week 0..T_max. Weeks with no active product keep
active_products = 0 but Sales and SPAP stay NaN (not observed), never zero (4.2.2, 4.2.5).
"""
import numpy as np
import pandas as pd

from ..constants import PRODUCT, STYLE, TIME, WEEK


def build_style_week(pw, style_of, t_max, origin, weighting="none", new_age_weeks=1):
    """(14) Sales, (15) A, (16) SPAP, (19) weighted price/discount, (21) style-level Google Trends."""
    df = pw.copy()
    df[STYLE] = df[PRODUCT].map(style_of)
    df = df[df[STYLE].notna()]
    df["_w"] = df["n_shops"].astype(float) if weighting == "n_shops" else 1.0
    df["_new"] = (df["age_weeks"] <= new_age_weeks).astype(float)
    wavg = ["price", "discount", *[c for c in df.columns if c.startswith("gt_")]]
    for c in wavg:
        valid = df[c].notna()
        df[f"_wx_{c}"] = (df[c] * df["_w"]).where(valid)
        df[f"_w_{c}"] = df["_w"].where(valid)
    named = {"sales": ("sales", "sum"),
             "active_products": (PRODUCT, "nunique"),
             "shop_exposure": ("n_shops", "sum"),
             "expected_sales": ("expected_sales", "sum"),
             "mean_age": ("age_weeks", "mean"),
             "share_new": ("_new", "mean")}
    for c in wavg:
        named[f"_wx_{c}"] = (f"_wx_{c}", "sum")
        named[f"_w_{c}"] = (f"_w_{c}", "sum")
    agg = df.groupby([STYLE, TIME]).agg(**named)
    for c in wavg:
        agg[c] = agg[f"_wx_{c}"] / agg[f"_w_{c}"].where(agg[f"_w_{c}"] > 0)
    agg = agg.drop(columns=[k for k in named if k.startswith("_")])

    grid = pd.MultiIndex.from_product([sorted(df[STYLE].unique()), range(t_max + 1)], names=[STYLE, TIME])
    panel = agg.reindex(grid).reset_index()
    panel["active_products"] = panel["active_products"].fillna(0).astype(int)
    panel["shop_exposure"] = panel["shop_exposure"].fillna(0).astype(int)
    panel["expected_sales"] = panel["expected_sales"].fillna(0.0)
    active = panel["active_products"] > 0
    panel["spap"] = (panel["sales"] / panel["active_products"]).where(active)
    panel[WEEK] = pd.Timestamp(origin) + pd.to_timedelta(7 * panel[TIME], unit="D")
    return panel


def conservation_check(pw, panel, style_of):
    """G3: total sales must be identical before and after aggregation (no loss, no zero-filling)."""
    mapped = pw[pw[PRODUCT].map(style_of).notna()]
    return {"product_week_sales": float(mapped["sales"].sum()),
            "style_week_sales": float(np.nansum(panel["sales"])),
            "equal": bool(np.isclose(mapped["sales"].sum(), np.nansum(panel["sales"])))}
