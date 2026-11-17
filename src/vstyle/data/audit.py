"""G0 - temporal and structural audit run before any style-level forecasting (Section 4, intro).

Every check returns plain numbers so the report can be regenerated and diffed.
"""
import numpy as np
import pandas as pd

from ..config import path
from . import gtrends as gtr
from .load import ATTRIBUTES, REL_WEEKS, image_decodes, product_table


def audit_sales(sales):
    products = product_table(sales)
    rd = sales["release_date"]
    per_product = sales.groupby("external_code")["release_date"].agg(["min", "max", "nunique"])
    spread = (per_product["max"] - per_product["min"]).dt.days // 7
    weekly = sales[REL_WEEKS]
    last_obs = rd + pd.to_timedelta(7 * 11, unit="D")
    return {
        "rows_item_shop": len(sales),
        "n_products": sales["external_code"].nunique(),
        "n_shops": sales["retail"].nunique(),
        "duplicate_item_shop": int(sales.duplicated(["external_code", "retail"]).sum()),
        "products_per_season": sales.groupby("season")["external_code"].nunique().to_dict(),
        "release_weekday_share_monday": float((rd.dt.dayofweek == 0).mean()),
        "release_first": rd.min(), "release_last": rd.max(),
        "last_observed_week": last_obs.max(),
        "share_products_multi_release_date": float((per_product["nunique"] > 1).mean()),
        "release_spread_weeks": spread.describe().round(2).to_dict(),
        "zero_share_by_relative_week": (weekly == 0).mean().round(4).tolist(),
        "negative_sales_cells": int((weekly < 0).sum().sum()),
        "negative_cells_by_relative_week": {int(h): int(n) for h, n in (weekly < 0).sum().items() if n},
        "min_sales_cell": float(weekly.min().min()),
        "missing_sales_cells": int(weekly.isna().sum().sum()),
        "attribute_cardinality": {a: int(products[a].nunique()) for a in ATTRIBUTES},
        "restock_to_12w_sales_ratio": (sales["restock"] / weekly.sum(axis=1).replace(0, np.nan))
                                      .describe().round(3).to_dict(),
    }


def audit_images(sales, image_dir):
    products = product_table(sales)
    exists = products["image_path"].map(lambda p: (image_dir / p).is_file())
    decodes = products["image_path"].map(lambda p: image_decodes(image_dir / p))
    return {"products_with_image": int(exists.sum()),
            "products_missing_image": products.loc[~exists, "external_code"].tolist(),
            "products_undecodable_image": products.loc[exists & ~decodes, "external_code"].tolist(),
            "images_per_product_max": int(sales.groupby("external_code")["image_path"].nunique().max())}


def audit_price(price, sales):
    disc = price[REL_WEEKS]
    keys = ["external_code", "retail"]
    joined = sales[keys].merge(price[keys], on=keys, how="outer", indicator=True)
    return {"rows": len(price),
            "duplicate_item_shop": int(price.duplicated(keys).sum()),
            "unmatched_vs_sales": joined["_merge"].astype(str).value_counts().to_dict(),
            "price_range": [float(price["price"].min()), float(price["price"].max())],
            "price_is_static_per_item_shop": True,
            "discount_range": [float(disc.min().min()), float(disc.max().max())],
            "share_discount_positive": float((disc > 0).mean().mean())}


def audit_restocks(restocks, sales):
    keys = ["external_code", "retail"]
    rs = restocks[keys].drop_duplicates()
    in_sales = rs.merge(sales[keys], on=keys, how="left", indicator=True)["_merge"].eq("both")
    return {"rows": len(restocks),
            "item_shops": len(rs),
            "item_shops_also_in_sales": int(in_sales.sum()),
            "year_counts": restocks["year"].value_counts().sort_index().to_dict(),
            "week_range": [int(restocks["week"].min()), int(restocks["week"].max())],
            "negative_qty": int((restocks["qty"] < 0).sum())}


def audit_gtrends(gt_raw, sales, sentinel=-1):
    gt = gtr.clean(gt_raw, sentinel)
    terms = set(gt.columns)
    missing_terms = {a: sorted(set(sales[a].unique()) - terms) for a in ATTRIBUTES}
    return {"n_weeks": len(gt), "n_terms": gt.shape[1],
            "first": gt.index.min(), "last": gt.index.max(),
            "weekday_share_monday": float((gt.index.dayofweek == 0).mean()),
            "is_weekly_without_gaps": gtr.is_weekly(gt.index),
            "sentinel_cells": int((gt_raw.drop(columns="date") == sentinel).sum().sum()),
            "terms_hitting_100": int((gt == 100).any().sum()),
            "attribute_terms_missing_in_gt": missing_terms}


def active_products_by_week(sales):
    """Number of products observed per calendar week (shows the SS/AW structure of the panel)."""
    from .calendar import to_calendar_week
    rows = [pd.DataFrame({"external_code": sales["external_code"],
                          "week_start": to_calendar_week(sales["release_date"], h)}) for h in range(12)]
    weeks = pd.concat(rows).drop_duplicates()
    return weeks.groupby("week_start")["external_code"].nunique()


def run(cfg):
    from .load import load_gtrends, load_price_discount, load_restocks, load_sales
    sales = load_sales(cfg)
    report = {
        "sales": audit_sales(sales),
        "images": audit_images(sales, path(cfg, "images")),
        "price_discount": audit_price(load_price_discount(cfg), sales),
        "restocks": audit_restocks(load_restocks(cfg), sales),
        "gtrends": audit_gtrends(load_gtrends(cfg), sales, cfg["panel"]["gtrends"]["missing_sentinel"]),
    }
    weekly = active_products_by_week(sales)
    full = pd.date_range(weekly.index.min(), weekly.index.max(), freq="7D")
    weekly = weekly.reindex(full, fill_value=0)
    report["panel"] = {"first_week": weekly.index.min(), "last_week": weekly.index.max(),
                       "n_weeks": int(len(weekly)),
                       "weeks_with_no_active_product": int((weekly == 0).sum()),
                       "active_products_per_week": weekly.describe().round(1).to_dict()}
    return report, weekly
