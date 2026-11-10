"""G3a | Nguoi 1 - item-shop-week calendar alignment (12) and product-week aggregation (13).

Outputs: outputs/interim/{item_shop_week,product_week,products}.parquet, calendar.json
"""
import numpy as np
import pandas as pd
from _common import log, parse

from vstyle.config import path
from vstyle.constants import TIME, WEEK
from vstyle.data import gtrends as gtr
from vstyle.data.calendar import add_week_index, to_item_shop_week
from vstyle.data.load import (REL_WEEKS, image_decodes, load_gtrends, load_price_discount, load_sales,
                              product_table)
from vstyle.data.product_week import build_product_week, lifecycle_curve
from vstyle.io import save_json, save_table
from vstyle.utils.repro import write_manifest

if __name__ == "__main__":
    args, cfg = parse(__doc__)
    sales, price = load_sales(cfg), load_price_discount(cfg)
    products = product_table(sales)
    images = path(cfg, "images")
    products["image_ok"] = products["image_path"].map(lambda p: image_decodes(images / p))  # D14
    isw = to_item_shop_week(sales, price, cfg["data"]["n_relative_weeks"])
    isw, origin = add_week_index(isw)
    if cfg["data"]["panel_end"]:
        isw = isw[isw[WEEK] <= pd.Timestamp(cfg["data"]["panel_end"])]
    t_max = int(isw[TIME].max())

    curve = lifecycle_curve(isw, pd.Timestamp(cfg["partition"]["calibration_end"]))   # age_adjusted target
    pw = build_product_week(isw, products, curve)
    if cfg["panel"]["gtrends"]["enabled"]:
        gt = gtr.clean(load_gtrends(cfg), cfg["panel"]["gtrends"]["missing_sentinel"])
        if cfg["panel"]["gtrends"]["renormalize"] == "expanding_max":
            gt = gtr.renormalize_expanding_max(gt)
        pw = gtr.attach_to_products(pw, products, gt, cfg["panel"]["gtrends"]["terms"])

    raw_total = float(sales[REL_WEEKS].to_numpy().sum())
    totals = {"sales_csv": raw_total, "item_shop_week": float(isw["sales"].sum()),
              "product_week": float(pw["sales"].sum())}
    assert np.isclose(totals["sales_csv"], totals["product_week"]) or cfg["data"]["panel_end"], totals
    calendar = {"origin_week": origin, "t_max": t_max, "last_week": isw[WEEK].max(),
                "n_item_shop_week": len(isw), "n_product_week": len(pw),
                "partial_first_weeks": int(isw["partial_first_week"].sum()),
                "products_without_usable_image": products.loc[~products["image_ok"], "external_code"].tolist(),
                "sales_conservation": totals,
                "lifecycle_curve_before_calibration_end": curve.round(4).to_dict(),
                "gt_missing_share": {c: float(pw[c].isna().mean()) for c in pw if c.startswith("gt_")}}

    out = path(cfg, "interim")
    files = [save_table(isw, out / "item_shop_week.parquet"), save_table(pw, out / "product_week.parquet"),
             save_table(products, out / "products.parquet"), save_json(calendar, out / "calendar.json")]
    write_manifest("01_build_product_week", cfg, outputs=files, extra=calendar)
    log(f"origin {origin:%Y-%m-%d}, t_max {t_max}, item-shop-weeks {len(isw):,}, product-weeks {len(pw):,}")
    log(f"sales conservation {totals}")
    log(f"GT missing share {calendar['gt_missing_share']}")
