"""Nguoi 1 - market and sales-behaviour insights that feed the business story (role table, row 1).

  * attribute performance: mean 12-week sales per product by category / color / fabric / season
  * lifecycle: share of 12-week sales by relative week
  * seasonality: total sales by ISO week-of-year
  * Google Trends lead-lag: corr(GT growth at t-k, category SPAP growth at t), k = -8..8
    (k > 0 means search interest moves first; association only, not causality)
Outputs: outputs/reports/P1_*.csv, P1_market_insights.json
"""
import numpy as np
import pandas as pd
from _common import log, parse

from vstyle.config import path
from vstyle.data import gtrends as gtr
from vstyle.data.load import REL_WEEKS, load_gtrends, load_sales
from vstyle.io import load_table, save_json

if __name__ == "__main__":
    args, cfg = parse(__doc__)
    out = path(cfg, "reports")
    sales = load_sales(cfg)
    g = sales.groupby("external_code")
    per_product = g[["season", "category", "color", "fabric"]].first()
    per_product["sales_12w"] = g[REL_WEEKS].sum().sum(axis=1)
    attr = {}
    for a in ("category", "color", "fabric", "season"):
        t = per_product.groupby(a)["sales_12w"].agg(["count", "mean", "median", "sum"]).sort_values("mean", ascending=False)
        t.to_csv(out / f"P1_performance_by_{a}.csv")
        attr[a] = {"top": t[t["count"] >= 20].head(5)["mean"].round(1).to_dict(),
                   "bottom": t[t["count"] >= 20].tail(5)["mean"].round(1).to_dict()}
    life = sales[REL_WEEKS].sum() / sales[REL_WEEKS].sum().sum()

    pw = load_table(path(cfg, "interim") / "product_week.parquet")
    products = load_table(path(cfg, "interim") / "products.parquet")
    pw = pw.merge(products[["external_code", "category"]], on="external_code")
    season_week = pw.groupby(pw["week_start"].dt.isocalendar().week)["sales"].sum()
    season_week.to_csv(out / "P1_sales_by_week_of_year.csv")

    gt = gtr.renormalize_expanding_max(gtr.clean(load_gtrends(cfg)))
    cat = pw.groupby(["category", "week_start"]).agg(sales=("sales", "sum"), active=("external_code", "nunique"))
    cat["spap"] = cat["sales"] / cat["active"]
    lags, rows = range(-8, 9), []
    for c, g in cat.reset_index().groupby("category"):
        if c not in gt or len(g) < 30:
            continue
        s = g.set_index("week_start")["spap"].asfreq("7D")
        both = pd.DataFrame({"s": np.log1p(s.clip(lower=0)).diff(4), "g": np.log1p(gt[c]).diff(4)}).dropna(how="all")
        rows.append({"category": c, **{k: both["s"].corr(both["g"].shift(k)) for k in lags}})
    leadlag = pd.DataFrame(rows).set_index("category")
    leadlag.to_csv(out / "P1_gtrends_leadlag.csv")
    mean_corr = leadlag.mean()
    report = {"attribute_performance": attr, "lifecycle_share_by_week": life.round(4).to_dict(),
              "gt_leadlag_mean_corr": mean_corr.round(3).to_dict(),
              "gt_best_lag": int(mean_corr.idxmax()), "gt_categories_used": len(leadlag)}
    save_json(report, out / "P1_market_insights.json")
    log(f"lifecycle share week0..3: {life.iloc[:4].round(3).tolist()}")
    log(f"GT lead-lag mean corr by k: {mean_corr.round(3).to_dict()}")
    log(f"best k = {report['gt_best_lag']} over {len(leadlag)} categories")
