"""4.8.4 - common product-week endpoint so that groupings are compared on identical labels (RQ3).

  * origins: product-weeks at product age 3..7 (4 past + 4 future weeks inside the observation window);
  * response: sales per active item-shop over 4 weeks (exposure-weighted), Delta and theta as in 4.4,
    theta calibrated on development product-weeks only;
  * configurations: own history | + attribute context | + FashionCLIP context | + ResNet-50 context;
  * context (68): leave-one-out, shop-weighted mean of the other products of the same group in the
    same week, i.e. (group sales - own sales) / (group shops - own shops); weeks <= t only.
"""
import numpy as np
import pandas as pd

from ..constants import TIME, WEEK

PRODUCT = "external_code"


def leave_one_out_context(df, group_col, value_col, weight_col=None, time_col=TIME):
    """(68) weighted mean of x_{j,t} over the other products j of the same group in the same week."""
    w = df[weight_col].astype(float) if weight_col else pd.Series(1.0, index=df.index)
    valid = df[value_col].notna()
    wx = (w * df[value_col]).where(valid, 0.0)
    ww = w.where(valid, 0.0)
    keys = [df[group_col], df[time_col]]
    num = wx.groupby(keys).transform("sum") - wx
    den = ww.groupby(keys).transform("sum") - ww
    return (num / den).where(den > 0)


def product_panel(pw):
    """Complete weekly grid per product between its first and last observed week."""
    rows = []
    for code, g in pw.groupby(PRODUCT):
        g = g.set_index(TIME).reindex(range(g[TIME].min(), g[TIME].max() + 1))
        g[PRODUCT] = code
        rows.append(g)
    pp = pd.concat(rows).rename_axis(TIME).reset_index()
    pp["n_shops"] = pp["n_shops"].fillna(0)
    pp["spap"] = (pp["sales"] / pp["n_shops"]).where(pp["n_shops"] > 0)
    return pp.sort_values([PRODUCT, TIME]).reset_index(drop=True)


def _roll(s, key, w):
    return s.groupby(key).transform(lambda v: v.rolling(w, min_periods=w).sum())


def _lp(x):
    return np.log1p(np.clip(x, 0, None))


def add_targets(pp, window=4, horizon=4):
    key = pp[PRODUCT]
    sales, shops = pp["sales"].fillna(0.0), pp["n_shops"].astype(float)
    full = _roll((shops > 0).astype(float), key, window) == window
    past = (_roll(sales, key, window) / _roll(shops, key, window)).where(full)
    pp["pmr_past4"] = past
    pp["pmr_future4"] = past.groupby(key).shift(-horizon)
    pp["delta"] = _lp(pp["pmr_future4"]) - _lp(pp["pmr_past4"])
    pp["pmr_change3"] = _lp(past) - _lp(past.groupby(key).shift(3))
    return pp


def add_own_features(pp, calendar_period=52):
    key = pp[PRODUCT]
    lp = _lp(pp["spap"])
    for h in (1, 2, 3):
        pp[f"own_spap__lag{h}"] = pp["spap"].groupby(key).shift(h)
    pp["own_spap"] = pp["spap"]
    pp["own_spap__growth1"] = lp - lp.groupby(key).shift(1)
    pp["own_spap__growth3"] = lp - lp.groupby(key).shift(3)
    pp["own_n_shops"] = pp["n_shops"]
    pp["own_n_shops__growth1"] = np.log1p(pp["n_shops"]) - np.log1p(pp["n_shops"].groupby(key).shift(1))
    pp["own_discount"] = pp["discount"]
    pp["own_discount__lag1"] = pp["discount"].groupby(key).shift(1)
    pp["own_price"] = pp["price"].groupby(key).transform("first")
    pp["own_age"] = pp["age_weeks"].groupby(key).ffill()
    pp[WEEK] = pp[WEEK].groupby(key).ffill()
    week = pp[WEEK].dt.isocalendar().week.astype(float)
    pp["cal__sin"] = np.sin(2 * np.pi * week / calendar_period)
    pp["cal__cos"] = np.cos(2 * np.pi * week / calendar_period)
    return pp


def add_context(pp, group_of, name):
    """Context features for grouping `name` (only products with a group get values)."""
    key = pp[PRODUCT]
    tmp = pp.assign(_g=pp[PRODUCT].map(group_of))
    sub = tmp[tmp["_g"].notna()]
    ctx = leave_one_out_context(sub, "_g", "spap", "n_shops").reindex(pp.index)
    lc = _lp(ctx)
    pp[f"ctx_{name}"] = ctx
    pp[f"ctx_{name}__lag1"] = ctx.groupby(key).shift(1)
    pp[f"ctx_{name}__growth3"] = lc - lc.groupby(key).shift(3)
    active = sub["n_shops"] > 0
    n_other = active.groupby([sub["_g"], sub[TIME]]).transform("sum") - active
    pp[f"ctx_{name}__n"] = n_other.reindex(pp.index)
    return pp


def own_columns(columns):
    return [c for c in columns if c.startswith(("own_", "cal__")) or c in ("pmr_past4", "pmr_change3")]
