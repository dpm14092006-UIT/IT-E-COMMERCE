"""4.4.1-4.4.2 - four-week exposure-weighted market response and three-class labels.

P_past(t)   = sum_{j=0..3} Sales_{t-j} / sum_{j=0..3} A_{t-j}      (36)
P_future(t) = sum_{h=1..4} Sales_{t+h} / sum_{h=1..4} A_{t+h}      (37) = P_past(t+4)
Delta(t)    = log(1+P_future) - log(1+P_past)                       (38)
"""
import numpy as np
import pandas as pd

from ..constants import DECLINING, EMERGING, STABLE, STYLE, TIME
from ..features.historical import check_full_grid

# 4.4.1 primary = "spap"; the others are the 4.9.4 operational alternatives (robustness only)
DENOMINATORS = {
    "spap": "active_products",        # sales per active product (primary)
    "raw_sales": None,                # mean weekly style sales (denominator = active weeks)
    "shop_exposure": "shop_exposure", # sales per active item-shop
    "age_adjusted": "expected_sales", # sales relative to the development lifecycle curve E[sales | h]
}


def window_response(panel, window, min_active_weeks, response="spap"):
    """Exposure-weighted response over weeks t-window+1..t; NaN unless every rule in 4.4.1 holds."""
    key = panel[STYLE]
    sales = panel["sales"].fillna(0.0)          # A = 0 weeks add 0 to numerator and denominator
    active = panel["active_products"].astype(float)
    col = DENOMINATORS[response]
    exposure = (active > 0).astype(float) if col is None else panel[col].fillna(0.0).astype(float)

    def roll(s):
        return s.groupby(key).transform(lambda v: v.rolling(window, min_periods=window).sum())

    num, den, n_active = roll(sales), roll(exposure), roll((active > 0).astype(float))
    valid = (den > 0) & (n_active >= min_active_weeks)
    return (num / den).where(valid)


def _log1p0(x):
    """log(1 + x) for a non-negative response; net sales with returns are floored at 0."""
    return np.log1p(np.clip(x, 0, None))


def add_market_response(panel, past_window=4, horizon=4, min_active_weeks=4, response="spap"):
    panel = panel.sort_values([STYLE, TIME]).reset_index(drop=True)
    check_full_grid(panel)
    key = panel[STYLE]
    past = window_response(panel, past_window, min_active_weeks, response)
    ending = window_response(panel, horizon, min(min_active_weeks, horizon), response)
    panel["pmr_past4"] = past                                                     # feature (B1)
    panel["pmr_change4"] = _log1p0(past) - _log1p0(past.groupby(key).shift(past_window))  # B2, persistence
    panel["pmr_future4"] = ending.groupby(key).shift(-horizon)                    # label input only
    panel["delta"] = _log1p0(panel["pmr_future4"]) - _log1p0(panel["pmr_past4"])     # (38)
    panel["label_ready_t"] = panel[TIME] + horizon
    return panel


def label_from_delta(delta, theta):
    """(40). Float codes (0=Declining, 1=Stable, 2=Emerging); NaN when unobservable."""
    d = np.asarray(delta, dtype=float)
    y = np.where(d > theta, EMERGING, np.where(d < -theta, DECLINING, STABLE)).astype(float)
    y[np.isnan(d)] = np.nan
    return pd.Series(y, index=getattr(delta, "index", None))
