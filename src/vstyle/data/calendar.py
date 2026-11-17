"""4.2.1 — release-relative weeks -> calendar weeks, at item–shop level.

Audit finding: release_date varies across shops of the same product (91.9% of products),
so (12) is applied per item–shop: t = T(r_{i,r}, h) = Monday(r_{i,r}) + 7h days.
"""
import numpy as np
import pandas as pd

from ..constants import KEYS_ITEM_SHOP, TIME, WEEK
from .load import REL_WEEKS


def monday(dates):
    dates = pd.to_datetime(dates)
    return (dates - pd.to_timedelta(dates.dt.dayofweek, unit="D")).dt.normalize()


def to_calendar_week(release_date, h):
    """(12) T(r, h): Monday of the release week plus h weeks."""
    return monday(release_date) + pd.to_timedelta(7 * np.asarray(h), unit="D")


def to_item_shop_week(sales, price_discount, n_weeks=12):
    """Long item–shop–week table with calendar weeks; only weeks inside each 12-week window exist."""
    rel = REL_WEEKS[:n_weeks]
    s = sales[[*KEYS_ITEM_SHOP, "release_date", *rel]].melt(
        id_vars=[*KEYS_ITEM_SHOP, "release_date"], value_vars=rel, var_name="h", value_name="sales")
    d = price_discount[[*KEYS_ITEM_SHOP, "price", *rel]].melt(
        id_vars=[*KEYS_ITEM_SHOP, "price"], value_vars=rel, var_name="h", value_name="discount")
    long = s.merge(d, on=[*KEYS_ITEM_SHOP, "h"], how="left", validate="one_to_one")
    long["h"] = long["h"].astype(int)
    long["partial_first_week"] = long["release_date"].dt.dayofweek.ne(0) & long["h"].eq(0)
    long[WEEK] = to_calendar_week(long["release_date"], long["h"])
    return long


def add_week_index(df, origin=None):
    """Integer calendar-week index; differences of t_idx are calendar weeks."""
    origin = df[WEEK].min() if origin is None else pd.Timestamp(origin)
    df[TIME] = ((df[WEEK] - origin).dt.days // 7).astype(int)
    return df, origin


def week_of(t_idx, origin):
    return pd.Timestamp(origin) + pd.to_timedelta(7 * np.asarray(t_idx), unit="D")


def to_t_idx(date, origin):
    return int((monday(pd.Series([pd.Timestamp(date)])).iloc[0] - pd.Timestamp(origin)).days // 7)
