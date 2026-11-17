"""4.2.4 — Google Trends as an external signal, without future-scale leakage.

Each downloaded series is g_t = 100 * x_t / M where M is the max over the whole retrieval
window (2015-10 .. 2019-12), i.e. it depends on the future. Expanding-max renormalisation
g_t / max_{s<=t} g_s = x_t / max_{s<=t} x_s removes M exactly, using only the past.
"""
import numpy as np
import pandas as pd

from ..constants import PRODUCT, WEEK


def clean(gt, sentinel=-1):
    gt = gt.rename(columns={"date": WEEK}).set_index(WEEK).sort_index()
    return gt.mask(gt == sentinel)


def renormalize_expanding_max(gt):
    running_max = gt.cummax()
    return 100 * gt / running_max.where(running_max > 0)


def attach_to_products(pw, products, gt, terms=("category", "color", "fabric")):
    """G_{i,t} of (20): the product's own terms looked up at calendar week t."""
    long = gt.reset_index().melt(id_vars=WEEK, var_name="term", value_name="gt")
    out = pw.merge(products[[PRODUCT, *terms]], on=PRODUCT, how="left")
    for term in terms:
        out = out.merge(long.rename(columns={"term": term, "gt": f"gt_{term}"}),
                        on=[WEEK, term], how="left")
    return out.drop(columns=list(terms))


def coverage(gt):
    return pd.DataFrame({"first": gt.apply(lambda s: s.first_valid_index()),
                         "last": gt.apply(lambda s: s.last_valid_index()),
                         "n_missing": gt.isna().sum(),
                         "hits_100": (gt == 100).any()})


def is_weekly(index):
    return bool((np.diff(index.values).astype("timedelta64[D]").astype(int) == 7).all())
