"""4.2.1 — item–shop–week -> product–week. Weeks outside every shop window are absent, not zero."""
from ..constants import PRODUCT, TIME, WEEK


def lifecycle_curve(isw, cutoff):
    """E[sales | shop-relative week h], estimated only on item-shop-weeks observed before `cutoff`."""
    return isw.loc[isw[WEEK] < cutoff].groupby("h")["sales"].mean()


def build_product_week(isw, products, curve=None):
    """(13) S_{i,t} = sum over shops observed in week t. Also store coverage and age."""
    isw = isw.assign(expected_sales=isw["h"].map(curve) if curve is not None else float("nan"))
    pw = (isw.groupby([PRODUCT, WEEK, TIME], as_index=False)
             .agg(sales=("sales", "sum"),
                  n_shops=("retail", "nunique"),
                  discount=("discount", "mean"),
                  price=("price", "mean"),
                  expected_sales=("expected_sales", "sum"),
                  shop_age_min=("h", "min"),
                  shop_age_max=("h", "max")))
    first = products.set_index(PRODUCT)["release_first"]
    pw["age_weeks"] = ((pw[WEEK] - pw[PRODUCT].map(first)).dt.days // 7).astype(int)
    pw["active"] = 1    # observation_window proxy: the row exists only if >=1 shop window covers t
    return pw
