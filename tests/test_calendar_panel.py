import numpy as np
import pandas as pd

from vstyle.data.calendar import add_week_index, to_calendar_week, to_item_shop_week
from vstyle.data.load import REL_WEEKS, product_table
from vstyle.data.product_week import build_product_week
from vstyle.panel.style_week import build_style_week, conservation_check


def _raw():
    sales = pd.DataFrame({"external_code": [1, 1, 2], "retail": [10, 11, 10],
                          "season": "SS17", "category": ["a", "a", "b"], "color": "red", "fabric": "x",
                          "image_path": ["p1", "p1", "p2"],
                          "release_date": pd.to_datetime(["2017-01-02", "2017-01-16", "2017-01-02"]),
                          "restock": 5})
    for h in REL_WEEKS:
        sales[h] = [1.0, 2.0, 3.0]
    price = sales[["external_code", "retail"]].copy()
    price["price"] = [0.1, 0.1, 0.2]
    for h in REL_WEEKS:
        price[h] = 0.0
    return sales, price


def test_calendar_mapping_monday_and_offset():
    r = pd.Series(pd.to_datetime(["2017-01-02", "2017-01-04"]))   # Monday, Wednesday
    out = to_calendar_week(r, [3, 0])
    assert out.iloc[0] == pd.Timestamp("2017-01-23")
    assert out.iloc[1] == pd.Timestamp("2017-01-02")           # floored to its Monday


def test_item_shop_alignment_and_conservation():
    sales, price = _raw()
    isw, origin = add_week_index(to_item_shop_week(sales, price))
    pw = build_product_week(isw, product_table(sales))
    # product 1: shop 10 from week 0, shop 11 from week 2 -> weeks 2..11 have both shops
    p1 = pw[pw.external_code == 1].set_index("t_idx")
    assert p1.loc[0, "sales"] == 1.0 and p1.loc[2, "sales"] == 3.0 and p1.loc[13, "sales"] == 2.0
    assert p1.loc[2, "n_shops"] == 2 and p1.index.max() == 13
    assert pw["sales"].sum() == sales[REL_WEEKS].to_numpy().sum()


def test_style_panel_grid_inactive_weeks_are_nan_not_zero():
    sales, price = _raw()
    isw, origin = add_week_index(to_item_shop_week(sales, price))
    pw = build_product_week(isw, product_table(sales))
    style_of = pd.Series({1: "s1", 2: "s2"})
    panel = build_style_week(pw, style_of, t_max=20, origin=origin)
    assert len(panel) == 2 * 21
    s2 = panel[panel["style"] == "s2"].set_index("t_idx")
    assert s2.loc[15, "active_products"] == 0
    assert np.isnan(s2.loc[15, "sales"]) and np.isnan(s2.loc[15, "spap"])
    assert conservation_check(pw, panel, style_of)["equal"]
