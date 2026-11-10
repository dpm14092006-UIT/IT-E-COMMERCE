"""4.3.1-4.3.4 - backward-looking features on the complete style-week grid.

All operators are groupwise shifts or trailing windows, so a value at t only uses rows <= t.
Windows require every observation in the window (min_periods = w), as (29) states.
"""
import numpy as np
import pandas as pd

from ..constants import STYLE, TIME, WEEK


def check_full_grid(panel):
    span = panel.groupby(STYLE)[TIME].agg(["min", "max", "count"])
    if not ((span["max"] - span["min"] + 1) == span["count"]).all():
        raise ValueError("panel is not a complete weekly grid; row shifts would not be calendar lags")


def add_history_features(panel, variables, lags, windows, growth_horizons, calendar_period=52):
    panel = panel.sort_values([STYLE, TIME]).reset_index(drop=True)
    check_full_grid(panel)
    key = panel[STYLE]
    new = {}
    for var, transforms in variables.items():
        if var not in panel or not transforms:
            continue
        x = panel[var].astype(float)
        gx = x.groupby(key)
        if "lag" in transforms:
            for h in lags:
                new[f"{var}__lag{h}"] = gx.shift(h)                                    # (23)
        for w in windows:
            ma = gx.transform(lambda s, w=w: s.rolling(w, min_periods=w).mean())       # (25)
            if "ma" in transforms:
                new[f"{var}__ma{w}"] = ma
            if "trend" in transforms:
                new[f"{var}__trend{w}"] = ma - ma.groupby(key).shift(w)                # (28)
            if "vol" in transforms:
                new[f"{var}__vol{w}"] = gx.transform(
                    lambda s, w=w: s.rolling(w, min_periods=w).std(ddof=1))           # (29)
        if "growth" in transforms:
            lx = np.log1p(x.clip(lower=0))    # net sales can be < 0 (returns); (27) is for x >= 0
            for h in growth_horizons:
                new[f"{var}__growth{h}"] = lx - lx.groupby(key).shift(h)               # (27)
    week = panel[WEEK].dt.isocalendar().week.astype(int)
    new["cal__sin"] = np.sin(2 * np.pi * week / calendar_period)                       # (30)
    new["cal__cos"] = np.cos(2 * np.pi * week / calendar_period)                       # (31)
    return pd.concat([panel, pd.DataFrame(new, index=panel.index)], axis=1)
