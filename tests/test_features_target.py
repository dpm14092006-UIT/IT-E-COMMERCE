import numpy as np
import pandas as pd

from vstyle.features.blocks import block_of, select_features
from vstyle.features.historical import add_history_features
from vstyle.target.eligibility import forecast_eligible
from vstyle.target.labels import add_market_response, label_from_delta
from vstyle.target.threshold import calibrate
from vstyle.utils.leakage import future_perturbation_check

VARS = {"sales": ["lag", "ma", "growth", "trend", "vol"], "spap": ["lag", "ma", "growth"],
        "active_products": ["lag"], "discount": ["lag", "ma"], "gt_color": ["lag", "growth", "trend"]}


def _features(p):
    p = add_market_response(p, 4, 4, 4)
    return add_history_features(p, VARS, [1, 2, 4, 8], [4, 8], [1, 4])


def test_lag_is_calendar_lag(panel):
    f = _features(panel)
    a = f[f["style"] == "A"].set_index("t_idx")
    assert a.loc[20, "sales__lag4"] == a.loc[16, "sales"]
    b = f[f["style"] == "B"].set_index("t_idx")
    assert np.isnan(b.loc[14, "sales__lag2"])            # week 12 is inactive -> NaN, not 0
    assert np.isnan(b.loc[14, "sales__ma4"])             # window touches an inactive week


def test_no_feature_uses_the_future(panel):
    f = _features(panel)
    cols = [c for c in f.columns if block_of(c)]
    for t in (15, 25):
        changed = future_perturbation_check(_features, panel, t, ["sales", "spap", "active_products",
                                                                 "discount", "gt_color"], cols)
        assert changed == [], changed


def test_labels_do_use_the_future(panel):
    changed = future_perturbation_check(_features, panel, 20, ["sales", "spap"], ["pmr_future4", "delta"])
    assert set(changed) == {"pmr_future4", "delta"}


def test_market_response_hand_example():
    p = pd.DataFrame({"style": "A", "t_idx": range(10), "active_products": [2] * 10,
                      "sales": [2, 2, 2, 2, 4, 4, 4, 4, 1, 1.0]})
    r = add_market_response(p, 4, 4, 4).set_index("t_idx")
    assert r.loc[3, "pmr_past4"] == 1.0                   # 8 / 8
    assert r.loc[3, "pmr_future4"] == 2.0                 # 16 / 8 over weeks 4..7
    assert np.isclose(r.loc[3, "delta"], np.log(3) - np.log(2))
    assert np.isnan(r.loc[6, "pmr_future4"])              # weeks 7..10 not all observed
    assert np.isnan(r.loc[2, "pmr_past4"])                # needs 4 past weeks


def test_label_rule_and_threshold():
    y = label_from_delta(pd.Series([0.5, 0.1, -0.1, -0.5, np.nan]), 0.2)
    assert y.iloc[:4].tolist() == [2.0, 1.0, 1.0, 0.0] and np.isnan(y.iloc[4])
    delta = pd.Series([0.1, -0.3, 0.2, 5.0])
    t = pd.Series([0, 1, 2, 10])
    th = calibrate(delta, t, calibration_end_t=6, horizon=4)
    assert th["n_calibration"] == 3 and np.isclose(th["theta"], 0.2)   # row t=10 is not matured
    zero = calibrate(pd.Series([0.0, 0.0, 0.0, 0.4]), pd.Series([0, 0, 0, 0]), 4, 4)
    assert zero["rule"] == "fallback_positive_quantile" and zero["theta"] == 0.4


def test_eligibility_requires_history(panel):
    p = add_market_response(panel, 4, 4, 4)
    e = forecast_eligible(p, 16)
    assert not e[p["t_idx"] < 15].any()
    assert e[(p["style"] == "A") & (p["t_idx"] == 20)].all()


def test_blocks():
    assert block_of("spap__lag4") == "B1" and block_of("spap__growth4") == "B2"
    assert block_of("discount__ma4") == "B3" and block_of("cal__sin") == "B4a"
    assert block_of("gt_color__trend8") == "B4b" and block_of("delta") is None
    assert select_features(["sales", "y", "pmr_change4", "label_ready_t"], ["B1", "B2"]) == ["sales", "pmr_change4"]
