import numpy as np
import pandas as pd
import pytest

from vstyle.data.gtrends import clean, renormalize_expanding_max
from vstyle.decision.rules import alert_quality, decision_signals
from vstyle.evaluation.bootstrap import paired_macro_f1_difference
from vstyle.evaluation.metrics import macro_f1, summary
from vstyle.experiments.common_endpoint import leave_one_out_context
from vstyle.models.tabular import AlignedClassifier, make_model
from vstyle.utils.leakage import LeakageError, assert_label_maturity, assert_no_target_columns
from vstyle.validation.walk_forward import run_walk_forward


class Spy:
    """Records the latest origin seen in each training set."""
    seen = []

    def fit(self, X, y):
        Spy.seen.append(int(X["t_idx"].max()))
        return self

    def predict_proba(self, X):
        return np.tile([0.2, 0.5, 0.3], (len(X), 1))


def _data(n=30):
    d = pd.DataFrame({"style": np.repeat(["A", "B"], n), "t_idx": np.tile(range(n), 2)})
    d["y"] = np.tile([0.0, 1.0, 2.0], 2 * n // 3)
    d["eligible"] = d["t_idx"] >= 5
    return d


def test_walk_forward_respects_label_maturity():
    Spy.seen = []
    origins = list(range(12, 25))
    preds = run_walk_forward(_data(), ["t_idx"], Spy, origins, horizon=4)
    assert all(s <= t - 4 for s, t in zip(Spy.seen, origins))
    assert set(preds["t_idx"]) == set(origins) and len(preds) == 2 * len(origins)
    assert np.allclose(preds[["p_declining", "p_stable", "p_emerging"]].sum(1), 1)


def test_guards():
    with pytest.raises(LeakageError):
        assert_label_maturity([1, 2, 9], t=12, horizon=4)
    with pytest.raises(LeakageError):
        assert_no_target_columns(["spap", "delta"])


def test_aligned_classifier_with_missing_class():
    X = pd.DataFrame({"x": np.arange(20.0)})
    y = np.array([0] * 10 + [2] * 10)                     # Stable never seen
    for name in ("random_forest", "lightgbm", "catboost"):
        m = make_model(name, {"n_estimators": 10} if name != "catboost" else {"iterations": 10}, 0)
        p = m.fit(X, y).predict_proba(X)
        assert p.shape == (20, 3) and np.allclose(p[:, 1], 0) and np.allclose(p.sum(1), 1)
    const = AlignedClassifier(None).fit(X, np.ones(20, int)).predict_proba(X)
    assert np.allclose(const, [0, 1, 0])


def test_random_forest_accepts_missing_values():
    X = pd.DataFrame({"x": [np.nan, 1.0, 2.0, np.nan, 4.0, 5.0] * 3})
    p = make_model("random_forest", {"n_estimators": 5}, 0).fit(X, np.array([0, 1, 2] * 6)).predict_proba(X)
    assert p.shape == (18, 3)


def test_metrics_and_bootstrap():
    y, yhat = np.array([0, 0, 1, 1, 2, 2]), np.array([0, 1, 1, 1, 2, 0])
    s = summary(y, yhat)
    assert np.isclose(s["macro_f1"], macro_f1(y, yhat)) and s["absent_classes"] == []
    assert summary([1, 1], [1, 1])["absent_classes"] == ["Declining", "Emerging"]
    df = pd.DataFrame({"t_idx": np.repeat(range(10), 3), "y_true": np.tile([0, 1, 2], 10)})
    df["a"] = df["y_true"]
    df["b"] = df["y_true"]
    d = paired_macro_f1_difference(df, "a", "b", n_boot=50, block_length=4)
    assert d["estimate"] == 0 and d["ci_low"] == 0 and d["ci_high"] == 0


def test_gtrends_renormalisation_removes_future_scale():
    raw = pd.DataFrame({"date": pd.date_range("2018-01-01", periods=6, freq="7D"),
                        "red": [10, 20, -1, 40, 30, 100]})
    gt = clean(raw)
    assert np.isnan(gt.loc["2018-01-15", "red"])
    a = renormalize_expanding_max(gt)
    b = renormalize_expanding_max(gt * 0.37)              # different (future-dependent) normaliser
    assert np.allclose(a, b, equal_nan=True)
    head = renormalize_expanding_max(gt.iloc[:4])
    assert np.allclose(head, a.iloc[:4], equal_nan=True)  # values never depend on later weeks


def test_leave_one_out_context():
    df = pd.DataFrame({"g": ["x", "x", "x", "y"], "t_idx": 0, "v": [1.0, 2.0, 6.0, 5.0]})
    c = leave_one_out_context(df, "g", "v")
    assert c.tolist()[:3] == [4.0, 3.5, 1.5] and np.isnan(c.iloc[3])


def test_decision_signals():
    pred = pd.DataFrame({"p_declining": [0.1, 0.7, 0.3], "p_stable": [0.2, 0.2, 0.35],
                         "p_emerging": [0.7, 0.1, 0.35], "y_true": [2.0, 1.0, 2.0], "y_pred": [2, 0, 1]})
    s = decision_signals(pred, min_prob=0.5, min_margin=0.1)
    assert s["signal"].tolist() == ["emerging_alert", "declining_alert", "hold"]
    q = alert_quality(s)
    assert q["emerging_alert_precision"] == 1.0 and q["declining_alert_precision"] == 0.0


def test_cluster_assign_and_profile():
    from vstyle.visual.clustering import assign, profile
    X = np.array([[1.0, 0.0], [0.9, 0.1], [0.0, 1.0]])
    C = np.array([[1.0, 0.0], [0.0, 1.0]])
    labels = assign(X, C)
    assert labels.tolist() == [0, 0, 1]
    products = pd.DataFrame({"category": ["a", "a", "b"], "color": "red", "fabric": "x", "season": "SS17"})
    p = profile(labels, products)
    assert p.loc[0, "n_products"] == 2 and p.loc[0, "top_category"] == "a 100%"


def test_product_context_excludes_self():
    from vstyle.experiments.common_endpoint import add_context, product_panel
    pw = pd.DataFrame({"external_code": [1, 1, 2, 2, 3], "t_idx": [0, 1, 0, 1, 1],
                       "week_start": pd.to_datetime(["2017-01-02", "2017-01-09"] * 2 + ["2017-01-09"]),
                       "sales": [4.0, 6.0, 2.0, 8.0, 3.0], "n_shops": [2, 2, 1, 2, 1],
                       "discount": 0.0, "price": 0.1, "age_weeks": [0, 1, 0, 1, 0]})
    pp = add_context(product_panel(pw), pd.Series({1: "g", 2: "g", 3: "h"}), "x")
    r = pp.set_index(["external_code", "t_idx"])
    assert r.loc[(1, 0), "ctx_x"] == 2.0          # product 2 only: 2 sales / 1 shop
    assert r.loc[(2, 1), "ctx_x"] == 3.0          # product 1 only: 6 / 2
    assert np.isnan(r.loc[(3, 1), "ctx_x"])       # alone in its group


def test_visual_novelty_and_knn():
    from vstyle.visual.vfeatures import knn_context, novelty
    Z = np.array([[1.0, 0.0], [0.8, 0.6], [0.0, 1.0]])
    nov = novelty(Z, pd.to_datetime(["2017-01-02", "2017-02-06", "2017-03-06"]))
    assert np.isnan(nov[0]) and np.isclose(nov[1], 0.2) and np.isclose(nov[2], 0.4)
    pp = pd.DataFrame({"external_code": [1, 2, 3], "t_idx": 0, "n_shops": 1, "spap": [10.0, 20.0, 40.0]})
    ctx = knn_context(pp, {1: Z[0], 2: Z[1], 3: Z[2]}, k=1)
    assert ctx.tolist() == [20.0, 10.0, 20.0]             # nearest other product (cos .8 > .6), never itself
