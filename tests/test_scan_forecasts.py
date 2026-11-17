import types

import numpy as np
import pandas as pd

from webapp.scan_forecasts import ScanForecaster


def test_refit_uses_only_mature_labels(monkeypatch, tmp_path):
    import webapp.scan_forecasts as module
    for name in ("random_forest", "lightgbm"):
        (tmp_path / f"resnet50_{name}_best.json").write_text(
            '{"params": {}, "class_weighting": "none"}')
    fitted = []
    class Model:
        def fit(self, x, y):
            fitted.append(x["marker"].tolist())
        def predict_proba(self, x):
            return np.tile([.2, .3, .5], (len(x), 1))
    monkeypatch.setattr(module, "make_model", lambda *args: Model())
    f = ScanForecaster(tmp_path, {"target": {"horizon": 4}, "paths": {"tuning": "."},
                                 "project": {"seed": 2026}}, {})
    data = pd.DataFrame({"style": [0]*5, "t_idx": [5, 6, 7, 9, 10],
                         "eligible": [True, True, True, False, True],
                         "y": [0, 1, 2, 1, 2], "marker": [5, 6, 7, 9, 10]})
    f.features = lambda encoder: (data, ["marker"])
    result = f.at_origin("resnet50", 10)
    assert fitted == [[5, 6], [5, 6]]
    assert set(result[0]) == {"random_forest", "lightgbm"}
    assert result[0]["lightgbm"]["fit_origin"] == 10


def test_novel_style_uses_proxy_for_each_model(tmp_path):
    names = ("random_forest", "lightgbm", "catboost", "tcn")
    forecasts = {str(s): {m: {"p": [.2, .3, .5], "source": "walk_forward"}
                          for m in names} for s in range(3)}
    f = ScanForecaster(tmp_path, {}, {"resnet50": types.SimpleNamespace(meta={"forecasts": {"10": forecasts}})})
    f.at_origin = lambda *args: {}
    result = f.for_scan("resnet50", 10, {"style": {"novel": True},
                        "ranking": [{"id": s, "distance": s+1} for s in range(3)]})
    assert set(result) == set(names)
    assert all(v["proxy"] and np.isclose(sum(v["p"]), 1) for v in result.values())
