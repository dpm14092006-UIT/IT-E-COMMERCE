"""Supplement packaged scan forecasts with time-safe RF/LightGBM predictions.

Rebuild the same style-week features from the frozen web assignments. Models
only see labels matured at the selected origin; cache results per encoder/week.
"""
import json
from functools import lru_cache

import numpy as np
import pandas as pd

from vstyle.features.blocks import select_features
from vstyle.features.historical import add_history_features
from vstyle.models.tabular import make_model
from vstyle.panel.style_week import build_style_week
from vstyle.target.eligibility import forecast_eligible
from vstyle.target.labels import add_market_response, label_from_delta
from vstyle.target.threshold import calibrate
from vstyle.validation.splits import origin_sets


class ScanForecaster:
    def __init__(self, root, cfg, spaces):
        self.root, self.cfg, self.spaces = root, cfg, spaces

    @lru_cache(maxsize=2)
    def features(self, encoder):
        cfg = self.cfg
        interim = self.root / cfg["paths"]["interim"]
        cal = json.loads((interim / "calendar.json").read_text(encoding="utf-8"))
        pw = pd.read_parquet(interim / "product_week.parquet")
        space = self.spaces[encoder]
        mapping = pd.Series(space.style_of, index=space.ids)
        panel = build_style_week(pw, mapping, cal["t_max"], cal["origin_week"],
                                 cfg["panel"]["weighting"], cfg["panel"]["new_product_age_weeks"])
        t, f = cfg["target"], cfg["features"]
        panel = add_market_response(panel, t["past_window"], t["horizon"],
                                    t["min_active_weeks_per_window"], t.get("response", "spap"))
        data = add_history_features(panel, f["variables"], f["lags"], f["windows"],
                                    f["growth_horizons"], f["calendar_period"])
        sets = origin_sets(cfg["partition"], cal["origin_week"], t["horizon"], cal["t_max"])
        theta = calibrate(data["delta"], data["t_idx"], sets["calibration_end"], t["horizon"],
                          t["threshold"]["rule"], t["threshold"]["fallback_quantile"])
        data["y"] = label_from_delta(data["delta"], theta["theta"])
        data["eligible"] = forecast_eligible(data, cfg["eligibility"]["min_history_weeks"],
                                              cfg["eligibility"]["require_active_at_origin"])
        return data, select_features(data.columns, f["primary_blocks"])

    @lru_cache(maxsize=32)
    def at_origin(self, encoder, origin):
        data, cols = self.features(encoder)
        horizon = self.cfg["target"]["horizon"]
        train = data[data["eligible"] & data["y"].notna() & (data["t_idx"] + horizon <= origin)]
        test = data[data["eligible"] & (data["t_idx"] == origin)]
        if train.empty or test.empty:
            return {}
        result = {int(s): {} for s in test["style"]}
        for name in ("random_forest", "lightgbm"):
            file = self.root / self.cfg["paths"]["tuning"] / f"resnet50_{name}_best.json"
            best = json.loads(file.read_text(encoding="utf-8"))
            model = make_model(name, best["params"], self.cfg["project"]["seed"], best["class_weighting"])
            model.fit(train[cols], train["y"].astype(int).to_numpy())
            for style, p in zip(test["style"], model.predict_proba(test[cols])):
                result[int(style)][name] = {"p": np.round(p, 4).tolist(),
                                           "source": "origin_refit", "fit_origin": int(origin)}
        return result

    def for_scan(self, encoder, origin, placement):
        packaged = self.spaces[encoder].meta["forecasts"].get(str(origin), {})
        extra = self.at_origin(encoder, origin)
        def models(style):
            return {**packaged.get(str(style), {}), **extra.get(int(style), {})}
        if not placement["style"]["novel"]:
            return {name: {**value, "proxy": False}
                    for name, value in models(placement["style"]["id"]).items()
                    if name in ("random_forest", "lightgbm", "catboost", "tcn")}
        result = {}
        for name in ("random_forest", "lightgbm", "catboost", "tcn"):
            near = [r for r in placement["ranking"] if name in models(r["id"])][:3]
            if not near:
                continue
            weights = np.array([1 / max(r["distance"], 1e-9) for r in near])
            weights /= weights.sum()
            p = sum(w * np.asarray(models(r["id"])[name]["p"]) for w, r in zip(weights, near))
            result[name] = {"p": np.round(p, 4).tolist(), "proxy": True,
                            "source": "nearest_styles_proxy", "styles": [r["id"] for r in near]}
        return result
