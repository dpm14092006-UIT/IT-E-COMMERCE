"""G4 + G5 | Nguoi 3 - historical features (4.3), market response, theta and labels (4.4), eligibility.

Outputs: outputs/features/features_{grouping}.parquet, target_{grouping}.json
"""
import numpy as np
from _common import log, parse

from vstyle.config import path
from vstyle.features.historical import add_history_features
from vstyle.io import load_json, load_table, save_json, save_table
from vstyle.target.eligibility import coverage, forecast_eligible
from vstyle.target.labels import add_market_response, label_from_delta
from vstyle.target.threshold import calibrate, class_distribution, quantile_threshold
from vstyle.utils.repro import write_manifest
from vstyle.validation.splits import origin_sets

if __name__ == "__main__":
    def extra(ap):
        ap.add_argument("--grouping", required=True, help="panel id, e.g. resnet50 or resnet50__k10")
        ap.add_argument("--variant", default=None, help="robustness variant name appended to the dataset id")
        ap.add_argument("--visual-features", default=None, help="encoder whose B5 visual block is added (run 2)")
    args, cfg = parse(__doc__, extra)
    did = f"{args.grouping}__{args.variant}" if args.variant else args.grouping
    t, f = cfg["target"], cfg["features"]
    cal = load_json(path(cfg, "interim") / "calendar.json")
    panel = load_table(path(cfg, "panels") / f"style_week_{args.grouping}.parquet")
    panel = add_market_response(panel, t["past_window"], t["horizon"], t["min_active_weeks_per_window"],
                                t.get("response", "spap"))
    variables = dict(f["variables"])
    if args.visual_features:
        from _visual import product_grid, visual_product_features
        from vstyle.visual.vfeatures import style_visual_panel
        enc = args.visual_features
        pp, products = product_grid(cfg)
        pp, nov, disp, style_of = visual_product_features(cfg, enc, pp, products)
        vis = style_visual_panel(pp, style_of, nov, disp, enc)
        panel = panel.merge(vis.astype({"style": panel["style"].dtype}), on=["style", "t_idx"], how="left")
        variables.update({"vis_novelty": ["lag"], "vis_novelty_new": [], "vis_dispersion": ["lag"],
                          "vis_knn": ["lag", "growth"], "vis_knn_growth3": ["lag"]})
    data = add_history_features(panel, variables, f["lags"], f["windows"], f["growth_horizons"],
                                f["calendar_period"])
    sets = origin_sets(cfg["partition"], cal["origin_week"], t["horizon"], cal["t_max"])
    theta = calibrate(data["delta"], data["t_idx"], sets["calibration_end"], t["horizon"],
                      t["threshold"]["rule"], t["threshold"]["fallback_quantile"])
    q = t["threshold"].get("quantile")
    if q is not None:                                            # 4.9.2 robustness only
        theta = {**theta, "theta": quantile_threshold(data["delta"], data["t_idx"], sets["calibration_end"],
                                                       t["horizon"], q), "rule": f"quantile_{q}"}
    data["y"] = label_from_delta(data["delta"], theta["theta"])
    data["eligible"] = forecast_eligible(data, cfg["eligibility"]["min_history_weeks"],
                                         cfg["eligibility"]["require_active_at_origin"])
    scorable = data["eligible"] & data["y"].notna()

    def part(origins):
        m = data["t_idx"].isin(origins)
        return {"origins": [int(min(origins)), int(max(origins))] if origins else None,
                **coverage(data.loc[m, "eligible"], scorable[m]),
                "class_distribution": class_distribution(data.loc[m & scorable, "y"]),
                "styles_with_eligible_rows": int(data.loc[m & data["eligible"], "style"].nunique())}

    calib = data["t_idx"] + t["horizon"] <= sets["calibration_end"]
    report = {"dataset": did, "response": t.get("response", "spap"), "theta": theta, "origin_sets": {k: v if np.isscalar(v) else [min(v), max(v)] for k, v in sets.items()},
              "calibration": {"class_distribution": class_distribution(data.loc[calib & scorable, "y"])},
              "validation": part(sets["validation"]), "test": part(sets["test"]),
              "n_features": int(sum("__" in c or c.startswith("cal") for c in data.columns))}
    out = path(cfg, "features")
    files = [save_table(data, out / f"features_{did}.parquet"),
             save_json(report, out / f"target_{did}.json")]
    write_manifest(f"05_features_targets_{did}", cfg, outputs=files, extra=report)
    log(f"theta = {theta}")
    for k in ("validation", "test"):
        r = report[k]
        shares = ", ".join(f"{c}: {v['share']:.2f}" for c, v in r["class_distribution"].items())
        log(f"{k}: origins {r['origins']} forecasted {r['n_forecasted']} scorable {r['n_scorable']} "
            f"coverage {r['coverage']:.3f} classes [{shares}]")
