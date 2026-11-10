"""G9a | Nguoi 3 - nested feature-block ablation (4.8.3). B4 is split: B4a calendar, B4b Google Trends.

Output: outputs/metrics/{grouping}_{stage}_ablation.csv  (Macro-F1 per model x configuration, + delta)
"""
import importlib.util
from pathlib import Path

import pandas as pd
from _common import log, parse

from vstyle.config import load_yaml, path
from vstyle.evaluation.metrics import macro_f1
from vstyle.features.blocks import mechanics_features, select_features
from vstyle.io import load_json, load_table
from vstyle.validation.splits import origin_sets

spec = importlib.util.spec_from_file_location("wf", Path(__file__).with_name("07_walk_forward.py"))
wf = importlib.util.module_from_spec(spec)
spec.loader.exec_module(wf)

if __name__ == "__main__":
    def extra(ap):
        ap.add_argument("--grouping", required=True)
        ap.add_argument("--stage", choices=["validation", "test"], default="validation")
        ap.add_argument("--refit-every", type=int, default=None)
    args, cfg = parse(__doc__, extra)
    exp = load_yaml("experiments.yaml")["ablation"]
    cal = load_json(path(cfg, "interim") / "calendar.json")
    data = load_table(path(cfg, "features") / f"features_{args.grouping}.parquet")
    theta = load_json(path(cfg, "features") / f"target_{args.grouping}.json")["theta"]["theta"]
    origins = origin_sets(cfg["partition"], cal["origin_week"], cfg["target"]["horizon"], cal["t_max"])[args.stage]
    rows = []
    for name, blocks in exp["configurations"].items():
        cols = (mechanics_features(data.columns) if blocks == ["M"] else select_features(data.columns, blocks))
        preds, _ = wf.run(data, cols, exp["models"], origins, cfg, args.grouping, theta, load_yaml("models.yaml"),
                          refit_every=args.refit_every)
        preds = preds[~preds["model"].isin(["majority", "persistence"])]
        # common sample across configurations = scorable eligible rows (same eligibility for all)
        for m, g in preds[preds["y_true"].notna()].groupby("model"):
            rows.append({"configuration": name, "model": m, "n_features": len(cols),
                         "macro_f1": macro_f1(g["y_true"].astype(int), g["y_pred"]), "n": len(g)})
        log(f"{name}: {len(cols)} features done")
    table = pd.DataFrame(rows).pivot(index="model", columns="configuration", values="macro_f1")
    table = table[list(exp["configurations"])]
    deltas = table.diff(axis=1).add_prefix("delta_").iloc[:, 1:]                     # (67)
    out = path(cfg, "metrics") / f"{args.grouping}_{args.stage}_ablation.csv"
    pd.concat([table, deltas], axis=1).to_csv(out)
    log("\n" + pd.concat([table, deltas], axis=1).round(3).to_string())
