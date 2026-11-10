"""Run 2 | Nguoi 4 - cost-sensitive decision rules on frozen model probabilities (4.10 decision layer).

y_hat = argmax_j  w_j * p_j  with w_Stable = 1. Multipliers are chosen on VALIDATION predictions only:
  rule "macro"     : maximise validation Macro-F1
  rule "recall70"  : maximise validation Macro-F1 subject to validation Emerging recall >= 0.70
then applied unchanged to the test probabilities. Models are not retrained.
Output: outputs/metrics/{grouping}_operating_points.csv / .json
"""
import itertools

import numpy as np
import pandas as pd
from _common import log, parse

from vstyle.config import path
from vstyle.constants import PROBA_COLS
from vstyle.evaluation.bootstrap import paired_macro_f1_difference
from vstyle.evaluation.metrics import summary
from vstyle.io import load_table, save_json

GRID_E = np.round(np.arange(0.8, 3.01, 0.1), 2)
GRID_D = np.round(np.arange(0.8, 2.01, 0.1), 2)


def apply(df, wd, we):
    return (df[list(PROBA_COLS)].to_numpy() * np.array([wd, 1.0, we])).argmax(1)


def choose(val):
    y = val["y_true"].astype(int).to_numpy()
    rows = []
    for wd, we in itertools.product(GRID_D, GRID_E):
        s = summary(y, apply(val, wd, we))
        rows.append((wd, we, s["macro_f1"], s["per_class"]["Emerging"]["recall"]))
    r = pd.DataFrame(rows, columns=["wd", "we", "f1", "rec_e"])
    macro = r.sort_values(["f1", "we"], ascending=[False, True]).iloc[0]
    ok = r[r["rec_e"] >= 0.70]
    rec = (ok.sort_values("f1", ascending=False) if len(ok) else r.sort_values("rec_e", ascending=False)).iloc[0]
    return {"macro": macro, "recall70": rec}


if __name__ == "__main__":
    def extra(ap):
        ap.add_argument("--grouping", default="resnet50")
        ap.add_argument("--val-tag", default="tuned")
        ap.add_argument("--test-tag", default="primary")
    args, cfg = parse(__doc__, extra)
    P = path(cfg, "predictions")
    val = load_table(P / f"{args.grouping}_validation_{args.val_tag}.parquet")
    test = load_table(P / f"{args.grouping}_test_{args.test_tag}.parquet")
    pers = test[test["model"] == "persistence"].set_index(["style", "t_idx"])["y_pred"].rename("persistence")
    b = cfg["evaluation"]["bootstrap"]
    rows, report = [], {}
    for m in [x for x in val["model"].unique() if x not in ("majority", "persistence")]:
        v = val[(val["model"] == m) & val["y_true"].notna()]
        t = test[(test["model"] == m) & test["y_true"].notna()].copy()
        if t.empty:
            continue
        t["argmax"] = t[list(PROBA_COLS)].to_numpy().argmax(1)
        for rule, w in choose(v).items():
            t[rule] = apply(t, w["wd"], w["we"])
            s = summary(t["y_true"].astype(int), t[rule])
            w2 = t.join(pers, on=["style", "t_idx"])
            d0 = paired_macro_f1_difference(w2, rule, "argmax", block_length=b["block_length"], n_boot=1000, seed=2026)
            d1 = paired_macro_f1_difference(w2, rule, "persistence", block_length=b["block_length"], n_boot=1000, seed=2026)
            rows.append({"model": m, "rule": rule, "w_declining": w["wd"], "w_emerging": w["we"],
                         "val_macro_f1": w["f1"], "val_recall_E": w["rec_e"], "test_macro_f1": s["macro_f1"],
                         "test_P_E": s["per_class"]["Emerging"]["precision"], "test_R_E": s["per_class"]["Emerging"]["recall"],
                         "test_P_D": s["per_class"]["Declining"]["precision"], "test_R_D": s["per_class"]["Declining"]["recall"],
                         "vs_argmax": f"{d0['estimate']:+.3f} [{d0['ci_low']:+.3f}, {d0['ci_high']:+.3f}]",
                         "vs_persistence": f"{d1['estimate']:+.3f} [{d1['ci_low']:+.3f}, {d1['ci_high']:+.3f}]"})
        sa = summary(t["y_true"].astype(int), t["argmax"])
        rows.append({"model": m, "rule": "argmax (run 1)", "test_macro_f1": sa["macro_f1"],
                     "test_P_E": sa["per_class"]["Emerging"]["precision"], "test_R_E": sa["per_class"]["Emerging"]["recall"],
                     "test_P_D": sa["per_class"]["Declining"]["precision"], "test_R_D": sa["per_class"]["Declining"]["recall"]})
    table = pd.DataFrame(rows)
    out = path(cfg, "metrics")
    table.to_csv(out / f"{args.grouping}_operating_points.csv", index=False)
    save_json(table.to_dict(orient="records"), out / f"{args.grouping}_operating_points.json")
    log("\n" + table.round(3).to_string(index=False))
