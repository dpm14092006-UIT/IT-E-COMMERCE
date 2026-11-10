"""G10 | Nguoi 4 - decision layer + dashboard table (4.10). Operating point chosen on VALIDATION only.

Outputs: outputs/reports/{grouping}_decision_scenarios.csv, {grouping}_{stage}_signals.csv,
         {grouping}_decision.json
"""
import numpy as np
from _common import log, parse

from vstyle.config import path
from vstyle.data.calendar import week_of
from vstyle.decision.rules import alert_quality, decision_signals, scenario_table
from vstyle.io import load_json, load_table, save_json

if __name__ == "__main__":
    def extra(ap):
        ap.add_argument("--grouping", required=True)
        ap.add_argument("--model", default="catboost")
        ap.add_argument("--stage", choices=["validation", "test"], default="validation")
        ap.add_argument("--tag", default="primary")
        ap.add_argument("--val-tag", default=None, help="validation predictions used to pick the operating point")
    args, cfg = parse(__doc__, extra)
    pdir, out = path(cfg, "predictions"), path(cfg, "reports")
    val = load_table(pdir / f"{args.grouping}_validation_{args.val_tag or args.tag}.parquet")
    val = val[val["model"] == args.model]
    scen = scenario_table(val)
    p, r = scen["emerging_alert_precision"].fillna(0), scen["emerging_recall"].fillna(0)
    scen["emerging_alert_f1"] = np.where(p + r > 0, 2 * p * r / (p + r), 0)
    best = scen.sort_values("emerging_alert_f1", ascending=False).iloc[0]
    scen.to_csv(out / f"{args.grouping}_decision_scenarios.csv", index=False)

    preds = load_table(pdir / f"{args.grouping}_{args.stage}_{args.tag}.parquet")
    preds = preds[preds["model"] == args.model]
    sig = decision_signals(preds, best["min_prob"], best["min_margin"])
    feats = load_table(path(cfg, "features") / f"features_{args.grouping}.parquet")
    sig = sig.merge(feats[["style", "t_idx", "spap", "spap__growth4", "active_products"]],
                    on=["style", "t_idx"], how="left")
    sig["week_start"] = week_of(sig["t_idx"], load_json(path(cfg, "interim") / "calendar.json")["origin_week"])
    sig.to_csv(out / f"{args.grouping}_{args.stage}_signals.csv", index=False)
    quality = alert_quality(sig)
    save_json({"operating_point": {"min_prob": best["min_prob"], "min_margin": best["min_margin"],
                                   "chosen_on": "validation"}, "stage": args.stage, "quality": quality},
              out / f"{args.grouping}_decision.json")
    log(f"operating point (validation): min_prob={best['min_prob']}, min_margin={best['min_margin']}")
    log(f"{args.stage} alert quality: {quality}")
