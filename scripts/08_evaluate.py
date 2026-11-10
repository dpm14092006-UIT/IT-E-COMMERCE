"""G8 | Nguoi 3 - matched-sample metrics, coverage, block-bootstrap CIs, per-period stability (4.7, 4.9.5-4.9.6).

Outputs: outputs/metrics/{grouping}_{stage}_{tag}.json and _table.csv
"""
import pandas as pd
from _common import log, parse

from vstyle.config import path
from vstyle.data.calendar import week_of
from vstyle.evaluation.bootstrap import paired_macro_f1_difference
from vstyle.evaluation.metrics import summary
from vstyle.io import load_json, load_table, save_json

if __name__ == "__main__":
    def extra(ap):
        ap.add_argument("--grouping", required=True)
        ap.add_argument("--stage", choices=["validation", "test"], default="validation")
        ap.add_argument("--tag", default="primary")
        ap.add_argument("--reference", default="persistence")
    args, cfg = parse(__doc__, extra)
    preds = load_table(path(cfg, "predictions") / f"{args.grouping}_{args.stage}_{args.tag}.parquet")
    cal = load_json(path(cfg, "interim") / "calendar.json")
    models = preds["model"].unique().tolist()

    # 4.7.4 matched sample: (style, origin) rows scored by every model
    wide = preds.pivot_table(index=["style", "t_idx"], columns="model", values="y_pred", aggfunc="first")
    truth = preds.drop_duplicates(["style", "t_idx"]).set_index(["style", "t_idx"])["y_true"]
    wide = wide.join(truth).dropna().reset_index()
    n_forecasted = preds.groupby("model").size().to_dict()

    rows, report = [], {"n_matched": len(wide), "n_forecasted_by_model": n_forecasted, "models": {}}
    for m in models:
        s = summary(wide["y_true"].astype(int), wide[m].astype(int), cfg["evaluation"]["zero_division"])
        report["models"][m] = s
        rows.append({"model": m, "macro_f1": s["macro_f1"], "balanced_accuracy": s["balanced_accuracy"],
                     "accuracy": s["accuracy"], "emerging_precision": s["per_class"]["Emerging"]["precision"],
                     "emerging_recall": s["per_class"]["Emerging"]["recall"], "n": s["n"],
                     "coverage": float(preds[(preds.model == m)]["y_true"].notna().mean())})
    table = pd.DataFrame(rows).sort_values("macro_f1", ascending=False)

    b = cfg["evaluation"]["bootstrap"]
    report["vs_reference"] = {
        m: paired_macro_f1_difference(wide, m, args.reference, alpha=b["alpha"], block_length=b["block_length"],
                                      n_boot=b["n_boot"], seed=cfg["project"]["seed"])
        for m in models if m != args.reference}
    wide["half_year"] = week_of(wide["t_idx"], cal["origin_week"]).to_period("Q").astype(str)
    report["by_quarter"] = {q: {m: summary(g["y_true"].astype(int), g[m].astype(int))["macro_f1"] for m in models}
                            for q, g in wide.groupby("half_year")}

    out = path(cfg, "metrics")
    save_json(report, out / f"{args.grouping}_{args.stage}_{args.tag}.json")
    table.to_csv(out / f"{args.grouping}_{args.stage}_{args.tag}_table.csv", index=False)
    log(f"matched rows {len(wide)}\n" + table.round(3).to_string(index=False))
    for m, d in report["vs_reference"].items():
        log(f"{m} - {args.reference}: {d['estimate']:+.3f} [{d['ci_low']:+.3f}, {d['ci_high']:+.3f}]")
