"""G9d | Nguoi 4 - explainability (4.10): permutation importance on Macro-F1 (+ SHAP if installed).

Model fitted with labels matured before the first validation origin, explained on validation rows.
Output: outputs/metrics/{grouping}_{model}_importance.csv, _blocks.csv
"""
from _common import log, parse

from vstyle.config import load_yaml, path
from vstyle.features.blocks import block_of, select_features
from vstyle.interpret.explain import permutation_importance
from vstyle.io import load_json, load_table
from vstyle.models.tabular import make_model
from vstyle.validation.splits import origin_sets

if __name__ == "__main__":
    def extra(ap):
        ap.add_argument("--grouping", required=True)
        ap.add_argument("--model", default="catboost")
        ap.add_argument("--stage", choices=["validation", "test"], default="validation")
    args, cfg = parse(__doc__, extra)
    H = cfg["target"]["horizon"]
    cal = load_json(path(cfg, "interim") / "calendar.json")
    data = load_table(path(cfg, "features") / f"features_{args.grouping}.parquet")
    sets = origin_sets(cfg["partition"], cal["origin_week"], H, cal["t_max"])
    cols = select_features(data.columns, cfg["features"]["primary_blocks"])
    ok = data["eligible"] & data["y"].notna()
    t0 = sets[args.stage][0]
    train = data[ok & (data["t_idx"] + H <= t0)]                       # labels matured before the stage
    val = data[ok & data["t_idx"].isin(sets[args.stage])]
    f = path(cfg, "tuning") / f"{args.grouping}_{args.model}_best.json"
    best = load_json(f) if f.exists() else {"params": load_yaml("models.yaml")[args.model]["defaults"],
                                            "class_weighting": "none"}
    model = make_model(args.model, best["params"], cfg["project"]["seed"], best["class_weighting"])
    model.fit(train[cols], train["y"].astype(int).to_numpy())
    imp = permutation_importance(model, val[cols], val["y"].astype(int).to_numpy(), n_repeats=5,
                                 seed=cfg["project"]["seed"])
    imp["block"] = imp["feature"].map(block_of)
    out = path(cfg, "metrics")
    imp.to_csv(out / f"{args.grouping}_{args.stage}_{args.model}_importance.csv", index=False)
    blocks = imp.groupby("block")["importance_mean"].sum().sort_values(ascending=False)
    blocks.to_csv(out / f"{args.grouping}_{args.stage}_{args.model}_importance_blocks.csv")
    log(f"train rows {len(train)} (origins <= {t0 - H}), validation rows {len(val)}")
    log("top features:\n" + imp.head(12).round(4).to_string(index=False))
    log("by block (sum of Macro-F1 drop; associations, not causal effects):\n" + blocks.round(4).to_string())
