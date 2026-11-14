"""G9b | Nguoi 3 - common product-week endpoint for RQ3 (4.8.4). Identical rows and labels for all configs.

Outputs: outputs/predictions/endpoint_{stage}.parquet, outputs/metrics/endpoint_{stage}.json/_table.csv
"""
import pandas as pd
from _common import log, parse

from vstyle.config import load_yaml, path
from vstyle.evaluation.bootstrap import paired_macro_f1_difference
from vstyle.evaluation.metrics import summary
from vstyle.experiments import common_endpoint as ce
from vstyle.io import load_json, load_table, save_json, save_table
from vstyle.models.tabular import make_model
from vstyle.target.labels import label_from_delta
from vstyle.target.threshold import calibrate, class_distribution
from vstyle.validation.splits import origin_sets
from vstyle.validation.walk_forward import run_walk_forward

PAIRS = [("knn_resnet50", "own_history"), ("knn_fashionclip", "own_history"),
         ("knn_novelty_resnet50", "own_history"), ("knn_novelty_fashionclip", "own_history"),
         ("knn_resnet50", "resnet50_context"),
         ("resnet50ft_context", "own_history"), ("resnet50ft_context", "resnet50_context"),
         ("fashionclipft_context", "own_history"), ("fashionclipft_context", "fashionclip_context"),
         ("resnet50_context", "own_history"), ("fashionclip_context", "own_history"),
         ("attribute_context", "own_history"), ("resnet50_context", "attribute_context"),
         ("fashionclip_context", "attribute_context")]

if __name__ == "__main__":
    def extra(ap):
        ap.add_argument("--stage", choices=["validation", "test"], default="validation")
        ap.add_argument("--models", nargs="*", choices=["lightgbm", "catboost", "random_forest"],
                        default=["lightgbm", "catboost", "random_forest"])
        ap.add_argument("--refit-every", type=int, default=4)
        ap.add_argument("--extensions", action="store_true", help="run 2: visual kNN context + novelty configs")
        ap.add_argument("--tag", default="")
    args, cfg = parse(__doc__, extra)
    H, seed = cfg["target"]["horizon"], cfg["project"]["seed"]
    interim = path(cfg, "interim")
    cal = load_json(interim / "calendar.json")
    products = load_table(interim / "products.parquet")
    usable = set(products.loc[products["image_ok"], "external_code"])
    pw = load_table(interim / "product_week.parquet")
    pp = ce.add_own_features(ce.add_targets(ce.product_panel(pw[pw["external_code"].isin(usable)])))
    k = cfg["visual"]["selected"]["k"]
    groupings = {"attribute": products.set_index("external_code")["category"]}
    for enc in ("fashionclip", "resnet50", "resnet50ft", "fashionclipft"):
        assignments = path(cfg, "clusters") / enc / f"assignments_k{k}.parquet"
        if assignments.is_file():
            groupings[enc] = load_table(assignments).set_index("external_code")["style"]
    for name, g in groupings.items():
        pp = ce.add_context(pp, g[g.index.isin(usable)], name)

    sets = origin_sets(cfg["partition"], cal["origin_week"], H, cal["t_max"])
    lo, hi = load_yaml("experiments.yaml")["common_endpoint"]["product_origin_age_weeks"]
    pp["eligible"] = pp["own_age"].between(lo, hi) & pp["pmr_past4"].notna() & (pp["n_shops"] > 0)
    e = pp["eligible"]
    theta = calibrate(pp.loc[e, "delta"], pp.loc[e, "t_idx"], sets["calibration_end"], H)
    pp["y"] = label_from_delta(pp["delta"], theta["theta"]).where(e)
    pp["style"] = pp["external_code"]                      # walk_forward keys rows by "style"

    own = ce.own_columns(pp.columns)
    configs = {"own_history": own}
    if args.extensions:
        from _visual import visual_product_features
        configs = {"own_history": own, "resnet50_context": None}
        for enc in ("resnet50", "fashionclip"):
            pp, nov, _, _ = visual_product_features(cfg, enc, pp, products)
            pp[f"novelty_{enc}"] = pp["external_code"].map(nov)
            configs[f"knn_{enc}"] = own + [c for c in pp.columns if c == f"knn_{enc}" or c.startswith(f"knn_{enc}__")]
            configs[f"knn_novelty_{enc}"] = configs[f"knn_{enc}"] + [f"novelty_{enc}"]
        configs["resnet50_context"] = own + [c for c in pp.columns if c == "ctx_resnet50" or c.startswith("ctx_resnet50__")]
    for name in ([] if args.extensions else groupings):
        configs[f"{name}_context"] = own + [c for c in pp.columns
                                            if c == f"ctx_{name}" or c.startswith(f"ctx_{name}__")]
    frames = []
    for cname, cols in configs.items():
        for m in args.models:
            p = load_yaml("models.yaml")[m]["defaults"]
            preds = run_walk_forward(pp, cols, lambda: make_model(m, p, seed), sets[args.stage], H,
                                     args.refit_every)
            frames.append(preds.assign(config=cname, model=m))
            log(f"{cname} / {m} done")
    preds = pd.concat(frames, ignore_index=True)
    save_table(preds, path(cfg, "predictions") / f"endpoint_{args.stage}{args.tag}.parquet")

    scored = preds[preds["y_true"].notna()]
    first = scored.drop_duplicates(["style", "t_idx"])
    report = {"theta": theta, "n_eligible_all_periods": int(e.sum()), "n_scored_rows": len(first),
              "class_distribution": class_distribution(first["y_true"]), "diffs": {}}
    rows, b = [], cfg["evaluation"]["bootstrap"]
    for m, g in scored.groupby("model"):
        wide = (g.pivot_table(index=["style", "t_idx"], columns="config", values="y_pred")
                 .join(first.set_index(["style", "t_idx"])["y_true"]).dropna().reset_index())
        for c in configs:
            s = summary(wide["y_true"].astype(int), wide[c].astype(int))
            rows.append({"model": m, "config": c, "macro_f1": s["macro_f1"]})
        for a_, b_ in [pr for pr in PAIRS if pr[0] in configs and pr[1] in configs]:
            report["diffs"][f"{m}: {a_} - {b_}"] = paired_macro_f1_difference(
                wide, a_, b_, alpha=b["alpha"], block_length=b["block_length"], n_boot=1000, seed=seed)
    table = pd.DataFrame(rows).pivot(index="model", columns="config", values="macro_f1")[list(configs)]
    out = path(cfg, "metrics")
    table.to_csv(out / f"endpoint_{args.stage}{args.tag}_table.csv")
    save_json(report, out / f"endpoint_{args.stage}{args.tag}.json")
    log(f"theta {theta['theta']:.3f}; scored product-weeks {len(first)}; classes {report['class_distribution']}")
    log("\n" + table.round(3).to_string())
    for k_, d in report["diffs"].items():
        log(f"{k_}: {d['estimate']:+.3f} [{d['ci_low']:+.3f}, {d['ci_high']:+.3f}]")
