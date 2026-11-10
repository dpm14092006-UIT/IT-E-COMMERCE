"""G6 | Nguoi 3 - hyperparameter search on validation origins only (4.6.4). Test origins are never touched.

Outputs: outputs/tuning/{grouping}_{model}.csv and {grouping}_{model}_best.json
"""
from _common import log, parse

from vstyle.config import load_yaml, path
from vstyle.features.blocks import select_features
from vstyle.io import load_json, load_table, save_json
from vstyle.models import FORECAST_MODELS
from vstyle.utils.repro import write_manifest
from vstyle.validation.splits import origin_sets
from vstyle.validation.tuning import search

if __name__ == "__main__":
    def extra(ap):
        ap.add_argument("--grouping", required=True)
        ap.add_argument("--models", nargs="*", choices=FORECAST_MODELS, default=list(FORECAST_MODELS))
        ap.add_argument("--n-iter", type=int, default=None)
    args, cfg = parse(__doc__, extra)
    spaces = load_yaml("models.yaml")
    cal = load_json(path(cfg, "interim") / "calendar.json")
    data = load_table(path(cfg, "features") / f"features_{args.grouping}.parquet")
    H = cfg["target"]["horizon"]
    sets = origin_sets(cfg["partition"], cal["origin_week"], H, cal["t_max"])
    cols = select_features(data.columns, cfg["features"]["primary_blocks"])
    v = cfg["validation"]
    for name in [m for m in args.models if m == "tcn"]:
        import importlib.util
        from pathlib import Path

        import pandas as pd
        from sklearn.model_selection import ParameterSampler

        from vstyle.evaluation.metrics import macro_f1
        spec = importlib.util.spec_from_file_location("wf", Path(__file__).with_name("07_walk_forward.py"))
        wf = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(wf)
        rows = []
        for params in ParameterSampler(spaces[name]["space"], n_iter=args.n_iter or v["n_iter"],
                                       random_state=cfg["project"]["seed"]):
            sp = {**spaces, name: {**spaces[name], "defaults": {**spaces[name]["defaults"], **params}}}
            preds, _ = wf.sequence_predictions(name, data, cols, sets["validation"], cfg, sp, n_repeats=1,
                                               grouping="__none__")
            sc = preds[preds["y_true"].notna()]
            rows.append({"model": name, "class_weighting": "none", "params": {**spaces[name]["defaults"], **params},
                         "macro_f1": macro_f1(sc["y_true"].astype(int), sc["y_pred"]), "n_scored": len(sc)})
            log(f"{name} {params} -> {rows[-1]['macro_f1']:.3f}")
        res = pd.DataFrame(rows).sort_values("macro_f1", ascending=False, ignore_index=True)
        out = path(cfg, "tuning")
        res.to_csv(out / f"{args.grouping}_{name}.csv", index=False)
        save_json(res.iloc[0].to_dict(), out / f"{args.grouping}_{name}_best.json")
        log(f"{name}: best validation Macro-F1 {res.iloc[0]['macro_f1']:.3f} {res.iloc[0]['params']}")
    for name in [m for m in args.models if m != "tcn"]:
        log(f"tuning {name} on {len(sets['validation'])} validation origins, {len(cols)} features")
        res = search(data, cols, name, spaces[name]["space"], spaces[name]["defaults"], sets["validation"], H,
                     args.n_iter or v["n_iter"], cfg["project"]["seed"], v["tuning_refit_every"],
                     v["class_weighting"])
        out = path(cfg, "tuning")
        res.to_csv(out / f"{args.grouping}_{name}.csv", index=False)
        best = res.iloc[0].to_dict()
        save_json(best, out / f"{args.grouping}_{name}_best.json")
        write_manifest(f"06_tune_{args.grouping}_{name}", cfg, outputs=[out / f"{args.grouping}_{name}_best.json"])
        log(f"{name}: best validation Macro-F1 {best['macro_f1']:.3f} ({best['class_weighting']}) {best['params']}")
