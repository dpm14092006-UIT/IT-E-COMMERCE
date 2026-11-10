"""G7 | Nguoi 3 (tabular, baselines) + Nguoi 4 (TCN) - expanding-window walk-forward (4.6.2).

--stage test is refused unless --frozen is given: every [CHOT] decision must be in docs/decision_log.md.
Output: outputs/predictions/{grouping}_{stage}_{tag}.parquet (one row per model x style x origin)
"""
import pandas as pd
from _common import log, parse

from vstyle.config import load_yaml, path
from vstyle.features.blocks import select_features
from vstyle.io import load_json, load_table, save_table
from vstyle.models.baselines import MajorityClass, Persistence
from vstyle.models import FORECAST_MODELS
from vstyle.models.tabular import make_model
from vstyle.utils.repro import write_manifest
from vstyle.validation.splits import origin_sets
from vstyle.validation.walk_forward import run_walk_forward


SEQ_CLASS_WEIGHTING = "none"   # set by --seq-class-weighting (run 2)
PARAMS_FROM = None   # set by --params-from: reuse another grouping's tuned hyperparameters


def tuned(cfg, grouping, name, spaces):
    f = path(cfg, "tuning") / f"{PARAMS_FROM or grouping.split('__')[0]}_{name}_best.json"
    if f.exists():
        best = load_json(f)
        return best["params"], best["class_weighting"], "tuned"
    return spaces[name]["defaults"], "none", "defaults"


def sequence_predictions(name, data, cols, origins, cfg, spaces, lookback=None, n_repeats=None, grouping=""):
    """TCN: one walk-forward per seed; the reported model averages the seed probabilities."""
    from vstyle.models.sequence import SequenceClassifier, build_tcn
    p = dict(spaces[name]["defaults"])
    best = path(cfg, "tuning") / f"{PARAMS_FROM or grouping.split('__')[0]}_{name}_best.json"
    if best.exists():
        p.update(load_json(best)["params"])
    if lookback:
        p["lookback"] = lookback
    H, base_seed = cfg["target"]["horizon"], cfg["project"]["seed"]
    if name != "tcn":
        raise ValueError(f"unknown sequence model {name}")
    net = lambda n: build_tcn(n, p["channels"], p["kernel_size"], p["dilations"], p["dropout"])  # noqa: E731
    runs = []
    for r in range(n_repeats or spaces[name]["n_repeats"]):
        make = lambda r=r: SequenceClassifier(  # noqa: E731
            data, cols, p["lookback"], net, p["learning_rate"], p["batch_size"], p["max_epochs"],
            p["patience"], class_weighting=SEQ_CLASS_WEIGHTING, seed=base_seed + r)
        runs.append(run_walk_forward(data, ["style", "t_idx"], make, origins, H,
                                     spaces[name]["refit_every"]).assign(model=name, seed=base_seed + r))
        log(f"{name} seed {base_seed + r} done")
    seeds = pd.concat(runs, ignore_index=True)
    mean = seeds.groupby(["style", "t_idx"], as_index=False).agg(
        y_true=("y_true", "first"), p_declining=("p_declining", "mean"), p_stable=("p_stable", "mean"),
        p_emerging=("p_emerging", "mean"), fit_origin=("fit_origin", "first"), n_train=("n_train", "first"))
    mean["y_pred"] = mean[["p_declining", "p_stable", "p_emerging"]].to_numpy().argmax(1)
    return mean.assign(model=name), seeds


def run(data, cols, models, origins, cfg, grouping, theta, spaces, lookback=None, refit_every=None,
        n_repeats=None):
    H, seed = cfg["target"]["horizon"], cfg["project"]["seed"]
    refit = refit_every or cfg["validation"]["refit_every"]
    frames, seed_frames = [], []
    for name, model, fc in [("majority", MajorityClass, ["pmr_change4"]),
                            ("persistence", lambda: Persistence(theta), ["pmr_change4"])]:
        frames.append(run_walk_forward(data, fc, model, origins, H, refit).assign(model=name))
    for name in models:
        if name == "tcn":
            mean, seeds = sequence_predictions(name, data, cols, origins, cfg, spaces, lookback, n_repeats, grouping)
            frames.append(mean)
            seed_frames.append(seeds)
            continue
        params, cw, source = tuned(cfg, grouping, name, spaces)
        log(f"{name}: {source} params, class_weighting={cw}")
        frames.append(run_walk_forward(data, cols, lambda: make_model(name, params, seed, cw),
                                       origins, H, refit).assign(model=name))
    seeds = pd.concat(seed_frames, ignore_index=True) if seed_frames else None
    return pd.concat(frames, ignore_index=True), seeds


if __name__ == "__main__":
    def extra(ap):
        ap.add_argument("--grouping", required=True)
        ap.add_argument("--stage", choices=["validation", "test"], default="validation")
        ap.add_argument("--models", nargs="*", choices=FORECAST_MODELS, default=list(FORECAST_MODELS))
        ap.add_argument("--blocks", nargs="*", default=None)
        ap.add_argument("--tag", default="primary")
        ap.add_argument("--frozen", action="store_true", help="confirm decision_log is complete")
        ap.add_argument("--lookback", type=int, default=None, help="robustness: TCN lookback")
        ap.add_argument("--refit-every", type=int, default=None)
        ap.add_argument("--n-repeats", type=int, default=None, help="seeds for TCN")
        ap.add_argument("--params-from", default=None, help="use tuned hyperparameters of this grouping")
        ap.add_argument("--seq-class-weighting", default="none", choices=["none", "balanced"])
    args, cfg = parse(__doc__, extra)
    if args.stage == "test" and not args.frozen:
        raise SystemExit("Final test needs --frozen (all [CHOT] decisions recorded). See docs/02_TIEU_CHI_DAT.md G6.")
    PARAMS_FROM = args.params_from
    SEQ_CLASS_WEIGHTING = args.seq_class_weighting
    cal = load_json(path(cfg, "interim") / "calendar.json")
    data = load_table(path(cfg, "features") / f"features_{args.grouping}.parquet")
    theta = load_json(path(cfg, "features") / f"target_{args.grouping}.json")["theta"]["theta"]
    sets = origin_sets(cfg["partition"], cal["origin_week"], cfg["target"]["horizon"], cal["t_max"])
    cols = select_features(data.columns, args.blocks or cfg["features"]["primary_blocks"])
    preds, seeds = run(data, cols, args.models, sets[args.stage], cfg, args.grouping, theta,
                       load_yaml("models.yaml"), args.lookback, args.refit_every, args.n_repeats)
    out = save_table(preds, path(cfg, "predictions") / f"{args.grouping}_{args.stage}_{args.tag}.parquet")
    if seeds is not None:
        save_table(seeds, path(cfg, "predictions") / f"{args.grouping}_{args.stage}_{args.tag}_seeds.parquet")
    write_manifest(f"07_walk_forward_{args.grouping}_{args.stage}_{args.tag}", cfg, outputs=[out],
                   extra={"n_features": len(cols), "origins": [min(sets[args.stage]), max(sets[args.stage])]})
    log(f"saved {len(preds):,} predictions -> {out.name}")
