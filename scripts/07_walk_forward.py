"""G7 | Nguoi 3 (tabular, baselines) + Nguoi 4 (TCN) - expanding-window walk-forward (4.6.2).

--stage test is refused unless --frozen is given: every [CHOT] decision must be in docs/decision_log.md.
Output: outputs/predictions/{grouping}_{stage}_{tag}.parquet (one row per model x style x origin)
"""
from _common import log, parse

from vstyle.config import load_yaml, path
from vstyle.experiments.walk_forward_experiment import WalkForwardExperiment
from vstyle.features.blocks import select_features
from vstyle.io import load_json, load_table, save_table
from vstyle.models import FORECAST_MODELS
from vstyle.utils.repro import write_manifest
from vstyle.validation.splits import origin_sets

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
    cal = load_json(path(cfg, "interim") / "calendar.json")
    data = load_table(path(cfg, "features") / f"features_{args.grouping}.parquet")
    theta = load_json(path(cfg, "features") / f"target_{args.grouping}.json")["theta"]["theta"]
    sets = origin_sets(cfg["partition"], cal["origin_week"], cfg["target"]["horizon"], cal["t_max"])
    cols = select_features(data.columns, args.blocks or cfg["features"]["primary_blocks"])
    experiment = WalkForwardExperiment(cfg, load_yaml("models.yaml"), args.grouping, theta, args.params_from,
                                       args.seq_class_weighting, log)
    preds, seeds = experiment.run(data, cols, args.models, sets[args.stage], args.lookback, args.refit_every,
                                  args.n_repeats)
    out = save_table(preds, path(cfg, "predictions") / f"{args.grouping}_{args.stage}_{args.tag}.parquet")
    if seeds is not None:
        save_table(seeds, path(cfg, "predictions") / f"{args.grouping}_{args.stage}_{args.tag}_seeds.parquet")
    write_manifest(f"07_walk_forward_{args.grouping}_{args.stage}_{args.tag}", cfg, outputs=[out],
                   extra={"n_features": len(cols), "origins": [min(sets[args.stage]), max(sets[args.stage])]})
    log(f"saved {len(preds):,} predictions -> {out.name}")
