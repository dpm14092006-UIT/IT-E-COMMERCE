"""One walk-forward experiment: baselines, tuned tabular models and the multi-seed TCN, as a class.

Replaces the module-level globals and the importlib hacks scripts 06/07/09 used to share this logic.
"""
import pandas as pd

from ..config import Config
from ..io import load_json
from ..models import SEQUENCE_MODELS
from ..models.baselines import MajorityClass, Persistence
from ..models.tabular import make_model
from ..validation.walk_forward import run_walk_forward

PROBA = ["p_declining", "p_stable", "p_emerging"]


class WalkForwardExperiment:
    def __init__(self, cfg: Config, spaces: dict, grouping: str, theta: float, params_from: str | None = None,
                 seq_class_weighting: str = "none", log=print):
        self.cfg, self.spaces, self.grouping, self.theta = cfg, spaces, grouping, theta
        self.params_from, self.seq_class_weighting, self.log = params_from, seq_class_weighting, log
        self.horizon, self.seed = cfg["target"]["horizon"], cfg["project"]["seed"]

    def _best_file(self, name):
        return self.cfg.path("tuning") / f"{self.params_from or self.grouping.split('__')[0]}_{name}_best.json"

    def tuned(self, name):
        """(params, class_weighting, source) from the tuning output, else the defaults."""
        f = self._best_file(name)
        if f.exists():
            best = load_json(f)
            return best["params"], best["class_weighting"], "tuned"
        return self.spaces[name]["defaults"], "none", "defaults"

    def sequence_predictions(self, name, data, cols, origins, lookback=None, n_repeats=None):
        """TCN: one walk-forward per seed; the reported model averages the seed probabilities."""
        from ..models.sequence import SequenceClassifier, build_tcn
        if name not in SEQUENCE_MODELS:
            raise ValueError(f"unknown sequence model {name}")
        p = dict(self.spaces[name]["defaults"])
        if self._best_file(name).exists():
            p.update(load_json(self._best_file(name))["params"])
        if lookback:
            p["lookback"] = lookback

        def net(n):
            return build_tcn(n, p["channels"], p["kernel_size"], p["dilations"], p["dropout"])

        runs = []
        for r in range(n_repeats or self.spaces[name]["n_repeats"]):
            def make(r=r):
                return SequenceClassifier(data, cols, p["lookback"], net, p["learning_rate"], p["batch_size"],
                                          p["max_epochs"], p["patience"],
                                          class_weighting=self.seq_class_weighting, seed=self.seed + r)
            runs.append(run_walk_forward(data, ["style", "t_idx"], make, origins, self.horizon,
                                         self.spaces[name]["refit_every"]).assign(model=name, seed=self.seed + r))
            self.log(f"{name} seed {self.seed + r} done")
        seeds = pd.concat(runs, ignore_index=True)
        mean = seeds.groupby(["style", "t_idx"], as_index=False).agg(
            y_true=("y_true", "first"), p_declining=("p_declining", "mean"), p_stable=("p_stable", "mean"),
            p_emerging=("p_emerging", "mean"), fit_origin=("fit_origin", "first"), n_train=("n_train", "first"))
        mean["y_pred"] = mean[PROBA].to_numpy().argmax(1)
        return mean.assign(model=name), seeds

    def run(self, data, cols, models, origins, lookback=None, refit_every=None, n_repeats=None):
        refit = refit_every or self.cfg["validation"]["refit_every"]
        frames, seed_frames = [], []
        for name, model in (("majority", MajorityClass), ("persistence", lambda: Persistence(self.theta))):
            frames.append(run_walk_forward(data, ["pmr_change4"], model, origins, self.horizon,
                                           refit).assign(model=name))
        for name in models:
            if name in SEQUENCE_MODELS:
                mean, seeds = self.sequence_predictions(name, data, cols, origins, lookback, n_repeats)
                frames.append(mean)
                seed_frames.append(seeds)
                continue
            params, cw, source = self.tuned(name)
            self.log(f"{name}: {source} params, class_weighting={cw}")
            frames.append(run_walk_forward(data, cols, lambda n=name, p=params, c=cw: make_model(n, p, self.seed, c),
                                           origins, self.horizon, refit).assign(model=name))
        seeds = pd.concat(seed_frames, ignore_index=True) if seed_frames else None
        return pd.concat(frames, ignore_index=True), seeds
