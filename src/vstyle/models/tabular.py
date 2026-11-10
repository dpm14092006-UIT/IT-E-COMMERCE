"""4.5.1-4.5.4 - tree ensembles behind one interface that always returns 3 aligned probabilities."""
import numpy as np
from sklearn.utils.class_weight import compute_sample_weight

from . import TABULAR_MODELS


class AlignedClassifier:
    """Fits on the classes present in the training window; predict_proba is always (n, 3)
    in CLASSES order, so a class missing from an early window cannot shift the columns."""

    def __init__(self, estimator, class_weighting="none"):
        self.estimator, self.class_weighting = estimator, class_weighting

    def fit(self, X, y):
        y = np.asarray(y, dtype=int)
        self.present_ = np.unique(y)
        self.constant_ = self.present_[0] if len(self.present_) == 1 else None
        if self.constant_ is None:
            enc = np.searchsorted(self.present_, y)
            sw = compute_sample_weight("balanced", enc) if self.class_weighting == "balanced" else None
            self.estimator.fit(X, enc, sample_weight=sw)       # (53): weights from training labels only
        return self

    def predict_proba(self, X):
        out = np.zeros((len(X), 3))
        if self.constant_ is not None:
            out[:, self.constant_] = 1.0
        else:
            out[:, self.present_] = self.estimator.predict_proba(X)
        return out

    def predict(self, X):
        return self.predict_proba(X).argmax(1)


def make_model(name, params, seed, class_weighting="none"):
    params = dict(params)
    if name == "random_forest":
        from sklearn.ensemble import RandomForestClassifier
        est = RandomForestClassifier(random_state=seed, n_jobs=-1, **params)
    elif name == "lightgbm":
        from lightgbm import LGBMClassifier
        est = LGBMClassifier(random_state=seed, n_jobs=-1, verbose=-1, **params)
    elif name == "catboost":
        from catboost import CatBoostClassifier
        est = CatBoostClassifier(random_seed=seed, loss_function="MultiClass", verbose=False,
                                 allow_writing_files=False, thread_count=-1, **params)
    else:
        raise ValueError(f"unknown tabular model {name}")
    return AlignedClassifier(est, class_weighting)
