"""4.5.1-4.5.4 - tree ensembles behind one interface that always returns 3 aligned probabilities."""
import numpy as np
from sklearn.utils.class_weight import compute_sample_weight

from .base import ForecastModel
from .registry import ESTIMATORS


class AlignedClassifier(ForecastModel):
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


@ESTIMATORS.register("random_forest")
def _random_forest(params, seed):
    from sklearn.ensemble import RandomForestClassifier
    return RandomForestClassifier(random_state=seed, n_jobs=-1, **params)


@ESTIMATORS.register("lightgbm")
def _lightgbm(params, seed):
    from lightgbm import LGBMClassifier
    return LGBMClassifier(random_state=seed, n_jobs=-1, verbose=-1, **params)


@ESTIMATORS.register("catboost")
def _catboost(params, seed):
    from catboost import CatBoostClassifier
    return CatBoostClassifier(random_seed=seed, loss_function="MultiClass", verbose=False,
                              allow_writing_files=False, thread_count=-1, **params)


def make_model(name, params, seed, class_weighting="none"):
    """Registry lookup; raises ValueError for an unregistered name."""
    return AlignedClassifier(ESTIMATORS.create(name, dict(params), seed), class_weighting)
