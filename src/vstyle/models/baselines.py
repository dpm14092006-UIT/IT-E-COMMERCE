"""4.8.1 - simple baselines with the same interface as the trained models."""
import numpy as np

from ..constants import DECLINING, EMERGING, STABLE


class MajorityClass:
    """(66) most frequent class in the matured training labels at each origin."""

    def fit(self, X, y):
        self.cls_ = int(np.bincount(np.asarray(y, dtype=int), minlength=3).argmax())
        return self

    def predict_proba(self, X):
        out = np.zeros((len(X), 3))
        out[:, self.cls_] = 1.0
        return out


class Persistence:
    """The last observed 4-week change (pmr_change4, fully past) classified with the same theta."""

    def __init__(self, theta, column="pmr_change4"):
        self.theta, self.column = theta, column

    def fit(self, X, y):
        return self

    def predict_proba(self, X):
        c = X[self.column].to_numpy(float)
        cls = np.where(c > self.theta, EMERGING, np.where(c < -self.theta, DECLINING, STABLE))
        cls[np.isnan(c)] = STABLE        # no history change available: count is reported separately
        out = np.zeros((len(X), 3))
        out[np.arange(len(X)), cls] = 1.0
        return out
