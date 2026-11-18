"""Common contract for every forecasting model, so walk-forward code depends on one abstraction."""
from abc import ABC, abstractmethod

import numpy as np


class ForecastModel(ABC):
    """fit on a labelled window, then return (n, 3) class probabilities in CLASSES order."""

    @abstractmethod
    def fit(self, X, y) -> "ForecastModel":
        """Train on matured labels only and return self."""

    @abstractmethod
    def predict_proba(self, X) -> np.ndarray:
        """Probabilities of shape (len(X), 3)."""

    def predict(self, X) -> np.ndarray:
        return self.predict_proba(X).argmax(1)
