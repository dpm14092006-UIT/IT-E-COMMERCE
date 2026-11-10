"""4.10 - predictive attributions (associations, not causal effects).

SHAP needs `pip install shap`; permutation importance on Macro-F1 works with any fitted model."""
import numpy as np
import pandas as pd

from ..evaluation.metrics import macro_f1


def permutation_importance(model, X, y, n_repeats=10, seed=0):
    rng = np.random.default_rng(seed)
    base = macro_f1(y, model.predict_proba(X).argmax(1))
    rows = []
    for col in X.columns:
        drops = []
        for _ in range(n_repeats):
            Xp = X.copy()
            Xp[col] = rng.permutation(Xp[col].to_numpy())
            drops.append(base - macro_f1(y, model.predict_proba(Xp).argmax(1)))
        rows.append({"feature": col, "importance_mean": np.mean(drops), "importance_sd": np.std(drops)})
    return pd.DataFrame(rows).sort_values("importance_mean", ascending=False, ignore_index=True)


def shap_values(model, X):
    import shap
    return shap.TreeExplainer(model.estimator).shap_values(X)
