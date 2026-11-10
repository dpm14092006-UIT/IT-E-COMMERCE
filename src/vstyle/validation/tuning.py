"""4.6.4 - bounded random search on validation origins inside development; Macro-F1 decides (59)."""
import pandas as pd
from sklearn.model_selection import ParameterSampler

from ..evaluation.metrics import macro_f1
from ..models.tabular import make_model
from .walk_forward import run_walk_forward


def search(data, feature_cols, model_name, space, defaults, origins, horizon, n_iter, seed,
           refit_every=4, class_weighting=("none",)):
    rows = []
    for params in ParameterSampler(space, n_iter=n_iter, random_state=seed):
        full = {**defaults, **params}
        for cw in class_weighting:
            preds = run_walk_forward(data, feature_cols,
                                     lambda: make_model(model_name, full, seed, cw),
                                     origins, horizon, refit_every)
            scored = preds[preds["y_true"].notna()]
            rows.append({"model": model_name, "class_weighting": cw, "params": full,
                         "macro_f1": macro_f1(scored["y_true"].astype(int), scored["y_pred"]),
                         "n_scored": len(scored)})
    return pd.DataFrame(rows).sort_values("macro_f1", ascending=False, ignore_index=True)
