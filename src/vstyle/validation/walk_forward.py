"""4.6.2-4.6.3 - expanding-window walk-forward with the label-maturity rule s + H <= t."""
import pandas as pd

from ..constants import PROBA_COLS, STYLE, TIME
from ..utils.leakage import assert_label_maturity, assert_no_target_columns


def run_walk_forward(data, feature_cols, make_model, origins, horizon, refit_every=1, label_col="y"):
    """(54)-(56). data needs: style, t_idx, eligible, label_col and the feature columns."""
    assert_no_target_columns([c for c in feature_cols if c not in (STYLE, TIME)])
    eligible = data["eligible"].astype(bool)
    labelled = eligible & data[label_col].notna()
    frames, model, fit_origin, n_train = [], None, None, 0
    for k, t in enumerate(origins):
        if model is None or k % refit_every == 0:
            train = data[labelled & (data[TIME] + horizon <= t)]
            if train.empty:
                continue
            assert_label_maturity(train[TIME], t, horizon)
            model = make_model()
            model.fit(train[feature_cols], train[label_col].astype(int).to_numpy())
            fit_origin, n_train = t, len(train)
        test = data[eligible & (data[TIME] == t)]
        if test.empty or model is None:
            continue
        proba = model.predict_proba(test[feature_cols])
        frame = test[[STYLE, TIME]].copy()
        frame["y_true"] = test[label_col].to_numpy()
        frame[list(PROBA_COLS)] = proba
        frame["y_pred"] = proba.argmax(1)
        frame["fit_origin"], frame["n_train"] = fit_origin, n_train
        frames.append(frame)
    return pd.concat(frames, ignore_index=True) if frames else pd.DataFrame()
