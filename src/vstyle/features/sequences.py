"""4.3.6 - multivariate sequences (33) for TCN, with masks that separate missing from zero."""
import numpy as np

from ..constants import STYLE, TIME


def build_sequences(panel, feature_cols, rows, lookback):
    """X[n, L, p] holding weeks t-L+1..t for each requested (style, t) row; NaN outside the style."""
    panel = panel.sort_values([STYLE, TIME]).reset_index(drop=True)
    arr = panel[feature_cols].to_numpy(float)
    start = panel.groupby(STYLE).cumcount().eq(0)
    first_pos = panel.index.to_series()[start]
    first_pos.index = panel.loc[start, STYLE].to_numpy()
    t_min = panel.groupby(STYLE)[TIME].min()
    pos = first_pos.loc[rows[STYLE]].to_numpy() + rows[TIME].to_numpy() - t_min.loc[rows[STYLE]].to_numpy()
    idx = pos[:, None] + np.arange(-lookback + 1, 1)[None, :]
    inside = idx >= first_pos.loc[rows[STYLE]].to_numpy()[:, None]
    X = arr[np.clip(idx, 0, len(arr) - 1)]
    X[~inside] = np.nan
    return X, ~np.isnan(X)


def fit_standardizer(X_train):
    """Training-only scaling parameters (4.3.6); validation/test reuse them unchanged."""
    flat = X_train.reshape(-1, X_train.shape[-1])
    mean = np.nanmean(flat, axis=0)
    std = np.nanstd(flat, axis=0)
    return mean, np.where(std > 0, std, 1.0)


def apply_standardizer(X, mean, std):
    Z = (X - mean) / std
    return np.nan_to_num(Z, nan=0.0)      # missingness is carried by the mask
