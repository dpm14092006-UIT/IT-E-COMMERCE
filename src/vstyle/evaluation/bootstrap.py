"""4.9.6 - temporal block bootstrap over forecast origins (all styles of an origin move together)."""
import numpy as np

from ..constants import TIME
from .metrics import macro_f1


def block_bootstrap(df, stat, block_length=4, n_boot=2000, seed=0, origin_col=TIME):
    origins = np.sort(df[origin_col].unique())
    n = len(origins)
    pos = {o: i for i, o in enumerate(origins)}
    groups = [[] for _ in range(n)]
    for row, o in enumerate(df[origin_col].to_numpy()):
        groups[pos[o]].append(row)
    groups = [np.array(g) for g in groups]
    L = min(block_length, n)
    rng = np.random.default_rng(seed)
    out = np.empty(n_boot)
    for b in range(n_boot):
        starts = rng.integers(0, n - L + 1, int(np.ceil(n / L)))
        sel = (starts[:, None] + np.arange(L)[None, :]).ravel()[:n]
        out[b] = stat(df.iloc[np.concatenate([groups[i] for i in sel])])
    return out


def paired_macro_f1_difference(df, pred_a, pred_b, y_col="y_true", alpha=0.05, **kw):
    """(69)-(70) d = MacroF1_A - MacroF1_B on the same rows; percentile CI of the bootstrap draws."""
    def d(x):
        y = x[y_col].astype(int)
        return macro_f1(y, x[pred_a]) - macro_f1(y, x[pred_b])
    draws = block_bootstrap(df, d, **kw)
    return {"estimate": float(d(df)), "ci_low": float(np.quantile(draws, alpha / 2)),
            "ci_high": float(np.quantile(draws, 1 - alpha / 2)), "n_boot": len(draws),
            "n_origins": int(df[kw.get("origin_col", TIME)].nunique())}
