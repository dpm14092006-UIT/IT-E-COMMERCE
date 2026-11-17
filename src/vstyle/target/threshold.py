"""4.4.3 - threshold calibration on matured development labels only."""
import numpy as np

from ..constants import CLASSES


def calibrate(delta, t_idx, calibration_end_t, horizon, rule="median_abs", fallback_quantile=0.5):
    """(41) theta = median |Delta| over rows whose label matured by calibration_end_t (s + H <= end)."""
    mask = delta.notna() & (t_idx + horizon <= calibration_end_t)
    a = delta[mask].abs()
    if a.empty:
        raise ValueError("no matured labels before the calibration end")
    theta, used = float(a.median()), rule
    if not theta > 0:
        theta, used = float(a[a > 0].quantile(fallback_quantile)), "fallback_positive_quantile"
    return {"theta": theta, "rule": used, "n_calibration": int(mask.sum()),
            "last_origin_used": int(t_idx[mask].max()), "calibration_end_t": int(calibration_end_t)}


def quantile_threshold(delta, t_idx, calibration_end_t, horizon, q):
    """4.9.2 - alternative thresholds, robustness only."""
    mask = delta.notna() & (t_idx + horizon <= calibration_end_t)
    return float(delta[mask].abs().quantile(q))


def class_distribution(y):
    y = np.asarray(y, dtype=float)
    y = y[~np.isnan(y)].astype(int)
    counts = np.bincount(y, minlength=3)
    return {c: {"n": int(n), "share": float(n / max(counts.sum(), 1))} for c, n in zip(CLASSES, counts)}
