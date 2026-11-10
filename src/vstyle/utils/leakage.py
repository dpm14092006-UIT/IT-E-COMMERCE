"""Guards that fail loudly when information from after the forecast origin is used (4.6.5)."""
import numpy as np

from ..constants import FORBIDDEN_FEATURE_PREFIXES


class LeakageError(AssertionError):
    pass


def assert_label_maturity(train_origins, t, horizon):
    """(57)-(58): every training row must satisfy s + H <= t."""
    s = np.asarray(train_origins)
    bad = s + horizon > t
    if bad.any():
        raise LeakageError(f"{bad.sum()} training rows have immature labels at origin {t} "
                           f"(max s={s.max()}, need s <= {t - horizon})")


def assert_fit_scope(fit_ids, allowed_ids, what):
    """Centroids, scalers, thresholds: fitted only on the allowed (historical) subset."""
    extra = set(fit_ids) - set(allowed_ids)
    if extra:
        raise LeakageError(f"{what} fitted on {len(extra)} ids outside the allowed scope")


def assert_no_target_columns(feature_cols):
    bad = [c for c in feature_cols if c.split("__")[0].startswith(FORBIDDEN_FEATURE_PREFIXES)]
    if bad:
        raise LeakageError(f"target-derived columns used as features: {bad}")


def future_perturbation_check(build, panel, t, value_cols, output_cols, time_col="t_idx", seed=0):
    """Rebuild after scrambling every value strictly after t; outputs at <= t must not move.

    `build(panel) -> DataFrame` must return rows aligned with `panel`.
    Returns the list of output columns that changed (empty list = no leakage).
    """
    rng = np.random.default_rng(seed)
    base = build(panel.copy())
    perturbed = panel.copy()
    future = perturbed[time_col] > t
    for c in value_cols:
        perturbed[c] = perturbed[c].astype(float)
        perturbed.loc[future, c] = rng.uniform(0, 1000, future.sum())
    after = build(perturbed)
    past = (panel[time_col] <= t).to_numpy()
    changed = []
    for c in output_cols:
        a, b = base.loc[past, c].to_numpy(float), after.loc[past, c].to_numpy(float)
        if not np.allclose(a, b, equal_nan=True):
            changed.append(c)
    return changed
