"""4.4.4 - forecast eligibility (known at t) versus target observability (known after t + H)."""
from ..constants import STYLE, TIME


def forecast_eligible(panel, min_history_weeks, require_active=True):
    active = panel["active_products"] > 0
    # First active week of the style. The condition t - first >= m - 1 can only hold when
    # first <= t, so the global minimum gives the same answer as a past-only computation.
    first = panel[TIME].where(active).groupby(panel[STYLE]).transform("min")
    ok = panel["pmr_past4"].notna() & (panel[TIME] - first >= min_history_weeks - 1)
    return ok & active if require_active else ok


def coverage(eligible, scorable):
    """(42) Coverage = N_scorable / N_forecasted."""
    n_f, n_s = int(eligible.sum()), int((eligible & scorable).sum())
    return {"n_forecasted": n_f, "n_scorable": n_s, "coverage": n_s / n_f if n_f else float("nan")}
