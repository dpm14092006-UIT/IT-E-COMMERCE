"""Nguoi 4 - decision layer (4.10). Turns E/S/D probabilities into review signals for buyers,
merchandisers and planners. It never outputs purchase quantities or replenishment orders.

Signal thresholds (min_prob, min_margin) are chosen on VALIDATION predictions, then frozen.
"""
import numpy as np
import pandas as pd

from ..constants import DECLINING, EMERGING, PROBA_COLS, STABLE

SIGNALS = {
    "emerging_alert": "Uu tien theo doi; can nhac mo rong assortment cua style",
    "declining_alert": "Ra soat nhap moi; can nhac markdown / giam do phu",
    "hold": "Duy tri; theo doi dinh ky",
}


def decision_signals(pred, min_prob=0.5, min_margin=0.1):
    p = pred[list(PROBA_COLS)].to_numpy()
    top = p.argmax(1)
    srt = np.sort(p, axis=1)
    margin = srt[:, -1] - srt[:, -2]
    confident = (p.max(1) >= min_prob) & (margin >= min_margin)
    out = pred.copy()
    out["signal"] = np.where(confident & (top == EMERGING), "emerging_alert",
                             np.where(confident & (top == DECLINING), "declining_alert", "hold"))
    out["confidence"], out["margin"] = p.max(1), margin
    return out


def _rate(num, den):
    return float(num / den) if den else float("nan")


def alert_quality(sig):
    """Missed emerging signals and false trend alerts (4.10), on scorable rows only."""
    s = sig[sig["y_true"].notna()]
    y = s["y_true"].astype(int).to_numpy()
    e, d = (s["signal"] == "emerging_alert").to_numpy(), (s["signal"] == "declining_alert").to_numpy()
    return {
        "n": int(len(s)),
        "emerging_alerts": int(e.sum()),
        "emerging_alert_precision": _rate((e & (y == EMERGING)).sum(), e.sum()),
        "emerging_recall": _rate((e & (y == EMERGING)).sum(), (y == EMERGING).sum()),
        "false_emerging_alerts": int((e & (y != EMERGING)).sum()),
        "missed_emerging_as_stable": int(((y == EMERGING) & (s["y_pred"].to_numpy() == STABLE)).sum()),
        "declining_alerts": int(d.sum()),
        "declining_alert_precision": _rate((d & (y == DECLINING)).sum(), d.sum()),
        "declining_recall": _rate((d & (y == DECLINING)).sum(), (y == DECLINING).sum()),
    }


def scenario_table(pred, prob_grid=(0.4, 0.5, 0.6, 0.7), margin_grid=(0.0, 0.1, 0.2)):
    """Scenario analysis: alert volume versus alert precision for each operating point."""
    rows = [{"min_prob": p, "min_margin": m, **alert_quality(decision_signals(pred, p, m))}
            for p in prob_grid for m in margin_grid]
    return pd.DataFrame(rows)
