"""4.7.2-4.7.3 - classification metrics with a fixed 3-class label set."""
import numpy as np
from sklearn.metrics import confusion_matrix, f1_score, precision_recall_fscore_support

from ..constants import CLASSES

LABELS = [0, 1, 2]


def macro_f1(y_true, y_pred, zero_division=0):
    """(64) averaged over all three classes, even when one is absent (reported separately)."""
    if len(y_true) == 0:
        return float("nan")
    return float(f1_score(y_true, y_pred, labels=LABELS, average="macro", zero_division=zero_division))


def summary(y_true, y_pred, zero_division=0):
    y_true, y_pred = np.asarray(y_true, dtype=int), np.asarray(y_pred, dtype=int)
    p, r, f, s = precision_recall_fscore_support(y_true, y_pred, labels=LABELS, zero_division=zero_division)
    present = s > 0
    return {
        "n": int(len(y_true)),
        "macro_f1": float(f.mean()),
        "balanced_accuracy": float(r[present].mean()) if present.any() else float("nan"),   # (65)
        "accuracy": float((y_true == y_pred).mean()) if len(y_true) else float("nan"),
        "absent_classes": [c for c, ok in zip(CLASSES, present) if not ok],
        "per_class": {c: {"precision": float(p[i]), "recall": float(r[i]), "f1": float(f[i]), "support": int(s[i])}
                      for i, c in enumerate(CLASSES)},
        "confusion_matrix": confusion_matrix(y_true, y_pred, labels=LABELS).tolist(),   # rows = true
    }
