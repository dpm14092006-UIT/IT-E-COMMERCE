"""Fixed conventions shared by every stage."""

# Class order is fixed everywhere: probability columns, confusion matrices, metrics.
CLASSES = ("Declining", "Stable", "Emerging")
DECLINING, STABLE, EMERGING = 0, 1, 2
PROBA_COLS = ("p_declining", "p_stable", "p_emerging")

KEYS_ITEM_SHOP = ["external_code", "retail"]
PRODUCT = "external_code"
STYLE = "style"
TIME = "t_idx"          # integer calendar-week index (0 = first observed Monday)
WEEK = "week_start"     # Monday of the calendar week

# Columns that hold future information. They may be used to build labels only,
# never as predictors (4.6.5).
FORBIDDEN_FEATURE_PREFIXES = ("pmr_future", "delta", "y", "label")
