import sys
from pathlib import Path

try:
    import torch  # noqa: F401  - load torch DLLs before sklearn on Windows
except (ImportError, OSError):
    pass

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
import pytest  # noqa: E402


def make_panel(n_weeks=40, styles=("A", "B"), seed=0, gaps=True):
    """Complete style-week grid with a few inactive weeks (A = 0, Sales = NaN)."""
    rng = np.random.default_rng(seed)
    rows = []
    for s in styles:
        for t in range(n_weeks):
            a = int(rng.integers(1, 6))
            if gaps and s == "B" and 10 <= t < 13:
                a = 0
            rows.append({"style": s, "t_idx": t, "active_products": a,
                         "sales": float(rng.integers(0, 30)) if a else np.nan,
                         "shop_exposure": a * 3, "price": rng.uniform(0.05, 0.1) if a else np.nan,
                         "discount": rng.uniform(0, 0.3) if a else np.nan,
                         "gt_color": rng.uniform(10, 100)})
    p = pd.DataFrame(rows)
    p["spap"] = (p["sales"] / p["active_products"]).where(p["active_products"] > 0)
    p["week_start"] = pd.Timestamp("2017-01-02") + pd.to_timedelta(7 * p["t_idx"], unit="D")
    return p


@pytest.fixture
def panel():
    return make_panel()
