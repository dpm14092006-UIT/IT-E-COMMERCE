"""Catalogue read-model: product, category and company views plus the style forecasts behind them.

Everything here reads data that already exists; nothing is re-trained:
  * catalogue + weekly sales      outputs/interim/products.parquet, product_week.parquet (scripts/01)
  * product -> style              outputs/web/resnet50/assets.npz (frozen K-Means, scripts/17)
  * style forecasts               outputs/web/{encoder}/meta.json - walk-forward test predictions, then deployment
  * review signals                outputs/reports/resnet50_decision.json - operating point chosen on VALIDATION (4.10)
A product's forecast is the forecast of its style: the models never forecast single products.
"""
import json
from pathlib import Path

import numpy as np
import pandas as pd
from fastapi import HTTPException

from .zero_shot import CATALOGUE_VI

PRIMARY = "resnet50"            # primary grouping of the study (configs/default.yaml)
MODEL = "catboost"              # the model the decision operating point was chosen for
TREND_VI = ["Đang giảm", "Ổn định", "Đang lên"]
SIGNAL_VI = {"emerging_alert": "Ưu tiên mở rộng", "declining_alert": "Rà soát nhập hàng",
             "hold": "Duy trì, theo dõi"}
SIGNAL_ACTION = {"emerging_alert": "Ưu tiên theo dõi; cân nhắc mở rộng assortment của style.",
                 "declining_alert": "Rà soát nhập mới; cân nhắc markdown hoặc giảm độ phủ.",
                 "hold": "Duy trì; theo dõi định kỳ."}
DEFAULT_WEEK = "2019-10-14"     # mid AW19 season, labels observed; chosen by season, not by accuracy
LATE_FROM = "2019-12-02"        # launches dry up (last release 2019-12-30): every style drifts down after this
LIFECYCLE = 12                 # VISUELLE records 12 weeks of sales per product


def vi(value):
    return CATALOGUE_VI.get(str(value), str(value))


class Catalogue:
    """In-memory product, category and company views; built once at start-up (a few seconds)."""

    def __init__(self, root: Path, cfg, spaces):
        self.root, self.cfg, self.spaces = root, cfg, spaces
        interim = root / cfg["paths"]["interim"]
        products = pd.read_parquet(interim / "products.parquet")
        pw = pd.read_parquet(interim / "product_week.parquet",
                             columns=["external_code", "t_idx", "sales", "price", "discount", "n_shops"])
        meta = spaces[PRIMARY].meta
        self.meta, self.t_max, self.week0 = meta, int(meta["t_max"]), pd.Timestamp(meta["week0"])
        sp = spaces[PRIMARY]
        style_of = dict(zip(sp.ids.tolist(), sp.style_of.tolist()))
        stats = pw.groupby("external_code").agg(units=("sales", "sum"), first_t=("t_idx", "min"),
                                                last_t=("t_idx", "max"), weeks=("t_idx", "nunique"),
                                                price=("price", "mean"), discount=("discount", "mean"),
                                                peak_shops=("n_shops", "max"))
        p = products.set_index("external_code").join(stats, how="left")
        p["units"] = p["units"].fillna(0)
        p["style"] = [style_of.get(int(c), -1) for c in p.index]
        p["release"] = p["release_first"].dt.strftime("%Y-%m-%d")
        self.products = p.sort_index()
        self.pw = pw.sort_values(["external_code", "t_idx"])
        self.shops = int(pd.read_parquet(interim / "item_shop_week.parquet", columns=["retail"])["retail"].nunique())
        self.weekly = self._series(pw)
        self.active = pw.groupby("t_idx")["external_code"].nunique().reindex(range(self.t_max + 1), fill_value=0)
        self.categories = sorted(p["category"].unique(), key=lambda c: -p.loc[p["category"] == c, "units"].sum())
        self.cat_weekly = {c: self._series(pw[pw["external_code"].isin(p.index[p["category"] == c])])
                           for c in self.categories}
        op = root / cfg["paths"]["reports"] / f"{PRIMARY}_decision.json"
        decision = json.loads(op.read_text(encoding="utf-8")) if op.exists() else {}
        self.op = decision.get("operating_point", {"min_prob": 0.5, "min_margin": 0.0})
        self.decision_quality = decision.get("quality")
        to_t = lambda d: int((pd.Timestamp(d) - self.week0).days // 7)  # noqa: E731
        default = to_t(DEFAULT_WEEK)
        self.default_t = default if default in meta["origins"] else int(meta["default_origin"])
        self.late_t = to_t(LATE_FROM)

    def _series(self, frame):
        return frame.groupby("t_idx")["sales"].sum().reindex(range(self.t_max + 1), fill_value=0).round(1).tolist()

    def week(self, t):
        return (self.week0 + pd.Timedelta(weeks=int(t))).strftime("%Y-%m-%d")

    def origin(self, t):
        origins = self.meta["origins"]
        if t is None:
            return self.default_t
        if int(t) not in origins:
            raise HTTPException(400, f"Thời điểm dự báo phải nằm trong tuần {origins[0]}–{origins[-1]}")
        return int(t)

    # ---------- forecasts ----------
    def signal(self, p):
        p = np.asarray(p, dtype=float)
        top, srt = int(p.argmax()), np.sort(p)
        confident = p.max() >= self.op["min_prob"] and srt[-1] - srt[-2] >= self.op["min_margin"]
        return {2: "emerging_alert", 0: "declining_alert"}.get(top, "hold") if confident else "hold"

    def _pack(self, p, extra=None):
        p = [round(float(x), 4) for x in p]
        k, s = int(np.argmax(p)), self.signal(p)
        return {"p": p, "trend": k, "trend_vi": TREND_VI[k], "confidence": max(p),
                "signal": s, "signal_vi": SIGNAL_VI[s], "action": SIGNAL_ACTION[s], **(extra or {})}

    def style_outlook(self, style, t, enc=PRIMARY):
        f = self.spaces[enc].meta["forecasts"].get(str(t), {}).get(str(style))
        if not f or MODEL not in f:
            return None
        m = f[MODEL]
        return self._pack(m["p"], {"style": int(style), "tcn_p": f.get("tcn", {}).get("p"),
                                   "y_true": m.get("y_true"), "source": m.get("source"),
                                   "actual_vi": TREND_VI[m["y_true"]] if m.get("y_true") is not None else None})

    def proxy_outlook(self, ranking, t, enc):
        """A new style has no history: inverse-distance mix of its 3 nearest styles, labelled as a proxy."""
        near = [r for r in ranking if self.style_outlook(r["id"], t, enc)][:3]
        if not near:
            return None
        w = np.array([1 / max(r["distance"], 1e-9) for r in near])
        w /= w.sum()
        p = sum(wi * np.asarray(self.style_outlook(r["id"], t, enc)["p"]) for wi, r in zip(w, near))
        return self._pack(p, {"proxy": True, "parts": [{"style": r["id"], "weight": round(float(wi), 3)}
                                                       for wi, r in zip(w, near)]})

    def is_active(self, codes, t):
        p = self.products.loc[codes]
        return (p["first_t"] <= t) & (p["last_t"] >= t)

    def group_outlook(self, codes, t):
        """Products weighted by their style's forecast; products on sale at t, otherwise all of them."""
        p = self.products.loc[codes]
        p = p[p["style"] >= 0]
        live = p[(p["first_t"] <= t) & (p["last_t"] >= t)]
        base = live if len(live) else p
        weights = base["style"].value_counts()
        rows = [(self.style_outlook(s, t), n) for s, n in weights.items()]
        rows = [(o, n) for o, n in rows if o]
        if not rows:
            return None
        total = sum(n for _, n in rows)
        mix = sum(np.asarray(o["p"]) * n for o, n in rows) / total
        share = {sig: round(sum(n for o, n in rows if o["signal"] == sig) / total, 3) for sig in SIGNAL_VI}
        return self._pack(mix, {"weighted_by": "active" if len(live) else "all", "n_weighted": int(total),
                                "signal_share": share})

    # ---------- views ----------
    def product_row(self, code, t=None):
        r = self.products.loc[code]
        row = {"code": int(code), "category": r["category"], "category_vi": vi(r["category"]),
               "color": r["color"], "color_vi": vi(r["color"]), "fabric": r["fabric"], "season": r["season"],
               "release": r["release"], "units": round(float(r["units"]), 0), "shops": int(r["n_shops"]),
               "style": int(r["style"]), "has_image": bool(r["image_ok"] and r["style"] >= 0),
               "price": None if pd.isna(r["price"]) else round(float(r["price"]), 4),
               "discount": None if pd.isna(r["discount"]) else round(float(r["discount"]), 3)}
        if t is not None:
            row["status"] = ("upcoming" if r["first_t"] > t else "ended" if r["last_t"] < t else "active") \
                if not pd.isna(r["first_t"]) else "no_sales"
            row["outlook"] = self.style_outlook(r["style"], t) if r["style"] >= 0 else None
        return row

    def style_name(self, style, enc=PRIMARY):
        return self.spaces[enc].meta["styles"][int(style)]["name"]

    def breakdown(self, frame, col, top=None, translate=True):
        g = frame.groupby(col)["units"].agg(["sum", "count"]).sort_values("sum", ascending=False)
        total = g["sum"].sum() or 1
        rows = [{"key": k, "label": vi(k) if translate else k, "units": round(float(v["sum"])),
                 "products": int(v["count"]), "share": round(float(v["sum"] / total), 4)} for k, v in g.iterrows()]
        return rows[:top] if top else rows
