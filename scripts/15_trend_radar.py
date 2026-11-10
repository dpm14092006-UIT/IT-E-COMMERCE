"""Nguoi 2 + Nguoi 4 - Visual Trend Radar: self-contained HTML dashboard of the final-test trend forecasts.

Output: outputs/reports/visual_trend_radar.html (open in any browser, no server needed)
"""
import json
from pathlib import Path

import pandas as pd
from _common import log, parse

from vstyle.config import path
from vstyle.io import load_json, load_table

if __name__ == "__main__":
    def extra(ap):
        ap.add_argument("--grouping", default="resnet50")
        ap.add_argument("--model", default="catboost")
    args, cfg = parse(__doc__, extra)
    origin = pd.Timestamp(load_json(path(cfg, "interim") / "calendar.json")["origin_week"])
    p = load_table(path(cfg, "predictions") / f"{args.grouping}_test_primary.parquet")
    p = p[p["model"] == args.model].copy()
    p["style"] = p["style"].astype(int)
    p["week"] = origin + pd.to_timedelta(7 * p["t_idx"], unit="D")
    p["month"] = p["week"].dt.to_period("M").astype(str)
    k = cfg["visual"]["selected"]["k"]
    names = pd.read_csv(path(cfg, "clusters") / args.grouping / f"style_names_k{k}_draft.csv").set_index("style")["draft_name"]
    s = p[p["y_true"].notna()]
    acc = s.groupby("style").apply(lambda x: (x["y_pred"] == x["y_true"]).mean(), include_groups=False)
    mp = s.groupby(["style", "month"])["y_pred"].agg(lambda x: x.mode().iloc[0]).unstack()
    mt = s.groupby(["style", "month"])["y_true"].agg(lambda x: x.mode().iloc[0]).unstack().reindex_like(mp)

    feats = load_table(path(cfg, "features") / f"features_{args.grouping}.parquet")[["style", "t_idx", "spap__growth4"]]
    feats["style"] = feats["style"].astype(int)
    last_t = int(p["t_idx"].max())
    last = p[p["t_idx"] == last_t].merge(feats, on=["style", "t_idx"], how="left")
    latest = [{"name": names[r.style], "cls": "DSE"[int(r.y_pred)], "pd": round(r.p_declining, 3),
               "ps": round(r.p_stable, 3), "pe": round(r.p_emerging, 3),
               "g": None if pd.isna(r.spap__growth4) else round(float(r.spap__growth4), 3),
               "acc": round(float(acc.get(r.style, float("nan"))), 3)}
              for r in last.sort_values("p_emerging", ascending=False).itertuples()]
    data = {"months": list(mp.columns), "names": [names[i] for i in mp.index],
            "pred": mp.fillna(-1).astype(int).values.tolist(), "true": mt.fillna(-1).astype(int).values.tolist(),
            "acc": acc.loc[mp.index].round(3).tolist(), "latest": latest}

    m = load_json(path(cfg, "metrics") / f"{args.grouping}_test_primary.json")["models"]
    dec = load_json(path(cfg, "reports") / f"{args.grouping}_decision.json")["quality"]
    pct = lambda x: f"{x * 100:.0f}%"  # noqa: E731
    fill = {"__DATA__": json.dumps(data, ensure_ascii=False), "__NSTYLES__": str(len(mp)),
            "__ENCODER__": {"resnet50": "ResNet-50"}.get(args.grouping, args.grouping), "__K__": str(k),
            "__MODEL__": args.model, "__START__": str(s["week"].min().date()), "__END__": str(s["week"].max().date()),
            "__LAST__": str((origin + pd.Timedelta(weeks=last_t)).date()),
            "__F1__": f"{m[args.model]['macro_f1']:.3f}", "__PF1__": f"{m['persistence']['macro_f1']:.3f}",
            "__ACC__": pct((s["y_pred"] == s["y_true"]).mean()), "__N__": f"{len(s):,}",
            "__PE__": pct(dec["emerging_alert_precision"]), "__RE__": pct(dec["emerging_recall"]),
            "__PD__": pct(dec["declining_alert_precision"]), "__RD__": pct(dec["declining_recall"])}
    html = (Path(__file__).parent / "templates" / "trend_radar.html").read_text(encoding="utf-8")
    for key, val in fill.items():
        html = html.replace(key, val)
    out = path(cfg, "reports") / "visual_trend_radar.html"
    out.write_text(html, encoding="utf-8")
    log(f"wrote {out}")
