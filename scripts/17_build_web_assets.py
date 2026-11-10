"""Nguoi 4 - training/packaging step for the Style Scanner web app (web/app.py).

What is trained here (once, offline) so the web app only has to run inference:
  * deployment TCN (tuned params, several seeds) + CatBoost, fitted on every label that has matured
    by the last observed week; they forecast the origins after the final walk-forward origin.
  * for earlier origins the app shows the walk-forward test predictions (each one made by a model
    that only saw data before that week), so every forecast shown is out-of-sample in time.
Also packaged: frozen K-Means centroids, catalogue embeddings (kNN), distance statistics used to flag
out-of-distribution uploads, style names/profiles, weekly style history.

Output: outputs/web/{grouping}/  (assets.npz, meta.json, tcn_seed*.pt, catboost.cbm)
Run once per encoder shown in the app: --grouping resnet50, --grouping fashionclip.
"""
import json

import numpy as np
import pandas as pd
from _common import log, parse

from vstyle.config import load_yaml, path
from vstyle.constants import PROBA_COLS
from vstyle.features.blocks import select_features
from vstyle.io import load_json, load_table
from vstyle.models.sequence import SequenceClassifier, build_tcn
from vstyle.models.tabular import make_model
from vstyle.visual.encoders import l2_normalize

MODELS = ("tcn", "catboost")


def deployment_forecasts(data, cols, cfg, params_from, seeds, out_dir):
    """Fit on all matured labels, predict every eligible origin after the last labelled one."""
    import torch
    base = cfg["project"]["seed"]
    eligible = data["eligible"].astype(bool)
    labelled = eligible & data["y"].notna()
    t_last_label = int(data.loc[labelled, "t_idx"].max())
    train = data[labelled]
    new = data[eligible & (data["t_idx"] > t_last_label)]
    log(f"deployment fit: {len(train):,} labelled rows (t <= {t_last_label}), forecasting {len(new)} rows")
    spaces = load_yaml("models.yaml")
    frames = []

    p = dict(spaces["tcn"]["defaults"])
    p.update(load_json(path(cfg, "tuning") / f"{params_from}_tcn_best.json")["params"])
    net = lambda n: build_tcn(n, p["channels"], p["kernel_size"], p["dilations"], p["dropout"])  # noqa: E731
    probs = []
    for r in range(seeds):
        m = SequenceClassifier(data, cols, p["lookback"], net, p["learning_rate"], p["batch_size"],
                               p["max_epochs"], p["patience"], seed=base + r)
        m.fit(train[["style", "t_idx"]], train["y"].astype(int).to_numpy())
        torch.save(m.net_.state_dict(), out_dir / f"tcn_seed{r}.pt")
        probs.append(m.predict_proba(new[["style", "t_idx"]]))
        log(f"tcn seed {base + r} trained")
    frames.append(new[["style", "t_idx"]].assign(model="tcn").reset_index(drop=True)
                  .join(pd.DataFrame(np.mean(probs, 0), columns=list(PROBA_COLS))))

    best = load_json(path(cfg, "tuning") / f"{params_from}_catboost_best.json")
    cb = make_model("catboost", best["params"], base, best["class_weighting"])
    cb.fit(train[cols], train["y"].astype(int).to_numpy())
    cb.estimator.save_model(str(out_dir / "catboost.cbm"))
    frames.append(new[["style", "t_idx"]].assign(model="catboost").reset_index(drop=True)
                  .join(pd.DataFrame(cb.predict_proba(new[cols]), columns=list(PROBA_COLS))))
    log("catboost trained")
    out = pd.concat(frames, ignore_index=True)
    out["y_true"] = np.nan
    out["source"] = "deployment"
    return out, t_last_label


if __name__ == "__main__":
    def extra(ap):
        ap.add_argument("--grouping", default="resnet50")
        ap.add_argument("--seeds", type=int, default=3, help="TCN seeds for the deployment model")
        ap.add_argument("--params-from", default="resnet50", help="grouping whose tuned hyperparameters are used")
        ap.add_argument("--default-origin", default="2019-12-30",
                        help="origin shown first in the app (before end-of-data censoring)")
    args, cfg = parse(__doc__, extra)
    enc, k = args.grouping, cfg["visual"]["selected"]["k"]
    out_dir = path(cfg, "reports").parent / "web" / enc
    out_dir.mkdir(parents=True, exist_ok=True)
    cal = load_json(path(cfg, "interim") / "calendar.json")
    week0 = pd.Timestamp(cal["origin_week"])

    # --- visual side: frozen centroids + catalogue embeddings -------------------------------
    ids = load_table(path(cfg, "embeddings") / f"{enc}_ids.parquet")["external_code"].to_numpy()
    Z = l2_normalize(np.load(path(cfg, "embeddings") / f"{enc}.npy"))
    C = np.load(path(cfg, "clusters") / enc / f"centroids_k{k}.npy")
    a = load_table(path(cfg, "clusters") / enc / f"assignments_k{k}.parquet").set_index("external_code")
    style = a.loc[ids, "style"].to_numpy()
    sq = (Z ** 2).sum(1)[:, None] - 2 * Z @ C.T + (C ** 2).sum(1)[None, :]
    dist = np.sqrt(np.maximum(sq, 0))
    d_own = dist[np.arange(len(Z)), style]
    ood_threshold = float(np.quantile(dist.min(1), 0.99))
    novel_radius = float(np.median(d_own))      # radius for grouping unseen uploads into new styles

    products = load_table(path(cfg, "interim") / "products.parquet").set_index("external_code").loc[ids]
    prof = pd.read_csv(path(cfg, "clusters") / enc / f"profile_k{k}.csv").set_index("style")
    names_csv = path(cfg, "clusters") / enc / f"style_names_k{k}_draft.csv"
    if names_csv.exists():
        names = pd.read_csv(names_csv).set_index("style")
    else:                                       # no hand-written names: use the dominant attributes
        first = lambda col: prof[col].str.split(";").str[0].str.rsplit(" ", n=1).str[0]  # noqa: E731
        names = pd.DataFrame({"draft_name": first("top_category") + " · " + first("top_color"),
                              "evidence": prof["top_category"].str.split(";").str[0]})
    styles = []
    for s in range(k):
        members = np.where(style == s)[0]
        rep = members[np.argsort(d_own[members])[:6]]
        styles.append({"id": s, "name": names.loc[s, "draft_name"], "evidence": names.loc[s, "evidence"],
                       "n_products": int(len(members)),
                       "top_category": prof.loc[s, "top_category"], "top_color": prof.loc[s, "top_color"],
                       "top_fabric": prof.loc[s, "top_fabric"], "top_season": prof.loc[s, "top_season"],
                       "dist_sorted": np.round(np.sort(d_own[members]), 4).tolist(),
                       "q99": round(float(np.quantile(d_own[members], 0.99)), 4),
                       "examples": [int(i) for i in ids[rep]]})

    # --- forecasting side ------------------------------------------------------------------
    data = load_table(path(cfg, "features") / f"features_{enc}.parquet")
    data["style"] = data["style"].astype(int)
    cols = select_features(data.columns, cfg["features"]["primary_blocks"])
    deploy, t_last_label = deployment_forecasts(data, cols, cfg, args.params_from, args.seeds, out_dir)
    wf = load_table(path(cfg, "predictions") / f"{enc}_test_primary.parquet")
    wf = wf[wf["model"].isin(MODELS)].assign(source="walk_forward")
    wf["style"] = wf["style"].astype(int)
    fc = pd.concat([wf[["style", "t_idx", "model", *PROBA_COLS, "y_true", "source"]],
                    deploy[deploy["t_idx"] > wf["t_idx"].max()]], ignore_index=True)
    forecasts = {}
    for r in fc.itertuples():
        forecasts.setdefault(str(int(r.t_idx)), {}).setdefault(str(r.style), {})[r.model] = {
            "p": [round(float(getattr(r, c)), 4) for c in PROBA_COLS],
            "y_true": None if pd.isna(r.y_true) else int(r.y_true), "source": r.source}
    origins = sorted(int(t) for t in forecasts)
    default_t = int((pd.Timestamp(args.default_origin) - week0).days // 7)
    default_t = min(origins, key=lambda t: abs(t - default_t))

    panel = load_table(path(cfg, "panels") / f"style_week_{enc}.parquet")
    panel["style"] = panel["style"].astype(int)
    hist = {}
    for s, g in panel.groupby("style"):
        g = g.set_index("t_idx").reindex(range(int(cal["t_max"]) + 1))
        hist[str(s)] = {"spap": g["spap"].fillna(0).round(4).tolist(),
                        "active": g["active_products"].fillna(0).astype(int).tolist()}

    metrics = load_json(path(cfg, "metrics") / f"{enc}_test_primary.json")["models"]
    meta = {"encoder": enc, "k": k, "week0": str(week0.date()), "t_max": int(cal["t_max"]),
            "horizon": cfg["target"]["horizon"], "classes": ["Declining", "Stable", "Emerging"],
            "ood_threshold": round(ood_threshold, 4), "novel_radius": round(novel_radius, 4),
            "params_from": args.params_from, "styles": styles, "forecasts": forecasts,
            "origins": origins, "default_origin": default_t, "t_last_label": t_last_label,
            "history": hist,
            "metrics": {m: {"macro_f1": round(metrics[m]["macro_f1"], 3),
                            "balanced_accuracy": round(metrics[m]["balanced_accuracy"], 3)}
                        for m in (*MODELS, "persistence", "majority") if m in metrics},
            "n_catalogue": int(len(ids)), "tcn_seeds": args.seeds}
    (out_dir / "meta.json").write_text(json.dumps(meta, ensure_ascii=False), encoding="utf-8")
    np.savez_compressed(out_dir / "assets.npz", centroids=C.astype(np.float32), Z=Z.astype(np.float16),
                        ids=ids, style=style, image_path=products["image_path"].to_numpy().astype(str),
                        category=products["category"].to_numpy().astype(str),
                        color=products["color"].to_numpy().astype(str),
                        season=products["season"].to_numpy().astype(str))
    log(f"wrote {out_dir} | origins {origins[0]}..{origins[-1]}, default {default_t}, OOD d>{ood_threshold:.3f}")
