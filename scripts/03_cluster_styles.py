"""G2 | Nguoi 2 (--encoder fashionclip) + Nguoi 4 (--encoder resnet50) - K-Means grid on the development catalogue, stability, coverage, contact sheets (4.1.3-4.1.4).

Outputs: outputs/clusters/{encoder}/grid.csv, assignments_k{K}.parquet, centroids_k{K}.npy, sheets/
The final (encoder, K) is a human decision recorded in docs/decision_log.md and configs (visual.selected).
"""
import numpy as np
import pandas as pd
from _common import log, parse

from vstyle.config import path
from vstyle.io import load_json, load_table, save_json, save_table
from vstyle.panel.style_week import build_style_week
from vstyle.target.eligibility import forecast_eligible
from vstyle.target.labels import add_market_response
from vstyle.utils.leakage import assert_fit_scope
from vstyle.utils.repro import write_manifest
from vstyle.validation.splits import catalogue_ids, origin_sets
from vstyle.visual import clustering as cl
from vstyle.visual.inspection import contact_sheet, representatives


def dev_coverage(pw, style_of, cal, cfg, dev_end):
    """Eligible development origins per style for a candidate partition (feasibility, 4.1.3)."""
    panel = build_style_week(pw[pw["t_idx"] <= dev_end], style_of, dev_end, cal["origin_week"])
    t = cfg["target"]
    panel = add_market_response(panel, t["past_window"], t["horizon"], t["min_active_weeks_per_window"])
    panel["eligible"] = forecast_eligible(panel, cfg["eligibility"]["min_history_weeks"])
    return panel.groupby("style")["eligible"].sum()


if __name__ == "__main__":
    args, cfg = parse(__doc__, lambda ap: ap.add_argument("--encoder", required=True,
                                                          choices=["fashionclip", "resnet50", "resnet50ft"]))
    km = cfg["visual"]["kmeans"]
    seed = cfg["project"]["seed"]
    interim = path(cfg, "interim")
    products, pw = load_table(interim / "products.parquet"), load_table(interim / "product_week.parquet")
    products = products[products["image_ok"]].reset_index(drop=True)          # D14
    cal = load_json(interim / "calendar.json")
    sets = origin_sets(cfg["partition"], cal["origin_week"], cfg["target"]["horizon"], cal["t_max"])
    X = np.load(path(cfg, "embeddings") / f"{args.encoder}.npy")
    ids = load_table(path(cfg, "embeddings") / f"{args.encoder}_ids.parquet")["external_code"].to_numpy()
    assert (ids == products["external_code"].to_numpy()).all()

    dev_ids = set(catalogue_ids(products, pd.Timestamp(cfg["partition"]["catalogue_cutoff"])))
    dev = np.array([i in dev_ids for i in ids])
    assert_fit_scope(ids[dev], dev_ids, "K-Means centroids")
    log(f"{args.encoder}: fitting on {dev.sum()} development products of {len(ids)}")

    grid = cl.evaluate_grid(X[dev], km["k_grid"], seed, km["stability_seeds"], km["n_init"], km["max_iter"])
    out = path(cfg, "clusters") / args.encoder
    (out / "sheets").mkdir(parents=True, exist_ok=True)
    cov_rows, agreement = [], {}
    for k in km["k_grid"]:
        model = cl.fit_kmeans(X[dev], k, seed, km["n_init"], km["max_iter"])
        labels = cl.assign(X, model.cluster_centers_)                   # (11) frozen centroids for all
        style_of = pd.Series(labels, index=ids)
        eligible = dev_coverage(pw, style_of, cal, cfg, sets["development_end"])
        cov_rows.append({"k": k, "min_eligible_dev_origins": int(eligible.min()),
                         "median_eligible_dev_origins": float(eligible.median())})
        agreement[k] = cl.attribute_agreement(labels[dev], products[dev])
        prof = cl.profile(labels, products).merge(
            eligible.rename("eligible_dev_origins").rename_axis("style").reset_index(), on="style", how="left")
        prof.to_csv(out / f"profile_k{k}.csv", index=False)
        np.save(out / f"centroids_k{k}.npy", model.cluster_centers_)
        save_table(pd.DataFrame({"external_code": ids, "style": labels, "in_dev_catalogue": dev}),
                   out / f"assignments_k{k}.parquet")
        reps = representatives(X, labels, model.cluster_centers_, **{
            "n_near": cfg["visual"]["inspection"]["n_near_centroid"],
            "n_random": cfg["visual"]["inspection"]["n_random"], "seed": seed})
        images = path(cfg, "images")
        for c, r in reps.items():
            contact_sheet([images / products["image_path"].iloc[i] for i in r["near"] + r["random"]],
                          out / "sheets" / f"k{k}_style{c:02d}.png")
    table = cl.passing_candidates(grid, pd.DataFrame(cov_rows), cfg["visual"]["selection"])
    table.to_csv(out / "grid.csv", index=False)
    save_json(agreement, out / "attribute_agreement.json")
    write_manifest(f"03_cluster_{args.encoder}", cfg, outputs=[out / "grid.csv"])
    log("\n" + table.round(3).to_string(index=False))
    log("Next: inspect sheets/, then set visual.selected and record the decision (G2).")
