"""G2 / RQ1 | Nguoi 2 + Nguoi 4 - FashionCLIP vs ResNet-50 on criteria that are comparable across
embedding spaces (ARI stability, NMI with catalogue attributes, coverage, size). Silhouette and DBI
are reported within each space only (4.7.1).

Outputs: outputs/reports/G2_encoder_comparison.{md,csv,png}
"""
import sys

import matplotlib
import pandas as pd
from _common import ROOT, log, parse

from vstyle.config import path
from vstyle.io import load_json, load_table
from vstyle.visual.inspection import overview_sheet, representatives

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

sys.path.insert(0, str(ROOT))
from figure_style import apply_theme  # noqa: E402

ENCODERS = tuple(e for e in ("fashionclip", "resnet50", "resnet50ft")
                 if (ROOT / "outputs" / "clusters" / e / "grid.csv").exists())

if __name__ == "__main__":
    args, cfg = parse(__doc__)
    frames = []
    for enc in ENCODERS:
        d = path(cfg, "clusters") / enc
        g = pd.read_csv(d / "grid.csv").assign(encoder=enc)
        agree = load_json(d / "attribute_agreement.json")
        for a in ("category", "color", "fabric"):
            g[f"nmi_{a}"] = g["k"].map(lambda k: agree[str(k)][a]["nmi"])
        frames.append(g)
    t = pd.concat(frames, ignore_index=True)
    cols = ["encoder", "k", "silhouette", "davies_bouldin", "ari_mean", "ari_sd", "size_min", "size_max",
            "min_eligible_dev_origins", "nmi_category", "nmi_color", "nmi_fabric",
            "pass_size", "pass_stability", "pass_coverage", "pass_all"]
    t = t[cols]
    out = path(cfg, "reports")
    t.to_csv(out / "G2_encoder_comparison.csv", index=False)

    apply_theme(journal=False)
    fig, axes = plt.subplots(1, 3, figsize=(12, 3.6), constrained_layout=True)
    colors = {"fashionclip": "#219EBC", "resnet50": "#F28E2B", "resnet50ft": "#8A60B0"}
    panels = [("ari_mean", "Stability: mean ARI across seeds", cfg["visual"]["selection"]["min_stability_ari"]),
              ("nmi_category", "NMI with category (higher = re-discovers category)", None),
              ("min_eligible_dev_origins", "Min eligible dev origins per style",
               cfg["visual"]["selection"]["min_eligible_dev_origins_per_style"])]
    for ax, (col, title, ref) in zip(axes, panels):
        for enc, g in t.groupby("encoder"):
            ax.plot(g["k"], g[col], marker="o", color=colors[enc], label=enc)
        if ref is not None:
            ax.axhline(ref, color="#63748B", lw=0.8, ls="--")
        ax.set_title(title)
        ax.set_xlabel("K")
        ax.grid(axis="y")
    axes[0].legend()
    fig.savefig(out / "G2_encoder_comparison.png", dpi=200)

    md = ["# G2 / RQ1 - So sánh FashionCLIP và ResNet-50", "",
          "Silhouette và DBI chỉ so trong cùng một encoder. Giữa hai encoder chỉ so ARI, NMI, coverage, kích thước.",
          "`pass_all` = qua gate đăng ký trước (size, stability, coverage). Chọn cuối cùng cần xem contact sheets.", "",
          t.round(3).to_markdown(index=False), "", "![](G2_encoder_comparison.png)", ""]
    (out / "G2_encoder_comparison.md").write_text("\n".join(md), encoding="utf-8")
    log("\n" + t.round(3).to_string(index=False))

    # one overview per (encoder, K): 8 near-centroid + 4 random images per style row
    import numpy as np
    products = load_table(path(cfg, "interim") / "products.parquet").set_index("external_code")
    images = path(cfg, "images")
    for enc in ENCODERS:
        X = np.load(path(cfg, "embeddings") / f"{enc}.npy")
        d = path(cfg, "clusters") / enc
        for k in cfg["visual"]["kmeans"]["k_grid"]:
            a = load_table(d / f"assignments_k{k}.parquet")
            reps = representatives(X, a["style"].to_numpy(), np.load(d / f"centroids_k{k}.npy"),
                                   n_near=8, n_random=4, seed=cfg["project"]["seed"])
            rows = [[images / products.loc[a["external_code"].iloc[i], "image_path"] for i in r["near"] + r["random"]]
                    for _, r in sorted(reps.items())]
            overview_sheet(rows, d / f"overview_k{k}.png")
    log("overview sheets written to outputs/clusters/{encoder}/overview_k{K}.png")
