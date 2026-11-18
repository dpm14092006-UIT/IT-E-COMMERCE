"""Evaluate image clustering on the Myntra validation split shared by both encoders."""
import os

import numpy as np
import pandas as pd
from _common import ROOT, log, parse

from vstyle.config import path
from vstyle.utils.repro import write_manifest
from vstyle.visual import clustering as cl
from vstyle.visual import encoders as enc
from vstyle.visual.finetune import extract, myntra_split, myntra_subset

if __name__ == "__main__":
    args, cfg = parse(__doc__)
    import torch
    from threadpoolctl import threadpool_limits

    torch.set_num_threads(min(8, os.cpu_count() or 1))
    visual = cfg["visual"]
    settings = visual["encoders"]
    source = settings["resnet50ft"]
    archive = ROOT / source["source_zip"]
    embeddings = path(cfg, "embeddings")
    backbones = {name: embeddings / f"{name}_backbone.pt" for name in ("resnet50ft", "fashionclipft")}
    for backbone in backbones.values():
        if not backbone.is_file():
            raise SystemExit(f"Missing {backbone}. Run stage 02 with --encoder all first.")
    seed = cfg["project"]["seed"]
    subset = myntra_subset(archive, tuple(source["genders"]), source["master_category"], source["min_class_count"])
    _, val = myntra_split(len(subset), seed)
    held_out = subset.iloc[val].reset_index(drop=True)
    image_dir = extract(archive, held_out, ROOT / source["image_dir"])
    images = [image_dir / f"{product_id}.jpg" for product_id in held_out["id"]]
    assignments = held_out[["id", "articleType", "baseColour"]].copy()
    assignments.insert(0, "source", "myntra")
    k = visual["selected"]["k"]
    km = visual["kmeans"]
    rows = []
    for name in ("resnet50", "resnet50ft", "fashionclip", "fashionclipft"):
        family = name.removesuffix("ft")
        model = settings[family]
        encode = enc.encode_resnet50 if family == "resnet50" else enc.encode_fashionclip
        weights = model["weights"] if family == "resnet50" else model["checkpoint"]
        log(f"{name}: encoding {len(images)} held-out Myntra images")
        z = encode(images, weights, settings[name]["batch_size"], visual["device"],
                   visual["alpha_background"], state_dict=backbones.get(name))
        checks = enc.check_embeddings(z, enc.EXPECTED_DIM[name])
        if not checks["dim_ok"] or checks["zero_norm"] or not np.isfinite(z).all():
            raise ValueError(f"Invalid {name} embeddings: {checks}")
        x = enc.l2_normalize(z)
        with threadpool_limits(limits=4):
            metrics = cl.evaluate_grid(x, [k], seed, km["stability_seeds"], km["n_init"], km["max_iter"])
            labels = cl.fit_kmeans(x, k, seed, km["n_init"], km["max_iter"]).labels_
        assignments[name] = labels
        agreement = cl.attribute_agreement(labels, held_out, ("articleType", "baseColour"))
        rows.append({"encoder": name, "n_images": len(x), **metrics.iloc[0].to_dict(),
                     "nmi_article_type": agreement["articleType"]["nmi"],
                     "nmi_base_colour": agreement["baseColour"]["nmi"]})
        log(rows[-1])
    out = path(cfg, "reports")
    metrics_file = out / "myntra_cluster_validation.csv"
    assignments_file = out / "myntra_cluster_assignments.csv"
    pd.DataFrame(rows).to_csv(metrics_file, index=False)
    assignments.to_csv(assignments_file, index=False)
    write_manifest("03c_evaluate_myntra_clusters", cfg, inputs=[archive, *backbones.values()],
                   outputs=[metrics_file, assignments_file], extra={"split": "Myntra validation", "k": k})
