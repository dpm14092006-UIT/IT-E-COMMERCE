"""G1 | Nguoi 4 - fine-tune the ResNet-50 CNN on Myntra women apparel (article type + colour), then
encode VISUELLE with the adapted backbone. Output: outputs/embeddings/resnet50ft.npy (+ ids, G1 json,
finetune history, backbone weights). External data only; no VISUELLE label is used.
"""
import numpy as np
import pandas as pd
from _common import ROOT, log, parse

from vstyle.config import path
from vstyle.io import load_table, save_json, save_table
from vstyle.utils.repro import write_manifest
from vstyle.visual import encoders as enc
from vstyle.visual.finetune import extract, finetune_resnet50, myntra_subset

# history of the 2026-10-12 fine-tuning run (backbone saved, VISUELLE encoding then hit an I/O error)
HISTORY_FROM_RUN = [
    {"epoch": 1, "train_loss": 2.7051, "val_acc_type": 0.7288, "val_acc_colour": 0.6169},
    {"epoch": 2, "train_loss": 1.6642, "val_acc_type": 0.7367, "val_acc_colour": 0.6531},
    {"epoch": 3, "train_loss": 1.3318, "val_acc_type": 0.7650, "val_acc_colour": 0.6621},
]

if __name__ == "__main__":
    args, cfg = parse(__doc__, lambda ap: ap.add_argument("--encode-only", action="store_true",
                                                          help="reuse the saved backbone, skip fine-tuning"))
    import torch
    torch.set_num_threads(16)
    c = cfg["visual"]["encoders"]["resnet50ft"]
    sub = myntra_subset(ROOT / c["source_zip"], tuple(c["genders"]), c["master_category"], c["min_class_count"])
    img_dir = extract(ROOT / c["source_zip"], sub, ROOT / c["image_dir"])
    log(f"Myntra subset: {len(sub)} images, {sub['type_id'].nunique()} article types, {sub['colour_id'].nunique()} colours")
    out = path(cfg, "embeddings")
    if args.encode_only:
        history = HISTORY_FROM_RUN
    else:
        net, history = finetune_resnet50(sub, img_dir, c["epochs"], c["learning_rate"], c["batch_size"],
                                         seed=cfg["project"]["seed"], train_layers=tuple(c["train_layers"]), log=log)
        torch.save(net.state_dict(), out / "resnet50ft_backbone.pt")
    products = load_table(path(cfg, "interim") / "products.parquet")
    products = products[products["image_ok"]].reset_index(drop=True)
    paths = [path(cfg, "images") / p for p in products["image_path"]]
    z = enc.encode_resnet50(paths, "IMAGENET1K_V2", c["batch_size"], cfg["visual"]["device"],
                            cfg["visual"]["alpha_background"], state_dict=out / "resnet50ft_backbone.pt")
    check = enc.check_embeddings(z, 2048)
    check.update({"finetune_history": history, "n_myntra": len(sub),
                  "article_types": sorted(sub["articleType"].unique()), "colours": sorted(sub["baseColour"].unique())})
    np.save(out / "resnet50ft.npy", enc.l2_normalize(z))
    files = [out / "resnet50ft.npy", out / "resnet50ft_backbone.pt",
             save_table(pd.DataFrame({"external_code": products["external_code"]}), out / "resnet50ft_ids.parquet"),
             save_json(check, out / "resnet50ft_G1.json")]
    write_manifest("02b_finetune_resnet", cfg, outputs=files, extra={k: v for k, v in check.items() if k != "colours"})
    log({k: v for k, v in check.items() if k not in ("article_types", "colours")})
