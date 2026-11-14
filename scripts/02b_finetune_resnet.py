"""G1 | Fine-tune a visual encoder on Myntra apparel (article type and colour) and encode VISUELLE.
"""
import os

import numpy as np
import pandas as pd
from _common import ROOT, log, parse

from vstyle.config import path
from vstyle.io import load_json, load_table, save_json, save_table
from vstyle.utils.repro import write_manifest
from vstyle.visual import encoders as enc
from vstyle.visual.finetune import extract, finetune_fashionclip, finetune_resnet50, myntra_subset

if __name__ == "__main__":
    def extra(ap):
        ap.add_argument("--encoder", default="resnet50ft", choices=["resnet50ft", "fashionclipft"])
        ap.add_argument("--encode-only", action="store_true", help="reuse the saved backbone and training history")
    args, cfg = parse(__doc__, extra)
    import torch
    torch.set_num_threads(min(8, os.cpu_count() or 1))
    name = args.encoder
    visual = cfg["visual"]
    source = visual["encoders"]["resnet50ft"]
    c = visual["encoders"][name]
    archive = ROOT / source["source_zip"]
    sub = myntra_subset(archive, tuple(source["genders"]), source["master_category"], source["min_class_count"])
    log(f"Myntra subset: {len(sub)} images, {sub['type_id'].nunique()} article types, {sub['colour_id'].nunique()} colours")
    out = path(cfg, "embeddings")
    backbone = out / f"{name}_backbone.pt"
    history_path = out / f"{name}_training.json"
    device = enc._device(visual["device"])
    checkpoint = visual["encoders"]["fashionclip"]["checkpoint"]
    seed = cfg["project"]["seed"]
    if args.encode_only:
        history = load_json(history_path) if history_path.is_file() else None
    else:
        img_dir = extract(archive, sub, ROOT / source["image_dir"])
        log(f"{name}: training on {device}")
        if name == "resnet50ft":
            net, history = finetune_resnet50(sub, img_dir, c["epochs"], c["learning_rate"], c["batch_size"],
                                            seed=seed, train_layers=tuple(c["train_layers"]), log=log, device=device)
        else:
            net, history = finetune_fashionclip(sub, img_dir, checkpoint, c["epochs"],
                                              c["learning_rate"], c["batch_size"], seed, log, device)
        pending = backbone.with_suffix(".tmp")
        torch.save(net.cpu().state_dict(), pending)
        pending.replace(backbone)
        save_json(history, history_path)
        del net
    products = load_table(path(cfg, "interim") / "products.parquet")
    products = products[products["image_ok"]].reset_index(drop=True)
    paths = [path(cfg, "images") / p for p in products["image_path"]]
    log(f"{name}: encoding {len(paths)} VISUELLE products")
    if name == "resnet50ft":
        z = enc.encode_resnet50(paths, visual["encoders"]["resnet50"]["weights"], c["batch_size"],
                                device, visual["alpha_background"], state_dict=backbone)
    else:
        z = enc.encode_fashionclip(paths, checkpoint, c["batch_size"], device,
                                  visual["alpha_background"], state_dict=backbone)
    check = enc.check_embeddings(z, enc.EXPECTED_DIM[name])
    if not check["dim_ok"] or check["zero_norm"] or not np.isfinite(z).all():
        raise ValueError(f"Invalid {name} embeddings: {check}")
    check.update({"finetune_history": history, "n_myntra": len(sub),
                  "article_types": sorted(sub["articleType"].unique()), "colours": sorted(sub["baseColour"].unique())})
    np.save(out / f"{name}.npy", enc.l2_normalize(z))
    files = [out / f"{name}.npy", backbone,
             save_table(pd.DataFrame({"external_code": products["external_code"]}), out / f"{name}_ids.parquet"),
             save_json(check, out / f"{name}_G1.json")]
    if history_path.is_file():
        files.append(history_path)
    write_manifest(f"02b_finetune_{name}", cfg, inputs=[archive], outputs=files, extra=check)
    log({k: v for k, v in check.items() if k not in ("article_types", "colours")})
