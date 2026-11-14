"""G1 | Extract baseline embeddings and train both visual encoders on Myntra.

Outputs: outputs/embeddings/{encoder}.npy (L2-normalised), {encoder}_ids.parquet, {encoder}_G1.json
"""
import subprocess
import sys

import numpy as np
import pandas as pd
from _common import ROOT, log, parse

from vstyle.config import path
from vstyle.io import load_table, save_json, save_table
from vstyle.utils.repro import write_manifest
from vstyle.visual import encoders as enc


def encode(name, paths, cfg):
    v = cfg["visual"]
    if name == "fashionclip":
        e = v["encoders"]["fashionclip"]
        return enc.encode_fashionclip(paths, e["checkpoint"], e["batch_size"], v["device"], v["alpha_background"])
    e = v["encoders"]["resnet50"]
    return enc.encode_resnet50(paths, e["weights"], e["batch_size"], v["device"], v["alpha_background"])


if __name__ == "__main__":
    args, cfg = parse(__doc__, lambda ap: ap.add_argument("--encoder", default="all",
                                                          choices=["all", "fashionclip", "resnet50",
                                                                   "fashionclipft", "resnet50ft"]))
    adapted = ("resnet50ft", "fashionclipft")
    for name in adapted if args.encoder == "all" else [args.encoder]:
        if name in adapted:
            subprocess.run([sys.executable, str(ROOT / "scripts/02b_finetune_resnet.py"),
                            "--encoder", name, "--config", *args.config], check=True)
    if args.encoder in adapted:
        raise SystemExit(0)
    products = load_table(path(cfg, "interim") / "products.parquet")
    products = products[products["image_ok"]].reset_index(drop=True)          # D14
    images = path(cfg, "images")
    paths = [images / p for p in products["image_path"]]
    out = path(cfg, "embeddings")
    for name in (["fashionclip", "resnet50"] if args.encoder == "all" else [args.encoder]):
        log(f"encoding {len(paths)} images with {name} ...")
        z = encode(name, paths, cfg)
        check = enc.check_embeddings(z, enc.EXPECTED_DIM[name])
        z_again = encode(name, paths[:32], cfg)                     # determinism (G1)
        cos = (enc.l2_normalize(z[:32]) * enc.l2_normalize(z_again)).sum(1)
        check["determinism_min_cosine"] = float(cos.min())
        zn = enc.l2_normalize(z)
        np.save(out / f"{name}.npy", zn)
        files = [out / f"{name}.npy",
                 save_table(pd.DataFrame({"external_code": products["external_code"]}), out / f"{name}_ids.parquet"),
                 save_json(check, out / f"{name}_G1.json")]
        write_manifest(f"02_embeddings_{name}", cfg, outputs=files, extra=check)
        log(f"{name}: {check}")
