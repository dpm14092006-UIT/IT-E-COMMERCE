"""Nguoi 4 - domain adaptation of the ResNet-50 CNN on an EXTERNAL fashion dataset (Myntra,
Fashion Product Images Small). Supervised on article type + base colour only: no VISUELLE sales,
trend labels or dates are used, so the 4.1.2 rule "not trained on future sales or trend labels"
still holds. The fine-tuned backbone becomes an additional representation (resnet50ft) for RQ1/RQ3.
"""
import io
import zipfile
from pathlib import Path

import numpy as np
import pandas as pd


def myntra_subset(zip_path, genders=("Women", "Girls"), master="Apparel", min_count=30):
    z = zipfile.ZipFile(zip_path)
    s = pd.read_csv(io.BytesIO(z.read("styles.csv")), on_bad_lines="skip")
    names = set(z.namelist())
    s = s[s["id"].map(lambda i: f"images/{i}.jpg" in names)]
    s = s[s["gender"].isin(genders) & (s["masterCategory"] == master)].copy()
    for col in ("articleType", "baseColour"):
        keep = s[col].value_counts()
        s = s[s[col].isin(keep[keep >= min_count].index)]
    s["type_id"] = s["articleType"].astype("category").cat.codes
    s["colour_id"] = s["baseColour"].astype("category").cat.codes
    return s.reset_index(drop=True)


def extract(zip_path, subset, out_dir):
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    z = zipfile.ZipFile(zip_path)
    for i in subset["id"]:
        f = out_dir / f"{i}.jpg"
        if not f.exists():
            f.write_bytes(z.read(f"images/{i}.jpg"))
    return out_dir


def finetune_resnet50(subset, image_dir, epochs=3, lr=1e-3, batch_size=64, val_frac=0.1, seed=0,
                      train_layers=("layer4",), log=print):
    import torch
    from PIL import Image
    from torch import nn
    from torchvision import transforms as T
    from torchvision.models import ResNet50_Weights, resnet50

    torch.manual_seed(seed)
    rng = np.random.default_rng(seed)
    idx = rng.permutation(len(subset))
    n_val = int(len(idx) * val_frac)
    val_idx, tr_idx = idx[:n_val], idx[n_val:]
    norm = T.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    train_tf = T.Compose([T.Resize(256), T.RandomResizedCrop(224, scale=(0.7, 1.0)), T.RandomHorizontalFlip(),
                          T.ToTensor(), norm])
    eval_tf = T.Compose([T.Resize(256), T.CenterCrop(224), T.ToTensor(), norm])
    paths = [Path(image_dir) / f"{i}.jpg" for i in subset["id"]]
    y1, y2 = subset["type_id"].to_numpy(), subset["colour_id"].to_numpy()

    def batch(ids, tf):
        x = torch.stack([tf(Image.open(paths[i]).convert("RGB")) for i in ids])
        return x, torch.tensor(y1[ids], dtype=torch.long), torch.tensor(y2[ids], dtype=torch.long)

    net = resnet50(weights=ResNet50_Weights.IMAGENET1K_V2)
    for name, p in net.named_parameters():
        p.requires_grad = any(name.startswith(l) for l in train_layers)
    net.fc = nn.Identity()
    heads = nn.ModuleDict({"article": nn.Linear(2048, int(y1.max()) + 1), "colour": nn.Linear(2048, int(y2.max()) + 1)})
    params = [p for p in net.parameters() if p.requires_grad] + list(heads.parameters())
    opt = torch.optim.AdamW(params, lr=lr, weight_decay=1e-4)
    ce = nn.CrossEntropyLoss()
    history = []
    for ep in range(epochs):
        net.train()
        for m in net.modules():                       # keep frozen BatchNorm statistics fixed
            if isinstance(m, nn.BatchNorm2d) and not any(p.requires_grad for p in m.parameters()):
                m.eval()
        order, losses = rng.permutation(tr_idx), []
        for b in range(0, len(order), batch_size):
            x, t1, t2 = batch(order[b:b + batch_size], train_tf)
            z = net(x)
            loss = ce(heads["article"](z), t1) + ce(heads["colour"](z), t2)
            opt.zero_grad()
            loss.backward()
            opt.step()
            losses.append(float(loss))
        net.eval()
        hit1 = hit2 = 0
        with torch.no_grad():
            for b in range(0, len(val_idx), batch_size):
                x, t1, t2 = batch(val_idx[b:b + batch_size], eval_tf)
                z = net(x)
                hit1 += int((heads["article"](z).argmax(1) == t1).sum())
                hit2 += int((heads["colour"](z).argmax(1) == t2).sum())
        history.append({"epoch": ep + 1, "train_loss": float(np.mean(losses)),
                        "val_acc_type": hit1 / len(val_idx), "val_acc_colour": hit2 / len(val_idx)})
        log(f"epoch {ep + 1}: {history[-1]}")
    return net, history
