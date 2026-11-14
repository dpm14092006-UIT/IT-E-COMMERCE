"""Nguoi 4 - domain adaptation of the ResNet-50 CNN on an EXTERNAL fashion dataset (Myntra,
Fashion Product Images Small). Supervised on article type + base colour only: no VISUELLE sales,
trend labels or dates are used, so the 4.1.2 rule "not trained on future sales or trend labels"
still holds. The fine-tuned backbone becomes an additional representation (resnet50ft) for RQ1/RQ3.
"""
import io
import tempfile
import zipfile
from pathlib import Path

import numpy as np
import pandas as pd


def myntra_subset(zip_path, genders=("Women", "Girls"), master="Apparel", min_count=30):
    with zipfile.ZipFile(zip_path) as z:
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
    with zipfile.ZipFile(zip_path) as z:
        for i in subset["id"]:
            f = out_dir / f"{i}.jpg"
            if not f.exists():
                with tempfile.NamedTemporaryFile(dir=out_dir, delete=False) as pending:
                    pending.write(z.read(f"images/{i}.jpg"))
                Path(pending.name).replace(f)
    return out_dir


def finetune_resnet50(subset, image_dir, epochs=3, lr=1e-3, batch_size=64, val_frac=0.1, seed=0,
                      train_layers=("layer4",), log=print, device="cpu"):
    from torch import nn
    from torchvision import transforms as T
    from torchvision.models import ResNet50_Weights, resnet50

    norm = T.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    train_tf = T.Compose([T.Resize(256), T.RandomResizedCrop(224, scale=(0.7, 1.0)), T.RandomHorizontalFlip(),
                          T.ToTensor(), norm])
    eval_tf = T.Compose([T.Resize(256), T.CenterCrop(224), T.ToTensor(), norm])
    net = resnet50(weights=ResNet50_Weights.IMAGENET1K_V2)
    for name, p in net.named_parameters():
        p.requires_grad = any(name.startswith(l) for l in train_layers)
    net.fc = nn.Identity()
    return _train_attributes(net, net, 2048, subset, image_dir, train_tf, eval_tf,
                             epochs, lr, batch_size, val_frac, seed, log, device)


def finetune_fashionclip(subset, image_dir, checkpoint, epochs, lr, batch_size, seed, log, device):
    from transformers import CLIPModel, CLIPProcessor
    from .encoders import fashionclip_features

    model = CLIPModel.from_pretrained(checkpoint).to(device).eval()
    model.requires_grad_(False)
    net = model.vision_model
    net.encoder.layers[-1].requires_grad_(True)
    net.post_layernorm.requires_grad_(True)
    processor = CLIPProcessor.from_pretrained(checkpoint)

    def preprocess(image):
        return processor(images=image, return_tensors="pt").pixel_values[0]

    def forward(x):
        return fashionclip_features(model, pixel_values=x)

    return _train_attributes(net, forward, model.config.projection_dim, subset, image_dir,
                             preprocess, preprocess, epochs, lr, batch_size, 0.1, seed, log, device)


def _train_attributes(net, forward, dim, subset, image_dir, train_tf, eval_tf,
                      epochs, lr, batch_size, val_frac, seed, log, device):
    import torch
    from PIL import Image
    from torch import nn
    torch.manual_seed(seed)
    rng = np.random.default_rng(seed)
    indices = rng.permutation(len(subset))
    n_val = int(len(indices) * val_frac)
    val_idx, tr_idx = indices[:n_val], indices[n_val:]
    paths = [Path(image_dir) / f"{i}.jpg" for i in subset["id"]]
    y1 = subset["type_id"].to_numpy(dtype=np.int64)
    y2 = subset["colour_id"].to_numpy(dtype=np.int64)

    def batch(ids, tf):
        images = []
        for i in ids:
            with Image.open(paths[i]) as image:
                images.append(tf(image.convert("RGB")))
        x = torch.stack(images).to(device)
        return x, torch.as_tensor(y1[ids], device=device), torch.as_tensor(y2[ids], device=device)

    net.to(device)
    heads = nn.ModuleDict({"article": nn.Linear(dim, int(y1.max()) + 1),
                           "colour": nn.Linear(dim, int(y2.max()) + 1)}).to(device)
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
            z = forward(x)
            loss = ce(heads["article"](z), t1) + ce(heads["colour"](z), t2)
            opt.zero_grad()
            loss.backward()
            opt.step()
            losses.append(loss.detach().item())
        net.eval()
        hit1 = hit2 = 0
        with torch.no_grad():
            for b in range(0, len(val_idx), batch_size):
                x, t1, t2 = batch(val_idx[b:b + batch_size], eval_tf)
                z = forward(x)
                hit1 += int((heads["article"](z).argmax(1) == t1).sum())
                hit2 += int((heads["colour"](z).argmax(1) == t2).sum())
        history.append({"epoch": ep + 1, "train_loss": float(np.mean(losses)),
                        "val_acc_type": hit1 / len(val_idx), "val_acc_colour": hit2 / len(val_idx)})
        log(f"epoch {ep + 1}: {history[-1]}")
    return net, history
