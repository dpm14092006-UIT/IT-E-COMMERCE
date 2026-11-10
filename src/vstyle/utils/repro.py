"""Seeds and per-stage manifests (4.11)."""
import hashlib
import importlib
import json
import platform
import random
import sys
from datetime import datetime
from pathlib import Path

import numpy as np

from ..config import ROOT
from ..io import save_json

_PACKAGES = ["numpy", "pandas", "sklearn", "lightgbm", "catboost", "torch",
             "torchvision", "transformers"]


def set_global_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    try:
        import torch
        torch.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
    except (ImportError, OSError):
        pass


def sha256(path, max_bytes=500_000_000):
    path = Path(path)
    if not path.is_file() or path.stat().st_size > max_bytes:
        return None
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def _versions():
    out = {}
    for name in _PACKAGES:
        try:
            out[name] = getattr(importlib.import_module(name), "__version__", "?")
        except (ImportError, OSError):
            out[name] = None
    return out


def write_manifest(stage, cfg, inputs=(), outputs=(), extra=None):
    """Record what a stage read, wrote and with which settings."""
    def files(paths):
        return [{"path": str(p), "sha256": sha256(p)} for p in paths]

    manifest = {
        "stage": stage,
        "time": datetime.now().isoformat(timespec="seconds"),
        "python": sys.version.split()[0],
        "platform": platform.platform(),
        "packages": _versions(),
        "config_sha256": hashlib.sha256(json.dumps(cfg, sort_keys=True, default=str).encode()).hexdigest(),
        "config": cfg,
        "inputs": files(inputs),
        "outputs": files(outputs),
        "extra": extra or {},
    }
    return save_json(manifest, ROOT / cfg["paths"]["manifests"] / f"{stage}.json")
