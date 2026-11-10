"""Configuration loading. Later files override earlier ones key by key."""
from copy import deepcopy
from pathlib import Path

import yaml

ROOT = Path(__file__).resolve().parents[2]
CONFIG_DIR = ROOT / "configs"


def _merge(base, override):
    out = deepcopy(base)
    for key, value in override.items():
        if isinstance(value, dict) and isinstance(out.get(key), dict):
            out[key] = _merge(out[key], value)
        else:
            out[key] = value
    return out


def load_config(*paths):
    paths = paths or (CONFIG_DIR / "default.yaml",)
    cfg = {}
    for p in paths:
        with open(p, encoding="utf-8") as f:
            cfg = _merge(cfg, yaml.safe_load(f) or {})
    return cfg


def load_yaml(name):
    with open(CONFIG_DIR / name, encoding="utf-8") as f:
        return yaml.safe_load(f)


def path(cfg, key):
    """Absolute path of a configured directory; output directories are created on demand."""
    p = ROOT / cfg["paths"][key]
    if key not in ("raw", "images"):
        p.mkdir(parents=True, exist_ok=True)
    return p
