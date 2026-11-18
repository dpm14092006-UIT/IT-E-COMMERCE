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


class Config(dict):
    """YAML configuration; a plain dict (so it serialises and indexes as before) with typed helpers."""

    @classmethod
    def from_files(cls, *paths) -> "Config":
        paths = paths or (CONFIG_DIR / "default.yaml",)
        data = {}
        for p in paths:
            with open(p, encoding="utf-8") as f:
                data = _merge(data, yaml.safe_load(f) or {})
        return cls(data)

    @property
    def seed(self) -> int:
        return self["project"]["seed"]

    def path(self, key: str) -> Path:
        """Absolute path of a configured directory; output directories are created on demand."""
        p = ROOT / self["paths"][key]
        if key not in ("raw", "images"):
            p.mkdir(parents=True, exist_ok=True)
        return p


def load_config(*paths) -> Config:
    return Config.from_files(*paths)


def load_yaml(name):
    with open(CONFIG_DIR / name, encoding="utf-8") as f:
        return yaml.safe_load(f)


def path(cfg, key):
    """Module-level shim kept for scripts: same as Config.path for any mapping."""
    return Config(cfg).path(key) if not isinstance(cfg, Config) else cfg.path(key)
