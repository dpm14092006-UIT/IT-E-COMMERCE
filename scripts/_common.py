"""Shared bootstrap for the numbered stage scripts: import path, config, logging."""
import argparse
import sys
from pathlib import Path

try:
    import torch  # noqa: F401  - on Windows torch must load its DLLs before sklearn/scipy (WinError 1114)
except (ImportError, OSError):
    pass

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from vstyle.config import CONFIG_DIR, load_config  # noqa: E402
from vstyle.utils.repro import set_global_seed  # noqa: E402

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")


def parse(description, extra=None):
    ap = argparse.ArgumentParser(description=description)
    ap.add_argument("--config", nargs="*", default=[str(CONFIG_DIR / "default.yaml")],
                    help="YAML files merged left to right")
    if extra:
        extra(ap)
    args = ap.parse_args()
    cfg = load_config(*args.config)
    set_global_seed(cfg["project"]["seed"])
    return args, cfg


def log(msg):
    print(f"[vstyle] {msg}", flush=True)
