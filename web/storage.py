"""Atomic persistence for the local saved-style registry."""
import json
import os
import tempfile
from pathlib import Path


def read_registry(path: Path):
    if not path.exists():
        return {"styles": []}
    registry = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(registry, dict) or not isinstance(registry.get("styles"), list):
        raise ValueError("Invalid saved-style registry")
    return registry


def write_registry(path: Path, registry):
    """Replace only after the complete JSON has reached disk; preserve the old file on failure."""
    content = json.dumps(registry, ensure_ascii=False, allow_nan=False)
    path.parent.mkdir(parents=True, exist_ok=True)
    pending = None
    try:
        with tempfile.NamedTemporaryFile(mode="w", encoding="utf-8", dir=path.parent,
                                         prefix=".registry-", suffix=".tmp", delete=False) as stream:
            pending = Path(stream.name)
            stream.write(content)
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(pending, path)
    finally:
        if pending is not None:
            pending.unlink(missing_ok=True)
