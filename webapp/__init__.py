"""Style Scanner web application (FastAPI)."""
import sys
from pathlib import Path

try:
    import vstyle  # noqa: F401  - installed with `pip install -e .`
except ImportError:                      # fallback for a checkout that has not been installed
    sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
