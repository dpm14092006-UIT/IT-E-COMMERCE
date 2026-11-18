"""Paths and tunable thresholds of the web application, gathered in one immutable object."""
from dataclasses import dataclass, field
from pathlib import Path

from vstyle.config import Config, load_config

ROOT = Path(__file__).resolve().parents[1]
ENCODERS = ("resnet50", "fashionclip")
LABELS = {"resnet50": "ResNet-50", "fashionclip": "FashionCLIP"}


@dataclass(frozen=True)
class WebSettings:
    root: Path = ROOT
    cfg: Config = field(default_factory=load_config)
    encoders: tuple = ENCODERS
    labels: dict = field(default_factory=lambda: dict(LABELS))
    max_bytes: int = 15 * 1024 * 1024
    near_duplicate: float = 0.95      # cosine above which the upload is treated as a catalogue product
    k_nn: int = 5                     # neighbours averaged for the look-alike (kNN) novelty test
    knn_quantile: float = 0.01        # catalogue leave-one-out quantile below which an upload has no look-alike
    token_limit: int = 50             # recent uploads kept so a new style can be registered

    @property
    def web_dir(self) -> Path:
        return self.root / "outputs" / "web"

    @property
    def images_dir(self) -> Path:
        return self.root / self.cfg["paths"]["images"]

    @property
    def registry_path(self) -> Path:
        return self.web_dir / "novel_styles.json"

    @property
    def static_dir(self) -> Path:
        return Path(__file__).parent / "static"

    @property
    def background(self) -> str:
        return self.cfg["visual"]["alpha_background"]
