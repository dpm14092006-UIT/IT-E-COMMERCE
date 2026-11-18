"""Style Scanner - upload one out-of-sample product image; compare ResNet-50 and FashionCLIP style spaces.

Inference only; everything trained lives in outputs/web/{encoder}/ (scripts/17_build_web_assets.py):
  1. preprocessing   same as training: RGBA -> white background. A photo with a person or a backdrop first goes
                     through garment isolation (webapp/segment.py, SegFormer clothes parsing) so that it is framed
                     like a catalogue image; then each encoder's own transforms
  2. encoders        ResNet-50 (frozen ImageNet, GAP of layer4, 2048-d) and FashionCLIP (ViT-B/32, 512-d),
                     both L2-normalised (2)(4)(5)                                     -> webapp/encoders.py
  3. style           nearest frozen K-Means centroid (11), novelty by centroid distance and kNN look-alikes
                                                                                      -> webapp/spaces.py
                     FashionCLIP's text encoder describes it zero-shot (webapp/zero_shot.py); new styles can be
                     registered so later uploads are matched against them            -> webapp/novel.py
  4. trend           TCN / CatBoost forecast of the style at the chosen origin (served from meta.json);
                     a new style has no sales history, so only a labelled proxy from its nearest styles is shown

Layout: settings.py (paths, thresholds) | analyzer.py (service) | scanner.py (routes) | manager.py (/api/m routes).
Pages: / is the manager console, /scanner is this technical scanner.
Run:  python -m uvicorn webapp.app:app --port 8000
"""
import torch  # noqa: F401  - load torch DLLs before numpy/scipy on Windows (see scripts/_common.py)
from fastapi import FastAPI
from fastapi.staticfiles import StaticFiles

from .analyzer import ScanAnalyzer
from .encoders import ClipEncoder, ResNetEncoder
from .manager import ManagerAPI
from .novel import NovelStyleRegistry
from .scanner import ScannerAPI
from .segment import GarmentSegmenter
from .settings import WebSettings
from .spaces import ClipStyleNamer, StyleSpace
from .zero_shot import text_bank


def create_app(settings: WebSettings | None = None) -> FastAPI:
    """Composition root: loads spaces and models once and wires services into the routers."""
    settings = settings or WebSettings()
    visual = settings.cfg["visual"]["encoders"]
    spaces = {e: StyleSpace(e, settings) for e in settings.encoders}
    encoders = {"resnet50": ResNetEncoder(visual["resnet50"]["weights"]),
                "fashionclip": ClipEncoder(visual["fashionclip"]["checkpoint"])}
    clip = encoders["fashionclip"]
    texts = text_bank(clip.text)
    ClipStyleNamer(spaces["fashionclip"], clip, texts).apply()
    registry = NovelStyleRegistry(settings, spaces["fashionclip"])
    analyzer = ScanAnalyzer(settings, spaces, encoders, GarmentSegmenter(), texts, registry)

    app = FastAPI(title="Style Scanner")
    app.mount("/static", StaticFiles(directory=settings.static_dir), name="static")
    app.include_router(ScannerAPI(settings, spaces, analyzer, registry).router)
    app.include_router(ManagerAPI(settings.root, settings.cfg, spaces, analyzer.analyze_bytes).router)
    return app


app = create_app()
