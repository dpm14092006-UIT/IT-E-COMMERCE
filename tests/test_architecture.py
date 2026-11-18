"""Object model: shared abstractions, registries, typed config and the webapp service classes."""
import json

import numpy as np
import pandas as pd
import pytest

from vstyle.config import CONFIG_DIR, Config, load_config
from vstyle.experiments.walk_forward_experiment import WalkForwardExperiment
from vstyle.models.base import ForecastModel
from vstyle.models.baselines import MajorityClass, Persistence
from vstyle.models.registry import ESTIMATORS, Registry
from vstyle.models.sequence import SequenceClassifier
from vstyle.models.tabular import AlignedClassifier, make_model
from vstyle.visual.encoders import FashionClipEncoder, ImageEncoder, ResNet50Encoder


def test_every_model_implements_the_forecast_contract():
    for cls in (MajorityClass, Persistence, AlignedClassifier, SequenceClassifier):
        assert issubclass(cls, ForecastModel)
    with pytest.raises(TypeError):
        ForecastModel()


def test_predict_defaults_to_argmax_of_probabilities():
    model = MajorityClass().fit(None, np.array([2, 2, 1]))
    assert model.predict(np.zeros((4, 1))).tolist() == [2, 2, 2, 2]


def test_registry_creates_by_name_and_rejects_unknown():
    assert set(ESTIMATORS.names()) == {"random_forest", "lightgbm", "catboost"}
    assert isinstance(make_model("random_forest", {"n_estimators": 3}, 0), AlignedClassifier)
    with pytest.raises(ValueError, match="unknown tabular model"):
        make_model("svm", {}, 0)
    registry = Registry("thing")
    registry.register("a")(lambda: 1)
    assert "a" in registry and registry.create("a") == 1


def test_config_is_a_dict_with_typed_helpers(tmp_path):
    cfg = load_config()
    assert isinstance(cfg, Config) and isinstance(cfg, dict)
    assert cfg.seed == cfg["project"]["seed"]
    assert cfg.path("raw").name
    override = tmp_path / "o.yaml"
    override.write_text("project:\n  seed: 7\n", encoding="utf-8")
    merged = Config.from_files(CONFIG_DIR / "default.yaml", override)
    assert merged.seed == 7 and merged["paths"] == cfg["paths"]


def test_image_encoder_is_abstract_with_concrete_dimensions():
    with pytest.raises(TypeError):
        ImageEncoder()
    assert (ResNet50Encoder().dim, FashionClipEncoder("x").dim) == (2048, 512)
    z = np.ones((2, 2048), dtype=np.float32)
    assert ResNet50Encoder().check(z)["dim_ok"]


def test_walk_forward_experiment_uses_defaults_without_tuning_output(tmp_path):
    cfg = load_config()
    spaces = {"catboost": {"defaults": {"iterations": 5}}}
    experiment = WalkForwardExperiment(cfg, spaces, "no_such_grouping__x", 0.1, params_from="missing_tag")
    assert experiment.tuned("catboost") == ({"iterations": 5}, "none", "defaults")
    with pytest.raises(ValueError, match="unknown sequence model"):
        experiment.sequence_predictions("lstm", pd.DataFrame(), [], [])


# ---------- webapp services ----------
def test_scan_store_roundtrip_and_rejects_bad_ids(tmp_path):
    from fastapi import HTTPException

    from webapp.scan_store import ScanStore
    store = ScanStore(tmp_path / "scans")
    assert store.count() == 0 and store.list() == []
    sid = "a" * 32
    store.save({"id": sid, "created": "2020-01-01T00:00:00"}, b"jpeg")
    assert store.count() == 1 and store.get(sid)["id"] == sid and store.image(sid) == b"jpeg"
    for bad in ("../x", "short", "Z" * 32):
        with pytest.raises(HTTPException):
            store.path(bad, "json")
    store.delete(sid)
    assert store.count() == 0


def test_dataset_files_lists_and_previews_csv(tmp_path):
    from fastapi import HTTPException

    from webapp.datafiles import DatasetFiles
    raw, interim, images = tmp_path / "raw", tmp_path / "interim", tmp_path / "images"
    for d in (raw, interim, images / "AI19"):
        d.mkdir(parents=True)
    pd.DataFrame({"a": [1, 2, 3], "b": ["x", "y", "z"]}).to_csv(raw / "sales.csv")
    (images / "AI19" / "1.png").write_bytes(b"x")
    files = DatasetFiles(raw, interim, images)
    listing = files.listing()
    assert listing["files"][0]["rows"] == 3 and listing["images"][0]["season"] == "AW19"
    assert files.preview("raw/sales.csv")["columns"] == ["a", "b"]
    with pytest.raises(HTTPException):
        files.preview("raw/missing.csv")


def test_novel_registry_register_match_and_delete(tmp_path):
    import types

    from fastapi import HTTPException

    from webapp.novel import NovelStyleRegistry
    from webapp.settings import WebSettings
    root = tmp_path
    (root / "outputs" / "web").mkdir(parents=True)
    settings = WebSettings(root=root)
    registry = NovelStyleRegistry(settings, types.SimpleNamespace(meta={"novel_radius": 0.5}))
    z = np.array([1.0, 0.0], dtype=np.float32)
    def fresh():
        return {"z": {"resnet50": [1.0, 0.0], "fashionclip": z.tolist()}, "description": "d", "thumb": "t"}
    registry.remember("tok", fresh())
    first = registry.register("tok", "Mine")
    assert first["id"] == "N1" and registry.register("tok", None)["id"] == "N1"      # idempotent per token
    registry.remember("tok2", fresh())
    assert registry.register("tok2", None)["n"] == 2                                  # merged into the match
    assert registry.match(z)["match"] is True
    assert json.loads(settings.registry_path.read_text(encoding="utf-8"))["styles"][0]["n"] == 2
    with pytest.raises(HTTPException):
        registry.register("unknown", None)
    assert registry.delete("N1") == {"ok": True} and registry.list() == []
