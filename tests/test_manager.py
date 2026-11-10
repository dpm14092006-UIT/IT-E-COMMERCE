"""Manager console API (web/manager.py) on the real outputs; no encoder is loaded."""
import json
import types
from pathlib import Path

import numpy as np
import pytest
from fastapi import FastAPI
from fastapi.testclient import TestClient

ROOT = Path(__file__).resolve().parents[1]
WEB = ROOT / "outputs" / "web"
pytestmark = pytest.mark.skipif(not (WEB / "resnet50" / "meta.json").exists(), reason="run scripts/17_build_web_assets.py")


@pytest.fixture(scope="module")
def client():
    from vstyle.config import load_config
    from web.manager import build_router
    spaces = {}
    for e in ("resnet50", "fashionclip"):
        a = np.load(WEB / e / "assets.npz")
        spaces[e] = types.SimpleNamespace(meta=json.loads((WEB / e / "meta.json").read_text(encoding="utf-8")),
                                          ids=a["ids"], style_of=a["style"])
    app = FastAPI()
    app.include_router(build_router(ROOT, load_config(), spaces, None))
    return TestClient(app)


def test_summary_covers_whole_catalogue(client):
    s = client.get("/api/m/summary").json()
    assert s["kpi"]["products"] == 5355 and s["kpi"]["units"] == 1599100
    assert len(s["weekly"]) == s["kpi"]["weeks"] and round(sum(s["weekly"])) == s["kpi"]["units"]
    assert all(abs(sum(x["outlook"]["p"]) - 1) < 1e-3 for x in s["styles"] if x["outlook"])


def test_origin_outside_forecast_range_rejected(client):
    assert client.get("/api/m/summary?t=5").status_code == 400


def test_category_units_add_up(client):
    cats = client.get("/api/m/categories").json()["categories"]
    assert sum(c["units"] for c in cats) == 1599100
    c = client.get("/api/m/categories/" + cats[0]["key"]).json()
    assert round(sum(c["weekly"])) == c["kpi"]["units"] == cats[0]["units"]
    assert client.get("/api/m/categories/not-a-category").status_code == 404


def test_products_filter_and_paginate(client):
    d = client.get("/api/m/products?color=blue&size=10&page=2").json()
    assert d["page"] == 2 and len(d["items"]) == 10 and all(p["color"] == "blue" for p in d["items"])
    p = client.get(f"/api/m/products/{d['items'][0]['code']}").json()
    assert p["outlook"]["style"] == p["style"] and p["weekly"]


def test_files_only_serve_listed_keys(client):
    keys = {f["key"] for f in client.get("/api/m/files").json()["files"]}
    assert "raw/sales.csv" in keys
    assert client.get("/api/m/files/download?key=../configs/default.yaml").status_code == 404
    assert client.get("/api/m/files/preview?key=raw/category_labels.pt").status_code == 415


def test_scan_ids_are_validated(client):
    assert client.get("/api/m/scans/..%2F..%2Fnovel_styles").status_code == 404
    assert client.delete("/api/m/scans/" + "0" * 32).status_code == 404
