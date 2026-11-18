"""Scans uploaded from the console are kept in outputs/web/scans/ ({id}.json + {id}.jpg) so they can be audited."""
import json
import os
import re
import tempfile
from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd
from fastapi import HTTPException

from .catalogue import PRIMARY, Catalogue, vi

SCAN_ID = re.compile(r"^[0-9a-f]{32}$")


def atomic_write(path: Path, data: bytes):
    path.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile(dir=path.parent, prefix=".tmp-", delete=False) as stream:
        stream.write(data)
        stream.flush()
        os.fsync(stream.fileno())
    os.replace(stream.name, path)


class ScanStore:
    """Persistence of saved scans: one JSON summary and one JPEG thumbnail per scan id."""

    def __init__(self, directory: Path):
        self.dir = directory

    def count(self) -> int:
        return len(list(self.dir.glob("*.json"))) if self.dir.exists() else 0

    def save(self, summary: dict, jpeg: bytes):
        atomic_write(self.dir / f"{summary['id']}.jpg", jpeg)
        atomic_write(self.dir / f"{summary['id']}.json", json.dumps(summary, ensure_ascii=False).encode("utf-8"))

    def path(self, scan_id: str, ext: str) -> Path:
        if not SCAN_ID.match(scan_id):
            raise HTTPException(404, "Không có lượt quét này.")
        f = self.dir / f"{scan_id}.{ext}"
        if not f.exists():
            raise HTTPException(404, "Không có lượt quét này.")
        return f

    def list(self):
        if not self.dir.exists():
            return []
        items = []
        for f in self.dir.glob("*.json"):
            try:
                items.append(json.loads(f.read_text(encoding="utf-8")))
            except (OSError, ValueError):
                continue
        return sorted(items, key=lambda s: s["created"], reverse=True)

    def get(self, scan_id: str):
        return json.loads(self.path(scan_id, "json").read_text(encoding="utf-8"))

    def image(self, scan_id: str) -> bytes:
        return self.path(scan_id, "jpg").read_bytes()

    def delete(self, scan_id: str):
        self.path(scan_id, "json").unlink()
        (self.dir / f"{scan_id}.jpg").unlink(missing_ok=True)


class ScanSummarizer:
    """Condenses a full /api/analyze payload into the manager's one-screen scan summary."""

    def __init__(self, catalogue: Catalogue):
        self.cat = catalogue

    def summarize(self, analysis, t, name):
        cat, E = self.cat, analysis["encoders"]
        enc = next((e for e in (PRIMARY, "fashionclip") if not E[e]["style"]["novel"]), None)
        if enc:
            s = E[enc]["style"]
            outlook = cat.style_outlook(s["id"], t, enc)
            style = {"encoder": enc, "id": s["id"], "name": cat.style_name(s["id"], enc),
                     "typicality_pct": s["typicality_pct"], "novel": False}
        else:
            s = E[PRIMARY]["style"]
            outlook = cat.proxy_outlook(E[PRIMARY]["ranking"], t, PRIMARY)
            style = {"encoder": PRIMARY, "id": s["id"], "name": cat.style_name(s["id"]), "novel": True,
                     "typicality_pct": s["typicality_pct"]}
        neighbors = []
        for n in E[PRIMARY]["neighbors"][:6]:
            r = cat.products.loc[n["code"]]
            neighbors.append({"code": n["code"], "similarity": n["similarity"], "category_vi": n["category_vi"],
                              "color_vi": n["color_vi"], "season": n["season"], "units": round(float(r["units"]))})
        cats = pd.Series([n["category"] for n in E[PRIMARY]["neighbors"]]).value_counts()
        zs = analysis["zero_shot"]
        category = zs["groups"][0]["items"][0]
        return {"id": analysis["token"], "name": name, "created": datetime.now().isoformat(timespec="seconds"),
                "t": t, "week": cat.week(t), "description": zs["short"],
                "category": {"vi": category["vi"], "p": category["p"], "known": category["known"],
                             "catalogue": cats.index[0], "catalogue_vi": vi(cats.index[0])},
                "colors": [{"vi": c["vi"], "hex": c.get("hex"), "share": c.get("share")} for c in zs["colors"][:3]],
                "style": style, "outlook": outlook,
                "near_duplicate": E[PRIMARY]["near_duplicate"], "novel_both": not enc,
                "neighbors": neighbors,
                "reference_units": round(float(np.mean([n["units"] for n in neighbors])), 1) if neighbors else None,
                "agreement": analysis["agreement"], "segmentation": analysis["segmentation"].get("applied", False)}
