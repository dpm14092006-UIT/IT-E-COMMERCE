"""Exercise the running Style Scanner API with real images; never change existing saved styles.

python scripts/18_check_web.py --check-saved-styles
Start the server first: python -m uvicorn web.app:app --port 8000
"""
import argparse
import concurrent.futures
import json
import time
import urllib.error
import urllib.request
import uuid
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--url", default="http://127.0.0.1:8000")
    parser.add_argument("--check-saved-styles", action="store_true",
                        help="Create, verify and delete a new temporary test style only.")
    args = parser.parse_args()
    checks = []
    analyses = []

    def request(path, method="GET", body=None, content_type=None):
        headers = {"Content-Type": content_type} if content_type else {}
        req = urllib.request.Request(args.url.rstrip("/") + path, data=body, method=method, headers=headers)
        start = time.monotonic()
        try:
            with urllib.request.urlopen(req, timeout=180) as response:
                return response.status, response.read(), {k.lower(): v for k, v in response.headers.items()}, time.monotonic() - start
        except urllib.error.HTTPError as error:
            return error.code, error.read(), {k.lower(): v for k, v in error.headers.items()}, time.monotonic() - start

    def check(name, condition, **details):
        checks.append({"name": name, "passed": bool(condition), **details})
        print(("PASS " if condition else "FAIL ") + name, flush=True)
        if not condition:
            raise AssertionError(name)

    def upload(raw, filename, mode):
        boundary = "scanner-" + uuid.uuid4().hex
        body = (f"--{boundary}\r\nContent-Disposition: form-data; name=\"segment\"\r\n\r\n{mode}\r\n"
                f"--{boundary}\r\nContent-Disposition: form-data; name=\"file\"; filename=\"{filename}\"\r\n"
                "Content-Type: image/png\r\n\r\n").encode() + raw + f"\r\n--{boundary}--\r\n".encode()
        return request("/api/analyze", "POST", body, f"multipart/form-data; boundary={boundary}")

    def post_json(path, data):
        return request(path, "POST", json.dumps(data).encode(), "application/json")

    try:
        status, body, headers, _ = request("/scanner")
        check("HTML served with cache refresh", status == 200 and b"previewScanner" in body
              and headers.get("cache-control") == "no-store")
        status, body, _, _ = request("/api/health")
        health = json.loads(body)
        check("All models ready", status == 200 and health["status"] == "ready"
              and set(health["encoders"]) == {"resnet50", "fashionclip"}, health=health)
        status, body, _, _ = request("/api/meta")
        meta = json.loads(body)
        check("Forecasts, histories and 20 groups per encoder", status == 200 and all(
            len(meta[e]["styles"]) == 20 and meta[e]["forecasts"] and meta[e]["history"]
            and str(meta[e]["default_origin"]) in meta[e]["forecasts"] for e in health["encoders"]))
        status, body, headers, _ = request("/catalog/582.jpg")
        check("Catalogue image", status == 200 and body.startswith(b"\xff\xd8")
              and headers.get("content-type", "").startswith("image/jpeg"))
        check("Unknown catalogue item rejected", request("/catalog/999999999.jpg")[0] == 404)
        check("Invalid image rejected", upload(b"not an image", "broken.png", "off")[0] == 400)
        check("Empty image rejected", upload(b"", "empty.png", "off")[0] == 400)
        check("Invalid segmentation mode rejected", upload(b"x", "x.png", "invalid")[0] == 400)
        check("Oversized upload rejected", upload(b"x" * (health["max_upload_bytes"] + 1), "big.png", "off")[0] == 413)
        check("Missing file rejected", post_json("/api/analyze", {})[0] == 422)
        check("Expired scan token rejected", post_json("/api/novel", {"token": "invalid"})[0] == 404)
        check("Missing saved style rejected", request("/api/novel/does-not-exist", "DELETE")[0] == 404)

        catalogue = ROOT / "Data/extracted/visuelle2/images/PE17/00582.png"
        person = ROOT / "Data/extracted/myntra_women_apparel/26960.jpg"
        for mode in ("auto", "off", "main", "all"):
            path = catalogue if mode in ("auto", "off") else person
            with concurrent.futures.ThreadPoolExecutor(max_workers=1) as pool:
                pending = pool.submit(upload, path.read_bytes(), path.name, mode)
                if mode == "auto":
                    time.sleep(0.25)
                    health_status, _, _, latency = request("/api/health")
                    check("API stays responsive during inference", health_status == 200 and latency < 2,
                          latency_seconds=round(latency, 3))
                status, body, _, elapsed = pending.result()
            result = json.loads(body)
            check(f"Real image analysis: {mode}", status == 200 and all(
                result.get("encoders", {}).get(e, {}).get("neighbors") for e in health["encoders"]),
                  seconds=round(elapsed, 2))
            check(f"Analysis contract: {mode}", all(result.get(k) is not None for k in
                  ("token", "preprocess", "segmentation", "zero_shot", "agreement"))
                  and len(result["zero_shot"]["groups"]) >= 5)
            for e, encoder in result["encoders"].items():
                forecasts = meta[e]["forecasts"][str(meta[e]["default_origin"])]
                check(f"Forecast lookup for {mode}/{e}", str(encoder["style"]["id"]) in forecasts)
            analyses.append(result)

        if args.check_saved_styles:
            original = json.loads(request("/api/novel")[1])
            candidate = next((r for r in analyses if not (r.get("registry_match") or {}).get("match")), None)
            check("Safe isolated save candidate available", candidate is not None)
            marker = "__integration_check_" + uuid.uuid4().hex
            saved_id = None
            try:
                status, body, _, _ = post_json("/api/novel", {"token": candidate["token"], "name": marker})
                saved = json.loads(body)
                check("Save style", status == 200 and saved["name"] == marker and saved["n"] == 1)
                saved_id = saved["id"]
                status, body, _, _ = post_json("/api/novel", {"token": candidate["token"], "name": marker})
                repeated = json.loads(body)
                check("Save retry is idempotent", status == 200 and repeated["id"] == saved_id and repeated["n"] == 1)
                listed = json.loads(request("/api/novel")[1])
                check("Saved style persists in list", any(s["id"] == saved_id for s in listed)
                      and all("proto" not in s for s in listed))
                check("Delete temporary style", request("/api/novel/" + saved_id, "DELETE")[0] == 200)
                saved_id = None
                check("Deleted style token cannot save again", post_json("/api/novel", {"token": candidate["token"]})[0] == 410)
                check("Existing saved styles unchanged", json.loads(request("/api/novel")[1]) == original)
            finally:
                if saved_id:
                    # Delete only the exact new record created by this run.
                    listed = json.loads(request("/api/novel")[1])
                    if any(s["id"] == saved_id and s["name"] == marker for s in listed):
                        request("/api/novel/" + saved_id, "DELETE")
    finally:
        report = ROOT / "outputs/web/ui-review/backend-check.json"
        report.parent.mkdir(parents=True, exist_ok=True)
        report.write_text(json.dumps({"checks": checks, "passed": sum(c["passed"] for c in checks),
                                      "total": len(checks)}, ensure_ascii=False, indent=2), encoding="utf-8")
        print(f"Report: {report}", flush=True)


if __name__ == "__main__":
    main()
