"""G9c | Nguoi 4 - robustness (4.9) on the final test origins. Each variant reruns 04/05 -> 07 -> 08 with
LightGBM + CatBoost (refit every 4 origins) next to a reference run with the same settings.
Variants whose labels differ are sensitivity analyses, not matched comparisons (4.9.1, 4.9.4).

Output: outputs/metrics/robustness_test.csv
"""
import subprocess
import sys
from pathlib import Path

import pandas as pd
import yaml
from _common import ROOT, log, parse

from vstyle.config import CONFIG_DIR, path
from vstyle.io import load_json

PY = [sys.executable]
S = ROOT / "scripts"
MODELS = ["lightgbm", "catboost"]


def sh(*args):
    r = subprocess.run(PY + [str(a) for a in args], cwd=ROOT, capture_output=True, text=True, encoding="utf-8")
    if r.returncode:
        raise RuntimeError(r.stderr[-2000:])


def override(name, data):
    d = CONFIG_DIR / "robustness"
    d.mkdir(exist_ok=True)
    f = d / f"{name}.yaml"
    f.write_text(yaml.safe_dump(data, allow_unicode=True), encoding="utf-8")
    return [str(CONFIG_DIR / "default.yaml"), str(f)]


def forecast(dataset, cfgs, models=MODELS, extra=()):
    sh(S / "07_walk_forward.py", "--grouping", dataset, "--stage", "test", "--frozen", "--tag", "robust",
       "--refit-every", 4, "--models", *models, *extra, "--config", *cfgs)
    sh(S / "08_evaluate.py", "--grouping", dataset, "--stage", "test", "--tag", "robust", "--config", *cfgs)


if __name__ == "__main__":
    args, cfg = parse(__doc__)
    base = [str(CONFIG_DIR / "default.yaml")]
    runs = [("reference", "resnet50", base, None)]
    for k in [10, 15, 25, 30]:                                      # 4.9.1
        sh(S / "04_build_style_panel.py", "--grouping", "resnet50", "--k", k)
        sh(S / "05_build_features_targets.py", "--grouping", f"resnet50__k{k}")
        runs.append((f"K={k}", f"resnet50__k{k}", base, None))
    for q in [0.33, 0.40, 0.60, 0.67]:                              # 4.9.2
        v = f"thq{int(q * 100)}"
        c = override(v, {"target": {"threshold": {"quantile": q}}})
        sh(S / "05_build_features_targets.py", "--grouping", "resnet50", "--variant", v, "--config", *c)
        runs.append((f"theta = q{int(q * 100)} |Delta|", f"resnet50__{v}", c, None))
    for resp in ["raw_sales", "shop_exposure", "age_adjusted"]:     # 4.9.4
        c = override(resp, {"target": {"response": resp}})
        sh(S / "05_build_features_targets.py", "--grouping", "resnet50", "--variant", resp, "--config", *c)
        runs.append((f"target = {resp}", f"resnet50__{resp}", c, None))
    c = override("wshops", {"panel": {"weighting": "n_shops"}})    # weighting of (19)(21)
    sh(S / "04_build_style_panel.py", "--grouping", "resnet50", "--panel-id", "resnet50__wshops", "--config", *c)
    sh(S / "05_build_features_targets.py", "--grouping", "resnet50__wshops", "--config", *c)
    runs.append(("weighting = n_shops", "resnet50__wshops", c, None))

    rows = []
    for label, dataset, cfgs, _ in runs:
        log(f"robustness: {label}")
        forecast(dataset, cfgs)
        m = load_json(path(cfg, "metrics") / f"{dataset}_test_robust.json")
        t = load_json(path(cfg, "features") / f"target_{dataset}.json")
        row = {"variant": label, "theta": t["theta"]["theta"], "n_matched": m["n_matched"],
               "share_emerging": t["test"]["class_distribution"]["Emerging"]["share"],
               "share_declining": t["test"]["class_distribution"]["Declining"]["share"]}
        for mod in MODELS + ["persistence", "majority"]:
            row[mod] = m["models"][mod]["macro_f1"]
        for mod in MODELS:
            d = m["vs_reference"][mod]
            row[f"{mod}_vs_persistence_ci"] = f"[{d['ci_low']:+.3f}, {d['ci_high']:+.3f}]"
        rows.append(row)
    for L in [4, 8, 12]:                                            # 4.9.3
        log(f"robustness: lookback {L}")
        sh(S / "07_walk_forward.py", "--grouping", "resnet50", "--stage", "test", "--frozen", "--tag", f"lb{L}",
           "--models", "tcn", "--lookback", L, "--n-repeats", 3)
        sh(S / "08_evaluate.py", "--grouping", "resnet50", "--stage", "test", "--tag", f"lb{L}")
        m = load_json(path(cfg, "metrics") / f"resnet50_test_lb{L}.json")
        rows.append({"variant": f"TCN lookback = {L}", "n_matched": m["n_matched"], "tcn": m["models"]["tcn"]["macro_f1"],
                     "persistence": m["models"]["persistence"]["macro_f1"], "majority": m["models"]["majority"]["macro_f1"]})
    table = pd.DataFrame(rows)
    table.to_csv(path(cfg, "metrics") / "robustness_test.csv", index=False)
    log("\n" + table.round(3).to_string(index=False))
