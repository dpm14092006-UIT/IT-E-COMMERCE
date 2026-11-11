"""Final report - collects every result table of Section 4 into outputs/reports/FINAL_RESULTS.md (+ figures).

Narrative interpretation is written in docs/04_KET_QUA.md; this file only assembles numbers.
"""
import sys

import matplotlib
import numpy as np
import pandas as pd
from _common import ROOT, log, parse

from vstyle.config import path
from vstyle.constants import CLASSES
from vstyle.io import load_json, load_table

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

sys.path.insert(0, str(ROOT))
from figure_style import apply_theme  # noqa: E402

PRIMARY = "resnet50"


def md(df, digits=3):
    return df.round(digits).to_markdown(index=False)


def model_table(m):
    rows = []
    for name, s in m["models"].items():
        d = m.get("vs_reference", {}).get(name)
        rows.append({"model": name, "Macro-F1": s["macro_f1"], "Balanced acc.": s["balanced_accuracy"],
                     "Accuracy": s["accuracy"], "P(E)": s["per_class"]["Emerging"]["precision"],
                     "R(E)": s["per_class"]["Emerging"]["recall"], "P(D)": s["per_class"]["Declining"]["precision"],
                     "R(D)": s["per_class"]["Declining"]["recall"],
                     "vs persistence [95% CI]": f"{d['estimate']:+.3f} [{d['ci_low']:+.3f}, {d['ci_high']:+.3f}]" if d else ""})
    return pd.DataFrame(rows).sort_values("Macro-F1", ascending=False)


if __name__ == "__main__":
    args, cfg = parse(__doc__)
    M, R, F = path(cfg, "metrics"), path(cfg, "reports"), path(cfg, "features")
    apply_theme(journal=False)
    out = ["# FINAL RESULTS - Visual-style trend forecasting (VISUELLE 2.0)", "",
           "Sinh tự động bởi `scripts/14_final_report.py`. Diễn giải: `docs/04_KET_QUA.md`.", ""]

    g0 = load_json(R / "G0_data_audit.json")
    cal = load_json(path(cfg, "interim") / "calendar.json")
    out += ["## 1. Dữ liệu", "",
            f"- Sản phẩm: {g0['sales']['n_products']} (dùng {g0['sales']['n_products'] - len(g0['images']['products_undecodable_image'])}, loại ảnh hỏng {g0['images']['products_undecodable_image']}); shop: {g0['sales']['n_shops']}; item–shop: {g0['sales']['rows_item_shop']:,}.",
            f"- Panel tuần lịch: {cal['origin_week'][:10]} → {str(cal['last_week'])[:10]} ({cal['t_max'] + 1} tuần).", ""]

    # RQ1
    g2 = pd.read_csv(R / "G2_encoder_comparison.csv")
    out += ["## 2. RQ1 - Visual style identification", "",
            md(g2[["encoder", "k", "silhouette", "davies_bouldin", "ari_mean", "ari_sd", "size_min",
                   "min_eligible_dev_origins", "nmi_category", "nmi_color", "nmi_fabric", "pass_all"]]), "",
            f"Cấu hình chính (D09): **{cfg['visual']['selected']['encoder']}, K = {cfg['visual']['selected']['k']}**. "
            "Ảnh cấu hình chính: `outputs/clusters/resnet50/overview_k20.png` (các overview khác sinh lại bởi `scripts/03_cluster_styles.py`).", ""]
    names = path(cfg, "clusters") / PRIMARY / f"style_names_k{cfg['visual']['selected']['k']}_draft.csv"
    if names.exists():
        out += [md(pd.read_csv(names)), ""]

    # RQ2
    t = load_json(F / f"target_{PRIMARY}.json")
    m = load_json(M / f"{PRIMARY}_test_primary.json")
    out += ["## 3. RQ2 - Final test, ResNet-50 K=20 (matched sample)", "",
            f"- θ = {t['theta']['theta']:.3f} ({t['theta']['rule']}, n calibration = {t['theta']['n_calibration']}).",
            f"- Test origins {t['test']['origins']}; dự báo {t['test']['n_forecasted']}, chấm được {t['test']['n_scorable']}, coverage {t['test']['coverage']:.3f}.",
            "- Phân bố lớp test: " + ", ".join(f"{c} {v['share']:.0%}" for c, v in t["test"]["class_distribution"].items()),
            f"- Matched rows: {m['n_matched']}.", "", md(model_table(m)), ""]
    seeds_f = path(cfg, "predictions") / f"{PRIMARY}_test_primary_seeds.parquet"
    if seeds_f.exists():
        from vstyle.evaluation.metrics import macro_f1
        sd = load_table(seeds_f)
        per = (sd[sd["y_true"].notna()].groupby(["model", "seed"])
                 .apply(lambda g: macro_f1(g["y_true"].astype(int), g["y_pred"])).groupby("model").agg(["mean", "std", "count"]))
        out += ["Seed của mô hình chuỗi (Macro-F1 từng seed; bảng trên dùng trung bình xác suất):", "",
                md(per.reset_index()), ""]
    q = pd.DataFrame(m["by_quarter"]).T
    out += ["Macro-F1 theo quý:", "", md(q.reset_index().rename(columns={"index": "quarter"})), ""]
    fig, ax = plt.subplots(figsize=(8, 3.4), constrained_layout=True)
    for col in q.columns:
        ax.plot(q.index, q[col], marker="o", label=col, lw=1.6 if col not in ("majority", "persistence") else 1, ls="-" if col not in ("majority", "persistence") else "--")
    ax.set_ylabel("Macro-F1")
    ax.set_title("Final test - Macro-F1 by quarter (ResNet-50 K=20)")
    ax.legend(ncol=4, fontsize=8)
    fig.savefig(R / "FINAL_by_quarter.png", dpi=200)
    best = model_table(m).iloc[0]["model"]
    cm = np.array(m["models"][best]["confusion_matrix"])
    fig, ax = plt.subplots(figsize=(3.8, 3.4), constrained_layout=True)
    ax.imshow(cm / cm.sum(1, keepdims=True), cmap="Blues", vmin=0, vmax=1)
    for i in range(3):
        for j in range(3):
            ax.text(j, i, f"{cm[i, j]}\n{cm[i, j] / cm[i].sum():.0%}", ha="center", va="center", fontsize=8)
    ax.set_xticks(range(3), CLASSES, rotation=20)
    ax.set_yticks(range(3), CLASSES)
    ax.set_xlabel("Predicted")
    ax.set_ylabel("True")
    ax.set_title(f"Confusion - {best}")
    fig.savefig(R / "FINAL_confusion.png", dpi=200)
    out += [f"Confusion matrix ({best}): `FINAL_confusion.png`; theo quý: `FINAL_by_quarter.png`.", ""]

    # RQ3 - style-level groupings (sensitivity only) + common endpoint (matched)
    rows = []
    for g in ("resnet50", "resnet50ft", "fashionclip", "attribute"):
        f = M / f"{g}_test_primary.json"
        if f.exists():
            mm = load_json(f)
            tt = load_json(F / f"target_{g}.json")
            bestg = max((k for k in mm["models"] if k not in ("majority", "persistence")),
                        key=lambda k: mm["models"][k]["macro_f1"])
            rows.append({"grouping": g, "n_styles": int(load_table(path(cfg, "panels") / f"style_week_{g}.parquet")["style"].nunique()),
                         "theta": tt["theta"]["theta"], "n_matched": mm["n_matched"], "best model": bestg,
                         "best Macro-F1": mm["models"][bestg]["macro_f1"],
                         "persistence": mm["models"]["persistence"]["macro_f1"], "majority": mm["models"]["majority"]["macro_f1"]})
    out += ["## 4. RQ3 - Giá trị của visual grouping", "",
            "### 4a. Style-level theo từng grouping (nhãn khác nhau → chỉ là sensitivity, không so trực tiếp)", "",
            md(pd.DataFrame(rows)), ""]
    ep = M / "endpoint_test_table.csv"
    if ep.exists():
        e = load_json(M / "endpoint_test.json")
        out += ["### 4b. Common endpoint product–week (cùng hàng, cùng nhãn → so sánh hợp lệ)", "",
                f"θ product = {e['theta']['theta']:.3f}; product-week chấm được: {e['n_scored_rows']}; lớp: "
                + ", ".join(f"{c} {v['share']:.0%}" for c, v in e["class_distribution"].items()), "",
                pd.read_csv(ep).round(3).to_markdown(index=False), "",
                md(pd.DataFrame([{"so sánh": k, "ΔMacro-F1": v["estimate"], "CI low": v["ci_low"], "CI high": v["ci_high"]}
                                 for k, v in e["diffs"].items()])), ""]

    ab = M / f"{PRIMARY}_test_ablation.csv"
    if ab.exists():
        a = pd.read_csv(ab)
        out += ["## 5. Feature ablation (test, ResNet-50 K=20)", "",
                "B1 sales level → B2 dynamics → B3 operations → B4a calendar → B4b Google Trends; "
                "M = chỉ rollout/vòng đời/calendar.", "", a.round(3).to_markdown(index=False), ""]
    rb = M / "robustness_test.csv"
    if rb.exists():
        out += ["## 6. Robustness (test; LightGBM + CatBoost, refit mỗi 4 origin)", "",
                pd.read_csv(rb).round(3).to_markdown(index=False), ""]
    imp = M / f"{PRIMARY}_test_catboost_importance.csv"
    if imp.exists():
        i = pd.read_csv(imp)
        blk = pd.read_csv(M / f"{PRIMARY}_test_catboost_importance_blocks.csv")
        out += ["## 7. Explainability (CatBoost, permutation importance trên Macro-F1 test; liên hệ, không nhân quả)", "",
                md(i.head(15)), "", md(blk), ""]
    dec = R / f"{PRIMARY}_decision.json"
    if dec.exists():
        d = load_json(dec)
        out += ["## 8. Decision layer (Người 4)", "",
                f"Điểm vận hành chọn trên validation: {d['operating_point']}; áp lên {d['stage']}:", "",
                md(pd.DataFrame([d["quality"]])), ""]
    (R / "FINAL_RESULTS.md").write_text("\n".join(out), encoding="utf-8")
    log(f"wrote {R / 'FINAL_RESULTS.md'}")
