"""G0 | Nguoi 1 - temporal and structural audit of VISUELLE 2.0 (Section 4 intro, 3.4, 4.2.1).

Outputs: outputs/reports/G0_data_audit.json, G0_data_audit.md, G0_active_products_by_week.csv
"""
from _common import log, parse

from vstyle.config import path
from vstyle.data import audit
from vstyle.io import save_json
from vstyle.utils.repro import write_manifest


def checks(r):
    s, g, img, pd_, rs = r["sales"], r["gtrends"], r["images"], r["price_discount"], r["restocks"]
    return [
        ("PASS" if s["duplicate_item_shop"] == 0 else "FAIL", "Khoa item-shop duy nhat trong sales.csv"),
        ("PASS" if pd_["duplicate_item_shop"] == 0 else "FAIL", "Khoa item-shop duy nhat trong price_discount_series.csv"),
        ("PASS" if not img["products_missing_image"] else "FAIL", f"Moi san pham co anh ({img['products_with_image']})"),
        ("PASS" if not img["products_undecodable_image"] else "DECIDE",
         f"Anh khong giai ma duoc: {img['products_undecodable_image']} -> loai khoi moi grouping (D14)"),
        ("PASS" if s["release_weekday_share_monday"] == 1 else "DECIDE", "release_date deu la thu Hai -> khong co partial week"),
        ("DECIDE" if s["share_products_multi_release_date"] > 0 else "PASS",
         f"release_date khac nhau giua cac shop ({s['share_products_multi_release_date']:.1%} san pham) -> (12) ap dung o cap item-shop"),
        ("DECIDE" if s["zero_share_by_relative_week"][0] < 0.1 * s["zero_share_by_relative_week"][1] else "PASS",
         f"Ti le tuan 0 bang 0 = {s['zero_share_by_relative_week'][0]:.2%} (tuan 1: {s['zero_share_by_relative_week'][1]:.2%}) "
         "-> release_date gan nhu la tuan ban dau tien (left truncation)"),
        ("PASS" if s["missing_sales_cells"] == 0 else "FAIL", "Khong co o sales bi thieu"),
        ("DECIDE" if s["negative_sales_cells"] else "PASS",
         f"{s['negative_sales_cells']} o sales am (net sales / hang tra lai?) tap trung o tuan "
         f"{s['negative_cells_by_relative_week']}"),
        ("PASS" if g["is_weekly_without_gaps"] and g["weekday_share_monday"] == 1 else "FAIL", "Google Trends la chuoi tuan (thu Hai) lien tuc"),
        ("DECIDE" if g["sentinel_cells"] else "PASS", f"Google Trends co {g['sentinel_cells']} o gia tri -1 (missing)"),
        ("DECIDE" if g["terms_hitting_100"] else "PASS",
         f"{g['terms_hitting_100']}/{g['n_terms']} chuoi GT dat 100 -> chuan hoa tren toan cua so (leak thang do)"),
        ("PASS" if not any(g["attribute_terms_missing_in_gt"].values()) else "FAIL", "Moi category/color/fabric co chuoi GT"),
        ("DECIDE", f"GT ket thuc {g['last']:%Y-%m-%d}, panel ban hang ket thuc {s['last_observed_week']:%Y-%m-%d}"),
        ("DECIDE", f"restocks.csv: {rs['item_shops']} item-shop ({rs['item_shops_also_in_sales']} co trong sales), "
                   f"nam {min(rs['year_counts'])}-{max(rs['year_counts'])} -> inventory tat cho toi khi chot ngu nghia"),
        ("PASS" if r["panel"]["weeks_with_no_active_product"] == 0 else "DECIDE",
         f"Tuan khong co san pham active: {r['panel']['weeks_with_no_active_product']}"),
    ]


def to_markdown(r, rows):
    lines = ["# G0 - Kiem toan du lieu VISUELLE 2.0", "",
             "Sinh tu `scripts/00_audit_data.py`. PASS = dat; FAIL = phai sua truoc khi di tiep; "
             "DECIDE = phai ghi quyet dinh vao `docs/decision_log.md`.", "",
             "| Trang thai | Kiem tra |", "| --- | --- |"]
    lines += [f"| {st} | {msg} |" for st, msg in rows]
    lines += ["", "## So lieu chinh", "", "```json"]
    import json
    lines += [json.dumps({k: r[k] for k in ("sales", "panel")}, indent=2, default=str, ensure_ascii=False), "```"]
    return "\n".join(lines) + "\n"


if __name__ == "__main__":
    args, cfg = parse(__doc__)
    log("running audit ...")
    report, weekly = audit.run(cfg)
    rows = checks(report)
    out = path(cfg, "reports")
    save_json(report, out / "G0_data_audit.json")
    (out / "G0_data_audit.md").write_text(to_markdown(report, rows), encoding="utf-8")
    weekly.rename("active_products").to_csv(out / "G0_active_products_by_week.csv", index_label="week_start")
    write_manifest("00_audit_data", cfg, outputs=[out / "G0_data_audit.json", out / "G0_data_audit.md"])
    for st, msg in rows:
        log(f"{st:6s} {msg}")
    n_fail = sum(st == "FAIL" for st, _ in rows)
    log(f"G0 automated checks: {'FAIL' if n_fail else 'PASS'} ({n_fail} fail, "
        f"{sum(st == 'DECIDE' for st, _ in rows)} decisions to record)")
