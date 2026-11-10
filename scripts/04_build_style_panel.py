"""G3b | Nguoi 2 (fashionclip) / Nguoi 4 (resnet50) / Nguoi 1 (attribute benchmark) - style-week panel (4.2.2-4.2.4).

--grouping fashionclip|resnet50 needs visual.selected.k set after G2; `attribute` needs nothing.
Output: outputs/panels/style_week_{grouping}.parquet
"""
import pandas as pd
from _common import log, parse

from vstyle.config import path
from vstyle.io import load_json, load_table, save_json, save_table
from vstyle.panel.style_week import build_style_week, conservation_check
from vstyle.utils.repro import write_manifest


def style_mapping(grouping, products, cfg, k=None):
    if grouping == "attribute":
        keys = cfg["grouping"]["attribute_keys"]
        usable = products[products["image_ok"]]                                # D14: same sample as visual
        return usable.set_index("external_code")[keys].astype(str).agg("|".join, axis=1)
    k = k or cfg["visual"]["selected"]["k"]
    if k is None:
        raise SystemExit("visual.selected.k is not set: finish G2 and record the choice first.")
    a = load_table(path(cfg, "clusters") / grouping / f"assignments_k{k}.parquet")
    return a.set_index("external_code")["style"]


if __name__ == "__main__":
    def extra(ap):
        ap.add_argument("--grouping", required=True, choices=["fashionclip", "resnet50", "resnet50ft", "attribute"])
        ap.add_argument("--k", type=int, default=None, help="robustness: K other than visual.selected.k")
        ap.add_argument("--panel-id", default=None, help="output id, default = grouping (or grouping__kK)")
    args, cfg = parse(__doc__, extra)
    pid = args.panel_id or (f"{args.grouping}__k{args.k}" if args.k else args.grouping)
    interim = path(cfg, "interim")
    products, pw = load_table(interim / "products.parquet"), load_table(interim / "product_week.parquet")
    cal = load_json(interim / "calendar.json")
    style_of = style_mapping(args.grouping, products, cfg, args.k)
    panel = build_style_week(pw, style_of, cal["t_max"], cal["origin_week"],
                             cfg["panel"]["weighting"], cfg["panel"]["new_product_age_weeks"])
    check = conservation_check(pw, panel, style_of)
    check.update({"n_styles": int(panel["style"].nunique()), "n_rows": len(panel),
                  "share_inactive_style_weeks": float((panel["active_products"] == 0).mean()),
                  "negative_style_week_sales": int((panel["sales"] < 0).sum())})
    assert check["equal"], check
    out = path(cfg, "panels")
    files = [save_table(panel, out / f"style_week_{pid}.parquet"),
             save_json(check, out / f"style_week_{pid}_G3.json")]
    write_manifest(f"04_style_panel_{pid}", cfg, outputs=files, extra=check)
    log(check)
