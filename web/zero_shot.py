"""Zero-shot description with the FashionCLIP text tower.

Each label is scored with several prompt templates whose text embeddings are averaged (prompt ensembling),
which is steadier than a single template. Groups mix the VISUELLE 2.0 vocabulary (known=True) with an open
vocabulary (known=False), so an item the catalogue never sold can still be named. Colour and stripes are
measured on the pixels instead (web/colors.py): CLIP is unreliable for both.
"""
import numpy as np

_CATEGORY_KNOWN = [
    ("capris", "quần lửng"), ("culottes", "quần ống rộng culottes"), ("doll dress", "váy babydoll"),
    ("drop sleeve top", "áo tay trễ vai"), ("gitana skirt", "chân váy xoè gitana"), ("jumpsuit", "jumpsuit"),
    ("kimono dress", "váy kimono"), ("long cardigan", "áo cardigan dài"), ("long coat", "áo khoác dáng dài"),
    ("long dress", "váy dài"), ("long duster coat", "áo khoác duster"), ("long sleeve top", "áo dài tay"),
    ("maxi dress", "váy maxi"), ("medium cardigan", "áo cardigan"), ("medium coat", "áo khoác dáng vừa"),
    ("miniskirt", "chân váy ngắn"), ("patterned top", "áo họa tiết"), ("printed shirt", "áo in hình"),
    ("blouse", "áo kiểu"),  # VISUELLE sells blouses, filed as long/short/drop sleeve tops
    ("sheath dress", "váy ôm"), ("shirt dress", "váy sơ mi"), ("short cardigan", "áo cardigan ngắn"),
    ("short coat", "áo khoác ngắn"), ("short sleeve top", "áo ngắn tay"), ("shorts", "quần short"),
    ("sleeveless top", "áo sát nách"), ("solid color top", "áo trơn"), ("trapeze dress", "váy suông chữ A"),
]
_CATEGORY_NEW = [
    ("soccer jersey", "áo đấu bóng đá"), ("basketball jersey", "áo bóng rổ"), ("sports t-shirt", "áo thể thao"),
    ("basic t-shirt", "áo thun"), ("polo shirt", "áo polo"), ("hoodie", "áo hoodie"), ("sweatshirt", "áo nỉ"),
    ("tracksuit", "bộ đồ thể thao"), ("denim jacket", "áo khoác jeans"), ("bomber jacket", "áo bomber"),
    ("puffer jacket", "áo phao"), ("leather jacket", "áo khoác da"), ("business suit", "bộ vest"),
    ("blazer", "áo blazer"), ("formal shirt", "áo sơ mi"), ("jeans", "quần jeans"),
    ("skinny jeans", "quần jeans bó"), ("cargo pants", "quần túi hộp"), ("trousers", "quần tây"), ("leggings", "quần legging"),
    ("swimsuit", "đồ bơi"), ("bikini", "bikini"), ("lingerie", "đồ lót"), ("pajamas", "đồ ngủ"),
    ("school uniform", "đồng phục học sinh"), ("vietnamese ao dai", "áo dài Việt Nam"), ("indian saree", "saree Ấn Độ"),
    ("wedding dress", "váy cưới"), ("evening gown", "đầm dạ hội"), ("sneakers", "giày sneaker"),
    ("handbag", "túi xách"), ("baseball cap", "mũ lưỡi trai"), ("scarf", "khăn choàng"),
]
_PATTERN = [("solid plain color", "trơn"), ("vertical stripes", "sọc dọc"), ("horizontal stripes", "sọc ngang"),
            ("floral print", "hoa"), ("paisley print", "hoa văn paisley"), ("checked plaid", "kẻ caro"),
            ("polka dots", "chấm bi"), ("animal print", "da thú"), ("graphic print", "in hình"),
            ("camouflage", "rằn ri"), ("lace texture", "ren"), ("color blocking", "phối mảng màu"),
            ("geometric print", "hình học"), ("tie-dye", "loang màu")]
_DETAIL = [("no visible details", "không có chi tiết nổi bật"), ("a sponsor logo", "logo nhà tài trợ"),
           ("a team crest badge", "huy hiệu đội"), ("a brand logo", "logo thương hiệu"), ("embroidery", "thêu"),
           ("buttons", "hàng cúc"), ("a zipper", "khoá kéo"), ("ruffles", "bèo nhún"), ("a hood", "mũ trùm"),
           ("pockets", "túi"), ("a belt", "thắt lưng"), ("sequins", "kim sa"), ("a bow", "nơ"), ("pleats", "xếp ly")]
_STYLE = [("sporty athletic", "thể thao"), ("streetwear", "đường phố"), ("casual everyday", "thường ngày"),
          ("formal business", "công sở"), ("elegant evening", "dạ tiệc, sang trọng"), ("bohemian", "bohemian"),
          ("vintage retro", "cổ điển"), ("minimalist", "tối giản"), ("romantic feminine", "nữ tính"),
          ("preppy", "học đường"), ("rock punk", "cá tính"), ("beachwear", "đi biển")]
_FABRIC_KNOWN = [("cotton", "cotton"), ("linen", "vải lanh"), ("lace", "ren"), ("velvet", "nhung"), ("fur", "lông"),
                 ("faux leather", "da giả"), ("light denim", "jeans sáng màu"), ("dark denim", "jeans tối màu"),
                 ("georgette", "voan georgette"), ("silky satin", "lụa satin"), ("tulle", "vải tulle"),
                 ("crepe", "crepe"), ("mohair wool", "len mohair"), ("nylon", "nylon"), ("matte jersey", "thun lì"),
                 ("shiny jersey", "thun bóng"), ("jacquard", "jacquard"), ("sequined fabric", "vải kim sa"),
                 ("pleated fabric", "vải xếp ly")]
_FABRIC_NEW = [("polyester sports mesh", "lưới polyester thể thao"), ("knitted wool", "len đan"),
               ("genuine leather", "da thật"), ("silk", "lụa"), ("fleece", "nỉ"), ("spandex", "thun co giãn")]
_CONTEXT = [("a product photo of a garment on a plain white background", "ảnh sản phẩm nền trơn", True),
            ("a photo of a person wearing the clothes", "ảnh có người mặc", False),
            ("a photo of a football player in a team kit", "ảnh cầu thủ", False),
            ("a fashion runway photo", "ảnh sàn diễn", False),
            ("a street style photo", "ảnh đường phố", False),
            ("a flat lay photo of clothes", "ảnh trải phẳng", False)]

GROUPS = [
    {"key": "category", "label": "Loại sản phẩm",
     "templates": ["a photo of a {}.", "a product photo of a {}.", "a photo of a person wearing a {}.",
                   "a close-up photo of a {}, fashion item."],
     "items": [(t, v, True) for t, v in _CATEGORY_KNOWN] + [(t, v, False) for t, v in _CATEGORY_NEW]},
    {"key": "pattern", "label": "Họa tiết",
     "templates": ["a photo of a garment with a {} pattern.", "clothing with {}.", "fabric with {}."],
     "items": [(t, v, True) for t, v in _PATTERN]},
    {"key": "detail", "label": "Chi tiết",
     "templates": ["a photo of a garment with {}.", "clothing featuring {}."],
     "items": [(t, v, True) for t, v in _DETAIL]},
    {"key": "style", "label": "Phong cách",
     "templates": ["a photo of {} style clothing.", "a {} fashion outfit.", "a garment in {} style."],
     "items": [(t, v, True) for t, v in _STYLE]},
    {"key": "fabric", "label": "Chất liệu",
     "templates": ["clothing made of {} fabric.", "a close-up of {} fabric.", "a garment in {}."],
     "items": [(t, v, True) for t, v in _FABRIC_KNOWN] + [(t, v, False) for t, v in _FABRIC_NEW]},
    {"key": "context", "label": "Ngữ cảnh ảnh", "templates": ["{}"], "items": _CONTEXT},
]
STRIPES_VI = {"vertical": "sọc dọc", "horizontal": "sọc ngang"}


def text_bank(encode_text):
    """Ensembled, normalised text embeddings per group; encode_text(list[str]) -> (n, d) normalised."""
    bank = {}
    for g in GROUPS:
        e = np.stack([encode_text([t.format(x) for x, _, _ in g["items"]]) for t in g["templates"]]).mean(0)
        bank[g["key"]] = e / np.linalg.norm(e, axis=1, keepdims=True)
    return bank


def describe(z, bank, scale, known_only=False, top=3, pixels=None):
    """Softmax over each group; `known_only` restricts to the VISUELLE vocabulary. `pixels`
    (web/colors.analyse output) replaces the pattern with measured stripes and supplies the colours."""
    groups, best = [], {}
    for g in GROUPS:
        logits = scale * (bank[g["key"]] @ z)
        if known_only:
            logits = np.where([k for _, _, k in g["items"]], logits, -np.inf)
        p = np.exp(logits - logits.max())
        p /= p.sum()
        items = [{"term": g["items"][i][0], "vi": g["items"][i][1], "known": g["items"][i][2],
                  "p": round(float(p[i]), 4), "source": "clip"} for i in np.argsort(-p)[:top]]
        direction = (pixels or {}).get("stripes")
        if g["key"] == "pattern" and direction:
            term = direction + " stripes"
            measured = {"term": term, "vi": STRIPES_VI[direction], "known": True, "p": None, "source": "pixels"}
            items = [measured] + [i for i in items if i["term"] != term][:top - 1]
        best[g["key"]] = items[0]
        groups.append({"key": g["key"], "label": g["label"], "items": items})
    colors = (pixels or {}).get("colors", [])
    short_vi, short_en = [best["category"]["vi"]], [best["category"]["term"]]
    if best["pattern"]["term"] != "solid plain color":
        short_vi.append(best["pattern"]["vi"])
        short_en.append(best["pattern"]["term"])
    if colors:
        short_vi.append(" – ".join((pixels or {}).get("families", [c["vi"] for c in colors])[:2]))
        short_en.append(" and ".join(c["en"] for c in colors[:2]))
    unknown = [g["label"] for g in groups if g["key"] in ("category", "fabric") and not g["items"][0]["known"]]
    return {"groups": groups, "colors": colors, "short": " · ".join(short_vi), "short_en": " · ".join(short_en),
            "outside_vocabulary": unknown}

# VISUELLE catalogue attribute values -> Vietnamese (for naming the FashionCLIP clusters from their profile)
CATALOGUE_VI = {
    "capris": "quần lửng", "culottes": "quần culottes", "doll dress": "váy babydoll", "drop sleeve": "áo trễ vai",
    "gitana skirt": "chân váy gitana", "jumpsuit": "jumpsuit", "kimono dress": "váy kimono",
    "long cardigan": "cardigan dài", "long coat": "áo khoác dài", "long dress": "váy dài",
    "long duster": "áo khoác duster", "long sleeve": "áo dài tay", "maxi": "váy maxi",
    "medium cardigan": "cardigan", "medium coat": "áo khoác vừa", "miniskirt": "chân váy ngắn",
    "patterned top": "áo họa tiết", "printed shirt": "áo in hình", "sheath dress": "váy ôm",
    "shirt dress": "váy sơ mi", "short cardigan": "cardigan ngắn", "short coat": "áo khoác ngắn",
    "short sleeves": "áo ngắn tay", "shorts": "quần short", "sleeveless": "áo sát nách",
    "solid color top": "áo trơn", "trapeze dress": "váy suông",
    "black": "đen", "blue": "xanh dương", "brown": "nâu", "green": "xanh lá", "grey": "xám", "orange": "cam",
    "red": "đỏ", "violet": "tím", "white": "trắng", "yellow": "vàng",
}
