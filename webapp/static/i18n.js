"use strict";
// Language selection is shared by the dashboard and technical scanner.
window.I18N = (() => {
  const url = new URL(location.href);
  let remembered = "vi";
  try { remembered = localStorage.getItem("style-scanner-language") || "vi"; } catch {}
  const requested = url.searchParams.get("lang");
  const lang = requested === "en" || requested === "vi" ? requested : remembered === "en" ? "en" : "vi";
  if (!["en", "vi"].includes(requested) && lang !== document.documentElement.lang) {
    url.searchParams.set("lang", lang);
    location.replace(url.href);
  }
  try { localStorage.setItem("style-scanner-language", lang); } catch {}
  const dictionary = window.EN_DICTIONARY || {};
  const keys = Object.keys(dictionary).sort((a, b) => b.length - a.length);
  const escapeRegex = (s) => s.replace(/[.*+?^${}()|[\]\\]/g, "\\$&");
  const pattern = new RegExp("(?<![\\p{L}\\p{N}])(?:" + keys.map(escapeRegex).join("|") + ")(?![\\p{L}\\p{N}])", "gu");
  function text(value) {
    if (lang !== "en" || typeof value !== "string") return value;
    if (Object.hasOwn(dictionary, value)) return dictionary[value];
    if (!/[ăâđêôơưĂÂĐÊÔƠƯ\u1ea0-\u1ef9àáãèéìíòóõùúý]/u.test(value)) return value;
    return value.replace(pattern, (match) => dictionary[match]);
  }
  const displayFields = new Set(["vi", "label", "name", "name_note", "description", "short", "category_vi", "color_vi", "catalogue_vi", "style_name", "reason", "evidence_note", "action", "actual_vi", "trend_vi", "signal_vi", "kept_vi", "outside_vocabulary", "family", "top_category", "top_color", "top_fabric", "evidence"]);
  const immutableFields = new Set(["data", "columns", "dtypes", "key", "term", "path", "image_path", "thumbs", "crop", "overlay", "isolated"]);
  function data(value, field = "") {
    if (lang !== "en" || immutableFields.has(field)) return value;
    if (Array.isArray(value)) return value.map((v) => data(v, field));
    if (value && typeof value === "object") {
      const result = Object.fromEntries(Object.entries(value).map(([k, v]) => [k, data(v, k)]));
      if (typeof value.vi === "string" && typeof value.term === "string") result.vi = text(value.term);
      if (typeof value.vi === "string" && typeof value.en === "string") result.vi = value.en;
      if (typeof value.short_en === "string") result.short = value.short_en;
      const groupLabels = { category: "Garment category", pattern: "Pattern", style: "Style", detail: "Details", fabric: "Fabric", context: "Image context" };
      if (Array.isArray(value.items) && groupLabels[value.key]) result.label = groupLabels[value.key];
      // Filenames and user-entered style names are preserved.
      if ("created" in value && "name" in value) result.name = value.name;
      return result;
    }
    return displayFields.has(field) ? text(value) : value;
  }
  function date(d, year = true) {
    return new Intl.DateTimeFormat("en-GB", { day: "2-digit", month: "short", ...(year ? { year: "numeric" } : {}), timeZone: "UTC" }).format(d);
  }
  function query(value) {
    const aliases = { "long-sleeved top": "long sleeve", "long sleeved top": "long sleeve", "short-sleeved top": "short sleeves", "sleeveless top": "sleeveless", "dropped-sleeve top": "drop sleeve", "babydoll dress": "doll dress", "plain top": "solid color top", "mid-length coat": "medium coat", "mid-length cardigan": "medium cardigan", "maxi dress": "maxi", "long duster coat": "long duster" };
    return lang === "en" ? aliases[value.trim().toLowerCase()] || value : value;
  }
  document.addEventListener("DOMContentLoaded", () => {
    const select = document.createElement("select");
    select.className = "language-select";
    select.setAttribute("aria-label", lang === "en" ? "Interface language" : "Ngôn ngữ giao diện");
    select.innerHTML = '<option value="vi">Tiếng Việt</option><option value="en">English</option>';
    select.value = lang;
    select.addEventListener("change", () => {
      const next = new URL(location.href);
      next.searchParams.set("lang", select.value);
      const origin = document.getElementById("origin");
      if (origin && /^\d+$/.test(origin.value)) next.searchParams.set("t", origin.value);
      const image = document.querySelector('#scanResult img[src^="/api/m/scans/"]');
      const scanId = image?.getAttribute("src")?.match(/\/scans\/([0-9a-f]{32})\/image/);
      if (scanId) next.searchParams.set("scan", scanId[1]);
      try { localStorage.setItem("style-scanner-language", select.value); } catch {}
      location.assign(next.href);
    });
    document.querySelector(".topbar-actions")?.prepend(select);
    document.querySelectorAll('a[href="/"],a[href="/scanner"]').forEach((a) => {
      const target = new URL(a.href);
      target.searchParams.set("lang", lang);
      a.href = target.href;
    });
  });
  return { lang, text, data, date, query };
})();
