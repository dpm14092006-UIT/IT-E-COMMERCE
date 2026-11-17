"use strict";
/* Manager console. Talks to web/manager.py (/api/m/*) and, for the detailed technical view, links to /scanner. */

const TREND = [
  { vi: "Đang giảm", icon: "↘" },
  { vi: "Ổn định", icon: "→" },
  { vi: "Đang lên", icon: "↗" },
];
const SIGNAL = {
  emerging_alert: { vi: "Ưu tiên mở rộng", icon: "▲" },
  declining_alert: { vi: "Rà soát nhập hàng", icon: "▼" },
  hold: { vi: "Duy trì, theo dõi", icon: "●" },
};
const SIGNAL_ORDER = ["emerging_alert", "declining_alert", "hold"];
const SEGMENT = [["auto", "Tự động"], ["main", "Chỉ giữ món đồ chính"], ["all", "Giữ toàn bộ trang phục"], ["off", "Không tách nền"]];
const NDASH = "–";

const $ = (id) => document.getElementById(id);
const ESC = { "&": "&amp;", "<": "&lt;", ">": "&gt;", '"': "&quot;", "'": "&#39;" };
const esc = (s) => String(s ?? "").replace(/[&<>"']/g, (c) => ESC[c]);
const cap = (s) => (s ? s.charAt(0).toUpperCase() + s.slice(1) : s);
const int = (v) => Math.round(v).toLocaleString("vi-VN");
const dec = (v, d = 1) => Number(v).toLocaleString("vi-VN", { minimumFractionDigits: d, maximumFractionDigits: d });
const pct = (v, d = 0) => (v * 100).toLocaleString("vi-VN", { maximumFractionDigits: d }) + "%";
const compact = (v) => (v >= 1e6 ? dec(v / 1e6, 1) + " tr" : v >= 1e3 ? dec(v / 1e3, v >= 1e4 ? 0 : 1) + "K" : int(v));
const bytes = (b) => (b >= 1 << 20 ? dec(b / (1 << 20), 1) + " MB" : b >= 1024 ? dec(b / 1024, 0) + " KB" : b + " B");

const state = { origins: null, t: null, connected: false, scan: { file: null, url: null, busy: false, result: null },
  storage: { q: "", category: "", color: "", season: "", fabric: "", sort: "units", page: 1 }, catSort: "units", catQuery: "" };
const CHARTS = new Map();
let renderToken = 0;

/* ---------- api ---------- */
async function api(path, options = {}, timeoutMs = 20000) {
  const controller = new AbortController();
  const timer = setTimeout(() => controller.abort(), timeoutMs);
  try {
    const response = await fetch(path, { ...options, cache: "no-store", signal: controller.signal });
    const text = await response.text();
    let body = null;
    try { body = text ? JSON.parse(text) : null; } catch { body = null; }
    if (!response.ok) throw new Error((body && body.detail) || `Máy chủ trả lỗi ${response.status}`);
    return body;
  } catch (error) {
    if (error.name === "AbortError") throw new Error("Máy chủ phản hồi quá lâu. Hãy thử lại.");
    if (error instanceof TypeError) throw new Error("Không kết nối được máy chủ.");
    throw error;
  } finally {
    clearTimeout(timer);
  }
}
const withT = (path) => path + (path.includes("?") ? "&" : "?") + "t=" + state.t;

/* ---------- dates ---------- */
function weekDate(t) {
  const d = new Date(state.origins.week0 + "T00:00:00Z");
  d.setUTCDate(d.getUTCDate() + 7 * t);
  return d;
}
const fmt = (d) => `${String(d.getUTCDate()).padStart(2, "0")}/${String(d.getUTCMonth() + 1).padStart(2, "0")}/${d.getUTCFullYear()}`;
const fmtShort = (d) => `${String(d.getUTCDate()).padStart(2, "0")}/${String(d.getUTCMonth() + 1).padStart(2, "0")}`;
const iso = (s) => fmt(new Date(s.slice(0, 10) + "T00:00:00Z"));
function windowText(t) {
  const end = weekDate(t + state.origins.horizon);
  end.setUTCDate(end.getUTCDate() + 6);
  return `${fmtShort(weekDate(t + 1))} ${NDASH} ${fmt(end)}`;
}
const seasonVi = (s) => ({ SS: "Xuân Hè", AW: "Thu Đông" }[s.slice(0, 2)] ? `${{ SS: "Xuân Hè", AW: "Thu Đông" }[s.slice(0, 2)]} 20${s.slice(2)}` : s);

/* ---------- small UI ---------- */
function toast(msg) {
  const el = document.createElement("div");
  el.className = "toast";
  el.setAttribute("role", "status");
  el.textContent = msg;
  document.body.appendChild(el);
  setTimeout(() => el.remove(), 3600);
}
function connection(stateName, label) {
  $("connection").dataset.state = stateName;
  $("connectionLabel").textContent = label;
}
const kpi = (label, value, hint = "") => `<div class="kpi"><div class="label">${label}</div><div class="value">${value}</div>${hint ? `<div class="hint">${hint}</div>` : ""}</div>`;
const thumb = (code, ok = true, size = 120) => ok
  ? `<img class="thumb" src="/catalog/${code}.jpg?size=${size}" alt="Sản phẩm ${code}" loading="lazy">`
  : `<span class="thumb ph"><svg class="icon"><use href="#i-image"/></svg></span>`;
const trendBadge = (o) => o ? `<span class="badge t${o.trend}">${TREND[o.trend].icon} ${TREND[o.trend].vi} · ${pct(o.confidence)}</span>` : `<span class="badge plain">Chưa có dự báo</span>`;
const signalBadge = (o) => o ? `<span class="badge s-${o.signal}">${SIGNAL[o.signal].icon} ${SIGNAL[o.signal].vi}</span>` : "";
function tbar(p, tipLabel = "") {
  return `<div class="tbar" role="img" aria-label="Xác suất: giảm ${pct(p[0])}, ổn định ${pct(p[1])}, tăng ${pct(p[2])}">${p.map((v, i) =>
    v > 0.004 ? `<i class="c${i}" style="flex:${v}" data-tip="${esc(tipLabel)}<b>${TREND[i].vi}</b>: ${pct(v, 1)}"></i>` : "").join("")}</div>`;
}
const tlegend = (p) => `<div class="tlegend">${[0, 1, 2].map((i) => `<span><i class="tbar-dot" style="background:${["var(--down)", "#a3adbd", "var(--up)"][i]}"></i>${TREND[i].vi} ${pct(p[i])}</span>`).join("")}</div>`;
function actualText(o) {
  if (!o || o.actual_vi == null) return "";
  const hit = o.y_true === o.trend;
  return `<span class="small ${hit ? "" : "muted"}">Thực tế sau đó: <b>${esc(o.actual_vi)}</b> ${hit ? "✓ đúng" : "✗ lệch"}</span>`;
}
function outlookBlock(o, subject) {
  if (!o) return `<p class="muted small">Chưa có dự báo cho tuần này.</p>`;
  return `<div class="outlook-hero t${o.trend}"><span class="arrow" aria-hidden="true">${TREND[o.trend].icon}</span><div>
      <h3>${TREND[o.trend].vi}</h3><p>${esc(subject)} · khả năng ${pct(o.confidence)} · 4 tuần ${windowText(state.t)}</p></div></div>
    ${tbar(o.p)}${tlegend(o.p)}
    <div class="action s-${o.signal}"><b>${SIGNAL[o.signal].icon} Khuyến nghị: ${SIGNAL[o.signal].vi}</b>${esc(o.action)}</div>
    ${o.actual_vi != null ? `<p style="margin:10px 0 0">${actualText(o)}</p>` : ""}`;
}

/* tooltips: any element with data-tip */
document.addEventListener("mouseover", (e) => {
  const el = e.target.closest("[data-tip]");
  if (!el) return;
  showTip(el.dataset.tip, e.clientX, e.clientY);
});
document.addEventListener("mousemove", (e) => { if (e.target.closest("[data-tip]")) moveTip(e.clientX, e.clientY); });
document.addEventListener("mouseout", (e) => { if (e.target.closest("[data-tip]")) hideTip(); });
function showTip(html, x, y) { const t = $("tooltip"); t.innerHTML = html; t.hidden = false; moveTip(x, y); }
function moveTip(x, y) {
  const t = $("tooltip"), w = t.offsetWidth, h = t.offsetHeight;
  t.style.left = Math.min(x + 14, innerWidth - w - 8) + "px";
  t.style.top = (y + h + 20 > innerHeight ? y - h - 12 : y + 16) + "px";
}
function hideTip() { $("tooltip").hidden = true; }

/* ---------- charts ---------- */
const niceMax = (v) => { const p = 10 ** Math.floor(Math.log10(Math.max(v, 1))); return Math.ceil(v / p / (v / p > 5 ? 2 : 1)) * p * (v / p > 5 ? 2 : 1); };
/* One series over calendar weeks; optional forecast window [t, t+H]. Crosshair tooltip bound in bindCharts(). */
function lineChart(id, values, { from = 0, to = values.length - 1, t = null, unit = "đơn vị", label = "", points = false, height = 190, weeks = null } = {}) {
  const W = 720, H = height, pad = { l: 40, r: 10, t: 14, b: 26 };
  const idx = [];
  for (let i = from; i <= to; i++) idx.push(i);
  const ymax = niceMax(Math.max(...idx.map((i) => values[i] || 0), 1));
  const X = (i) => pad.l + ((i - from) / Math.max(to - from, 1)) * (W - pad.l - pad.r);
  const Y = (v) => pad.t + (1 - v / ymax) * (H - pad.t - pad.b);
  const path = "M" + idx.map((i) => X(i).toFixed(1) + "," + Y(values[i] || 0).toFixed(1)).join("L");
  const ticks = [0, ymax / 2, ymax].map((v) => `<line x1="${pad.l}" x2="${W - pad.r}" y1="${Y(v)}" y2="${Y(v)}" stroke="var(--line)" ${v ? 'stroke-dasharray="2 4"' : ""}/>
      <text x="${pad.l - 6}" y="${Y(v) + 4}" text-anchor="end">${compact(v)}</text>`).join("");
  const wk = weeks || ((i) => weekDate(i));
  const years = [];
  if (!weeks) {
    for (let i = from; i <= to; i++) { const d = wk(i); if (d.getUTCMonth() === 0 && d.getUTCDate() <= 7) years.push(i); }
  }
  const xlabels = (years.length ? years.map((i) => `<text x="${X(i)}" y="${H - 6}" text-anchor="middle">${wk(i).getUTCFullYear()}</text><line x1="${X(i)}" x2="${X(i)}" y1="${H - pad.b}" y2="${H - pad.b + 4}" stroke="var(--line-strong)"/>`)
    : [from, to].map((i, k) => `<text x="${X(i)}" y="${H - 6}" text-anchor="${k ? "end" : "start"}">${fmt(wk(i))}</text>`)).join("");
  const Hz = state.origins.horizon;
  const win = t != null && t >= from && t <= to
    ? `<rect x="${X(t)}" y="${pad.t}" width="${Math.max(X(Math.min(t + Hz, to)) - X(t), 3)}" height="${H - pad.t - pad.b}" fill="var(--accent-soft)" rx="3"/>
       <line x1="${X(t)}" x2="${X(t)}" y1="${pad.t}" y2="${H - pad.b}" stroke="var(--accent)" stroke-dasharray="3 3"/>
       <text x="${X(t) > W * 0.7 ? X(t) - 6 : X(t) + 6}" y="${pad.t + 11}" text-anchor="${X(t) > W * 0.7 ? "end" : "start"}" style="fill:var(--accent)">Tuần điều hành</text>` : "";
  const dots = points ? idx.map((i) => `<circle cx="${X(i)}" cy="${Y(values[i] || 0)}" r="4" fill="var(--accent)" stroke="#fff" stroke-width="2"/>`).join("") : "";
  CHARTS.set(id, { values, from, to, X, Y, unit, label, wk, pad, W, H });
  return `<svg class="chart" id="${id}" data-chart="${id}" viewBox="0 0 ${W} ${H}" role="img" aria-label="${esc(label)}">
      ${ticks}${win}<path d="${path}" fill="none" stroke="var(--accent)" stroke-width="2" stroke-linejoin="round"/>${dots}${xlabels}
      <line class="xh" x1="0" x2="0" y1="${pad.t}" y2="${H - pad.b}" stroke="var(--text-2)" stroke-width="1" visibility="hidden"/>
      <circle class="xd" r="5" fill="var(--accent)" stroke="#fff" stroke-width="2" visibility="hidden"/>
      <rect class="hit" x="${pad.l}" y="0" width="${W - pad.l - pad.r}" height="${H}"/></svg>`;
}
function bindCharts(root = document) {
  root.querySelectorAll("svg[data-chart]").forEach((svg) => {
    const c = CHARTS.get(svg.dataset.chart);
    if (!c || svg.dataset.bound) return;
    svg.dataset.bound = "1";
    const xh = svg.querySelector(".xh"), xd = svg.querySelector(".xd");
    svg.addEventListener("mousemove", (e) => {
      const r = svg.getBoundingClientRect(), x = ((e.clientX - r.left) / r.width) * c.W;
      const i = Math.min(c.to, Math.max(c.from, Math.round(c.from + ((x - c.pad.l) / (c.W - c.pad.l - c.pad.r)) * (c.to - c.from))));
      const v = c.values[i] || 0;
      xh.setAttribute("x1", c.X(i)); xh.setAttribute("x2", c.X(i)); xh.setAttribute("visibility", "visible");
      xd.setAttribute("cx", c.X(i)); xd.setAttribute("cy", c.Y(v)); xd.setAttribute("visibility", "visible");
      showTip(`Tuần ${fmt(c.wk(i))}<br><b>${dec(v, v % 1 ? 1 : 0)}</b> ${esc(c.unit)}`, e.clientX, e.clientY);
    });
    svg.addEventListener("mouseleave", () => { xh.setAttribute("visibility", "hidden"); xd.setAttribute("visibility", "hidden"); hideTip(); });
  });
}
function hbars(rows, { value = (r) => r.units, label = (r) => cap(r.label), text = (r) => int(value(r)), tip = null, href = null } = {}) {
  const max = Math.max(...rows.map(value), 1e-9);
  return `<div class="hbars">${rows.map((r) => {
    const inner = `<span class="name">${esc(label(r))}</span><span class="track"><i style="width:${((value(r) / max) * 100).toFixed(1)}%"></i></span><span class="val">${text(r)}</span>`;
    const t = tip ? ` data-tip="${esc(tip(r))}"` : "";
    return href ? `<button class="hbar" data-href="${esc(href(r))}"${t}>${inner}</button>` : `<div class="hbar"${t}>${inner}</div>`;
  }).join("")}</div>`;
}

/* End of the dataset: no launches after the last release, so every style drifts down whatever the model does. */
function lateBanner() {
  const o = state.origins;
  if (state.t < o.late_from) return "";
  return `<div class="notice late" role="note"><b>Dữ liệu cuối kỳ:</b> từ ${fmt(weekDate(o.late_from))} chuỗi gần như không còn sản phẩm mới
    (lần ra mắt cuối ${iso(o.last_release)}) và dữ liệu dừng ở ${fmt(weekDate(o.t_max))}, nên hầu hết nhóm style trông như đang giảm.
    Đây là giới hạn của dataset, không phải tín hiệu kinh doanh. <button class="btn" id="toDefault" style="min-height:28px;padding:3px 10px;margin-left:6px">Về tuần giữa mùa ${fmt(weekDate(o.default))}</button></div>`;
}
document.addEventListener("click", (e) => {
  if (e.target.id !== "toDefault") return;
  state.t = state.origins.default;
  $("origin").value = state.t;
  render();
});

/* ---------- router ---------- */
function route() {
  const parts = location.hash.replace(/^#\/?/, "").split("/").map(decodeURIComponent);
  return { page: parts[0] || "overview", arg: parts[1] || "" };
}
function setNav(page, crumbs) {
  document.querySelectorAll("[data-nav]").forEach((a) => {
    if (a.dataset.nav === page) a.setAttribute("aria-current", "page");
    else a.removeAttribute("aria-current");
  });
  $("crumbs").innerHTML = crumbs;
}
async function render() {
  if (!state.connected) return;
  const token = ++renderToken, { page, arg } = route();
  const main = $("main");
  CHARTS.clear();
  hideTip();
  main.setAttribute("aria-busy", "true");
  if (page !== "scan") main.innerHTML = `<div class="page-loading"><span class="spinner"></span> Đang tải dữ liệu…</div>`;
  try {
    let html;
    if (page === "scan") html = await pageScan();
    else if (page === "categories" && arg) html = await pageCategory(arg);
    else if (page === "categories") html = await pageCategories();
    else if (page === "storage") html = await pageStorage(arg || "products");
    else html = await pageOverview();
    if (token !== renderToken) return;
    main.innerHTML = lateBanner() + html;
    bindCharts(main);
    bindPage(page, arg);
  } catch (error) {
    if (token !== renderToken) return;
    main.innerHTML = `<div class="error-banner" role="alert">${esc(error.message)}</div><button class="btn" id="retry">Thử lại</button>`;
    $("retry").addEventListener("click", render);
  } finally {
    main.setAttribute("aria-busy", "false");
  }
}
window.addEventListener("hashchange", () => { render(); $("main").focus({ preventScroll: true }); scrollTo(0, 0); });
document.addEventListener("click", (e) => {
  const go = e.target.closest("[data-href]");
  if (go) { location.hash = go.dataset.href; return; }
  const p = e.target.closest("[data-product]");
  if (p) openProduct(+p.dataset.product);
});

/* ---------- overview ---------- */
async function pageOverview() {
  const s = await api(withT("/api/m/summary"));
  setNav("overview", "<b>Tổng quan điều hành</b>");
  const k = s.kpi, styles = s.styles.filter((x) => x.outlook);
  const counts = Object.fromEntries(SIGNAL_ORDER.map((sig) => [sig, styles.filter((x) => x.outlook.signal === sig)]));
  const activeTotal = styles.reduce((a, x) => a + x.active, 0) || 1;
  const order = (x) => SIGNAL_ORDER.indexOf(x.outlook.signal) * 10 - x.outlook.confidence;
  const scored = styles.filter((x) => x.outlook.y_true != null);
  const hits = scored.filter((x) => x.outlook.y_true === x.outlook.trend).length;
  const m = s.metrics, d = s.decision;
  const modelRows = [["catboost", "CatBoost (dùng cho khuyến nghị)"], ["tcn", "TCN"], ["persistence", "Giữ nguyên xu hướng gần nhất"], ["majority", "Luôn đoán lớp phổ biến"]]
    .filter(([key]) => m[key]).map(([key, label]) => ({ label, v: m[key].macro_f1 }));
  return `<div class="page-head"><div><p class="eyebrow">Tuần điều hành ${fmt(weekDate(s.t))}</p><h1>Tổng quan điều hành</h1>
      <p>Toàn chuỗi ${int(k.shops)} cửa hàng · ${iso(k.first_week)} ${NDASH} ${iso(k.last_week)} · dự báo cho 4 tuần ${windowText(s.t)}</p></div>
      <a class="btn btn-primary" href="#/scan"><svg class="icon"><use href="#i-scan"/></svg> Quét ảnh sản phẩm mới</a></div>
    <div class="kpis">${kpi("Sản phẩm trong kho dữ liệu", int(k.products), `${k.categories} loại · ${k.colors} màu`)}
      ${kpi("Tổng đơn vị đã bán", int(k.units), `${k.weeks} tuần dữ liệu`)}
      ${kpi("Sản phẩm đang bán", int(k.active_now), `tại tuần ${fmt(weekDate(s.t))}`)}
      ${kpi("Nhóm style theo dõi", k.styles, "học từ ảnh sản phẩm")}
      ${kpi("Ảnh đã quét", int(s.scans), `<a href="#/storage/scans" style="color:var(--accent)">Xem lịch sử</a>`)}</div>
    <div class="grid-main">
      <section class="card"><div class="card-head"><div><h2 class="card-title">Doanh số toàn chuỗi theo tuần</h2><p class="card-sub">Đơn vị bán mỗi tuần; vùng xanh là 4 tuần được dự báo</p></div></div>
        ${lineChart("chWeekly", s.weekly, { t: s.t, label: "Doanh số toàn chuỗi theo tuần" })}</section>
      <section class="card"><h2 class="card-title">Triển vọng 4 tuần tới</h2><p class="card-sub">${styles.length} nhóm style · tính theo sản phẩm đang bán</p>
        ${SIGNAL_ORDER.map((sig) => {
          const act = counts[sig].reduce((a, x) => a + x.active, 0);
          return `<div class="action s-${sig}" style="margin-top:10px"><b>${SIGNAL[sig].icon} ${SIGNAL[sig].vi}: ${counts[sig].length} nhóm style</b>${int(act)} sản phẩm đang bán (${pct(act / activeTotal)})</div>`;
        }).join("")}
        ${scored.length ? `<p class="small muted" style="margin:12px 0 0">Đối chiếu thực tế tuần này: dự báo đúng xu hướng ${hits}/${scored.length} nhóm style.</p>` : `<p class="small muted" style="margin:12px 0 0">Chưa có số liệu thực tế sau tuần này để đối chiếu.</p>`}</section>
    </div>
    <section class="card" style="margin-top:20px"><div class="card-head"><div><h2 class="card-title">Bảng theo dõi nhóm style</h2><p class="card-sub">Cảnh báo được xếp lên đầu. Ngưỡng cảnh báo chọn trên dữ liệu validation, không chỉnh theo kết quả test.</p></div></div>
      <div class="table-wrap"><table><thead><tr><th>Nhóm style</th><th class="r">SP trong nhóm</th><th class="r">Đang bán</th><th>Xu hướng 4 tuần</th><th style="min-width:130px">Phân bố xác suất</th><th>Khuyến nghị</th><th>Thực tế</th></tr></thead><tbody>
      ${styles.slice().sort((a, b) => order(a) - order(b)).map((x) => `<tr><td><div style="display:flex;gap:10px;align-items:center">${x.examples.slice(0, 2).map((c) => `<button class="prod" style="padding:2px;border-radius:6px" data-product="${c}" aria-label="Xem sản phẩm ${c}">${thumb(c, true, 96)}</button>`).join("")}<b style="font-weight:500">${esc(x.name)}</b></div></td>
        <td class="r">${int(x.n_products)}</td><td class="r">${int(x.active)}</td><td>${trendBadge(x.outlook)}</td><td>${tbar(x.outlook.p, esc(x.name) + "<br>")}</td><td>${signalBadge(x.outlook)}</td>
        <td>${x.outlook.actual_vi != null ? (x.outlook.y_true === x.outlook.trend ? `<span class="badge t${x.outlook.y_true}">✓ ${TREND[x.outlook.y_true].vi}</span>` : `<span class="badge plain">✗ ${TREND[x.outlook.y_true].vi}</span>`) : `<span class="muted small">Chưa có</span>`}</td></tr>`).join("")}
      </tbody></table></div></section>
    <div class="grid-3" style="margin-top:20px">
      <section class="card"><h2 class="card-title">Loại sản phẩm bán chạy</h2><p class="card-sub">Tổng đơn vị bán · bấm để mở dashboard</p>
        ${hbars(s.categories, { text: (r) => compact(r.units), tip: (r) => `<b>${cap(r.label)}</b><br>${int(r.units)} đơn vị · ${int(r.products)} SP · ${pct(r.share, 1)}`, href: (r) => "/categories/" + encodeURIComponent(r.key) })}
        <a class="btn" style="margin-top:14px" href="#/categories">Tất cả ${k.categories} loại <svg class="icon"><use href="#i-arrow"/></svg></a></section>
      <section class="card"><h2 class="card-title">Doanh số theo mùa</h2><p class="card-sub">Tổng đơn vị bán theo bộ sưu tập</p>
        ${hbars(s.seasons, { label: (r) => seasonVi(r.key), text: (r) => compact(r.units), tip: (r) => `<b>${seasonVi(r.key)}</b><br>${int(r.units)} đơn vị · ${int(r.products)} sản phẩm` })}</section>
      <section class="card"><h2 class="card-title">Độ tin cậy của dự báo</h2><p class="card-sub">Macro-F1 trên giai đoạn test 2019 (càng cao càng tốt)</p>
        ${hbars(modelRows, { value: (r) => r.v, label: (r) => r.label, text: (r) => dec(r.v, 3) })}
        ${d ? `<p class="small" style="margin:14px 0 0">Khi hệ thống báo <b>“Ưu tiên mở rộng”</b>, ${pct(d.emerging_alert_precision)} trường hợp style thật sự tăng; báo <b>“Rà soát nhập hàng”</b> đúng ${pct(d.declining_alert_precision)}.</p>
        <p class="small muted" style="margin:6px 0 0">Hệ thống bắt được ${pct(d.emerging_recall)} các style tăng và ${pct(d.declining_recall)} các style giảm. Đây là tín hiệu để rà soát, không phải số lượng đặt hàng.</p>` : ""}</section>
    </div>`;
}

/* ---------- categories ---------- */
async function pageCategories() {
  const data = await api(withT("/api/m/categories"));
  setNav("categories", "<b>Loại sản phẩm</b>");
  state.catData = data;
  return `<div class="page-head"><div><p class="eyebrow">Tuần điều hành ${fmt(weekDate(data.t))}</p><h1>Dashboard theo loại sản phẩm</h1>
      <p>${data.categories.length} loại sản phẩm. Dự báo của mỗi loại là trung bình dự báo các nhóm style, theo số sản phẩm đang bán.</p></div></div>
    <div class="filters"><label class="search"><svg class="icon"><use href="#i-search"/></svg><input id="catQuery" type="search" placeholder="Tìm loại sản phẩm…" value="${esc(state.catQuery)}" aria-label="Tìm loại sản phẩm"></label>
      <select id="catSort" aria-label="Sắp xếp">${[["units", "Bán chạy nhất"], ["products", "Nhiều sản phẩm nhất"], ["emerging", "Triển vọng tăng"], ["declining", "Cảnh báo giảm"]].map(([v, l]) => `<option value="${v}" ${state.catSort === v ? "selected" : ""}>${l}</option>`).join("")}</select></div>
    <div class="cat-grid" id="catGrid">${catCards()}</div>`;
}
function catCards() {
  const q = state.catQuery.trim().toLowerCase();
  const key = { units: (c) => -c.units, products: (c) => -c.products, emerging: (c) => -(c.outlook ? c.outlook.p[2] : 0), declining: (c) => -(c.outlook ? c.outlook.p[0] : 0) }[state.catSort];
  const rows = state.catData.categories.filter((c) => !q || (c.label + " " + c.key).toLowerCase().includes(q)).sort((a, b) => key(a) - key(b));
  if (!rows.length) return `<p class="muted">Không có loại sản phẩm phù hợp.</p>`;
  return rows.map((c) => `<a class="cat-card" href="#/categories/${encodeURIComponent(c.key)}"><img src="/catalog/${c.example}.jpg?size=160" alt="" loading="lazy">
      <div><h3>${esc(cap(c.label))}</h3><div class="meta">${int(c.products)} SP · ${compact(c.units)} đơn vị · ${int(c.active)} đang bán</div>
      ${trendBadge(c.outlook)}${c.outlook ? tbar(c.outlook.p, esc(cap(c.label)) + "<br>") : ""}</div></a>`).join("");
}

async function pageCategory(key) {
  const c = await api(withT("/api/m/categories/" + encodeURIComponent(key)));
  setNav("categories", `<a href="#/categories">Loại sản phẩm</a> / <b>${esc(cap(c.label))}</b>`);
  const k = c.kpi, o = c.outlook;
  const share = o ? o.signal_share : null;
  return `<a class="back" href="#/categories"><svg class="icon"><use href="#i-back"/></svg> Tất cả loại sản phẩm</a>
    <div class="page-head"><div><p class="eyebrow">${esc(c.key)} · tuần điều hành ${fmt(weekDate(c.t))}</p><h1>${esc(cap(c.label))}</h1>
      <p>${int(k.products)} sản phẩm · chiếm ${pct(k.share, 1)} tổng doanh số toàn chuỗi</p></div>
      <a class="btn" href="#/storage/products" data-filter-category="${esc(c.key)}"><svg class="icon"><use href="#i-db"/></svg> Xem trong kho dữ liệu</a></div>
    <div class="kpis">${kpi("Tổng đơn vị bán", int(k.units))}${kpi("Bán TB mỗi sản phẩm", dec(k.units_per_product), "đơn vị trong 12 tuần")}
      ${kpi("Đang bán tại tuần điều hành", int(k.active))}${kpi("Chiết khấu trung bình", pct(k.discount, 1))}${kpi("Chỉ số giá TB", dec(k.price, 3), "giá đã chuẩn hoá trong dataset")}</div>
    <div class="grid-main">
      <section class="card"><h2 class="card-title">Doanh số theo tuần</h2><p class="card-sub">Đơn vị bán mỗi tuần của toàn bộ ${esc(c.label)}</p>
        ${lineChart("chCat", c.weekly, { t: c.t, label: "Doanh số theo tuần của " + c.label })}</section>
      <section class="card"><h2 class="card-title">Dự báo 4 tuần tới</h2><p class="card-sub">${o ? `Trung bình theo ${int(o.n_weighted)} sản phẩm ${o.weighted_by === "active" ? "đang bán" : "của loại này (không có SP đang bán)"}` : ""}</p>
        ${outlookBlock(o, cap(c.label))}
        ${share ? `<p class="small muted" style="margin:12px 0 0">Trong số sản phẩm này: ${pct(share.emerging_alert)} thuộc style “ưu tiên mở rộng”, ${pct(share.declining_alert)} thuộc style “rà soát nhập hàng”.</p>` : ""}</section>
    </div>
    <div class="grid-3" style="margin-top:20px">
      <section class="card"><h2 class="card-title">Theo màu</h2><p class="card-sub">Tổng đơn vị bán</p>${hbars(c.colors, { text: (r) => compact(r.units), tip: (r) => `<b>${cap(r.label)}</b><br>${int(r.units)} đơn vị · ${int(r.products)} SP` })}</section>
      <section class="card"><h2 class="card-title">Theo mùa</h2><p class="card-sub">Tổng đơn vị bán</p>${hbars(c.seasons, { label: (r) => seasonVi(r.key), text: (r) => compact(r.units), tip: (r) => `<b>${seasonVi(r.key)}</b><br>${int(r.units)} đơn vị · ${int(r.products)} SP` })}</section>
      <section class="card"><h2 class="card-title">Theo chất liệu</h2><p class="card-sub">8 chất liệu bán chạy nhất</p>${hbars(c.fabrics, { text: (r) => compact(r.units), tip: (r) => `<b>${esc(r.label)}</b><br>${int(r.units)} đơn vị · ${int(r.products)} SP` })}</section>
    </div>
    <section class="card" style="margin-top:20px"><h2 class="card-title">Nhóm style trong loại này</h2><p class="card-sub">Mỗi sản phẩm thuộc một nhóm style; dự báo được tính cho nhóm style</p>
      <div class="table-wrap"><table><thead><tr><th>Nhóm style</th><th class="r">Số SP của loại này</th><th>Xu hướng 4 tuần</th><th style="min-width:130px">Xác suất</th><th>Khuyến nghị</th></tr></thead><tbody>
      ${c.styles.map((s) => `<tr><td>${esc(s.name)}</td><td class="r">${int(s.products)}</td><td>${trendBadge(s.outlook)}</td><td>${s.outlook ? tbar(s.outlook.p, esc(s.name) + "<br>") : ""}</td><td>${signalBadge(s.outlook)}</td></tr>`).join("")}
      </tbody></table></div></section>
    <section class="card" style="margin-top:20px"><h2 class="card-title">Sản phẩm bán chạy nhất</h2><p class="card-sub">Bấm vào một dòng để xem chi tiết và dự báo của sản phẩm</p>${productTable(c.top_products)}</section>`;
}

const STATUS = { active: "Đang bán", ended: "Đã hết vòng đời", upcoming: "Chưa ra mắt", no_sales: "Không có doanh số" };
function productTable(items) {
  if (!items.length) return `<div class="empty-state"><h3>Không có sản phẩm phù hợp</h3><p>Thử bỏ bớt bộ lọc.</p></div>`;
  return `<div class="table-wrap"><table><thead><tr><th>Ảnh</th><th>Mã</th><th>Loại</th><th>Màu</th><th>Chất liệu</th><th>Mùa</th><th>Ra mắt</th><th class="r">Cửa hàng</th><th class="r">Đơn vị bán</th><th>Trạng thái</th><th>Dự báo style</th></tr></thead><tbody>
    ${items.map((p) => `<tr class="clickable" data-product="${p.code}" tabindex="0"><td>${thumb(p.code, p.has_image)}</td><td class="num">#${p.code}</td><td>${esc(cap(p.category_vi))}</td><td>${esc(cap(p.color_vi))}</td><td>${esc(p.fabric)}</td><td>${esc(p.season)}</td><td class="num">${iso(p.release)}</td><td class="r">${int(p.shops)}</td><td class="r">${int(p.units)}</td><td><span class="small">${STATUS[p.status] || ""}</span></td><td>${trendBadge(p.outlook)}</td></tr>`).join("")}
    </tbody></table></div>`;
}

/* ---------- storage ---------- */
async function pageStorage(tab) {
  setNav("storage", `<a href="#/storage">Kho dữ liệu</a> / <b>${{ products: "Sản phẩm", files: "Tệp dữ liệu gốc", scans: "Ảnh đã quét" }[tab] || ""}</b>`);
  const tabs = `<nav class="tabs" aria-label="Khu vực kho dữ liệu">${[["products", "Sản phẩm"], ["files", "Tệp dữ liệu gốc"], ["scans", "Ảnh đã quét"]].map(([k, l]) =>
    `<a href="#/storage/${k}" ${k === tab ? 'aria-current="page"' : ""}>${l}</a>`).join("")}</nav>`;
  const head = `<div class="page-head"><div><p class="eyebrow">Kho lưu trữ</p><h1>Kho dữ liệu</h1><p>Toàn bộ dữ liệu VISUELLE 2.0: sản phẩm, ảnh, doanh số, tệp gốc và lịch sử ảnh đã quét.</p></div></div>`;
  if (tab === "files") return head + tabs + (await storageFiles());
  if (tab === "scans") return head + tabs + (await storageScans());
  return head + tabs + `<div id="productsArea">${await storageProducts()}</div>`;
}
async function storageProducts() {
  const f = state.storage;
  const qs = new URLSearchParams({ q: f.q, category: f.category, color: f.color, season: f.season, fabric: f.fabric, sort: f.sort, page: f.page, size: 20 });
  const d = await api(withT("/api/m/products?" + qs));
  f.page = d.page;
  const sel = (name, label, values, fmtv = (v) => v) => `<select data-filter="${name}" aria-label="${label}"><option value="">${label}: tất cả</option>${values.map((v) =>
    `<option value="${esc(v)}" ${f[name] === v ? "selected" : ""}>${esc(fmtv(v))}</option>`).join("")}</select>`;
  return `<div class="filters"><label class="search"><svg class="icon"><use href="#i-search"/></svg><input id="pq" type="search" placeholder="Tìm mã, loại, màu, chất liệu, mùa…" value="${esc(f.q)}" aria-label="Tìm sản phẩm"></label>
      ${sel("category", "Loại", d.facets.category)}${sel("color", "Màu", d.facets.color)}${sel("season", "Mùa", d.facets.season, seasonVi)}${sel("fabric", "Chất liệu", d.facets.fabric)}
      <select data-filter="sort" aria-label="Sắp xếp">${[["units", "Bán chạy nhất"], ["units_asc", "Bán chậm nhất"], ["release", "Mới ra mắt"], ["shops", "Nhiều cửa hàng"], ["code", "Theo mã"]].map(([v, l]) => `<option value="${v}" ${f.sort === v ? "selected" : ""}>${l}</option>`).join("")}</select>
      ${f.q || f.category || f.color || f.season || f.fabric ? `<button class="btn" id="clearFilters">Xoá bộ lọc</button>` : ""}</div>
    ${productTable(d.items)}
    <div class="pager"><span>${int(d.total)} sản phẩm · trang ${d.page}/${d.pages}</span><div>
      <button class="btn" data-page="${d.page - 1}" ${d.page <= 1 ? "disabled" : ""}>Trước</button><button class="btn" data-page="${d.page + 1}" ${d.page >= d.pages ? "disabled" : ""}>Sau</button></div></div>`;
}
async function refreshProducts() {
  const area = $("productsArea");
  if (!area) return;
  area.setAttribute("aria-busy", "true");
  try { area.innerHTML = await storageProducts(); bindProducts(); } catch (error) { toast(error.message); } finally { area.setAttribute("aria-busy", "false"); }
}
function bindProducts() {
  let timer;
  const q = $("pq");
  if (q) q.addEventListener("input", () => { clearTimeout(timer); timer = setTimeout(() => { state.storage.q = q.value; state.storage.page = 1; refreshProducts().then(() => { const n = $("pq"); n.focus(); n.setSelectionRange(n.value.length, n.value.length); }); }, 350); });
  document.querySelectorAll("[data-filter]").forEach((s) => s.addEventListener("change", () => { state.storage[s.dataset.filter] = s.value; state.storage.page = 1; refreshProducts(); }));
  document.querySelectorAll("[data-page]").forEach((b) => b.addEventListener("click", () => { state.storage.page = +b.dataset.page; refreshProducts(); }));
  if ($("clearFilters")) $("clearFilters").addEventListener("click", () => { Object.assign(state.storage, { q: "", category: "", color: "", season: "", fabric: "", page: 1 }); refreshProducts(); });
}
async function storageFiles() {
  const d = await api("/api/m/files", {}, 60000);
  const raw = d.files.filter((f) => f.group === "raw"), interim = d.files.filter((f) => f.group === "interim");
  const table = (rows) => `<div class="table-wrap"><table><thead><tr><th>Tệp</th><th>Nội dung</th><th class="r">Số dòng</th><th class="r">Số cột</th><th class="r">Dung lượng</th><th></th></tr></thead><tbody>
    ${rows.map((f) => `<tr><td><span style="display:flex;gap:8px;align-items:center"><svg class="icon" style="color:var(--muted)"><use href="#i-file"/></svg><b style="font-weight:500">${esc(f.name)}</b></span></td>
      <td class="small">${esc(f.description)}</td><td class="r">${f.rows != null ? int(f.rows) : "—"}</td><td class="r">${f.columns ?? "—"}</td><td class="r">${bytes(f.bytes)}</td>
      <td><span style="display:flex;gap:6px;justify-content:flex-end">${f.previewable ? `<button class="btn" data-preview="${esc(f.key)}"><svg class="icon"><use href="#i-eye"/></svg> Xem</button>` : ""}
      <a class="btn" href="/api/m/files/download?key=${encodeURIComponent(f.key)}" download><svg class="icon"><use href="#i-download"/></svg> Tải</a></span></td></tr>`).join("")}</tbody></table></div>`;
  return `<div class="kpis">${kpi("Tệp dữ liệu gốc", raw.length, bytes(d.raw_bytes))}${kpi("Ảnh sản phẩm", int(d.image_total), `${d.images.length} thư mục mùa`)}
      ${kpi("Bảng đã xử lý", interim.length, "dùng cho dashboard và mô hình")}</div>
    <section class="card"><h2 class="card-title">Dữ liệu gốc VISUELLE 2.0</h2><p class="card-sub">Data/extracted/visuelle2 · xem trước 50 dòng đầu hoặc tải nguyên tệp</p>${table(raw)}</section>
    <section class="card" style="margin-top:20px"><h2 class="card-title">Ảnh sản phẩm</h2><p class="card-sub">Mỗi sản phẩm một ảnh PNG; xem từng ảnh ở tab “Sản phẩm”</p>
      ${hbars(d.images.map((i) => ({ label: `${i.folder} · ${seasonVi(i.season)}`, units: i.count })), { label: (r) => r.label, text: (r) => int(r.units) + " ảnh" })}</section>
    <section class="card" style="margin-top:20px"><h2 class="card-title">Bảng đã xử lý</h2><p class="card-sub">outputs/interim · tạo bởi scripts/01_build_product_week.py</p>${table(interim)}</section>`;
}
async function openPreview(key) {
  const dlg = $("previewDialog");
  $("previewBody").innerHTML = `<div class="dialog-head"><h2 id="pvTitle">${esc(key.split("/").pop())}</h2><button class="icon-btn" data-close aria-label="Đóng"><svg class="icon"><use href="#i-close"/></svg></button></div><div class="dialog-body"><div class="page-loading"><span class="spinner"></span> Đang đọc tệp…</div></div>`;
  dlg.showModal();
  try {
    const d = await api("/api/m/files/preview?key=" + encodeURIComponent(key), {}, 60000);
    dlg.querySelector(".dialog-body").innerHTML = `<p class="small muted" style="margin:0 0 12px">${d.rows != null ? int(d.rows) + " dòng · " : ""}${d.columns.length} cột · hiển thị ${d.data.length} dòng đầu</p>
      <div class="table-wrap" style="max-height:60vh"><table><thead><tr>${d.columns.map((c, i) => `<th title="${esc(d.dtypes[i])}">${esc(c)}</th>`).join("")}</tr></thead><tbody>
      ${d.data.map((r) => `<tr>${r.map((v) => `<td class="num">${v == null ? '<span class="muted">—</span>' : esc(typeof v === "number" && !Number.isInteger(v) ? +v.toFixed(4) : v)}</td>`).join("")}</tr>`).join("")}</tbody></table></div>`;
  } catch (error) {
    dlg.querySelector(".dialog-body").innerHTML = `<div class="error-banner">${esc(error.message)}</div>`;
  }
}
async function storageScans() {
  const items = await api("/api/m/scans");
  if (!items.length) return `<section class="card"><div class="empty-state"><svg class="icon"><use href="#i-image"/></svg><h3>Chưa có ảnh nào được quét</h3><p>Mỗi ảnh quét ở trang “Quét ảnh &amp; dự báo” sẽ được lưu tại đây cùng kết quả dự báo.</p><a class="btn btn-primary" href="#/scan">Quét ảnh đầu tiên</a></div></section>`;
  return `<p class="small muted" style="margin:0 0 14px">${items.length} lượt quét · lưu tại outputs/web/scans. Dự báo giữ nguyên theo tuần điều hành lúc quét.</p>
    <div class="scan-grid">${items.map((s) => `<article class="scan-item"><button class="open" data-scan="${s.id}" aria-label="Mở kết quả quét ${esc(s.name)}"><img src="/api/m/scans/${s.id}/image" alt="" loading="lazy">
      <div class="body"><b>${esc(cap(s.description))}</b><span class="muted">${esc(s.name)}</span><br><span class="muted">${new Date(s.created).toLocaleString("vi-VN")}</span></div></button>
      <div class="foot">${trendBadge(s.outlook)}<button class="icon-btn" data-delete-scan="${s.id}" aria-label="Xoá lượt quét" title="Xoá"><svg class="icon"><use href="#i-trash"/></svg></button></div></article>`).join("")}</div>`;
}

/* ---------- scan ---------- */
async function pageScan() {
  setNav("scan", "<b>Quét ảnh &amp; dự báo</b>");
  const s = state.scan;
  return `<div class="page-head"><div><p class="eyebrow">Tuần điều hành ${fmt(weekDate(state.t))}</p><h1>Quét ảnh &amp; dự báo</h1>
      <p>Tải ảnh một sản phẩm: hệ thống nhận diện, xếp vào nhóm style gần nhất và dự báo xu hướng 4 tuần ${windowText(state.t)}.</p></div></div>
    <div class="scan-layout"><section class="card">
      <div class="drop" id="drop" tabindex="0" role="button" aria-label="Chọn ảnh để quét">${s.url ? `<img src="${s.url}" alt="Ảnh đã chọn">` : `<div><span class="upload-icon"><svg class="icon"><use href="#i-upload"/></svg></span><b>Kéo thả hoặc bấm để chọn ảnh</b><span class="small">JPG, PNG, WEBP · tối đa 15 MB · hoặc Ctrl + V</span></div>`}</div>
      <input type="file" id="file" accept="image/jpeg,image/png,image/webp" hidden>
      ${s.file ? `<p class="small muted" style="margin:8px 0 0;overflow:hidden;text-overflow:ellipsis;white-space:nowrap">${esc(s.file.name)} · ${bytes(s.file.size)}</p>` : ""}
      <div class="field"><label for="segment">Tách nền</label><select id="segment">${SEGMENT.map(([v, l]) => `<option value="${v}">${l}</option>`).join("")}</select></div>
      <button class="btn btn-primary btn-block" id="runScan" style="margin-top:16px" ${!s.file || s.busy ? "disabled" : ""}>${s.busy ? `<span class="spinner" style="border-color:#fff6;border-top-color:#fff"></span> Đang phân tích…` : `<svg class="icon"><use href="#i-scan"/></svg> Quét &amp; dự báo`}</button>
      <p class="small muted" style="margin:12px 0 0">Ảnh và kết quả được lưu vào <a href="#/storage/scans" style="color:var(--accent)">Kho dữ liệu › Ảnh đã quét</a>.</p></section>
      <div id="scanResult">${s.busy ? `<section class="card"><div class="page-loading"><span class="spinner"></span> Đang nhận diện và dự báo… ảnh có người mẫu có thể mất thêm thời gian.</div></section>`
        : s.result ? scanResult(s.result.scan)
        : `<section class="card"><div class="empty-state"><svg class="icon"><use href="#i-trend"/></svg><h3>Chưa có kết quả</h3><p>Chọn ảnh bên trái rồi bấm “Quét &amp; dự báo”.</p></div></section>`}</div></div>`;
}
function scanResult(s) {
  const o = s.outlook, st = s.style;
  const notices = [];
  if (s.near_duplicate) notices.push(`<div class="notice info">Ảnh này gần như trùng với sản phẩm <button class="btn" style="min-height:26px;padding:2px 8px" data-product="${s.near_duplicate}">#${s.near_duplicate}</button> đã có trong kho dữ liệu.</div>`);
  if (s.novel_both) notices.push(`<div class="notice">Style mới: món đồ khác rõ với cả 20 nhóm style đã có. Chưa có lịch sử bán, nên dự báo bên dưới chỉ là tham khảo từ 3 nhóm gần nhất.</div>`);
  if (s.t !== state.t) notices.push(`<div class="notice info">Dự báo này được tính tại tuần điều hành ${iso(s.week)} (lúc quét), khác với tuần đang chọn.</div>`);
  const colors = s.colors.length ? s.colors.map((c) => `<span style="margin-right:10px">${c.hex ? `<span class="swatch" style="background:${esc(c.hex)}"></span>` : ""}${esc(cap(c.vi))}</span>`).join("") : "—";
  return `${notices.join("")}<section class="card"><div class="result-hero"><img src="/api/m/scans/${s.id}/image" alt="Ảnh đã quét">
      <div><p class="eyebrow">Kết quả nhận diện</p><h2>${esc(cap(s.category.vi))}</h2><p class="small muted" style="margin:0">${esc(cap(s.description))} · độ khớp ${pct(s.category.p)}</p>
      <dl class="facts"><dt>Màu chủ đạo</dt><dd>${colors}</dd><dt>Nhóm style</dt><dd>${esc(st.name)}${st.novel ? ' <span class="badge proxy">gần nhất</span>' : ""}</dd>
        <dt>Mức điển hình</dt><dd>${dec(st.typicality_pct, 0)}% sản phẩm trong nhóm khác biệt hơn ảnh này</dd>
        <dt>Loại gần nhất trong kho</dt><dd><a href="#/categories/${encodeURIComponent(s.category.catalogue)}" style="color:var(--accent)">${esc(cap(s.category.catalogue_vi))} › xem dashboard</a></dd>
        <dt>Tham chiếu doanh số</dt><dd>${s.reference_units != null ? `${int(s.reference_units)} đơn vị / sản phẩm tương tự (12 tuần đầu)` : "—"}</dd></dl></div></div></section>
    <section class="card" style="margin-top:20px"><h2 class="card-title">Dự báo 4 tuần tới cho nhóm style của sản phẩm</h2><p class="card-sub">${o && o.proxy ? "Ước lượng tham khảo từ các nhóm style gần nhất" : "Mô hình CatBoost, áp dụng cho cả nhóm style"}</p>
      ${o && o.proxy ? '<p style="margin:0 0 10px"><span class="badge proxy">Ước lượng tham khảo</span></p>' : ""}${outlookBlock(o, st.name)}</section>
    <section class="card" style="margin-top:20px"><div class="card-head"><div><h2 class="card-title">Sản phẩm tương tự trong kho dữ liệu</h2><p class="card-sub">Theo ResNet-50 · số đơn vị đã bán giúp ước lượng nhu cầu</p></div>
      <a class="btn" href="/scanner"><svg class="icon"><use href="#i-lab"/></svg> Phân tích kỹ thuật</a></div>
      <div class="prod-grid">${s.neighbors.map((n) => `<button class="prod" data-product="${n.code}"><img src="/catalog/${n.code}.jpg?size=240" alt="" loading="lazy"><b>${esc(cap(n.category_vi))}</b>${esc(cap(n.color_vi))} · ${esc(n.season)}<br>${int(n.units)} đơn vị · giống ${pct(n.similarity)}</button>`).join("")}</div></section>`;
}
function setScanFile(f) {
  const s = state.scan;
  if (!f) return;
  if (!/^image\/(jpeg|png|webp)$/.test(f.type)) { toast("Chỉ nhận ảnh JPG, PNG hoặc WEBP."); return; }
  if (f.size > 15 * 1024 * 1024) { toast("Ảnh quá lớn (tối đa 15 MB)."); return; }
  if (s.url) URL.revokeObjectURL(s.url);
  Object.assign(s, { file: f, url: URL.createObjectURL(f), result: null });
  render();
}
async function runScan() {
  const s = state.scan;
  if (!s.file || s.busy) return;
  const segment = $("segment").value;
  const form = new FormData();
  form.append("file", s.file, s.file.name || "upload.png");
  form.append("segment", segment);
  form.append("t", state.t);
  s.busy = true;
  await render();
  try {
    s.result = await api("/api/m/scans", { method: "POST", body: form }, 180000);
    toast("Đã quét và lưu vào kho dữ liệu.");
  } catch (error) {
    toast(error.message);
  } finally {
    s.busy = false;
    if (route().page === "scan") await render();
  }
}

/* ---------- product dialog ---------- */
async function openProduct(code) {
  const dlg = $("productDialog");
  $("productBody").innerHTML = `<div class="dialog-head"><h2 id="pdTitle">Sản phẩm #${code}</h2><button class="icon-btn" data-close aria-label="Đóng"><svg class="icon"><use href="#i-close"/></svg></button></div><div class="dialog-body"><div class="page-loading"><span class="spinner"></span> Đang tải…</div></div>`;
  if (!dlg.open) dlg.showModal();
  try {
    const p = await api(withT("/api/m/products/" + code));
    const w = p.weekly, units = w.map((r) => r.units);
    const s = p.style_info;
    $("productBody").innerHTML = `<div class="dialog-head"><div><p class="eyebrow">Mã #${p.code} · ${esc(p.category)}</p><h2 id="pdTitle">${esc(cap(p.category_vi))} ${esc(p.color_vi)}</h2></div>
        <button class="icon-btn" data-close aria-label="Đóng"><svg class="icon"><use href="#i-close"/></svg></button></div>
      <div class="dialog-body"><div class="pd-top">${p.has_image ? `<img src="/catalog/${p.code}.jpg?size=480" alt="Ảnh sản phẩm ${p.code}">` : `<div class="thumb ph" style="width:100%;height:200px"><svg class="icon"><use href="#i-image"/></svg></div>`}
        <div><dl class="facts" style="margin-top:0"><dt>Loại</dt><dd>${esc(cap(p.category_vi))} <span class="muted">(${esc(p.category)})</span></dd><dt>Màu</dt><dd>${esc(cap(p.color_vi))}</dd><dt>Chất liệu</dt><dd>${esc(p.fabric)}</dd>
          <dt>Mùa</dt><dd>${seasonVi(p.season)}</dd><dt>Ra mắt</dt><dd>${iso(p.release)}</dd><dt>Trạng thái tại tuần điều hành</dt><dd>${STATUS[p.status] || "—"}</dd></dl>
          <div class="mini-kpis"><div><span>Đơn vị đã bán</span><b>${int(p.units)}</b></div><div><span>Số cửa hàng</span><b>${int(p.shops)}</b></div>
            <div><span>Xếp hạng trong loại</span><b>${int(p.category_rank.rank)}/${int(p.category_rank.of)}</b></div>
            <div><span>Trung vị của loại</span><b>${int(p.category_rank.median_units)}</b></div><div><span>Chiết khấu TB</span><b>${p.discount != null ? pct(p.discount, 1) : "—"}</b></div>
            <div><span>Tuần có doanh số</span><b>${w.length}</b></div></div></div></div>
        ${w.length ? `<h3 class="section-title">Doanh số theo tuần của sản phẩm</h3>${lineChart("chProd", units, { points: units.length <= 20, height: 160, unit: "đơn vị", label: "Doanh số theo tuần của sản phẩm", weeks: (i) => new Date(w[i].week + "T00:00:00Z") })}` : ""}
        ${s ? `<h3 class="section-title">Dự báo cho nhóm style “${esc(s.name)}”</h3>
          <p class="small muted" style="margin:-4px 0 12px">${int(s.products)} sản phẩm cùng nhóm, bán TB ${dec(s.units_per_product)} đơn vị. Mô hình dự báo xu hướng của nhóm style, không dự báo riêng một sản phẩm.</p>
          ${outlookBlock(p.outlook, s.name)}
          ${p.launch_outlook ? `<div class="action" style="margin-top:14px"><b>Dự báo tại tuần ra mắt (${fmt(weekDate(p.launch_t))})</b>${trendBadge(p.launch_outlook)} ${signalBadge(p.launch_outlook)} ${actualText(p.launch_outlook)}</div>` : ""}
          <h3 class="section-title">Doanh số TB mỗi sản phẩm của nhóm style</h3>${lineChart("chStyle", s.history, { t: state.t, height: 150, unit: "đơn vị / SP", label: "Doanh số trung bình mỗi sản phẩm của nhóm style" })}` : `<p class="muted">Sản phẩm không có ảnh hợp lệ nên chưa được xếp nhóm style.</p>`}</div>`;
    bindCharts($("productBody"));
  } catch (error) {
    dlg.querySelector(".dialog-body").innerHTML = `<div class="error-banner">${esc(error.message)}</div>`;
  }
}
document.addEventListener("click", (e) => {
  if (e.target.closest("[data-close]")) e.target.closest("dialog").close();
});
["productDialog", "previewDialog"].forEach((id) => $(id).addEventListener("click", (e) => { if (e.target === $(id)) $(id).close(); }));
document.addEventListener("keydown", (e) => {
  const row = e.target.closest && e.target.closest("tr[data-product]");
  if (row && (e.key === "Enter" || e.key === " ")) { e.preventDefault(); openProduct(+row.dataset.product); }
});

/* ---------- per-page bindings ---------- */
function bindPage(page) {
  if (page === "categories" && $("catGrid")) {
    $("catQuery").addEventListener("input", (e) => { state.catQuery = e.target.value; $("catGrid").innerHTML = catCards(); });
    $("catSort").addEventListener("change", (e) => { state.catSort = e.target.value; $("catGrid").innerHTML = catCards(); });
  }
  document.querySelectorAll("[data-filter-category]").forEach((a) => a.addEventListener("click", () => {
    Object.assign(state.storage, { q: "", category: a.dataset.filterCategory, color: "", season: "", fabric: "", page: 1 });
  }));
  if (page === "storage") {
    bindProducts();
    document.querySelectorAll("[data-preview]").forEach((b) => b.addEventListener("click", () => openPreview(b.dataset.preview)));
    document.querySelectorAll("[data-scan]").forEach((b) => b.addEventListener("click", async () => {
      const dlg = $("previewDialog");
      try {
        const s = await api("/api/m/scans/" + b.dataset.scan);
        $("previewBody").innerHTML = `<div class="dialog-head"><div><p class="eyebrow">${esc(s.name)} · ${new Date(s.created).toLocaleString("vi-VN")}</p><h2 id="pvTitle">Kết quả quét</h2></div><button class="icon-btn" data-close aria-label="Đóng"><svg class="icon"><use href="#i-close"/></svg></button></div><div class="dialog-body">${scanResult(s)}</div>`;
        dlg.showModal();
      } catch (error) { toast(error.message); }
    }));
    document.querySelectorAll("[data-delete-scan]").forEach((b) => b.addEventListener("click", async () => {
      if (!confirm("Xoá vĩnh viễn ảnh và kết quả của lượt quét này?")) return;
      try { await api("/api/m/scans/" + b.dataset.deleteScan, { method: "DELETE" }); toast("Đã xoá lượt quét."); render(); } catch (error) { toast(error.message); }
    }));
  }
  if (page === "scan") {
    const drop = $("drop"), input = $("file");
    drop.addEventListener("click", () => !state.scan.busy && input.click());
    drop.addEventListener("keydown", (e) => { if (e.key === "Enter" || e.key === " ") { e.preventDefault(); input.click(); } });
    input.addEventListener("change", () => setScanFile(input.files[0]));
    ["dragenter", "dragover"].forEach((ev) => drop.addEventListener(ev, (e) => { e.preventDefault(); drop.classList.add("over"); }));
    ["dragleave", "drop"].forEach((ev) => drop.addEventListener(ev, (e) => { e.preventDefault(); drop.classList.remove("over"); }));
    drop.addEventListener("drop", (e) => setScanFile(e.dataTransfer.files[0]));
    $("runScan").addEventListener("click", runScan);
  }
}
document.addEventListener("paste", (e) => {
  if (route().page !== "scan") return;
  const item = [...(e.clipboardData?.items || [])].find((i) => i.type.startsWith("image/"));
  if (item) setScanFile(new File([item.getAsFile()], "anh-dan.png", { type: item.type }));
});

/* ---------- start ---------- */
async function init() {
  connection("connecting", "Đang kết nối…");
  try {
    const health = await api("/api/health");
    if (health.status !== "ready") throw new Error("Mô hình chưa sẵn sàng.");
    state.origins = await api("/api/m/origins");
    state.t = state.origins.default;
    const sel = $("origin");
    sel.innerHTML = state.origins.origins.slice().reverse().map((o) =>
      `<option value="${o.t}">${fmt(weekDate(o.t))}${o.t === state.origins.default ? " (mặc định)" : ""}${o.t >= state.origins.late_from ? " · cuối kỳ" : ""}${o.t > state.origins.t_last_label ? " · chưa có thực tế" : ""}</option>`).join("");
    sel.value = state.t;
    sel.disabled = false;
    sel.addEventListener("change", () => { state.t = +sel.value; render(); });
    state.connected = true;
    connection("ready", "Đã kết nối");
    render();
  } catch (error) {
    connection("error", "Mất kết nối máy chủ");
    $("main").innerHTML = `<div class="error-banner" role="alert">${esc(error.message)}</div><button class="btn" id="retryInit">Kết nối lại</button>`;
    $("retryInit").addEventListener("click", init);
  }
}
init();
