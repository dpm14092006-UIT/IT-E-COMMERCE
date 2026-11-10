"use strict";
/* Style Scanner front end. Talks to web/app.py: GET /api/meta, POST /api/analyze, /api/novel. */

const TREND = [
  { vi: "Đang giảm", icon: "\u2198" },
  { vi: "Ổn định", icon: "\u2192" },
  { vi: "Đang lên", icon: "\u2197" },
];
const ENC = ["resnet50", "fashionclip"];
const ENC_LABEL = { resnet50: "ResNet-50", fashionclip: "FashionCLIP" };
const ENC_ABOUT = {
  resnet50: "Mạng CNN học từ ảnh tổng quát (ImageNet). Nhạy với màu sắc, kết cấu bề mặt và hình dáng.",
  fashionclip: "Mô hình học từ ảnh thời trang kèm mô tả bằng chữ. Nhạy với loại sản phẩm và ý nghĩa thời trang.",
};
const SEASON = { SS: "Xuân Hè", AW: "Thu Đông" };
const NDASH = "\u2013";

const $ = (id) => document.getElementById(id);
const ESC = { "&": "&amp;", "<": "&lt;", ">": "&gt;", "\u0022": "&quot;", "\u0027": "&#39;" };
const esc = (s) => String(s).replace(/[&<>\u0022\u0027]/g, (c) => ESC[c]);
const cap = (s) => (s ? s.charAt(0).toUpperCase() + s.slice(1) : s);
const pct = (v) => Math.round(v * 100) + "%";
const num = (v, d = 2) => Number(v).toLocaleString("vi-VN", { minimumFractionDigits: d, maximumFractionDigits: d });
const argmax = (p) => p.indexOf(Math.max(...p));
let savedItems = [], mediaItems = [], mediaIndex = 0;

const state = { meta: null, file: null, result: null, tab: "overview", nbEnc: "fashionclip", heroView: "isolated", procEnc: "resnet50", busy: false, previewUrl: null, scanTimer: null, connected: false, initializing: false, imageReady: false, maxUploadBytes: 15 * 1024 * 1024, saving: false, savedStyle: null, sampleLoading: false, selectionVersion: 0, elapsedTimer: null, draftName: null };

async function api(path, options = {}, timeoutMs = 15000) {
  const controller = new AbortController();
  const timer = setTimeout(() => controller.abort(), timeoutMs);
  try {
    const response = await fetch(path, { ...options, cache: "no-store", signal: controller.signal });
    const body = await response.json().catch(() => null);
    if (!response.ok) {
      const detail = typeof body?.detail === "string" ? body.detail : `Yêu cầu thất bại (HTTP ${response.status}).`;
      const error = new Error(detail);
      error.status = response.status;
      throw error;
    }
    if (body === null) throw new Error("Máy chủ trả về dữ liệu không hợp lệ.");
    return body;
  } catch (error) {
    if (error.name === "AbortError") throw new Error("Máy chủ phản hồi quá lâu. Hãy thử lại.");
    if (error instanceof TypeError) {
      state.connected = false;
      connectionStatus("error", "Mất kết nối máy chủ");
      updateRunButton();
      throw new Error("Không kết nối được máy chủ. Kiểm tra kết nối rồi thử kết nối lại.");
    }
    throw error;
  } finally {
    clearTimeout(timer);
  }
}

function updateRunButton() {
  $("run").disabled = state.busy || !state.connected || !state.imageReady || !state.file;
  $("clearImage").disabled = state.busy;
  document.querySelectorAll("[data-sample]").forEach((button) => { button.disabled = state.busy || state.sampleLoading; });
}

function workflow(stage) {
  const stages = ["upload", "analyze", "result"], current = stages.indexOf(stage);
  $("workflow").dataset.stage = stage;
  document.querySelectorAll("#workflow li").forEach((item, index) => {
    if (index === current) item.setAttribute("aria-current", "step");
    else item.removeAttribute("aria-current");
    item.classList.toggle("complete", index < current);
  });
}

function connectionStatus(status, label) {
  $("connection").dataset.state = status;
  $("connectionLabel").textContent = label;
  $("retryConnection").hidden = status !== "error";
}

/* ---------- dates ---------- */
function weekDate(t) {
  const d = new Date(state.meta.resnet50.week0 + "T00:00:00Z");
  d.setUTCDate(d.getUTCDate() + 7 * t);
  return d;
}
const fmt = (d) => `${String(d.getUTCDate()).padStart(2, "0")}/${String(d.getUTCMonth() + 1).padStart(2, "0")}/${d.getUTCFullYear()}`;
const fmtShort = (d) => `${String(d.getUTCDate()).padStart(2, "0")}/${String(d.getUTCMonth() + 1).padStart(2, "0")}`;
function windowText(t) {
  const H = state.meta.resnet50.horizon, end = weekDate(t + H);
  end.setUTCDate(end.getUTCDate() + 6);
  return `${fmtShort(weekDate(t + 1))} ${NDASH} ${fmt(end)}`;
}
const season = (s) => (SEASON[s.slice(0, 2)] ? `${SEASON[s.slice(0, 2)]} 20${s.slice(2)}` : s);

/* ---------- small UI helpers ---------- */
function toast(msg) {
  const el = document.createElement("div");
  el.className = "toast";
  el.setAttribute("role", "status");
  el.textContent = msg;
  document.body.appendChild(el);
  setTimeout(() => el.remove(), 4000);
}
const styleName = (enc, id) => state.meta[enc].styles[id].name;
/* FashionCLIP clusters are named after their plurality category/colour; say so when neither is a majority */
function mixedNote(enc, id) {
  const st = state.meta[enc].styles[id];
  return st.mixed ? ` <span class="small muted">(nhóm pha trộn: ${esc(st.name_note)})</span>` : "";
}
const trendPill = (k) => `<span class="pill t${k}">${TREND[k].icon} ${TREND[k].vi}</span>`;
function bar3(p) {
  return `<div class="bar3" role="img" aria-label="${TREND.map((t, i) => `${t.vi} ${pct(p[i])}`).join(", ")}">${
    p.map((v, i) => `<span class="c${i}" style="width:${(v * 100).toFixed(1)}%"></span>`).join("")}</div>
    <div class="legend">${TREND.map((t, i) => `<span><i class="c${i}"></i>${t.vi} ${pct(p[i])}</span>`).join("")}</div>`;
}
function product(n) {
  return `<div class="prod"><button class="ph product-open" data-product="${n.code}" aria-haspopup="dialog" aria-label="Xem ảnh ${esc(cap(n.category_vi))}, mã ${n.code}"><img loading="lazy" src="/catalog/${n.code}.jpg" alt="${esc(cap(n.category_vi))}"><span class="product-zoom" aria-hidden="true"><svg class="icon"><use href="#i-zoom"/></svg></span></button>
    <b>${esc(cap(n.category_vi))}</b>${esc(cap(n.color_vi))} · ${esc(season(n.season))}<br>Độ giống ${pct(n.similarity)}</div>`;
}

/* ---------- product photos and saved-style gallery ---------- */
function showMedia(index) {
  if (!mediaItems.length) return;
  mediaIndex = (index + mediaItems.length) % mediaItems.length;
  const item = mediaItems[mediaIndex];
  $("mediaImage").src = item.src;
  $("mediaImage").alt = item.alt;
  $("mediaCaption").textContent = `${mediaItems.length > 1 ? `Ảnh ${mediaIndex + 1}/${mediaItems.length} · ` : ""}${item.caption}`;
  $("mediaPrev").hidden = $("mediaNext").hidden = $("mediaKeyHint").hidden = mediaItems.length < 2;
  $("mediaThumbs").hidden = mediaItems.length < 2;
  document.querySelectorAll("[data-media-index]").forEach((button) => button.setAttribute("aria-pressed", String(+button.dataset.mediaIndex === mediaIndex)));
}

function openMedia(title, eyebrow, subtitle, items) {
  mediaItems = items;
  $("mediaTitle").textContent = title;
  $("mediaEyebrow").textContent = eyebrow;
  $("mediaSubtitle").textContent = subtitle;
  $("mediaThumbs").innerHTML = items.length > 1 ? items.map((item, index) => `<button data-media-index="${index}" aria-label="Xem ảnh ${index + 1}" aria-pressed="false"><img src="${esc(item.src)}" alt=""></button>`).join("") : "";
  showMedia(0);
  $("mediaDialog").showModal();
}

function openProduct(code) {
  if (!state.result) return;
  const n = ENC.flatMap((enc) => state.result.encoders[enc].neighbors).find((item) => String(item.code) === String(code));
  if (!n) return;
  openMedia(cap(n.category_vi), "SẢN PHẨM TRONG BỘ SƯU TẬP", `${cap(n.color_vi)} · ${season(n.season)} · Mã ${n.code}`,
    [{ src: `/catalog/${n.code}.jpg?size=480`, alt: `${cap(n.category_vi)} ${n.color_vi}`, caption: "Ảnh sản phẩm từ bộ sưu tập VISUELLE 2.0." }]);
}

function openSaved(id) {
  const item = savedItems.find((s) => s.id === id);
  if (!item?.thumbs?.length) return;
  openMedia(cap(item.name), "PHONG CÁCH ĐÃ LƯU", `${item.id} · ${item.n} ảnh đã ghi nhận · Tạo ngày ${fmt(new Date(item.created.slice(0, 10) + "T00:00:00Z"))}`,
    item.thumbs.map((src) => ({ src, alt: `Ảnh đã lưu trong phong cách ${item.name}`, caption: item.n > item.thumbs.length ? `${item.thumbs.length} ảnh gần nhất của phong cách này.` : "Ảnh xem trước được lưu cùng phong cách." })));
}

function renderWithFocus(renderSection, selector) {
  renderSection();
  document.querySelector(selector)?.focus({ preventScroll: true });
}
function typicality(s) {
  if (s.novel) return "khác biệt rõ với nhóm";
  if (s.far_from_centroid) return "nằm ngoài rìa nhóm nhưng vẫn có sản phẩm rất giống trong bộ sưu tập";
  if (s.typicality_pct >= 50) return "rất điển hình của nhóm";
  if (s.typicality_pct >= 20) return "điển hình của nhóm";
  return "nằm ở rìa nhóm";
}

/* ---------- start-up ---------- */
async function init() {
  if (state.initializing) return;
  state.initializing = true;
  state.connected = false;
  updateRunButton();
  connectionStatus("connecting", "Đang kết nối backend…");
  $("retryConnection").disabled = true;
  try {
    const health = await api("/api/health");
    if (health.status !== "ready") throw new Error("Mô hình chưa sẵn sàng.");
    const meta = await api("/api/meta");
    if (ENC.some((enc) => !Array.isArray(meta[enc]?.styles) || !meta[enc].styles.length || !meta[enc].origins?.length || !meta[enc].forecasts || !meta[enc].history)) {
      throw new Error("Dữ liệu mô hình chưa đầy đủ.");
    }
    state.meta = meta;
    state.maxUploadBytes = health.max_upload_bytes || state.maxUploadBytes;
    const m = state.meta.resnet50;
    $("origin").innerHTML = m.origins.slice().reverse().map((t) =>
      `<option value="${t}">${fmt(weekDate(t))}${t === m.default_origin ? " (mặc định)" : ""}</option>`).join("");
    $("origin").value = m.default_origin;
    $("origin").disabled = false;
    state.connected = true;
    updateRunButton();
    connectionStatus("ready", "Sẵn sàng phân tích");
    originHelp();
    $("footer").innerHTML = `Dữ liệu bán hàng: VISUELLE 2.0, chuỗi bán lẻ thời trang tại Ý, ${fmt(weekDate(0))} ${NDASH} ${fmt(weekDate(m.t_max))}. ` +
      `Nhóm style được học từ ${m.n_catalogue.toLocaleString("vi-VN")} sản phẩm. Dự báo áp dụng cho cả nhóm style, không phải cho riêng một sản phẩm.`;
    $("workspaceStatus").textContent = state.result ? "Phân tích hoàn tất" : state.file ? "Ảnh sẵn sàng" : "Chưa chọn ảnh";
    await refreshSaved();
  } catch (error) {
    state.connected = false;
    $("origin").disabled = true;
    connectionStatus("error", "Mất kết nối máy chủ");
    $("workspaceStatus").textContent = "Chưa kết nối máy chủ";
    $("saved").innerHTML = '<p class="small muted">Chưa tải được phong cách đã lưu.</p>';
    toast(error.message);
  } finally {
    state.initializing = false;
    $("retryConnection").disabled = false;
    updateRunButton();
  }
}

function originHelp() {
  const t = +$("origin").value, m = state.meta.resnet50;
  const parts = [t > m.t_last_label
    ? "Sau thời điểm này chưa có số liệu thực tế để đối chiếu."
    : "Có số liệu thực tế 4 tuần sau để đối chiếu với dự báo."];
  if (t > m.default_origin) parts.push(`<span class="warn">Cuối kỳ dữ liệu, số liệu bán thưa dần nên dự báo kém tin cậy.</span>`);
  $("originHelp").innerHTML = parts.join(" ");
}

/* ---------- upload ---------- */
function stopImageScan() {
  clearTimeout(state.scanTimer);
  state.scanTimer = null;
  $("previewScanner").classList.remove("is-scanning");
  $("analysisScanner").classList.remove("is-scanning");
  $("drop").classList.remove("scanning");
}

function startImageScan(analyzing = false) {
  stopImageScan();
  $("previewScanLabel").textContent = analyzing ? "Đang phân tích ảnh" : "Quét ảnh xem trước";
  // Restart the sweep when another image is chosen, including during an existing preview sweep.
  void $("previewScanner").offsetWidth;
  $("previewScanner").classList.add("is-scanning");
  $("drop").classList.add("scanning");
  if (analyzing) {
    $("analysisScanner").classList.add("is-scanning");
  } else {
    state.scanTimer = setTimeout(stopImageScan, 3600);
  }
}

function setFile(f) {
  if (state.busy) { toast("Ảnh hiện tại đang được phân tích. Bạn có thể đổi ảnh khi hoàn tất."); return; }
  if (!f) return;
  if (!f.type.startsWith("image/")) { toast("File này không phải ảnh."); return; }
  if (f.size > state.maxUploadBytes) { toast("Ảnh lớn hơn 15 MB. Hãy chọn ảnh có dung lượng nhỏ hơn."); return; }
  state.selectionVersion++;
  document.querySelectorAll("[data-sample]").forEach((button) => button.setAttribute("aria-pressed", "false"));
  stopImageScan();
  if (state.previewUrl) URL.revokeObjectURL(state.previewUrl);
  state.file = f;
  state.imageReady = false;
  state.savedStyle = null;
  state.draftName = null;
  state.previewUrl = URL.createObjectURL(f);
  $("preview").src = state.previewUrl;
  $("analysisPreview").src = state.previewUrl;
  $("preview").hidden = false;
  $("previewScanner").hidden = false;
  $("dropEmpty").hidden = true;
  $("fileName").hidden = false;
  $("fileInfo").hidden = false;
  $("fileName").textContent = `${f.name} · ${f.size < 1024 * 1024 ? Math.max(1, Math.round(f.size / 1024)) + " KB" : (f.size / (1024 * 1024)).toFixed(1) + " MB"}`;
  updateRunButton();
  $("drop").classList.add("has-file");
  $("replaceImage").hidden = false;
  state.result = null;
  state.tab = "overview";
  workflow("upload");
  $("results").classList.add("hidden");
  $("empty").classList.remove("hidden");
  $("analysisError").hidden = true;
  $("emptyTitle").textContent = "Ảnh đã được chọn";
  $("emptyDescription").textContent = "Ảnh của bạn đã sẵn sàng. Bấm “Quét ảnh” để nhận diện trang phục và khám phá nhóm phong cách tương ứng.";
  $("workspaceStatus").textContent = "Đang đọc ảnh…";
}

function clearImage() {
  if (state.busy) return;
  state.selectionVersion++;
  stopImageScan();
  if (state.previewUrl) URL.revokeObjectURL(state.previewUrl);
  Object.assign(state, { file: null, imageReady: false, previewUrl: null, result: null, savedStyle: null, draftName: null, tab: "overview" });
  $("file").value = "";
  $("preview").removeAttribute("src");
  $("analysisPreview").removeAttribute("src");
  $("preview").hidden = true;
  $("previewScanner").hidden = true;
  $("dropEmpty").hidden = false;
  $("replaceImage").hidden = true;
  $("fileName").hidden = true;
  $("fileInfo").hidden = true;
  $("drop").classList.remove("has-file");
  $("results").classList.add("hidden");
  $("analysisError").hidden = true;
  $("empty").classList.remove("hidden");
  $("emptyTitle").innerHTML = "Một bức ảnh.<br> Nhiều điều để khám phá.";
  $("emptyDescription").textContent = "Chọn ảnh trang phục để nhận diện đặc điểm, tìm sản phẩm tương tự và khám phá xu hướng của nhóm phong cách.";
  $("workspaceStatus").textContent = state.connected ? "Chưa chọn ảnh" : "Chưa kết nối máy chủ";
  document.querySelectorAll("[data-sample]").forEach((button) => button.setAttribute("aria-pressed", "false"));
  workflow("upload");
  updateRunButton();
  $("drop").focus({ preventScroll: true });
}

async function selectSample(button) {
  if (state.busy || state.sampleLoading) return;
  state.sampleLoading = true;
  const selectionVersion = ++state.selectionVersion;
  updateRunButton();
  button.setAttribute("aria-busy", "true");
  const controller = new AbortController();
  const timeout = setTimeout(() => controller.abort(), 15000);
  try {
    const response = await fetch(`/catalog/${button.dataset.sample}.jpg`, { signal: controller.signal });
    if (!response.ok) throw new Error("Ảnh mẫu chưa tải được. Hãy thử lại hoặc chọn ảnh từ thiết bị.");
    const blob = await response.blob();
    if (selectionVersion !== state.selectionVersion) return;
    setFile(new File([blob], `visuelle-${button.dataset.sample}.jpg`, { type: blob.type || "image/jpeg" }));
    button.setAttribute("aria-pressed", "true");
    $("emptyTitle").textContent = `Đã chọn ${button.dataset.name.toLocaleLowerCase("vi-VN")}`;
    $("emptyDescription").textContent = "Bấm “Quét ảnh” để xem kết quả phân tích ảnh mẫu từ bộ sưu tập VISUELLE 2.0.";
    if (window.matchMedia("(max-width: 760px)").matches) $("up-title").scrollIntoView({ behavior: window.matchMedia("(prefers-reduced-motion: reduce)").matches ? "instant" : "smooth", block: "start" });
    $("drop").focus({ preventScroll: true });
  } catch (error) {
    if (selectionVersion === state.selectionVersion) toast(error.name === "AbortError" ? "Tải ảnh mẫu quá lâu. Hãy thử lại." : error.message);
  } finally {
    clearTimeout(timeout);
    state.sampleLoading = false;
    button.removeAttribute("aria-busy");
    updateRunButton();
  }
}

function exportResult() {
  if (!state.result || state.busy) return;
  const t = +$("origin").value, zs = state.result.zero_shot;
  const lines = ["STYLE SCANNER — TÓM TẮT PHÂN TÍCH", "", `Ảnh: ${state.file.name}`, `Thời điểm dự báo: ${fmt(weekDate(t))}`, `Khoảng dự báo: ${windowText(t)}`, "", "ĐẶC ĐIỂM TRANG PHỤC"];
  zs.groups.forEach((g) => lines.push(`${g.label}: ${g.items.slice(0, 3).map((i) => `${cap(i.vi)} (${i.source === "pixels" ? "đo từ ảnh" : pct(i.p)})`).join(", ")}`));
  lines.push(`Màu chủ đạo: ${zs.colors.map((c) => `${cap(c.vi)} ${pct(c.share)}`).join(", ") || "Không xác định"}`, "", "NHÓM PHONG CÁCH VÀ XU HƯỚNG");
  ENC.forEach((enc) => {
    const s = state.result.encoders[enc].style, f = forecast(enc, t);
    lines.push(`${ENC_LABEL[enc]}: ${styleName(enc, s.id)} — ${s.novel ? "Phong cách mới" : typicality(s)}.`);
    if (f.kind === "style") lines.push(`Dự báo TCN: ${TREND[argmax(f.tcn.p)].vi}. ${TREND.map((v, i) => `${v.vi}: ${pct(f.tcn.p[i])}`).join("; ")}.`);
    else if (f.kind === "proxy") lines.push(`Chưa thể dự báo trực tiếp. Tham khảo các nhóm gần nhất: ${TREND.map((v, i) => `${v.vi}: ${pct(f.p[i])}`).join("; ")}.`);
    else lines.push("Chưa đủ số liệu để dự báo tại thời điểm đã chọn.");
    lines.push("");
  });
  lines.push("LƯU Ý", "Thuộc tính nhận diện và chất liệu là ước đoán từ ảnh; màu sắc được đo từ điểm ảnh.", "Dự báo dựa trên lịch sử VISUELLE 2.0 và tuần được chọn, áp dụng cho nhóm phong cách, không phải một sản phẩm riêng lẻ hay ngày hiện tại.", "Phong cách mới chỉ có xu hướng tham khảo từ các nhóm gần nhất.");
  const url = URL.createObjectURL(new Blob(["\uFEFF" + lines.join("\r\n")], { type: "text/plain;charset=utf-8" }));
  const link = document.createElement("a");
  link.href = url; link.download = `style-scanner-${fmt(weekDate(t)).replaceAll("/", "-")}.txt`;
  document.body.appendChild(link); link.click(); link.remove();
  setTimeout(() => URL.revokeObjectURL(url), 1000);
  toast("Đã tải bản tóm tắt kết quả.");
}

function bindUpload() {
  const drop = $("drop");
  $("preview").addEventListener("load", () => {
    state.imageReady = true;
    updateRunButton();
    $("workspaceStatus").textContent = "Ảnh sẵn sàng";
    startImageScan(state.busy);
  });
  $("preview").addEventListener("error", () => {
    stopImageScan();
    state.file = null;
    state.imageReady = false;
    $("run").disabled = true;
    $("previewScanner").hidden = true;
    $("dropEmpty").hidden = false;
    $("replaceImage").hidden = true;
    $("fileName").hidden = true;
    $("fileInfo").hidden = true;
    drop.classList.remove("has-file");
    $("workspaceStatus").textContent = "Hãy chọn một ảnh khác";
    $("emptyTitle").textContent = "Không đọc được ảnh";
    $("emptyDescription").textContent = "Chọn một ảnh JPG, PNG hoặc WEBP hợp lệ.";
    toast("Không đọc được ảnh này. Hãy thử một ảnh JPG, PNG hoặc WEBP khác.");
  });
  const chooseFile = () => {
    if (state.busy) return;
    $("file").value = "";
    $("file").click();
  };
  drop.addEventListener("click", chooseFile);
  drop.addEventListener("keydown", (e) => { if (e.key === "Enter" || e.key === " ") { e.preventDefault(); chooseFile(); } });
  $("file").addEventListener("change", (e) => setFile(e.target.files[0]));
  ["dragenter", "dragover"].forEach((ev) => drop.addEventListener(ev, (e) => { e.preventDefault(); drop.classList.add("over"); }));
  ["dragleave", "drop"].forEach((ev) => drop.addEventListener(ev, (e) => { e.preventDefault(); drop.classList.remove("over"); }));
  drop.addEventListener("drop", (e) => setFile(e.dataTransfer.files[0]));
  document.addEventListener("paste", (e) => {
    const item = [...(e.clipboardData?.items || [])].find((i) => i.type.startsWith("image/"));
    if (item) setFile(item.getAsFile());
  });
  $("run").addEventListener("click", analyze);
  $("clearImage").addEventListener("click", clearImage);
  $("exportResult").addEventListener("click", exportResult);
  document.querySelectorAll("[data-sample]").forEach((button) => {
    button.setAttribute("aria-pressed", "false");
    button.addEventListener("click", () => selectSample(button));
  });
  $("openGuide").addEventListener("click", () => $("guideDialog").showModal());
  ["closeGuide", "guideDone"].forEach((id) => $(id).addEventListener("click", () => $("guideDialog").close()));
  $("guideDialog").addEventListener("click", (event) => {
    const rect = $("guideDialog").getBoundingClientRect();
    if (event.target === $("guideDialog") && (event.clientX < rect.left || event.clientX > rect.right || event.clientY < rect.top || event.clientY > rect.bottom)) $("guideDialog").close();
  });
  $("closeMedia").addEventListener("click", () => $("mediaDialog").close());
  $("mediaPrev").addEventListener("click", () => showMedia(mediaIndex - 1));
  $("mediaNext").addEventListener("click", () => showMedia(mediaIndex + 1));
  $("mediaThumbs").addEventListener("click", (event) => {
    const button = event.target.closest("[data-media-index]");
    if (button) showMedia(+button.dataset.mediaIndex);
  });
  $("mediaDialog").addEventListener("keydown", (event) => {
    if (event.key !== "ArrowLeft" && event.key !== "ArrowRight") return;
    event.preventDefault();
    showMedia(mediaIndex + (event.key === "ArrowRight" ? 1 : -1));
  });
  $("mediaDialog").addEventListener("click", (event) => {
    const rect = $("mediaDialog").getBoundingClientRect();
    if (event.target === $("mediaDialog") && (event.clientX < rect.left || event.clientX > rect.right || event.clientY < rect.top || event.clientY > rect.bottom)) $("mediaDialog").close();
  });
  $("content").addEventListener("click", (event) => {
    const productButton = event.target.closest("[data-product]");
    if (productButton) openProduct(productButton.dataset.product);
  });
  $("retryConnection").addEventListener("click", init);
  window.addEventListener("online", () => { if (!state.connected && !state.busy) init(); });
  window.addEventListener("offline", () => {
    state.connected = false;
    connectionStatus("error", "Mất kết nối máy chủ");
    updateRunButton();
  });
  $("origin").addEventListener("change", () => { originHelp(); if (state.result) render(); });
  $("segment").addEventListener("change", () => { if (state.file && state.result) analyze(); });
  const tabs = [...document.querySelectorAll(".tabs button")];
  tabs.forEach((b, index) => {
    b.addEventListener("click", () => { state.tab = b.dataset.tab; showTab(); });
    b.addEventListener("keydown", (e) => {
      const next = e.key === "ArrowRight" ? (index + 1) % tabs.length : e.key === "ArrowLeft" ? (index + tabs.length - 1) % tabs.length : e.key === "Home" ? 0 : e.key === "End" ? tabs.length - 1 : null;
      if (next === null) return;
      e.preventDefault(); state.tab = tabs[next].dataset.tab; showTab(); tabs[next].focus();
    });
  });
}

/* ---------- analysis ---------- */
async function analyze() {
  if (!state.file || !state.meta || !state.connected || !state.imageReady || state.busy) return;
  const previousResult = state.result;
  const previousSaved = state.savedStyle;
  const started = performance.now();
  $("elapsed").textContent = "0 giây";
  clearInterval(state.elapsedTimer);
  state.elapsedTimer = setInterval(() => { $("elapsed").textContent = `${Math.floor((performance.now() - started) / 1000)} giây`; }, 1000);
  state.busy = true;
  workflow("analyze");
  updateRunButton();
  $("analysisError").hidden = true;
  $("drop").setAttribute("aria-disabled", "true");
  $("content").setAttribute("aria-busy", "true");
  $("runLabel").textContent = "Đang phân tích…";
  $("workspaceStatus").textContent = "Đang xử lý ảnh…";
  $("origin").disabled = true;
  $("segment").disabled = true;
  $("run").disabled = true;
  $("empty").classList.add("hidden");
  $("results").classList.add("hidden");
  $("loading").classList.remove("hidden");
  startImageScan(true);
  if (window.matchMedia("(max-width: 620px)").matches) {
    $("loading").scrollIntoView({ behavior: window.matchMedia("(prefers-reduced-motion: reduce)").matches ? "instant" : "smooth", block: "start" });
  }
  try {
    const fd = new FormData();
    fd.append("file", state.file);
    fd.append("segment", $("segment").value);
    const body = await api("/api/analyze", { method: "POST", body: fd }, 180000);
    if (!body.token || !body.segmentation || !body.preprocess || ENC.some((enc) => !body.encoders?.[enc]?.style || !body.encoders[enc].neighbors?.length) || ["category", "pattern", "style", "detail", "fabric"].some((key) => !body.zero_shot?.groups?.find((g) => g.key === key)?.items?.length)) {
      throw new Error("Máy chủ trả về kết quả không đầy đủ. Hãy quét lại ảnh.");
    }
    state.result = body;
    state.savedStyle = null;
    state.heroView = "isolated";
    render();
    $("resultDuration").textContent = `· ${num((performance.now() - started) / 1000, 1)} giây`;
    $("results").classList.remove("hidden");
    if (window.matchMedia("(max-width: 620px)").matches) {
      $("results").scrollIntoView({ behavior: window.matchMedia("(prefers-reduced-motion: reduce)").matches ? "instant" : "smooth", block: "start" });
    }
  } catch (e) {
    state.result = previousResult;
    state.savedStyle = previousSaved;
    if (state.result) render();
    $("analysisError").textContent = "Không phân tích được ảnh: " + e.message;
    $("analysisError").hidden = false;
    $((state.result ? "results" : "empty")).classList.remove("hidden");
  } finally {
    clearInterval(state.elapsedTimer);
    state.elapsedTimer = null;
    stopImageScan();
    $("loading").classList.add("hidden");
    state.busy = false;
    updateRunButton();
    $("drop").setAttribute("aria-disabled", "false");
    $("origin").disabled = false;
    $("segment").disabled = false;
    $("content").setAttribute("aria-busy", "false");
    $("runLabel").textContent = "Quét ảnh";
    $("workspaceStatus").textContent = state.result ? "Phân tích hoàn tất" : "Sẵn sàng thử lại";
    workflow(state.result ? "result" : "upload");
  }
}

function showTab() {
  document.querySelectorAll(".tabs button").forEach((b) => {
    const selected = b.dataset.tab === state.tab;
    b.setAttribute("aria-selected", String(selected));
    b.tabIndex = selected ? 0 : -1;
  });
  ["overview", "compare", "process"].forEach((t) => $("tab-" + t).classList.toggle("hidden", t !== state.tab));
}

function render() {
  renderOverview();
  renderCompare();
  renderProcess();
  showTab();
}

/* ---------- saved new styles ---------- */
async function refreshSaved() {
  try {
    await loadSaved();
  } catch (error) {
    $("savedCount").textContent = "–";
    $("saved").innerHTML = `<p class="small muted">${esc(error.message)}</p><button class="retry-btn" id="retrySaved">Tải lại danh sách</button>`;
    $("retrySaved").addEventListener("click", refreshSaved);
  }
}

async function loadSaved() {
  const items = await api("/api/novel");
  if (!Array.isArray(items)) throw new Error("Danh sách phong cách không hợp lệ.");
  savedItems = items;
  $("savedCount").textContent = items.length;
  $("saved").innerHTML = items.length ? items.map((s) => `<div class="saved-item">
      <button class="saved-open" data-saved-open="${esc(s.id)}" aria-haspopup="dialog" aria-label="Xem phong cách ${esc(s.name)}"><img src="${esc(s.thumbs[s.thumbs.length - 1])}" alt="">
      <span><span class="name" title="${esc(s.name)}">${esc(cap(s.name))}</span>
        <span class="small muted">${s.n} ảnh · ${fmt(new Date(s.created.slice(0, 10) + "T00:00:00Z"))}</span></span></button>
      <button class="icon-btn" data-ask-delete aria-label="Xoá phong cách ${esc(s.name)}" title="Xoá" aria-expanded="false">&#10005;</button>
      <div class="saved-confirm" hidden><span>Xoá phong cách này?</span><button class="retry-btn" data-cancel-delete>Huỷ</button><button class="retry-btn" data-del="${esc(s.id)}">Xoá</button></div></div>`).join("")
    : `<div class="saved-empty"><svg class="icon"><use href="#i-bookmark"/></svg><div><p>Bộ sưu tập của bạn bắt đầu ở đây.</p><p>Phong cách mới có thể được lưu lại sau khi quét ảnh.</p></div></div>`;
  document.querySelectorAll("[data-saved-open]").forEach((button) => button.addEventListener("click", () => openSaved(button.dataset.savedOpen)));
  document.querySelectorAll("[data-ask-delete]").forEach((button) => button.addEventListener("click", () => {
    button.nextElementSibling.hidden = false;
    button.setAttribute("aria-expanded", "true");
  }));
  document.querySelectorAll("[data-cancel-delete]").forEach((button) => button.addEventListener("click", () => {
    button.parentElement.hidden = true;
    button.parentElement.previousElementSibling.setAttribute("aria-expanded", "false");
    button.parentElement.previousElementSibling.focus({ preventScroll: true });
  }));
  document.querySelectorAll("[data-del]").forEach((button) => button.addEventListener("click", async () => {
    button.disabled = true;
    try {
      await api("/api/novel/" + encodeURIComponent(button.dataset.del), { method: "DELETE" });
      if (state.savedStyle?.id === button.dataset.del) state.savedStyle.deleted = true;
      if (state.result?.registry_match?.id === button.dataset.del) state.result.registry_match = null;
      if (state.result) renderOverview();
      await refreshSaved();
      $("saved-title").tabIndex = -1;
      $("saved-title").focus({ preventScroll: true });
      toast("Đã xoá phong cách.");
    } catch (error) {
      toast("Không xoá được: " + error.message);
      button.disabled = false;
      if (error.status === 404) await refreshSaved();
    }
  }));
}

function savedMessage() {
  if (!state.savedStyle) return "";
  return state.savedStyle.deleted
    ? `<p class="small muted" style="margin:12px 0 0">Phong cách đã được xoá. Quét lại ảnh để lưu mới.</p>`
    : `<p class="small" style="margin:12px 0 0;color:var(--up)">Đã lưu vào “${esc(cap(state.savedStyle.name))}” (${state.savedStyle.n} ảnh).</p>`;
}

async function saveNovel() {
  if (state.saving || !state.result || state.savedStyle) return;
  const token = state.result.token;
  const name = $("saveName") ? $("saveName").value.trim() : null;
  const button = $("saveBtn");
  state.saving = true;
  if (button) { button.disabled = true; button.textContent = "Đang lưu…"; }
  try {
    const saved = await api("/api/novel", { method: "POST", headers: { "Content-Type": "application/json" }, body: JSON.stringify({ token, name }) });
    if (state.result?.token === token) {
      state.savedStyle = saved;
      renderOverview();
    }
    await refreshSaved();
  } catch (error) {
    toast("Không lưu được: " + error.message);
    if (state.result?.token === token && $("saveBtn")) {
      $("saveBtn").disabled = false;
      $("saveBtn").textContent = "Lưu phong cách";
    }
  } finally {
    state.saving = false;
  }
}

/* ---------- forecasts ---------- */
function forecast(enc, t) {
  const r = state.result.encoders[enc], F = state.meta[enc].forecasts[t] || {};
  if (!r.style.novel) {
    const f = F[r.style.id];
    return f && f.tcn ? { kind: "style", enc, style: r.style.id, tcn: f.tcn, cb: f.catboost } : { kind: "none", enc, style: r.style.id };
  }
  const near = r.ranking.slice(0, 3).filter((x) => F[x.id] && F[x.id].tcn);
  if (!near.length) return { kind: "none", enc, style: r.style.id };
  const w = near.map((x) => 1 / Math.max(x.distance, 1e-9)), sw = w.reduce((a, b) => a + b, 0);
  const p = [0, 1, 2].map((c) => near.reduce((a, x, i) => a + (w[i] / sw) * F[x.id].tcn.p[c], 0));
  return { kind: "proxy", enc, p, parts: near.map((x, i) => ({ id: x.id, w: w[i] / sw, k: argmax(F[x.id].tcn.p) })) };
}
function primaryEnc() {
  const E = state.result.encoders;
  if (!E.resnet50.style.novel) return "resnet50";
  if (!E.fashionclip.style.novel) return "fashionclip";
  return "resnet50";
}

/* ---------- overview ---------- */
const group = (key) => state.result.zero_shot.groups.find((g) => g.key === key);
function chips(items) {
  return items.map((i) => `<span class="chip">${esc(cap(i.vi))}</span>`).join("");
}

function verdictBlock() {
  const R = state.result, E = R.encoders, zs = R.zero_shot;
  const novel = ENC.filter((e) => E[e].style.novel);
  const match = R.registry_match && R.registry_match.match ? R.registry_match : null;
  let html;
  if (novel.length === 2) {
    html = `<div class="verdict novel"><div class="head">Style mới, chưa có trong bộ sưu tập</div>
      <p>Món đồ này khác rõ với cả 20 nhóm style học từ dữ liệu bán hàng, theo cả hai mô hình. Gần nhất là
      “${esc(styleName("resnet50", E.resnet50.style.id))}” và “${esc(styleName("fashionclip", E.fashionclip.style.id))}”.</p>`;
  } else if (novel.length === 1) {
    const known = ENC.find((e) => !E[e].style.novel);
    html = `<div class="verdict maybe"><div class="head">Có thể là style mới</div>
      <p>${ENC_LABEL[known]} xếp món đồ vào nhóm “${esc(styleName(known, E[known].style.id))}”, nhưng ${ENC_LABEL[novel[0]]}
      thấy nó khác biệt với mọi nhóm đã biết.</p>`;
  } else {
    const s = E.resnet50.style, st = state.meta.resnet50.styles[s.id];
    html = `<div class="verdict known"><div class="head">Thuộc nhóm style “${esc(st.name)}”</div>
      <p>Nhóm có ${st.n_products.toLocaleString("vi-VN")} sản phẩm trong bộ sưu tập; món đồ ${typicality(s)}.
      FashionCLIP xếp vào nhóm “${esc(styleName("fashionclip", E.fashionclip.style.id))}”${mixedNote("fashionclip", E.fashionclip.style.id)};
      món đồ ${typicality(E.fashionclip.style)}.</p>`;
  }
  if (state.savedStyle) {
    html += savedMessage();
  } else if (match) {
    html += `<p>Giống style bạn đã lưu: <b>${esc(cap(match.name))}</b> (${match.n} ảnh).</p>
      <div id="saveBox"><div class="save-row"><button class="btn btn-ghost" id="saveBtn">Thêm ảnh này vào style đó</button></div></div>`;
  } else if (novel.length) {
    html += `<div id="saveBox"><div class="save-row"><input id="saveName" maxlength="80" aria-label="Tên style mới" value="${esc(state.draftName ?? cap(zs.short))}">
      <button class="btn btn-ghost" id="saveBtn">Lưu style mới</button></div></div>`;
  }
  return html + `</div>`;
}

function heroCard() {
  const R = state.result, zs = R.zero_shot, seg = R.segmentation;
  const cat = group("category").items, top = cat[0];
  const confident = top.p >= 0.5;
  const alt = !confident && cat[1] ? ` <span class="muted" style="font-weight:500">hoặc ${esc(cat[1].vi)}</span>` : "";
  const pattern = group("pattern").items[0];
  const details = group("detail").items.filter((i) => i.term !== "no visible details" && i.p >= 0.15).slice(0, 2);
  const styles = group("style").items.filter((i, j) => j === 0 || i.p >= 0.15).slice(0, 2);
  const fabrics = group("fabric").items, fabric = fabrics[0];
  const fabricSure = fabric.p >= 0.5;
  const img = seg.applied && state.heroView === "isolated" ? seg.isolated : $("preview").src;
  const toggle = seg.applied ? `<div class="seg-toggle" role="group" aria-label="Chọn ảnh hiển thị">
      <button data-hero="isolated" aria-pressed="${state.heroView === "isolated"}">Đã tách nền</button>
      <button data-hero="original" aria-pressed="${state.heroView === "original"}">Ảnh gốc</button></div>` : "";
  const colors = zs.colors.map((c) => `<span class="chip"><span class="swatch" style="background:${c.hex}"></span>${esc(cap(c.vi))} <span class="pct">${pct(c.share)}</span></span>`).join("");
  const sub = `${confident ? "Độ tin cậy" : "Chưa chắc chắn, độ tin cậy"} ${pct(top.p)}${top.known ? "" : " · loại hàng chưa có trong dữ liệu bán"}`;
  return `<section class="card"><div class="hero">
    <div><div class="hero-img"><img src="${img}" alt="Ảnh món đồ"></div>${toggle}</div>
    <div>
      <p class="eyebrow">Kết quả nhận diện</p>
      <h2 class="item-name">${esc(cap(top.vi))}${alt}</h2>
      <div class="item-sub">${sub}</div>
      <dl class="attrs">
        <dt>Màu chủ đạo</dt><dd>${colors || `<span class="muted">Không xác định</span>`}</dd>
        <dt>Họa tiết</dt><dd><span class="chip">${esc(cap(pattern.vi))}</span></dd>
        <dt>Phong cách</dt><dd>${chips(styles)}</dd>
        <dt>Chi tiết</dt><dd>${details.length ? chips(details) : `<span class="muted">Không có chi tiết nổi bật</span>`}</dd>
        <dt>Chất liệu</dt><dd>${fabricSure
          ? `<span class="chip">${esc(cap(fabric.vi))}</span><span class="small muted">ước đoán từ ảnh</span>`
          : `${fabrics.slice(0, 3).map((i) => `<span class="chip">${esc(cap(i.vi))} <span class="pct">${pct(i.p)}</span></span>`).join("")}<span class="small muted">chưa chắc chắn, ước đoán từ ảnh</span>`}</dd>
      </dl>
      ${verdictBlock()}
    </div></div></section>`;
}

function trendCard() {
  const t = +$("origin").value, enc = primaryEnc(), f = forecast(enc, t);
  const head = `<div class="trend-head"><h2 class="card-title" style="margin:0">Xu hướng 4 tuần tới</h2>
    <span class="small muted">${windowText(t)}</span></div>`;
  if (f.kind === "none") {
    return `<section class="card">${head}<p class="muted">Nhóm style này không đủ số liệu bán tại thời điểm đã chọn. Hãy chọn thời điểm khác trong phần Tuỳ chọn.</p></section>`;
  }
  if (f.kind === "proxy") {
    return `<section class="card">${head}
      <div class="trend-big"><div class="trend-icon tq">?</div><div><div class="label">Chưa thể dự báo</div>
        <div class="small muted">Style mới chưa có lịch sử bán hàng nên chưa có số liệu để dự báo trực tiếp.</div></div></div>
      <p class="small" style="margin:0 0 8px"><b>Tham khảo</b> xu hướng của các nhóm gần nhất (không phải dự báo):</p>
      ${bar3(f.p)}
      <div class="refs">${f.parts.map((x) => `<div><span>${esc(styleName(enc, x.id))}</span>${trendPill(x.k)}</div>`).join("")}</div>
    </section>`;
  }
  const k = argmax(f.tcn.p), metric = state.meta[enc].metrics.tcn.macro_f1;
  let outcome = "";
  if (f.tcn.y_true != null) {
    const ok = f.tcn.y_true === k;
    outcome = `<div class="outcome">Thực tế sau 4 tuần: <b>${TREND[f.tcn.y_true].vi}</b> \u2014 <span class="${ok ? "ok" : "bad"}">${ok ? "dự báo đúng" : "dự báo chưa đúng"}</span></div>`;
  }
  return `<section class="card">${head}<div class="trend-body"><div>
    <div class="trend-big"><div class="trend-icon t${k}">${TREND[k].icon}</div><div>
      <div class="label">${TREND[k].vi}</div>
      <div class="small muted">Nhóm “${esc(styleName(enc, f.style))}” · khả năng ${pct(f.tcn.p[k])}</div></div></div>
    ${bar3(f.tcn.p)}${outcome}
    <p class="small muted" style="margin:12px 0 0">Mô hình TCN, điểm F1 trên dữ liệu kiểm tra ${num(metric)} (đoán ngẫu nhiên khoảng 0,33).</p>
    </div><div>${chart(enc, f.style, t)}</div></div>
  </section>`;
}

function chart(enc, styleId, t) {
  const m = state.meta[enc], h = m.history[styleId].spap, H = m.horizon;
  const t0 = Math.max(0, t - 39), t1 = Math.min(m.t_max, t + H);
  const W = 640, Ht = 150, pad = { l: 8, r: 8, t: 12, b: 22 };
  const ymax = Math.max(...h.slice(t0, t1 + 1), 1e-9) * 1.12;
  const X = (i) => pad.l + ((i - t0) / Math.max(t + H - t0, 1)) * (W - pad.l - pad.r);
  const Y = (v) => pad.t + (1 - v / ymax) * (Ht - pad.t - pad.b);
  const line = (a, b) => { const pts = []; for (let i = a; i <= b; i++) pts.push(X(i).toFixed(1) + "," + Y(h[i]).toFixed(1)); return "M" + pts.join("L"); };
  const after = t1 > t ? `<path d="${line(t, t1)}" fill="none" stroke="var(--muted)" stroke-width="1.6" stroke-dasharray="4 4"/>` : "";
  return `<svg class="chart" viewBox="0 0 ${W} ${Ht}" role="img" aria-label="Doanh số trung bình mỗi sản phẩm của nhóm style theo tuần">
      <rect x="${X(t)}" y="${pad.t}" width="${X(t + H) - X(t)}" height="${Ht - pad.t - pad.b}" fill="var(--accent-soft)" rx="4"/>
      <line x1="${pad.l}" x2="${W - pad.r}" y1="${Y(0)}" y2="${Y(0)}" stroke="var(--line)"/>
      <path d="${line(t0, t)}" fill="none" stroke="var(--accent)" stroke-width="2.2" stroke-linejoin="round"/>${after}
      <circle cx="${X(t)}" cy="${Y(h[t])}" r="4" fill="var(--accent)"/>
      <text x="${pad.l}" y="${Ht - 6}" font-size="11" fill="var(--muted)">${fmt(weekDate(t0))}</text>
      <text x="${X(t)}" y="${Ht - 6}" font-size="11" text-anchor="middle" fill="var(--text)">${fmt(weekDate(t))}</text>
      <text x="${X(t + H) - 4}" y="${pad.t + 12}" font-size="11" text-anchor="end" fill="var(--accent)">4 tuần dự báo</text>
    </svg>
    <div class="legend"><span><i style="background:var(--accent)"></i>Doanh số trung bình mỗi sản phẩm của nhóm, theo tuần</span>${
      t1 > t ? `<span><i style="background:var(--muted)"></i>Thực tế sau đó</span>` : ""}</div>`;
}

function productsCard() {
  const E = state.result.encoders, enc = state.nbEnc, s = E[enc].style;
  const note = s.novel
    ? `<p class="small muted" style="margin:-6px 0 14px">Không có sản phẩm nào thật sự giống: độ giống trung bình của ${s.knn_k} sản phẩm gần nhất là ${pct(s.knn_sim)}, thấp hơn 99% sản phẩm trong bộ sưu tập (ngưỡng ${pct(s.knn_threshold)}). Đây chỉ là các món gần nhất để tham khảo.</p>`
    : "";
  return `<section class="card"><div class="row-between"><h2 class="card-title">${s.novel ? "Sản phẩm gần nhất trong bộ sưu tập" : "Sản phẩm tương tự trong bộ sưu tập"}</h2>
      <div class="switch" role="group" aria-label="Chọn mô hình">${ENC.map((e) =>
        `<button data-nb="${e}" aria-pressed="${e === enc}">Theo ${ENC_LABEL[e]}</button>`).join("")}</div></div>
    ${note}<div class="grid">${E[enc].neighbors.map(product).join("")}</div></section>`;
}

function attributesCard() {
  const groups = ["category", "pattern", "style", "detail", "fabric"].map(group);
  return `<section class="card"><h2 class="card-title">Đặc điểm chi tiết</h2>
    <p class="small muted" style="margin:-6px 0 14px">Mức độ khớp giữa ảnh và từng mô tả. Nhãn <span class="chip new" style="padding:0 7px">mới</span> là thuộc tính không có trong dữ liệu bán hàng.</p>
    <div class="bars">${groups.map((g) => `<div><h4>${g.label}</h4>${g.items.map((i) => {
      const measured = i.source === "pixels";
      return `<div class="bar-row"><span>${esc(cap(i.vi))}${i.known ? "" : ` <span class="chip new" style="padding:0 7px">mới</span>`}</span>
        <span class="track"><i style="width:${measured ? 100 : (i.p * 100).toFixed(1)}%"></i></span>
        <span class="num">${measured ? "đo" : pct(i.p)}</span></div>`;
    }).join("")}</div>`).join("")}</div>
    <p class="small muted" style="margin:12px 0 0">Họa tiết sọc (nếu có) được đo trực tiếp trên điểm ảnh; các mục còn lại do FashionCLIP so khớp ảnh với mô tả bằng chữ.</p></section>`;
}

function renderOverview() {
  $("tab-overview").innerHTML = heroCard() + trendCard() + productsCard() + attributesCard();
  document.querySelectorAll("[data-hero]").forEach((b) => b.addEventListener("click", () => { state.heroView = b.dataset.hero; renderWithFocus(renderOverview, `[data-hero="${state.heroView}"]`); }));
  document.querySelectorAll("[data-nb]").forEach((b) => b.addEventListener("click", () => { state.nbEnc = b.dataset.nb; renderWithFocus(renderOverview, `[data-nb="${state.nbEnc}"]`); }));
  if ($("saveName")) $("saveName").addEventListener("input", (event) => { state.draftName = event.target.value; });
  if ($("saveBtn")) $("saveBtn").addEventListener("click", saveNovel);
}

/* ---------- compare ---------- */
function renderCompare() {
  const E = state.result.encoders, t = +$("origin").value, ag = state.result.agreement;
  const cell = (fn) => ENC.map((e) => `<td>${fn(e, E[e].style)}</td>`).join("");
  const trendCell = (e) => {
    const f = forecast(e, t);
    if (f.kind === "style") return trendPill(argmax(f.tcn.p)) + ` <span class="small muted">${pct(Math.max(...f.tcn.p))}</span>`;
    if (f.kind === "proxy") return `<span class="muted">Chưa thể dự báo (style mới)</span>`;
    return `<span class="muted">Không đủ số liệu</span>`;
  };
  const rows = [
    ["Nhóm style gần nhất", (e, s) => esc(styleName(e, s.id)) + mixedNote(e, s.id)],
    ["Kết luận", (e, s) => s.novel ? `<span class="pill novel">Style mới</span>` : `<span class="pill known">Thuộc nhóm này</span>`],
    ["Mức độ giống nhóm", (e, s) => cap(typicality(s))],
    ["Xu hướng 4 tuần", trendCell],
    ["Độ chính xác dự báo (F1)", (e) => num(state.meta[e].metrics.tcn.macro_f1)],
  ];
  const intro = `<section class="card"><h2 class="card-title">Hai cách nhìn một món đồ</h2>
    <div class="two-col">${ENC.map((e) => `<div><b>${ENC_LABEL[e]}</b><p class="small muted" style="margin:4px 0 0">${ENC_ABOUT[e]}</p></div>`).join("")}</div>
    <p class="small muted" style="margin:12px 0 0">Mỗi mô hình tự chia bộ sưu tập thành 20 nhóm style riêng, nên tên nhóm của hai bên không trùng nhau.</p>
    <div class="table-wrap" style="margin-top:16px"><table class="cmp"><thead><tr><th></th>${ENC.map((e) => `<th>${ENC_LABEL[e]}</th>`).join("")}</tr></thead>
    <tbody>${rows.map(([h, fn]) => `<tr><th>${h}</th>${cell(fn)}</tr>`).join("")}</tbody></table></div></section>`;
  const focus = `<section class="card"><h2 class="card-title">Mô hình nhìn vào đâu</h2>
    <p class="small muted" style="margin:-6px 0 14px">Vùng màu cam là phần ảnh kéo món đồ về nhóm style gần nhất.</p>
    <div class="two-col">${ENC.map((e) => `<figure><img src="${E[e].evidence}" alt="Vùng ảnh quan trọng theo ${ENC_LABEL[e]}">
      <figcaption>${ENC_LABEL[e]}</figcaption></figure>`).join("")}</div></section>`;
  const similar = `<section class="card"><h2 class="card-title">Sản phẩm gần nhất theo từng mô hình</h2>
    <p class="small muted" style="margin:-6px 0 14px">Hai mô hình chọn chung ${ag.neighbor_overlap}/${ag.n} sản phẩm.</p>
    <div class="two-col">${ENC.map((e) => `<div><h3 class="small" style="margin:0 0 8px">${ENC_LABEL[e]}</h3>
      <div class="grid">${E[e].neighbors.slice(0, 6).map(product).join("")}</div></div>`).join("")}</div></section>`;
  $("tab-compare").innerHTML = intro + focus + similar;
}

/* ---------- process (technical) ---------- */
function stepCard(n, title, lead, body) {
  return `<section class="card step"><div class="n">${n}</div><div><h3>${title}</h3><p class="lead">${lead}</p>${body}</div></section>`;
}
const kv = (pairs) => `<dl class="kv">${pairs.map(([k, v]) => `<dt>${k}</dt><dd>${v}</dd>`).join("")}</dl>`;

function segmentationStep() {
  const g = state.result.segmentation, E = state.result.encoders;
  let body = `<p style="margin:0 0 12px">${esc(g.reason)}.</p>`;
  if (g.applied) {
    body += `<div class="figs">
      <figure><img src="${$("preview").src}" alt="Ảnh gốc"><figcaption>Ảnh gốc</figcaption></figure>
      <figure><img src="${g.overlay}" alt="Bản đồ phân vùng"><figcaption>Phân vùng: phần giữ lại có màu, phần bỏ đi làm mờ</figcaption></figure>
      <figure><img src="${g.isolated}" alt="Món đồ đã tách"><figcaption>Ảnh đưa vào mô hình</figcaption></figure></div>`;
    const cell = (e, s) => `${esc(styleName(e, s.id))}<br><span class="small muted">khoảng cách ${num(s.distance, 3)} / ngưỡng ${num(s.q99, 3)}</span>
      ${s.novel ? `<span class="pill novel">mới</span>` : `<span class="pill known">thuộc nhóm</span>`}`;
    body += `<h4 class="small" style="margin:16px 0 4px">Tác động của bước tách nền</h4><div class="table-wrap"><table class="cmp">
      <thead><tr><th></th><th>Không tách nền</th><th>Có tách nền</th></tr></thead><tbody>${
      ENC.map((e) => `<tr><th>${ENC_LABEL[e]}</th><td>${cell(e, g.before[e])}</td><td>${cell(e, E[e].style)}</td></tr>`).join("")}</tbody></table></div>`;
  }
  if (g.classes && g.classes.length) {
    body += `<h4 class="small" style="margin:16px 0 6px">Các vùng nhận ra trong ảnh gốc</h4><div>${g.classes.map((c) =>
      `<span class="chip" style="margin:0 6px 6px 0${c.garment ? ";background:var(--accent-soft);color:var(--accent)" : ""}">${esc(cap(c.vi))} ${pct(c.share)}</span>`).join("")}</div>`;
  }
  return stepCard(0, "Tách món đồ khỏi nền",
    "Mô hình phân vùng quần áo SegFormer gán từng điểm ảnh vào 18 loại (áo, quần, váy, mặt, tay, tóc, nền…). Phần trang phục được giữ lại, đặt lên nền trắng và cắt sát như ảnh trong bộ sưu tập.", body);
}

function renderProcess() {
  const enc = state.procEnc, r = state.result.encoders[enc], i = r.info, s = r.style, p = state.result.preprocess;
  const clip = enc === "fashionclip", seg = state.result.segmentation, m = state.meta[enc];
  const sw = `<div class="switch" role="group" aria-label="Chọn mô hình">${ENC.map((e) =>
    `<button data-proc="${e}" aria-pressed="${e === enc}">${ENC_LABEL[e]}</button>`).join("")}</div>`;
  const mx = Math.max(...r.preview.map(Math.abs), 1e-9);
  const barcode = `<div class="barcode" aria-hidden="true">${r.preview.map((v) =>
    `<span style="background:${v >= 0 ? "var(--accent)" : "var(--down)"};opacity:${(0.08 + 0.92 * Math.abs(v) / mx).toFixed(3)}"></span>`).join("")}</div>`;
  const top5 = r.ranking.slice(0, 5), dMin = top5[0].distance, dMax = top5[4].distance;
  const rank = `<div class="rank">${top5.map((x, j) => {
    const over = x.distance > m.styles[x.id].q99;
    return `<div class="r"${j === 0 ? ` style="font-weight:600"` : ""}><span>${esc(styleName(enc, x.id))}</span>
      <span class="track"><i style="width:${(100 - 60 * (x.distance - dMin) / Math.max(dMax - dMin, 1e-9)).toFixed(0)}%;${over ? "background:var(--warn)" : ""}"></i></span>
      <span class="num">${num(x.distance, 3)}</span></div>`;
  }).join("")}</div>`;
  const st = m.styles[s.id];
  const f = forecast(enc, +$("origin").value);
  const source = f.kind === "style" ? (f.tcn.source === "walk_forward"
    ? "Dự báo lấy từ lần kiểm tra lùi theo thời gian (walk-forward): mô hình chỉ được học dữ liệu trước thời điểm dự báo."
    : "Dự báo từ mô hình huấn luyện trên toàn bộ số liệu đã có kết quả.") : "";
  const models = ["tcn", "catboost", "persistence", "majority"];
  const MODEL = { tcn: "TCN (mạng tích chập theo thời gian)", catboost: "CatBoost (cây quyết định)", persistence: "Giữ nguyên xu hướng gần nhất", majority: "Luôn đoán lớp phổ biến nhất" };
  const quality = `<div class="table-wrap"><table class="cmp"><thead><tr><th>Mô hình dự báo</th>${ENC.map((e) => `<th>${ENC_LABEL[e]}</th>`).join("")}</tr></thead>
    <tbody>${models.map((md) => `<tr><th>${MODEL[md]}</th>${ENC.map((e) => `<td>${state.meta[e].metrics[md] ? num(state.meta[e].metrics[md].macro_f1) : "\u2013"}</td>`).join("")}</tr>`).join("")}</tbody></table></div>
    <p class="small muted" style="margin:8px 0 0">Điểm F1 trung bình ba lớp (Đang giảm, Ổn định, Đang lên) trên dữ liệu kiểm tra 01/2019 ${NDASH} 02/2020. Càng cao càng tốt, tối đa 1.</p>`;

  $("tab-process").innerHTML =
    `<div class="row-between"><p class="small muted" style="margin:0">Các bước xử lý bên trong, dành cho người muốn xem kỹ thuật.</p>${sw}</div>` +
    segmentationStep() +
    stepCard(1, "Chuẩn hoá ảnh", clip
      ? "Đổi kích thước cạnh ngắn về 224 điểm ảnh, cắt vuông ở giữa, chuẩn hoá màu theo FashionCLIP."
      : "Đổi kích thước về 232, cắt vuông 224 ở giữa, chuẩn hoá màu theo ImageNet, giống hệt lúc huấn luyện.",
      `<div class="figs"><figure><img src="${r.crop}" alt="Ảnh đầu vào của mô hình"><figcaption>Ảnh mô hình thực sự nhìn thấy</figcaption></figure>
       <div>${kv([["Ảnh gốc", `${p.original_size[0]} × ${p.original_size[1]} px`], ["Đã tách nền", seg.applied ? "có" : "không"], ["Đầu vào", i.input_shape.join(" × ")]])}</div></div>`) +
    stepCard(2, "Trích đặc trưng hình ảnh", clip
      ? "FashionCLIP (ViT-B/32) chia ảnh thành 49 ô 32×32 và tóm tắt cả ảnh thành một vector 512 số."
      : "ResNet-50 tạo bản đồ đặc trưng 2048 × 7 × 7 rồi lấy trung bình thành một vector 2048 số.",
      `<div class="figs"><figure><img src="${r.evidence}" alt="Vùng ảnh quan trọng"><figcaption>${esc(i.evidence_note)}</figcaption></figure>
       <div>${kv([["Số chiều", i.dim.toLocaleString("vi-VN")], ["Bản đồ đặc trưng", i.fmap_shape.join(" × ")], ["Độ lớn vector", num(i.norm)]])}
       <p class="small muted" style="margin:12px 0 0">64 giá trị đầu của vector (sau chuẩn hoá):</p>${barcode}</div></div>`) +
    stepCard(3, "Đối chiếu với 20 nhóm style", `Mỗi nhóm có một điểm trung tâm học từ bộ sưu tập. Món đồ chỉ được coi là style mới khi thoả cả hai điều kiện: (1) xa điểm trung tâm gần nhất hơn 99% sản phẩm của chính nhóm đó, và (2) ${s.knn_k} sản phẩm gần nhất trong bộ sưu tập kém giống hơn so với 99% sản phẩm. Chỉ thoả (1) nghĩa là món đồ nằm ở rìa nhóm nhưng vẫn có sản phẩm giống.`,
      kv([["Nhóm gần nhất", esc(st.name)], ["Khoảng cách", num(s.distance, 3)], ["Ngưỡng style mới của nhóm", num(s.q99, 3)],
          [`Độ giống trung bình ${s.knn_k} sản phẩm gần nhất`, pct(s.knn_sim)], ["Ngưỡng không có sản phẩm giống", pct(s.knn_threshold)],
          ["Kết luận", s.novel ? "Style mới" : `Thuộc nhóm (${typicality(s)})`]]) +
      `<h4 class="small" style="margin:16px 0 4px">5 nhóm gần nhất (khoảng cách càng nhỏ càng giống)</h4>${rank}
       <h4 class="small" style="margin:16px 0 8px">Sản phẩm tiêu biểu của nhóm “${esc(st.name)}”</h4>
       <div class="grid">${st.examples.map((c) => `<div class="prod"><div class="ph"><img loading="lazy" src="/catalog/${c}.jpg" alt=""></div></div>`).join("")}</div>`) +
    stepCard(4, "Dự báo xu hướng", "Mô hình đọc chuỗi doanh số theo tuần của nhóm style tới thời điểm dự báo và phân loại 4 tuần tiếp theo thành Đang giảm, Ổn định hoặc Đang lên.",
      (source ? `<p class="small" style="margin:0 0 12px">${source}</p>` : "") + quality);
  document.querySelectorAll("[data-proc]").forEach((b) => b.addEventListener("click", () => { state.procEnc = b.dataset.proc; renderWithFocus(renderProcess, `[data-proc="${state.procEnc}"]`); }));
}

bindUpload();
init();
