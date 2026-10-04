// When served by FastAPI (http://localhost:8000) the API is same-origin.
// If you open index.html some other way (e.g. VS Code Live Server on :5500) it talks to :8000.
const API_BASE = location.port === "8000" ? "" : "http://localhost:8000";
const MAX_MB = 25;

const $ = (id) => document.getElementById(id);
const el = {
  dropzone: $("dropzone"), fileInput: $("fileInput"), dzEmpty: $("dzEmpty"), dzPreview: $("dzPreview"),
  previewImg: $("previewImg"), fileName: $("fileName"), clearBtn: $("clearBtn"),
  refInput: $("refInput"), refName: $("refName"), refClear: $("refClear"), refBox: $("refBox"),
  chips: $("hazeChips"), dehazeBtn: $("dehazeBtn"), errorBox: $("errorBox"),
  statusPill: $("statusPill"), statusText: $("statusText"),
  results: $("results"), outHazy: $("outHazy"), outClear: $("outClear"), downloadBtn: $("downloadBtn"),
  tags: $("resultTags"), psnrVal: $("psnrVal"), ssimVal: $("ssimVal"), psnrBar: $("psnrBar"), ssimBar: $("ssimBar"),
  timeVal: $("timeVal"), deviceNote: $("deviceNote"), densityVal: $("densityVal"), densityNote: $("densityNote"),
  outDiff: $("outDiff"), dlPlot: $("dlPlot"), dlRaw: $("dlRaw"), diffBadge: $("diffBadge"), legendMax: $("legendMax"), maeVal: $("maeVal"),
  banner: $("metricBanner"),
};

const state = { file: null, ref: null, hazeType: "auto", busy: false, previewUrl: null };

/* ------------------------------ backend status ------------------------------ */
async function checkHealth() {
  try {
    const r = await fetch(`${API_BASE}/api/health`);
    const j = await r.json();
    if (j.status === "ok") {
      el.statusPill.className = "status-pill ok";
      el.statusText.textContent = `Model ready · ${j.device.toUpperCase()} · ${j.parameters_million}M params`;
      return;
    }
    throw new Error("loading");
  } catch {
    el.statusPill.className = "status-pill err";
    el.statusText.textContent = "Backend offline — start it with: python app.py";
    setTimeout(checkHealth, 4000);
  }
}

/* ------------------------------ file handling ------------------------------ */
function showError(msg) {
  el.errorBox.textContent = msg;
  el.errorBox.hidden = !msg;
}

function validate(file) {
  if (!file.type.startsWith("image/")) return "Please choose an image file (PNG, JPG, BMP, TIFF).";
  if (file.size > MAX_MB * 1024 * 1024) return `Image is larger than ${MAX_MB} MB.`;
  return null;
}

function setFile(file) {
  const err = file && validate(file);
  if (err) return showError(err);
  showError("");

  if (state.previewUrl) URL.revokeObjectURL(state.previewUrl);
  state.file = file || null;

  if (file) {
    state.previewUrl = URL.createObjectURL(file);
    el.previewImg.src = state.previewUrl;
    el.fileName.textContent = `${file.name} · ${(file.size / 1024).toFixed(0)} KB`;
  }
  el.dzEmpty.hidden = !!file;
  el.dzPreview.hidden = !file;
  el.dehazeBtn.disabled = !file || state.busy;
}

el.fileInput.addEventListener("change", () => setFile(el.fileInput.files[0]));
el.clearBtn.addEventListener("click", (e) => {
  e.preventDefault(); // the dropzone is a <label>; don't open the picker
  el.fileInput.value = "";
  setFile(null);
});

["dragenter", "dragover"].forEach((ev) =>
  el.dropzone.addEventListener(ev, (e) => { e.preventDefault(); el.dropzone.classList.add("drag"); }));
["dragleave", "drop"].forEach((ev) =>
  el.dropzone.addEventListener(ev, (e) => { e.preventDefault(); el.dropzone.classList.remove("drag"); }));
el.dropzone.addEventListener("drop", (e) => setFile(e.dataTransfer.files[0]));
el.dropzone.addEventListener("keydown", (e) => { if (e.key === "Enter" || e.key === " ") { e.preventDefault(); el.fileInput.click(); } });

// optional reference image
el.refInput.addEventListener("change", () => {
  const f = el.refInput.files[0];
  const err = f && validate(f);
  if (err) { showError(err); el.refInput.value = ""; return; }
  showError("");
  state.ref = f || null;
  el.refName.textContent = f ? f.name : "No reference selected";
  el.refClear.hidden = !f;
});
el.refClear.addEventListener("click", () => {
  el.refInput.value = ""; state.ref = null;
  el.refName.textContent = "No reference selected"; el.refClear.hidden = true;
});

/* ------------------------------ haze type chips ------------------------------ */
el.chips.addEventListener("click", (e) => {
  const chip = e.target.closest(".chip");
  if (!chip) return;
  state.hazeType = chip.dataset.type;
  el.chips.querySelectorAll(".chip").forEach((c) => {
    const on = c === chip;
    c.classList.toggle("active", on);
    c.setAttribute("aria-checked", on);
  });
});

/* ------------------------------ run dehazing ------------------------------ */
function setBusy(b) {
  state.busy = b;
  el.dehazeBtn.disabled = b || !state.file;
  el.dehazeBtn.querySelector(".spinner").hidden = !b;
  el.dehazeBtn.querySelector(".btn-label").textContent = b ? "Dehazing…" : "Dehaze Image";
}

el.dehazeBtn.addEventListener("click", async () => {
  if (!state.file || state.busy) return;
  showError("");
  setBusy(true);

  const form = new FormData();
  form.append("image", state.file);
  form.append("haze_type", state.hazeType);
  if (state.ref) form.append("reference", state.ref);

  try {
    const res = await fetch(`${API_BASE}/api/dehaze`, { method: "POST", body: form });
    const data = await res.json().catch(() => ({}));
    if (!res.ok) throw new Error(data.detail || `Server error (${res.status})`);
    renderResult(data);
  } catch (err) {
    const offline = err instanceof TypeError;
    showError(offline ? "Cannot reach the backend. Is it running on http://localhost:8000 ?" : err.message);
  } finally {
    setBusy(false);
  }
});

/* ------------------------------ render ------------------------------ */
function countUp(node, to, decimals, ms = 1000) {
  const t0 = performance.now();
  const step = (t) => {
    const p = Math.min(1, (t - t0) / ms);
    node.textContent = (to * (1 - Math.pow(1 - p, 3))).toFixed(decimals);
    if (p < 1) requestAnimationFrame(step);
  };
  requestAnimationFrame(step);
}

const TYPE_LABEL = { auto: "Auto-detect", thin: "Thin haze", moderate: "Moderate haze", thick: "Thick haze" };

function renderResult(d) {
  el.outHazy.src = d.hazy;
  el.outClear.src = d.dehazed;
  el.downloadBtn.href = d.dehazed;
  el.downloadBtn.download = (state.file.name.replace(/\.[^.]+$/, "") || "image") + "_dehazed.png";

  const base = state.file.name.replace(/\.[^.]+$/, "") || "image";
  el.outDiff.src = d.error_map;
  el.dlPlot.href = d.error_map_plot; el.dlPlot.download = `error_map_plot_${base}.png`;
  el.dlRaw.href = d.error_map;       el.dlRaw.download = `error_map_raw_${base}.png`;
  el.legendMax.textContent = d.error_vmax;
  el.diffBadge.textContent = d.metrics.mode === "reference" ? "Error Map · vs Ground Truth" : "Change Map · vs Hazy Input";
  el.maeVal.textContent = d.metrics.mae.toFixed(4);

  const chosen = TYPE_LABEL[d.haze_type];
  const resized = d.original_size[0] !== d.width || d.original_size[1] !== d.height;
  el.tags.innerHTML = [
    `Mode: ${chosen}`,
    `${d.width} × ${d.height} px` + (resized ? " (resized)" : ""),
  ].map((t) => `<span class="tag">${t}</span>`).join("");

  // metrics (bottom)
  const m = d.metrics;
  countUp(el.psnrVal, m.psnr, 2);
  countUp(el.ssimVal, m.ssim, 4);
  el.psnrBar.style.width = "0%"; el.ssimBar.style.width = "0%";
  requestAnimationFrame(() => {
    el.psnrBar.style.width = Math.min(100, (m.psnr / 40) * 100) + "%";
    el.ssimBar.style.width = Math.max(0, Math.min(100, m.ssim * 100)) + "%";
  });

  el.timeVal.textContent = d.inference_seconds.toFixed(2);
  el.deviceNote.textContent = `Ran on ${d.device.toUpperCase()}`;
  el.densityVal.textContent = Math.round(d.haze_estimate.density * 100);
  el.densityNote.textContent = `looks ${d.haze_estimate.label} · heuristic`;

  if (m.mode === "reference") {
    el.banner.className = "metric-banner ref";
    el.banner.textContent = "✔ PSNR, SSIM and the error map are computed between the dehazed output and your ground-truth reference image (error = mean absolute RGB difference, colour range 0–0.3).";
  } else {
    el.banner.className = "metric-banner noref";
    el.banner.textContent =
      "ⓘ No ground-truth image was provided, so PSNR, SSIM and the map compare the output with the hazy input (they show how much " +
      "the image changed, not its quality). Add a clear reference image in step 1 for a true error map.";
  }

  el.results.hidden = false;
  el.results.scrollIntoView({ behavior: "smooth", block: "start" });
}

checkHealth();
