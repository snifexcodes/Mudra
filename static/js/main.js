// Mudra front end: wires the camera, the overlay and the three views together.

import { api, packHands } from "./api.js";
import { Tracker } from "./tracker.js";
import { Overlay, introAnimation } from "./render.js";
import { Translator, esc } from "./translate.js";
import { Studio } from "./studio.js";
import { renderResults } from "./results.js";

const $ = (id) => document.getElementById(id);
const SETTINGS_KEY = "mudra.settings";
const DEFAULTS = {
  source: "model", holdMs: 900, minConf: 60, autoSpace: true, speak: false,
  twoHands: false, trails: true, mirror: true, cameraId: "", samplesPerRecording: 100,
};

const settings = loadSettings();
const state = { view: "live", status: null, modelReady: false, cameraReady: false, injected: null };

const video = $("video");
const tracker = new Tracker(video);
const overlay = new Overlay($("overlay"), video);

// ---------- toast ----------
let toastTimer = 0;
function toast(message, action) {
  const t = $("toast"), btn = $("toast-action");
  $("toast-text").textContent = message;
  btn.hidden = !action;
  btn.onclick = null;
  if (action) {
    btn.textContent = action.label;
    btn.onclick = () => { t.classList.remove("on"); action.run(); };
  }
  t.classList.add("on");
  clearTimeout(toastTimer);
  toastTimer = setTimeout(() => t.classList.remove("on"), action ? 7000 : 4200);
}

// ---------- views ----------
const translator = new Translator(
  { bubble: $("bubble"), glyph: $("glyph"), meta: $("bubble-meta"), ringFill: $("ring-fill"), caption: $("caption"), hint: $("hint") },
  settings,
  { source: effectiveSource, onPrediction: () => {}, toast }
);

const studio = new Studio(
  {
    addForm: $("add-form"), newSign: $("new-sign"), presets: [...document.querySelectorAll("[data-preset]")],
    list: $("sign-list"), datasetLine: $("dataset-line"), trainNote: $("train-note"), progress: $("train-progress"),
    train: $("btn-train"), record: $("btn-record"), recordText: $("record-text"), recHint: $("rec-hint"),
    dock: $("rec-dock"), hud: $("rec-hud"), countBig: $("count-big"), capture: $("capture"),
    captureFill: $("capture-fill"), captureLabel: $("capture-label"), captureN: $("capture-n"), captureWarn: $("capture-warn"),
  },
  settings,
  {
    toast,
    starterLabels: () => (state.status ? state.status.starter.map((s) => s.label) : []),
    cameraReady: () => state.cameraReady,
    onTrained: async (msg) => {
      await refreshStatus();
      settings.source = "model";
      saveSettings();
      syncSettingsUI();
      toast(`${msg} Translate now uses your signs.`, { label: "See results", run: () => setView("results") });
    },
  }
);

function setView(view) {
  state.view = view;
  document.body.dataset.view = view;
  for (const v of ["live", "train", "results"]) $(`view-${v}`).hidden = v !== view;
  document.querySelectorAll(".tab").forEach((t) => t.setAttribute("aria-selected", String(t.dataset.view === view)));
  if (view !== "train" && studio.rec) studio.cancelRecording();
  if (view === "train") studio.render();
  if (view === "results") loadResults();
  $("tooltip").classList.remove("on");
}

async function loadResults() {
  const root = $("results-inner");
  let report = null;
  try { report = await api.report(); } catch (e) { if (e.status !== 404) { toast(e.message); } }
  renderResults(root, report, { onGoTrain: () => setView("train") });
}

document.querySelectorAll(".tab").forEach((t) => t.addEventListener("click", () => setView(t.dataset.view)));

// ---------- source of predictions ----------
function effectiveSource() {
  return settings.source === "model" && state.modelReady ? "model" : "rules";
}

function syncSourceUI() {
  const src = effectiveSource();
  const chip = $("source-chip");
  chip.classList.toggle("model", src === "model");
  $("source-text").textContent = src === "model" ? "My model" : "Starter gestures";
  const m = state.status && state.status.model;
  chip.title = src === "model" && m ? `${m.name}, ${m.labels.length} signs. Change in Settings.` : "Change in Settings";
  document.querySelectorAll("[data-source]").forEach((b) => {
    b.setAttribute("aria-pressed", String(b.dataset.source === src));
    if (b.dataset.source === "model") b.disabled = !state.modelReady;
  });
  $("source-help").textContent = state.modelReady
    ? `Your model knows ${m.labels.length} signs (${m.name}, trained ${m.trained_at}).`
    : "Train a model in the Train tab to use your own signs.";
  renderGuide();
}

function renderGuide() {
  const src = effectiveSource();
  const list = $("guide-list");
  if (src === "model" && state.status && state.status.model) {
    $("guide-intro").textContent = "Your trained model knows these signs. Hold one steady until the ring fills to add it.";
    list.innerHTML = state.status.model.labels.map((l) => {
      const ctrl = l === "SPACE" ? "Adds a space" : l === "DELETE" ? "Deletes the last character" : "Your recorded sign";
      return `<li><span class="g${[...l].length > 1 ? " word" : ""}">${esc(l)}</span><span class="d">${ctrl}</span></li>`;
    }).join("");
  } else if (state.status) {
    $("guide-intro").textContent = "Starter gestures work without training. Hold one steady until the ring fills to add it. Lower your hand to add a space.";
    list.innerHTML = state.status.starter.map((s) =>
      `<li><span class="g${[...s.label].length > 1 ? " word" : ""}">${esc(s.label)}</span><span class="d">${esc(s.how)}</span></li>`
    ).join("");
  }
}

async function refreshStatus() {
  const st = await api.status();
  state.status = st;
  state.modelReady = !!st.model;
  studio.setDataset(st.dataset);
  syncSourceUI();
  if (st.training && st.training.state === "running") studio.watchTraining();
  return st;
}

// ---------- drawers ----------
let lastFocus = null;
function openDrawer(id) {
  closeDrawers();
  lastFocus = document.activeElement;
  const d = $(id);
  $("scrim").hidden = false;
  d.classList.add("open");
  d.setAttribute("aria-hidden", "false");
  const first = d.querySelector("button, input, select");
  if (first) setTimeout(() => first.focus(), 50);
}
function closeDrawers() {
  document.querySelectorAll(".drawer.open").forEach((d) => { d.classList.remove("open"); d.setAttribute("aria-hidden", "true"); });
  $("scrim").hidden = true;
  if (lastFocus) { lastFocus.focus(); lastFocus = null; }
}
$("open-settings").addEventListener("click", () => openDrawer("settings"));
$("source-chip").addEventListener("click", () => openDrawer("settings"));
$("open-guide").addEventListener("click", () => openDrawer("guide"));
$("scrim").addEventListener("click", closeDrawers);
document.querySelectorAll("[data-close]").forEach((b) => b.addEventListener("click", closeDrawers));

// ---------- settings ----------
function loadSettings() {
  try { return { ...DEFAULTS, ...JSON.parse(localStorage.getItem(SETTINGS_KEY) || "{}") }; }
  catch { return { ...DEFAULTS }; }
}
function saveSettings() {
  try { localStorage.setItem(SETTINGS_KEY, JSON.stringify(settings)); } catch { /* storage unavailable */ }
}
function syncSettingsUI() {
  $("s-hold").value = settings.holdMs;
  $("o-hold").textContent = `${(settings.holdMs / 1000).toFixed(1)} s`;
  $("s-conf").value = settings.minConf;
  $("o-conf").textContent = `${settings.minConf}%`;
  $("s-autospace").checked = settings.autoSpace;
  $("s-speak").checked = settings.speak;
  $("s-twohands").checked = settings.twoHands;
  $("s-trails").checked = settings.trails;
  $("s-mirror").checked = settings.mirror;
  video.classList.toggle("mirror", settings.mirror);
  overlay.mirror = settings.mirror;
  syncSourceUI();
}
$("s-hold").addEventListener("input", (e) => { settings.holdMs = +e.target.value; saveSettings(); syncSettingsUI(); });
$("s-conf").addEventListener("input", (e) => { settings.minConf = +e.target.value; saveSettings(); syncSettingsUI(); });
$("s-autospace").addEventListener("change", (e) => { settings.autoSpace = e.target.checked; saveSettings(); });
$("s-speak").addEventListener("change", (e) => { settings.speak = e.target.checked; saveSettings(); });
$("s-trails").addEventListener("change", (e) => { settings.trails = e.target.checked; saveSettings(); });
$("s-mirror").addEventListener("change", (e) => { settings.mirror = e.target.checked; saveSettings(); syncSettingsUI(); });
$("s-twohands").addEventListener("change", async (e) => {
  settings.twoHands = e.target.checked; saveSettings();
  try { await tracker.setNumHands(settings.twoHands ? 2 : 1); } catch (err) { toast(err.message); }
});
document.querySelectorAll("[data-source]").forEach((b) => b.addEventListener("click", () => {
  settings.source = b.dataset.source; saveSettings(); syncSourceUI();
}));
$("s-camera").addEventListener("change", async (e) => {
  settings.cameraId = e.target.value; saveSettings();
  try { await tracker.startCamera(settings.cameraId); toast("Camera switched."); }
  catch (err) { toast(err.message); }
});

async function listCameras() {
  if (!navigator.mediaDevices || !navigator.mediaDevices.enumerateDevices) return;
  const devices = (await navigator.mediaDevices.enumerateDevices()).filter((d) => d.kind === "videoinput");
  const current = tracker.currentDeviceId();
  $("s-camera").innerHTML = devices.map((d, i) =>
    `<option value="${esc(d.deviceId)}" ${d.deviceId === current ? "selected" : ""}>${esc(d.label || `Camera ${i + 1}`)}</option>`
  ).join("") || "<option>No camera found</option>";
}

// ---------- caption buttons + keyboard ----------
$("btn-space").addEventListener("click", () => translator.space());
$("btn-back").addEventListener("click", () => translator.backspace());
$("btn-clear").addEventListener("click", () => translator.clear());
$("btn-copy").addEventListener("click", () => translator.copy());
$("btn-speak").addEventListener("click", () => translator.speakAll());

document.addEventListener("keydown", (e) => {
  if (e.key === "Escape") {
    if (document.querySelector(".drawer.open")) { closeDrawers(); return; }
    if (studio.rec) { studio.cancelRecording(); return; }
  }
  const tag = (e.target.tagName || "").toLowerCase();
  if (["input", "textarea", "select"].includes(tag) || e.ctrlKey || e.metaKey || e.altKey) return;
  if (!$("intro").hidden && !$("intro").classList.contains("leaving")) return;
  if (e.target.closest && e.target.closest("button") && (e.key === " " || e.key === "Enter")) return;
  if (["1", "2", "3"].includes(e.key)) { setView(["live", "train", "results"][+e.key - 1]); return; }
  if (state.view === "live") {
    if (e.key === " ") { e.preventDefault(); translator.space(); }
    else if (e.key === "Backspace") { e.preventDefault(); translator.backspace(); }
    else if (e.key === "Enter") { e.preventDefault(); translator.speakAll(); }
  } else if (state.view === "train") {
    if (e.key === "r" || e.key === "R") { e.preventDefault(); studio.startRecording(); }
  }
});

// ---------- main loop ----------
let fpsShown = 0;
function onFrame(result, now) {
  if (state.injected) result = state.injected;
  const vw = video.videoWidth || 1280, vh = video.videoHeight || 720;
  const hands = packHands(result, vw, vh, settings.twoHands ? 2 : 1);
  overlay.draw(state.view === "results" ? null : result, now, { trails: settings.trails });
  if (state.view === "live") translator.frame(result, hands, now, overlay);
  if (state.view === "train") studio.frame(hands, now);
  if (now - fpsShown > 500) {
    fpsShown = now;
    $("fps").textContent = state.cameraReady && tracker.fps ? `${tracker.fps} fps` : "";
  }
}

// ---------- start up ----------
const stopIntro = introAnimation($("intro-canvas"));
const introStatus = (msg, error = false) => {
  const el = $("intro-status");
  el.textContent = msg;
  el.classList.toggle("error", error);
};

function leaveIntro() {
  const intro = $("intro");
  intro.classList.add("leaving");
  setTimeout(() => { intro.hidden = true; stopIntro(); }, 700);
}

let loadPromise = null;
async function boot() {
  syncSettingsUI();
  try {
    await refreshStatus();
  } catch (e) {
    introStatus(e.message, true);
    return;
  }
  const st = state.status;
  if (st.model_error) toast(st.model_error);
  loadPromise = tracker.load(st.assets, settings.twoHands ? 2 : 1, (m) => introStatus(m))
    .then(() => introStatus(""))
    .catch((e) => { introStatus(e.message, true); throw e; });
  loadPromise.catch(() => {});
}

$("btn-start").addEventListener("click", async () => {
  const btn = $("btn-start");
  btn.disabled = true;
  try {
    if (!loadPromise) throw new Error("Still connecting to the Mudra server. Try again in a moment.");
    introStatus("Loading hand tracking…");
    await loadPromise;
    introStatus("Starting camera…");
    await tracker.startCamera(settings.cameraId);
    state.cameraReady = true;
    video.classList.add("live");
    leaveIntro();
    tracker.start(onFrame);
    studio.render();
    listCameras();
  } catch (e) {
    introStatus(e.message, true);
    btn.disabled = false;
    if (String(e.message).includes("couldn't load") || String(e.message).includes("couldn't start")) boot();
  }
});

// Developer hook for testing the UI without a camera: open /?debug
if (new URLSearchParams(location.search).has("debug")) {
  window.mudra = {
    inject: (res) => { state.injected = res; },
    skipIntro: () => { leaveIntro(); state.cameraReady = true; studio.render(); tracker.start(onFrame); },
    translator, studio, setView, settings,
  };
}

boot();
