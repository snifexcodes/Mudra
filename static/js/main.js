// Mudra front end: wires the camera, the overlay, the sign-language packs and
// the four views (Translate, Talk, Train, Results) together.

import { api, packHands, packPose } from "./api.js";
import { Tracker } from "./tracker.js";
import { Overlay, introAnimation } from "./render.js";
import { Translator, esc } from "./translate.js";
import { Studio } from "./studio.js";
import { Talk } from "./talk.js";
import { renderResults } from "./results.js";
import { speak, voices } from "./speech.js";

const $ = (id) => document.getElementById(id);
const VIEWS = ["live", "talk", "train", "results"];
const SETTINGS_KEY = "mudra.settings";
const DEFAULTS = {
  pack: "", source: "model", holdMs: 900, minConf: 60, autoSpace: true, speak: false,
  twoHands: false, trails: true, mirror: true, cameraId: "", samplesPerRecording: 100,
  repsPerRecording: 10, voiceURI: "", rate: 1, listenLang: "",
};

const settings = loadSettings();
const state = {
  view: "live", status: null, packs: [], pack: null, cameraReady: false, injected: null,
  resultsKind: null,
};

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
  toastTimer = setTimeout(() => t.classList.remove("on"), action ? 7000 : 4600);
}

// ---------- speech ----------
function say(text) {
  return speak(text, { lang: (state.pack && state.pack.spoken) || "en", voiceURI: settings.voiceURI, rate: settings.rate });
}

// ---------- pack helpers ----------
const packCode = () => (state.pack ? state.pack.code : settings.pack || "isl");
const models = () => (state.pack && state.pack.models) || {};
function effectiveSource() {
  return settings.source === "model" && models().still ? "model" : "rules";
}

// ---------- views ----------
const translator = new Translator(
  { bubble: $("bubble"), glyph: $("glyph"), meta: $("bubble-meta"), ringFill: $("ring-fill"), caption: $("caption"), hint: $("hint") },
  settings,
  {
    pack: packCode, source: effectiveSource, hasMoving: () => !!models().moving,
    speak: say, toast, onSpeakAll: (text) => talk.add("you", text, "signed"),
  }
);

const studio = new Studio(
  {
    addForm: $("add-form"), newSign: $("new-sign"), kindButtons: [...document.querySelectorAll(".kind-pick [data-kind]")],
    presets: $("presets"), presetsBox: $("presets-box"), packName: $("train-pack-name"),
    list: $("sign-list"), datasetLine: $("dataset-line"), trainNote: $("train-note"), progress: $("train-progress"),
    train: $("btn-train"), record: $("btn-record"), recordText: $("record-text"), recHint: $("rec-hint"),
    dock: $("rec-dock"), hud: $("rec-hud"), countBig: $("count-big"), capture: $("capture"),
    captureFill: $("capture-fill"), captureLabel: $("capture-label"), captureN: $("capture-n"), captureWarn: $("capture-warn"),
  },
  settings,
  {
    pack: packCode, packInfo: () => state.pack || {}, toast,
    cameraReady: () => state.cameraReady,
    onPackChanged: (info) => applyPack(info, { keepSelection: true }),
    onTrained: async (msg) => {
      await loadPack(packCode(), { keepSelection: true });
      settings.source = "model";
      saveSettings();
      syncSourceUI();
      toast(`${msg} Translate now uses your signs.`, { label: "See results", run: () => setView("results") });
    },
  }
);

const talk = new Talk(
  {
    cats: $("talk-cats"), board: $("talk-board"), edit: $("talk-edit"), log: $("talk-log"),
    typeForm: $("talk-type"), typeInput: $("talk-input"), listen: $("talk-listen"), lang: $("talk-lang"),
    listenNote: $("talk-listen-note"), clear: $("talk-clear"), show: $("talk-show"),
    big: $("talk-big"), bigText: $("talk-big-text"),
  },
  settings,
  { speak: say, toast, save: () => saveSettings() }
);

function setView(view) {
  state.view = view;
  document.body.dataset.view = view;
  for (const v of VIEWS) $(`view-${v}`).hidden = v !== view;
  document.querySelectorAll(".tab").forEach((t) => t.setAttribute("aria-selected", String(t.dataset.view === view)));
  if (view !== "train" && studio.rec) studio.cancelRecording();
  if (view !== "talk") talk.stop();
  if (view === "train") studio.render();
  if (view === "results") loadResults();
  if (view === "live") translator.reset();
  $("tooltip").classList.remove("on");
}

async function loadResults() {
  const root = $("results-inner");
  let reports = null;
  try { reports = await api.report(packCode()); } catch (e) { if (e.status !== 404) toast(e.message); }
  state.resultsKind = renderResults(root, reports, {
    onGoTrain: () => setView("train"), packName: state.pack ? state.pack.name : "", kind: state.resultsKind,
  });
}

document.querySelectorAll(".tab").forEach((t) => t.addEventListener("click", () => setView(t.dataset.view)));

// ---------- packs ----------
async function loadPack(code, opts = {}) {
  const info = await api.pack(code);
  applyPack(info, opts);
  return info;
}

function applyPack(info, { keepSelection = false } = {}) {
  const changed = !state.pack || state.pack.code !== info.code;
  state.pack = info;
  const i = state.packs.findIndex((p) => p.code === info.code);
  const summary = { ...info, sign_count: (info.dataset ? info.dataset.signs.length : info.sign_count) };
  if (i >= 0) state.packs[i] = { ...state.packs[i], ...summary }; else state.packs.push(summary);
  if (changed) {
    settings.pack = info.code;
    if (settings.twoHands !== info.two_hands) {
      settings.twoHands = info.two_hands;
      tracker.setNumHands(info.two_hands ? 2 : 1).catch(() => {});
    }
    saveSettings();
    translator.reset();
    talk.setDefaultLanguage(info.spoken);
    for (const err of Object.values(info.model_errors || {})) toast(err);
  }
  studio.setPack(info, { keepSelection: keepSelection && !changed });
  syncSettingsUI();
  renderPacks();
  if (state.view === "results" && changed) loadResults();
}

function renderPacks() {
  const list = $("pack-list");
  list.innerHTML = state.packs.map((p) => {
    const trained = Object.entries(p.models || {}).filter(([, v]) => v).map(([k]) => k);
    const status = trained.length ? `Trained: ${trained.join(" and ")} signs` : (p.code === "asl" ? "Starter gestures, not trained" : "Not trained yet");
    const sel = state.pack && p.code === state.pack.code;
    return `<li><button type="button" class="pack-item" data-pack="${esc(p.code)}" aria-pressed="${sel}">
        <span class="short">${esc(p.short)}</span>
        <span class="info"><b>${esc(p.name)}</b><small>${esc(p.region || "")}${p.region ? ". " : ""}${p.sign_count || 0} signs. ${status}.</small></span>
      </button>${p.builtin ? "" : `<button type="button" class="icon-btn" data-del-pack="${esc(p.code)}" aria-label="Delete ${esc(p.name)}" title="Delete"><svg><use href="#i-trash"/></svg></button>`}</li>`;
  }).join("");
  $("pack-export").href = api.exportUrl(packCode());
  $("pack-export").setAttribute("download", `mudra-${packCode()}.zip`);
}

$("pack-list").addEventListener("click", async (ev) => {
  const pick = ev.target.closest("[data-pack]");
  const del = ev.target.closest("[data-del-pack]");
  if (pick) {
    try { await loadPack(pick.dataset.pack); toast(`Switched to ${state.pack.name}.`); }
    catch (e) { toast(e.message); }
  } else if (del) {
    const code = del.dataset.delPack;
    if (del.dataset.confirm !== "1") {
      del.dataset.confirm = "1";
      del.classList.add("danger-btn");
      del.title = "Click again to delete this pack and all its recordings";
      setTimeout(() => { del.dataset.confirm = ""; del.classList.remove("danger-btn"); }, 3500);
      return;
    }
    try {
      const out = await api.deletePack(code);
      state.packs = out.packs;
      if (packCode() === code) await loadPack("isl"); else renderPacks();
      toast("Pack deleted.");
    } catch (e) { toast(e.message); }
  }
});

$("new-pack").addEventListener("submit", async (ev) => {
  ev.preventDefault();
  const f = ev.target;
  try {
    const info = await api.createPack({
      name: f.name.value, short: f.short.value, region: f.region.value,
      spoken: f.spoken.value || "en", two_hands: f.two_hands.checked,
    });
    state.packs.push(info);
    applyPack(info);
    f.reset();
    f.two_hands.checked = true;
    toast(`Added ${info.name}. Open Train to add its signs.`);
  } catch (e) { toast(e.message); }
});

$("pack-import").addEventListener("change", async (ev) => {
  const file = ev.target.files[0];
  ev.target.value = "";
  if (!file) return;
  try {
    const out = await api.importPack(file);
    if (!out.merged) state.packs.push(out.pack);
    applyPack(out.pack);
    toast(out.merged
      ? `Added ${out.added} new recordings to ${out.pack.name}. Train again to use them.`
      : `Imported ${out.pack.name} with ${out.added} recordings. Train it to use them.`);
  } catch (e) { toast(e.message); }
});

// ---------- source of predictions ----------
function syncSourceUI() {
  const src = effectiveSource();
  const p = state.pack;
  const chip = $("pack-chip");
  chip.classList.toggle("model", src === "model" || !!models().moving);
  $("pack-short").textContent = p ? p.short : "…";
  const starterCount = p && p.starter ? p.starter.length : 0;
  $("source-text").textContent = src === "model" || models().moving ? "My model" : (starterCount ? "Starter gestures" : "Not trained");
  chip.title = p ? `${p.name}. Click to choose another sign language.` : "Choose a sign language";
  document.querySelectorAll("[data-source]").forEach((b) => {
    b.setAttribute("aria-pressed", String(b.dataset.source === src));
    if (b.dataset.source === "model") b.disabled = !models().still;
  });
  const still = models().still;
  $("source-help").textContent = still
    ? `Your still-sign model knows ${still.labels.length} signs (${still.name}, trained ${still.trained_at}).`
    : "Train still signs in the Train tab to use your own hand shapes.";
  renderGuide();
}

function renderGuide() {
  const p = state.pack;
  if (!p) return;
  const list = $("guide-list");
  const rows = [];
  const m = models();
  const describe = (l, kind) => {
    if (l === "SPACE") return "Adds a space";
    if (l === "DELETE") return "Deletes the last character";
    if (l.startsWith("NO SIGN")) return "Recognised, but never added";
    return kind === "moving" ? "Your recorded moving sign" : "Your recorded still sign";
  };
  if (effectiveSource() === "model" && m.still) m.still.labels.forEach((l) => rows.push([l, describe(l, "still")]));
  else (p.starter || []).forEach((s) => rows.push([s.label, s.how]));
  if (m.moving) m.moving.labels.forEach((l) => rows.push([l, describe(l, "moving")]));
  $("guide-title").textContent = `Signs I can read in ${p.short}`;
  $("guide-intro").textContent = rows.length
    ? "Hold a still sign until the ring fills. Make a moving sign once, then pause. Lower your hand to add a space."
    : `${p.name} has no starter gestures and nothing trained yet. Open the Train tab to record signs.`;
  list.innerHTML = rows.map(([l, d]) =>
    `<li><span class="g${[...l].length > 1 ? " word" : ""}">${esc(l)}</span><span class="d">${esc(d)}</span></li>`).join("");
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
$("pack-chip").addEventListener("click", () => openDrawer("packs"));
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
  $("s-reps").value = settings.repsPerRecording;
  $("o-reps").textContent = String(settings.repsPerRecording);
  $("s-rate").value = settings.rate;
  $("o-rate").textContent = `${(+settings.rate).toFixed(1)}×`;
  $("s-autospace").checked = settings.autoSpace;
  $("s-speak").checked = settings.speak;
  $("s-twohands").checked = settings.twoHands;
  $("s-trails").checked = settings.trails;
  $("s-mirror").checked = settings.mirror;
  video.classList.toggle("mirror", settings.mirror);
  overlay.mirror = settings.mirror;
  syncSourceUI();
}
const bindRange = (id, key, num = Number) => $(id).addEventListener("input", (e) => {
  settings[key] = num(e.target.value); saveSettings(); syncSettingsUI();
});
bindRange("s-hold", "holdMs");
bindRange("s-conf", "minConf");
bindRange("s-reps", "repsPerRecording");
bindRange("s-rate", "rate", parseFloat);
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
$("s-voice").addEventListener("change", (e) => { settings.voiceURI = e.target.value; saveSettings(); say("Hello"); });

function listVoices() {
  const all = voices();
  const lang = ((state.pack && state.pack.spoken) || "en").split("-")[0];
  const sorted = [...all].sort((a, b) => (b.lang.startsWith(lang) - a.lang.startsWith(lang)) || a.name.localeCompare(b.name));
  $("s-voice").innerHTML = `<option value="">Automatic (${esc((state.pack && state.pack.spoken) || "en")})</option>` +
    sorted.map((v) => `<option value="${esc(v.voiceURI)}" ${v.voiceURI === settings.voiceURI ? "selected" : ""}>${esc(v.name)} (${esc(v.lang)})</option>`).join("");
}
if ("speechSynthesis" in window) speechSynthesis.addEventListener("voiceschanged", listVoices);

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
    if (!$("talk-big").hidden) { $("talk-big").hidden = true; return; }
    if (document.querySelector(".drawer.open")) { closeDrawers(); return; }
    if (studio.rec) { studio.cancelRecording(); return; }
  }
  const tag = (e.target.tagName || "").toLowerCase();
  if (["input", "textarea", "select"].includes(tag) || e.ctrlKey || e.metaKey || e.altKey) return;
  if (!$("intro").hidden && !$("intro").classList.contains("leaving")) return;
  if (e.target.closest && e.target.closest("button") && (e.key === " " || e.key === "Enter")) return;
  if (["1", "2", "3", "4"].includes(e.key)) { setView(VIEWS[+e.key - 1]); return; }
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
function onFrame(result, now, fresh, poseResult) {
  if (state.injected) { result = state.injected.hands; poseResult = state.injected.pose; }
  const vw = video.videoWidth || 1280, vh = video.videoHeight || 720;
  const hands = packHands(result, vw, vh, settings.twoHands ? 2 : 1);
  const pose = packPose(poseResult, vw, vh);
  const drawable = state.view === "results" || state.view === "talk" ? null : result;
  overlay.draw(drawable, now, { trails: settings.trails });
  if (state.view === "live") translator.frame(result, hands, pose, now, overlay);
  if (state.view === "train") studio.frame(hands, pose, now);
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
    const st = await api.status();
    state.status = st;
    state.packs = st.packs;
    const code = st.packs.some((p) => p.code === settings.pack) ? settings.pack : st.default;
    await loadPack(code);
    if (st.training && st.training.state === "running" && st.training.pack === code) studio.watchTraining();
  } catch (e) {
    introStatus(e.message, true);
    return;
  }
  listVoices();
  loadPromise = tracker.load(state.status.assets, settings.twoHands ? 2 : 1, (m) => introStatus(m))
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
    inject: (hands, pose = null) => { state.injected = hands ? { hands, pose } : null; },
    skipIntro: () => { leaveIntro(); state.cameraReady = true; studio.render(); tracker.start(onFrame); },
    translator, studio, talk, setView, settings, state,
  };
}

boot();
