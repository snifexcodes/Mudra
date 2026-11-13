// Train view: manage signs, record samples from the webcam, train the model.

import { api, newSessionId } from "./api.js";
import { esc } from "./translate.js";

const TARGET = 300;           // samples per sign that fills the row's bar
const MIN_SAMPLES = 20;       // must match core/model.py MIN_SAMPLES_PER_SIGN
const CAPTURE_EVERY_MS = 66;  // ~15 frames per second
const CAPTURE_RING = 628.3;
const PENDING_KEY = "mudra.pendingSigns";

const PRESETS = {
  alphabet: "ABCDEFGHIKLMNOPQRSTUVWXY".split(""),
  words: ["HELLO", "THANK YOU", "YES", "NO", "PLEASE", "SORRY", "GOOD", "I LOVE YOU", "HELP", "WATER"],
  controls: ["SPACE", "DELETE"],
};

export class Studio {
  constructor(els, settings, hooks) {
    this.els = els;
    this.settings = settings;
    this.hooks = hooks;            // { toast(msg, action), onTrained(), starterLabels() }
    this.signs = [];               // from the server
    this.pending = loadPending();  // added but not recorded yet
    this.selected = null;
    this.rec = null;               // active recording state
    this.polling = null;
    this.confirmDelete = null;
    this.bind();
  }

  bind() {
    const e = this.els;
    e.addForm.addEventListener("submit", (ev) => {
      ev.preventDefault();
      const name = e.newSign.value;
      if (this.addSign(name)) e.newSign.value = "";
    });
    e.presets.forEach((b) => b.addEventListener("click", () => this.addPreset(b.dataset.preset)));
    e.record.addEventListener("click", () => this.startRecording());
    e.train.addEventListener("click", () => this.train());
    e.list.addEventListener("click", (ev) => this.onListClick(ev));
    e.list.addEventListener("keydown", (ev) => {
      const row = ev.target.closest(".sign-row");
      if (!row || ev.target.tagName === "INPUT") return;
      if (ev.key === "Enter" || ev.key === " ") { ev.preventDefault(); this.select(row.dataset.label); }
      if (ev.key === "ArrowDown" || ev.key === "ArrowUp") {
        ev.preventDefault();
        const rows = [...e.list.querySelectorAll(".sign-row")];
        const i = rows.indexOf(row) + (ev.key === "ArrowDown" ? 1 : -1);
        if (rows[i]) { rows[i].focus(); this.select(rows[i].dataset.label); }
      }
    });
  }

  setDataset(summary) {
    if (this.total !== undefined && summary.total !== this.total) this.outcome = null;  // data changed
    this.signs = summary.signs || [];
    this.total = summary.total || 0;
    this.last = summary.last;
    const known = new Set(this.signs.map((s) => s.label));
    this.pending = this.pending.filter((l) => !known.has(l));
    savePending(this.pending);
    this.render();
  }

  allSigns() {
    const rows = this.signs.map((s) => ({ ...s }));
    for (const l of this.pending) rows.push({ label: l, samples: 0, sessions: 0 });
    return rows.sort((a, b) => a.label.localeCompare(b.label, undefined, { numeric: true }));
  }

  addSign(raw, quiet = false) {
    const name = String(raw || "").replace(/\s+/g, " ").trim().toUpperCase();
    if (!name) { if (!quiet) this.hooks.toast("Type a name for the sign first."); return false; }
    if (name.length > 24) { this.hooks.toast("Keep sign names under 24 characters."); return false; }
    if (/[<>"\\]/.test(name)) { this.hooks.toast('Sign names can\'t contain < > " or \\.'); return false; }
    const exists = this.allSigns().some((s) => s.label === name);
    if (!exists) { this.pending.push(name); savePending(this.pending); }
    if (!quiet) this.select(name);
    this.render();
    return true;
  }

  addPreset(key) {
    const list = key === "starter" ? this.hooks.starterLabels() : PRESETS[key];
    list.forEach((l) => this.addSign(l, true));
    if (!this.selected) this.select(list[0]);
    this.render();
    if (key === "alphabet") this.hooks.toast("Added A–Z without J and Z: those letters need movement, which a single frame can't show.");
  }

  select(label) {
    this.selected = label;
    this.confirmDelete = null;
    this.render();
  }

  onListClick(ev) {
    const row = ev.target.closest(".sign-row");
    if (!row) return;
    const label = row.dataset.label;
    const btn = ev.target.closest("button");
    if (!btn) { this.select(label); return; }
    ev.stopPropagation();
    if (btn.dataset.act === "rename") this.startRename(row, label);
    if (btn.dataset.act === "delete") this.deleteSign(label);
  }

  startRename(row, label) {
    const nameEl = row.querySelector(".name");
    const input = document.createElement("input");
    input.className = "field";
    input.value = label;
    input.maxLength = 24;
    input.setAttribute("aria-label", `New name for ${label}`);
    nameEl.replaceWith(input);
    input.focus();
    input.select();
    let done = false;
    const finish = async (save) => {
      if (done) return;
      done = true;
      const next = input.value.replace(/\s+/g, " ").trim().toUpperCase();
      if (!save || !next || next === label) { this.render(); return; }
      if (this.pending.includes(label)) {
        this.pending = this.pending.map((l) => (l === label ? next : l));
        savePending(this.pending);
        if (this.selected === label) this.selected = next;
        this.render();
        return;
      }
      try {
        const res = await api.rename(label, next);
        if (this.selected === label) this.selected = next;
        this.setDataset(res.dataset);
        this.hooks.toast(`Renamed ${label} to ${next}. Train again to use the new name.`);
      } catch (e) { this.hooks.toast(e.message); this.render(); }
    };
    input.addEventListener("keydown", (ev) => {
      if (ev.key === "Enter") finish(true);
      if (ev.key === "Escape") finish(false);
      ev.stopPropagation();
    });
    input.addEventListener("blur", () => finish(true));
  }

  async deleteSign(label) {
    const sign = this.signs.find((s) => s.label === label);
    if (!sign) {
      this.pending = this.pending.filter((l) => l !== label);
      savePending(this.pending);
      if (this.selected === label) this.selected = null;
      this.render();
      return;
    }
    if (this.confirmDelete !== label) {
      this.confirmDelete = label;
      this.render();
      clearTimeout(this._confirmTimer);
      this._confirmTimer = setTimeout(() => { this.confirmDelete = null; this.render(); }, 3500);
      return;
    }
    try {
      const res = await api.deleteLabel(label);
      if (this.selected === label) this.selected = null;
      this.confirmDelete = null;
      this.setDataset(res.dataset);
      this.hooks.toast(`Deleted ${label} and its ${res.removed} samples.`);
    } catch (e) { this.hooks.toast(e.message); }
  }

  render() {
    const e = this.els;
    const rows = this.allSigns();
    if (!rows.length) {
      e.list.innerHTML = '<li class="empty-list">No signs yet. Type a name above, or quick-add a set. Then pick a sign and press Record.</li>';
    } else {
      e.list.innerHTML = rows.map((s) => {
        const sel = s.label === this.selected;
        const word = [...s.label].length > 1;
        const pct = Math.min(100, (s.samples / TARGET) * 100);
        const confirming = this.confirmDelete === s.label;
        const stats = s.samples
          ? `${s.samples} samples, ${s.sessions} recording${s.sessions === 1 ? "" : "s"}`
          : "Not recorded yet";
        return `<li class="sign-row" role="option" tabindex="${sel || (!this.selected && s === rows[0]) ? 0 : -1}" aria-selected="${sel}" data-label="${esc(s.label)}">
          <div class="name${word ? " word" : ""}">${esc(s.label)}</div>
          <div><div class="stats">${stats}</div><div class="bar"><i class="${s.samples < MIN_SAMPLES ? "low" : ""}" style="width:${pct}%"></i></div></div>
          <div class="row-actions">
            <button class="icon-btn" data-act="rename" aria-label="Rename ${esc(s.label)}" title="Rename"><svg><use href="#i-pen"/></svg></button>
            <button class="icon-btn ${confirming ? "danger-btn" : ""}" data-act="delete" aria-label="${confirming ? "Click again to delete" : "Delete"} ${esc(s.label)}" title="${confirming ? "Click again to delete" : "Delete"}"><svg><use href="#i-trash"/></svg></button>
          </div>
        </li>`;
      }).join("");
    }

    const ready = this.signs.filter((s) => s.samples >= MIN_SAMPLES);
    const n = this.signs.length;
    e.datasetLine.textContent = n
      ? `${n} sign${n === 1 ? "" : "s"} recorded, ${this.total.toLocaleString()} samples in total.`
      : "Add a sign, then record yourself making it.";

    const training = this.polling !== null;
    if (!training && this.outcome) {
      e.trainNote.className = this.outcome.ok ? "note ok" : "note error";
      e.trainNote.textContent = this.outcome.message;
    } else if (!training) {
      e.trainNote.className = "note";
      if (ready.length < 2) {
        e.trainNote.textContent = `To train, record at least two signs with ${MIN_SAMPLES} or more samples each.`;
      } else {
        const short = this.signs.filter((s) => s.samples < MIN_SAMPLES).map((s) => s.label);
        e.trainNote.textContent = short.length
          ? `Record more for ${short.join(", ")} first (at least ${MIN_SAMPLES} samples each).`
          : `Ready to train on ${n} signs.`;
      }
    }
    const canTrain = ready.length >= 2 && ready.length === n;
    e.train.disabled = training || !canTrain || !!this.rec;

    const sel = this.selected;
    e.record.disabled = !sel || !!this.rec || !this.hooks.cameraReady();
    e.recordText.textContent = sel ? `Record ${sel}` : "Record";
    if (!this.rec) {
      e.recHint.innerHTML = sel
        ? `Hold <b>${esc(sel)}</b> and move your hand a little while recording: closer, farther, slightly tilted. Press <b>R</b> to start.`
        : "Pick a sign from your list, then press <b>R</b> to record.";
    }
  }

  // ---------- recording ----------
  startRecording() {
    if (!this.selected || this.rec || !this.hooks.cameraReady()) return;
    const label = this.selected;
    this.rec = { label, session: newSessionId(), frames: [], phase: "count", countFrom: performance.now(), lastCap: 0, lastHand: 0 };
    this.els.hud.hidden = false;
    this.els.capture.hidden = true;
    this.els.countBig.hidden = false;
    this.els.dock.hidden = true;
    this.render();
  }

  cancelRecording() {
    if (!this.rec) return;
    this.rec = null;
    this.els.hud.hidden = true;
    this.els.dock.hidden = false;
    this.hooks.toast("Recording cancelled.");
    this.render();
  }

  // Called every animation frame while the Train view is open.
  frame(hands, now) {
    const r = this.rec;
    if (!r) return;
    const e = this.els;
    if (r.phase === "count") {
      const left = 3 - Math.floor((now - r.countFrom) / 1000);
      if (left <= 0) {
        r.phase = "capture";
        r.started = now;
        e.countBig.hidden = true;
        e.capture.hidden = false;
        e.captureLabel.textContent = r.label;
        e.captureLabel.className = "label" + ([...r.label].length > 1 ? " word" : "");
      } else if (e.countBig.textContent !== String(left)) {
        e.countBig.textContent = String(left);
      }
      return;
    }
    if (r.phase !== "capture") return;
    const target = this.settings.samplesPerRecording;
    if (hands.length) {
      r.lastHand = now;
      if (now - r.lastCap >= CAPTURE_EVERY_MS) {
        r.lastCap = now;
        r.frames.push({ hands });
      }
    }
    const n = r.frames.length;
    e.captureFill.style.strokeDashoffset = String(CAPTURE_RING * (1 - n / target));
    e.captureN.textContent = `${n} of ${target}`;
    e.captureWarn.textContent = hands.length ? "" : "Show your hand";
    const timedOut = now - r.started > 30000;
    if (n >= target || timedOut) this.finishRecording(timedOut);
  }

  async finishRecording(timedOut) {
    const r = this.rec;
    r.phase = "saving";
    this.els.captureN.textContent = "Saving…";
    try {
      if (r.frames.length < 10) throw new Error("Your hand wasn't visible long enough. Nothing was saved, try again.");
      const res = await api.addSamples(r.label, r.session, r.frames);
      this.setDataset(res.dataset);
      const msg = `Saved ${res.added} samples for ${r.label}${timedOut ? " (stopped after 30 seconds)" : ""}.`;
      const session = r.session;
      this.hooks.toast(msg, {
        label: "Undo",
        run: async () => {
          try {
            const out = await api.deleteSession(session);
            this.setDataset(out.dataset);
            this.hooks.toast("Recording removed.");
          } catch (e) { this.hooks.toast(e.message); }
        },
      });
    } catch (e) {
      this.hooks.toast(e.message);
    } finally {
      this.rec = null;
      this.els.hud.hidden = true;
      this.els.dock.hidden = false;
      this.render();
    }
  }

  // ---------- training ----------
  async train() {
    try {
      await api.train();
    } catch (err) {
      if (err.status !== 409) { this.hooks.toast(err.message); return; }
    }
    this.watchTraining();
  }

  watchTraining() {
    const e = this.els;
    if (this.polling) return;
    e.progress.hidden = false;
    this.outcome = null;
    e.trainNote.className = "note";
    this.polling = setInterval(async () => {
      let s;
      try { s = await api.trainStatus(); } catch { return; }
      e.progress.querySelector("i").style.width = `${Math.round(s.progress * 100)}%`;
      if (s.state === "running") {
        e.trainNote.textContent = `${s.stage}…`;
        return;
      }
      clearInterval(this.polling);
      this.polling = null;
      e.progress.hidden = true;
      if (s.state === "done" || s.state === "error") this.outcome = { ok: s.state === "done", message: s.message };
      this.render();
      if (s.state === "done") this.hooks.onTrained(s.message);
    }, 400);
    this.render();
  }
}

function loadPending() {
  try { return JSON.parse(localStorage.getItem(PENDING_KEY) || "[]").filter((x) => typeof x === "string"); }
  catch { return []; }
}
function savePending(list) {
  try { localStorage.setItem(PENDING_KEY, JSON.stringify(list)); } catch { /* storage unavailable */ }
}
