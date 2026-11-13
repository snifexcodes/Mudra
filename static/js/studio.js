// Train view: manage a pack's signs, record them from the webcam, train.
// Still signs: ~7 seconds of frames while you hold the hand shape.
// Moving signs: a series of repetitions, one short clip each.

import { api, newSessionId, isNothing } from "./api.js";
import { esc } from "./translate.js";

const TARGET = { still: 300, moving: 20 };   // fills the row's bar
const MIN = { still: 20, moving: 5 };        // must match core/model.py
const CAPTURE_EVERY_MS = 66;                 // ~15 frames per second
const CAPTURE_RING = 628.3;
const REP_MS = 2200;                         // one repetition of a moving sign
const REST_MS = 900;

const ALPHA = "ABCDEFGHIJKLMNOPQRSTUVWXYZ".split("");
const DIGITS = "0123456789".split("");
const PRESETS = {
  asl: [
    { key: "alpha", label: "Alphabet A–Y (still)", signs: ALPHA.filter((c) => c !== "J" && c !== "Z"), kind: "still" },
    { key: "jz", label: "J and Z (moving)", signs: ["J", "Z"], kind: "moving" },
    { key: "starter", label: "Starter set", starter: true, kind: "still" },
    { key: "words", label: "Common words (moving)", kind: "moving",
      signs: ["HELLO", "THANK YOU", "PLEASE", "SORRY", "YES", "NO", "HELP", "WATER", "EAT", "MORE"] },
  ],
  isl: [
    { key: "alpha", label: "Alphabet A–Z (two hands, still)", signs: ALPHA, kind: "still" },
    { key: "digits", label: "Numbers 0–9 (still)", signs: DIGITS, kind: "still" },
    { key: "words", label: "Common words (moving)", kind: "moving",
      signs: ["NAMASTE", "THANK YOU", "SORRY", "PLEASE", "YES", "NO", "HELP", "WATER", "FOOD", "GOOD"] },
  ],
  bsl: [
    { key: "alpha", label: "Alphabet A–Z (two hands, still)", signs: ALPHA, kind: "still" },
    { key: "digits", label: "Numbers 0–9 (still)", signs: DIGITS, kind: "still" },
    { key: "words", label: "Common words (moving)", kind: "moving",
      signs: ["HELLO", "THANK YOU", "PLEASE", "SORRY", "YES", "NO", "HELP", "WATER", "TOILET", "GOOD"] },
  ],
  other: [
    { key: "alpha", label: "Alphabet A–Z (still)", signs: ALPHA, kind: "still" },
    { key: "digits", label: "Numbers 0–9 (still)", signs: DIGITS, kind: "still" },
    { key: "words", label: "Common words (moving)", kind: "moving",
      signs: ["HELLO", "THANK YOU", "PLEASE", "SORRY", "YES", "NO", "HELP", "WATER"] },
  ],
};
const NOTHING = { key: "nothing", label: "Nothing (rest and random moves)", pairs: [["NO SIGN", "still"], ["NO SIGN MOVING", "moving"]] };

export class Studio {
  constructor(els, settings, hooks) {
    this.els = els;
    this.settings = settings;
    this.hooks = hooks;  // { pack(), packInfo(), toast(msg, action), onTrained(msg), onPackChanged(info), cameraReady() }
    this.signs = [];     // [{label, kind, samples, sessions}]
    this.selected = null;
    this.rec = null;
    this.polling = null;
    this.confirmDelete = null;
    this.newKind = "still";
    this.outcome = null;
    this.bind();
  }

  bind() {
    const e = this.els;
    e.addForm.addEventListener("submit", (ev) => {
      ev.preventDefault();
      this.addSign(e.newSign.value, this.newKind).then((ok) => { if (ok) e.newSign.value = ""; });
    });
    e.kindButtons.forEach((b) => b.addEventListener("click", () => {
      this.newKind = b.dataset.kind;
      e.kindButtons.forEach((x) => x.setAttribute("aria-pressed", String(x === b)));
    }));
    e.presets.addEventListener("click", (ev) => {
      const b = ev.target.closest("button[data-preset]");
      if (b) this.addPreset(b.dataset.preset);
    });
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

  presetList() {
    const code = this.hooks.pack();
    return [...(PRESETS[code] || PRESETS.other), NOTHING];
  }

  // Called when the pack changes or its data changes.
  setPack(info, { keepSelection = false } = {}) {
    const prevTotal = this.total;
    const ds = info.dataset || { signs: [], total: 0 };
    if (prevTotal !== undefined && ds.total !== prevTotal) this.outcome = null;
    this.total = ds.total;
    const rec = new Map(ds.signs.map((s) => [s.label, s]));
    const kinds = info.sign_kinds || {};
    const labels = new Set([...rec.keys(), ...Object.keys(kinds)]);
    this.signs = [...labels].map((l) => rec.get(l) || { label: l, kind: kinds[l] || "still", samples: 0, sessions: 0 })
      .sort((a, b) => a.label.localeCompare(b.label, undefined, { numeric: true }));
    if (!keepSelection || !labels.has(this.selected)) this.selected = null;
    this.els.presets.innerHTML = this.presetList()
      .map((p) => `<button type="button" data-preset="${p.key}">${esc(p.label)}</button>`).join("");
    if (this.els.presetsBox && (!keepSelection || prevTotal === undefined)) this.els.presetsBox.open = this.signs.length === 0;
    this.els.packName.textContent = info.name;
    this.render();
  }

  async refresh() {
    const info = await api.pack(this.hooks.pack());
    this.hooks.onPackChanged(info);
  }

  signKind(label) {
    const s = this.signs.find((x) => x.label === label);
    return s ? s.kind : null;
  }

  async addSign(raw, kind, quiet = false) {
    const name = String(raw || "").replace(/\s+/g, " ").trim().toUpperCase();
    if (!name) { if (!quiet) this.hooks.toast("Type a name for the sign first."); return false; }
    try {
      const res = await api.addSign(this.hooks.pack(), name, kind);
      if (!quiet) {
        this.hooks.onPackChanged(res.pack);
        this.select(res.label);
      }
      return res;
    } catch (e) {
      this.hooks.toast(e.message);
      return false;
    }
  }

  async addPreset(key) {
    const p = this.presetList().find((x) => x.key === key);
    if (!p) return;
    let pairs;
    if (p.pairs) pairs = p.pairs;
    else if (p.starter) pairs = (this.hooks.packInfo().starter || []).map((s) => [s.label, "still"]);
    else pairs = p.signs.map((s) => [s, p.kind]);
    let last = null, skipped = 0;
    for (const [label, kind] of pairs) {
      const existing = this.signKind(label);
      if (existing && existing !== kind) { skipped++; continue; }
      const res = await this.addSign(label, kind, true);
      if (res) last = res;
    }
    if (last) this.hooks.onPackChanged(last.pack);
    if (!this.selected && pairs.length) this.select(pairs[0][0]);
    if (key === "alpha" && this.hooks.pack() === "asl") {
      this.hooks.toast("Added A–Y as still signs. J and Z move: add them with the J and Z preset.");
    } else if (key === "nothing") {
      this.hooks.toast("Record yourself resting and making random moves as NO SIGN. Mudra will learn to ignore them.");
    } else if (skipped) {
      this.hooks.toast(`${skipped} sign${skipped === 1 ? " was" : "s were"} skipped: they already exist with the other type.`);
    }
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
      try {
        const res = await api.rename(this.hooks.pack(), label, next);
        if (this.selected === label) this.selected = next;
        this.hooks.onPackChanged(res.pack);
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
    if (sign && sign.samples && this.confirmDelete !== label) {
      this.confirmDelete = label;
      this.render();
      clearTimeout(this._confirmTimer);
      this._confirmTimer = setTimeout(() => { this.confirmDelete = null; this.render(); }, 3500);
      return;
    }
    try {
      const res = await api.deleteSign(this.hooks.pack(), label);
      if (this.selected === label) this.selected = null;
      this.confirmDelete = null;
      this.hooks.onPackChanged(res.pack);
      if (res.removed) this.hooks.toast(`Deleted ${label} and its ${res.removed} recordings.`);
    } catch (e) { this.hooks.toast(e.message); }
  }

  render() {
    const e = this.els;
    const rows = this.signs;
    if (!rows.length) {
      e.list.innerHTML = '<li class="empty-list">No signs yet. Type a name above and choose Still or Moving, or quick-add a set. Then pick a sign and press Record.</li>';
    } else {
      e.list.innerHTML = rows.map((s, idx) => {
        const sel = s.label === this.selected;
        const word = [...s.label].length > 1;
        const pct = Math.min(100, (s.samples / TARGET[s.kind]) * 100);
        const confirming = this.confirmDelete === s.label;
        const unit = s.kind === "moving" ? "repetition" : "sample";
        const stats = s.samples
          ? `${s.samples} ${unit}${s.samples === 1 ? "" : "s"}, ${s.sessions} recording${s.sessions === 1 ? "" : "s"}`
          : "Not recorded yet";
        const focusable = sel || (!this.selected && idx === 0);
        return `<li class="sign-row${isNothing(s.label) ? " nothing" : ""}" role="option" tabindex="${focusable ? 0 : -1}" aria-selected="${sel}" data-label="${esc(s.label)}">
          <div class="name${word ? " word" : ""}">${esc(s.label)}</div>
          <div><div class="stats"><span class="kind ${s.kind}">${s.kind === "moving" ? "Moving" : "Still"}</span> ${stats}</div><div class="bar"><i class="${s.samples < MIN[s.kind] ? "low" : ""}" style="width:${pct}%"></i></div></div>
          <div class="row-actions">
            <button class="icon-btn" data-act="rename" aria-label="Rename ${esc(s.label)}" title="Rename"><svg><use href="#i-pen"/></svg></button>
            <button class="icon-btn ${confirming ? "danger-btn" : ""}" data-act="delete" aria-label="${confirming ? "Click again to delete" : "Delete"} ${esc(s.label)}" title="${confirming ? "Click again to delete" : "Delete"}"><svg><use href="#i-trash"/></svg></button>
          </div>
        </li>`;
      }).join("");
    }

    const recorded = rows.filter((s) => s.samples > 0);
    const n = recorded.length;
    e.datasetLine.textContent = n
      ? `${n} sign${n === 1 ? "" : "s"} recorded, ${this.total.toLocaleString()} recordings in total.`
      : "Add a sign, then record yourself making it.";

    const groups = ["still", "moving"].map((k) => recorded.filter((s) => s.kind === k));
    const ready = groups.map((g, i) => g.length >= 2 && g.every((s) => s.samples >= MIN[["still", "moving"][i]]));
    const training = this.polling !== null;
    if (!training && this.outcome) {
      e.trainNote.className = this.outcome.ok ? "note ok" : "note error";
      e.trainNote.textContent = this.outcome.message;
    } else if (!training) {
      e.trainNote.className = "note";
      const short = recorded.filter((s) => s.samples < MIN[s.kind]).map((s) => s.label);
      if (!ready[0] && !ready[1]) {
        e.trainNote.textContent = short.length
          ? `Record more for ${short.slice(0, 6).join(", ")}${short.length > 6 ? "…" : ""} (still signs need ${MIN.still} samples, moving signs ${MIN.moving} repetitions).`
          : "To train, record at least two still signs or two moving signs.";
      } else {
        const parts = [];
        if (ready[0]) parts.push(`${groups[0].length} still`);
        if (ready[1]) parts.push(`${groups[1].length} moving`);
        e.trainNote.textContent = `Ready to train on ${parts.join(" and ")} signs.` + (short.length ? ` Not ready yet: ${short.slice(0, 4).join(", ")}.` : "");
      }
    }
    e.train.disabled = training || !(ready[0] || ready[1]) || !!this.rec;

    const sel = this.selected;
    const kind = sel ? this.signKind(sel) : null;
    e.record.disabled = !sel || !!this.rec || !this.hooks.cameraReady();
    e.recordText.textContent = sel ? `Record ${sel}` : "Record";
    if (!this.rec) {
      if (!sel) e.recHint.innerHTML = "Pick a sign from your list, then press <b>R</b> to record.";
      else if (isNothing(sel)) e.recHint.innerHTML = kind === "moving"
        ? "Make random movements that aren't signs: scratching, waving to stretch, reaching. Press <b>R</b> to start."
        : "Rest your hands in view, or hold shapes that aren't signs. Press <b>R</b> to start.";
      else if (kind === "moving") e.recHint.innerHTML = `Make <b>${esc(sel)}</b> once each time it says <b>Sign now</b>, ${this.settings.repsPerRecording} times. Rest your hands in between. Press <b>R</b> to start.`;
      else e.recHint.innerHTML = `Hold <b>${esc(sel)}</b> and move your hand a little while recording: closer, farther, slightly tilted. Press <b>R</b> to start.`;
    }
  }

  // ---------- recording ----------
  startRecording() {
    if (!this.selected || this.rec || !this.hooks.cameraReady()) return;
    const label = this.selected;
    const kind = this.signKind(label) || "still";
    this.rec = {
      label, kind, session: newSessionId(), frames: [], clips: [], rep: null,
      phase: "count", countFrom: performance.now(), lastCap: 0,
    };
    this.els.hud.hidden = false;
    this.els.capture.hidden = true;
    this.els.countBig.hidden = false;
    this.els.dock.hidden = true;
    this.render();
  }

  cancelRecording() {
    const r = this.rec;
    if (!r) return;
    if (r.kind === "moving" && r.clips.length >= 1) {   // keep finished repetitions
      this.finish(false);
      return;
    }
    this.rec = null;
    this.els.hud.hidden = true;
    this.els.dock.hidden = false;
    this.hooks.toast("Recording cancelled.");
    this.render();
  }

  // Called every animation frame while the Train view is open.
  frame(hands, pose, now) {
    const r = this.rec;
    if (!r) return;
    const e = this.els;
    if (r.phase === "count") {
      const left = 3 - Math.floor((now - r.countFrom) / 1000);
      if (left <= 0) {
        e.countBig.hidden = true;
        e.capture.hidden = false;
        e.captureLabel.textContent = r.label;
        e.captureLabel.className = "label" + ([...r.label].length > 1 ? " word" : "");
        if (r.kind === "moving") this.beginRep(now);
        else { r.phase = "capture"; r.started = now; }
      } else if (e.countBig.textContent !== String(left)) {
        e.countBig.textContent = String(left);
      }
      return;
    }
    if (r.kind === "moving") { this.frameMoving(hands, pose, now); return; }
    if (r.phase !== "capture") return;
    const target = this.settings.samplesPerRecording;
    if (hands.length && now - r.lastCap >= CAPTURE_EVERY_MS) {
      r.lastCap = now;
      r.frames.push({ hands });
    }
    const n = r.frames.length;
    e.captureFill.style.strokeDashoffset = String(CAPTURE_RING * (1 - n / target));
    e.captureN.textContent = `${n} of ${target}`;
    e.captureWarn.textContent = hands.length ? "" : "Show your hand";
    const timedOut = now - r.started > 30000;
    if (n >= target || timedOut) this.finish(timedOut);
  }

  beginRep(now) {
    const r = this.rec;
    r.phase = "rep";
    r.rep = { start: now, frames: [] };
    this.els.capture.classList.add("go");
    this.els.captureWarn.textContent = "";
  }

  frameMoving(hands, pose, now) {
    const r = this.rec, e = this.els;
    const total = this.settings.repsPerRecording;
    if (r.phase === "rep") {
      if (hands.length && now - r.lastCap >= CAPTURE_EVERY_MS) {
        r.lastCap = now;
        r.rep.frames.push({ t: +((now - r.rep.start) / 1000).toFixed(3), hands, pose });
      }
      const frac = Math.min(1, (now - r.rep.start) / REP_MS);
      e.captureFill.style.strokeDashoffset = String(CAPTURE_RING * (1 - frac));
      e.captureN.textContent = `Sign now (${r.clips.length + 1} of ${total})`;
      e.captureWarn.textContent = hands.length ? "" : "Show your hand";
      if (frac >= 1) {
        if (r.rep.frames.length >= 5) r.clips.push(r.rep.frames);
        r.phase = "rest";
        r.restFrom = now;
        e.capture.classList.remove("go");
        if (r.clips.length >= total) this.finish(false);
      }
    } else if (r.phase === "rest") {
      const frac = Math.min(1, (now - r.restFrom) / REST_MS);
      e.captureFill.style.strokeDashoffset = String(CAPTURE_RING * frac);
      e.captureN.textContent = `Rest (${r.clips.length} of ${total} done)`;
      if (frac >= 1) this.beginRep(now);
    }
  }

  async finish(timedOut) {
    const r = this.rec;
    r.phase = "saving";
    this.els.capture.classList.remove("go");
    this.els.captureN.textContent = "Saving…";
    const code = this.hooks.pack();
    try {
      let res;
      if (r.kind === "moving") {
        if (!r.clips.length) throw new Error("Your hand wasn't visible during the repetitions. Nothing was saved, try again.");
        res = await api.addClips(code, r.label, r.session, r.clips);
      } else {
        if (r.frames.length < 10) throw new Error("Your hand wasn't visible long enough. Nothing was saved, try again.");
        res = await api.addSamples(code, r.label, r.session, r.frames);
      }
      this.hooks.onPackChanged(res.pack);
      const what = r.kind === "moving" ? `${res.added} repetitions` : `${res.added} samples`;
      const session = r.session;
      this.hooks.toast(`Saved ${what} for ${r.label}${timedOut ? " (stopped after 30 seconds)" : ""}.`, {
        label: "Undo",
        run: async () => {
          try {
            const out = await api.deleteSession(code, session);
            this.hooks.onPackChanged(out.pack);
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
      await api.train(this.hooks.pack());
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
      if (s.state === "done") this.hooks.onTrained(s.message, s.pack);
    }, 400);
    this.render();
  }
}
