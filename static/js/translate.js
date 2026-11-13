// Live translation: asks the server what sign it sees, smooths the answers,
// and adds a sign to the caption once it has been held steady.

import { api } from "./api.js";

const RING = 477.5;               // circumference of the bubble ring
const SEND_EVERY_MS = 60;         // ~16 predictions per second
const WINDOW_MS = 650;            // smoothing window
const CONTROL = { SPACE: "space", DELETE: "delete", DEL: "delete", BACKSPACE: "delete" };

export class Translator {
  constructor(els, settings, hooks) {
    this.els = els;
    this.settings = settings;
    this.hooks = hooks;           // { source(), onPrediction(res), toast(msg) }
    this.text = "";
    this.fresh = "";              // last added chunk, highlighted briefly
    this.preds = [];
    this.inFlight = false;
    this.lastSent = 0;
    this.holdLabel = null;
    this.holdStart = 0;
    this.repeat = false;
    this.handSeenAt = 0;
    this.absentSince = performance.now();
    this.bubble = { x: -999, y: -999, on: false };
    this.current = null;
    this.latency = 0;
    this.render();
  }

  // Called every animation frame while the Translate view is open.
  frame(result, hands, now, overlay) {
    const present = hands.length > 0;
    if (present) {
      this.handSeenAt = now;
      this.absentSince = 0;
      if (!this.inFlight && now - this.lastSent >= SEND_EVERY_MS) this.send(hands, now);
    } else if (!this.absentSince) {
      this.absentSince = now;
    }

    this.preds = this.preds.filter((p) => now - p.t <= WINDOW_MS);
    const cur = present ? this.smoothed() : null;
    this.current = cur;

    // hold-to-add
    const label = cur && cur.sure ? cur.label : null;
    if (label !== this.holdLabel) {
      this.holdLabel = label;
      this.holdStart = now;
      this.repeat = false;
    }
    let progress = 0;
    if (label) {
      const need = this.settings.holdMs * (this.repeat ? 1.6 : 1);
      progress = Math.min(1, (now - this.holdStart) / need);
      if (progress >= 1) {
        this.commit(label);
        this.holdStart = now;
        this.repeat = true;
        progress = 0;
      }
    }

    // auto space after the hand drops
    if (!present && this.absentSince && this.settings.autoSpace && now - this.absentSince > 1100) {
      if (this.text && !this.text.endsWith(" ")) {
        const word = this.text.split(" ").pop();
        this.text += " ";
        this.render();
        if (this.settings.speak && word) this.say(word);
      }
    }

    this.drawBubble(result, overlay, cur, progress, present, now);
    const showHint = !present && (!this.absentSince || now - this.absentSince > 1500);
    this.els.hint.style.opacity = showHint ? "1" : "0";
  }

  async send(hands, now) {
    this.inFlight = true;
    this.lastSent = now;
    const t0 = performance.now();
    try {
      const res = await api.predict(hands, this.hooks.source(), 0);
      this.latency = performance.now() - t0;
      this.preds.push({ t: performance.now(), label: res.label, conf: res.confidence, top: res.top || [] });
      this.hooks.onPrediction(res);
    } catch (e) {
      if (!this._warned) { this.hooks.toast(e.message); this._warned = true; }
    } finally {
      this.inFlight = false;
    }
  }

  // Majority vote over recent predictions.
  smoothed() {
    if (!this.preds.length) return null;
    const min = this.settings.minConf / 100;
    const votes = new Map();
    for (const p of this.preds) {
      if (!p.label || p.conf < min) continue;
      const v = votes.get(p.label) || { n: 0, conf: 0 };
      v.n++; v.conf += p.conf;
      votes.set(p.label, v);
    }
    let best = null;
    for (const [label, v] of votes) if (!best || v.n > best.n) best = { label, ...v };
    const need = Math.max(2, Math.ceil(this.preds.length * 0.6));
    const last = this.preds[this.preds.length - 1];
    if (best && best.n >= need) {
      return { label: best.label, conf: best.conf / best.n, sure: true, top: last.top };
    }
    return { label: null, conf: last.conf, sure: false, top: last.top };
  }

  drawBubble(result, overlay, cur, progress, present, now) {
    const { bubble, glyph, meta, ringFill } = this.els;
    const visible = present || now - this.handSeenAt < 250;
    if (visible && present) {
      const box = overlay.handBox(result, 0);
      if (box) {
        const W = window.innerWidth, H = window.innerHeight;
        const bw = 176;
        let x = box.x1 + 28;
        if (x + bw > W - 12) x = box.x0 - bw - 28;
        x = Math.max(12, Math.min(W - bw - 12, x));
        let y = Math.max(76, Math.min(H - 380, box.y0 - 10));
        const b = this.bubble;
        if (!b.on || b.x < -500) { b.x = x; b.y = y; }
        const k = window.matchMedia("(prefers-reduced-motion: reduce)").matches ? 1 : 0.22;
        b.x += (x - b.x) * k;
        b.y += (y - b.y) * k;
        bubble.style.transform = `translate3d(${b.x.toFixed(1)}px, ${b.y.toFixed(1)}px, 0)`;
      }
    }
    this.bubble.on = visible;
    bubble.classList.toggle("on", visible);
    bubble.classList.toggle("repeat", this.repeat);
    if (!visible) return;

    let text = "", cls = "glyph", metaHtml = "";
    if (cur && cur.sure) {
      text = cur.label;
      if ([...text].length > 1) cls += " word";
      metaHtml = `<b>${Math.round(cur.conf * 100)}%</b> sure`;
      if (this.repeat) metaHtml += "<br>Keep holding to add again";
    } else if (cur) {
      text = "?";
      cls += " unsure";
      const guesses = (cur.top || []).filter((t) => t.confidence > 0.15).slice(0, 2).map((t) => esc(t.label));
      metaHtml = guesses.length ? `Maybe ${guesses.join(" or ")}` : "Not sure yet";
    } else {
      text = "…";
      cls += " unsure";
    }
    if (glyph.textContent !== text) glyph.textContent = text;
    if (glyph.className !== cls) glyph.className = cls;
    if (meta.innerHTML !== metaHtml) meta.innerHTML = metaHtml;
    ringFill.style.strokeDashoffset = String(RING * (1 - progress));
  }

  commit(label) {
    const action = CONTROL[label];
    if (action === "space") { this.space(); this.flash(); return; }
    if (action === "delete") { this.backspace(); this.flash(); return; }

    const isLetter = [...label].length === 1;
    let chunk;
    if (isLetter) {
      chunk = label;
    } else {
      chunk = (this.text && !this.text.endsWith(" ") ? " " : "") + label + " ";
      if (this.settings.speak) this.say(label);
    }
    this.fly(label);
    this.text += chunk;
    this.fresh = chunk;
    this.render();
    this.flash();
    clearTimeout(this._freshTimer);
    this._freshTimer = setTimeout(() => { this.fresh = ""; this.render(); }, 900);
  }

  space() {
    if (this.text && !this.text.endsWith(" ")) {
      const word = this.text.split(" ").pop();
      this.text += " ";
      if (this.settings.speak && word) this.say(word);
    }
    this.fresh = "";
    this.render();
  }

  backspace() {
    this.text = this.text.slice(0, -1);
    this.fresh = "";
    this.render();
  }

  clear() {
    this.text = "";
    this.fresh = "";
    this.render();
  }

  speakAll() {
    const t = this.text.trim();
    if (!t) { this.hooks.toast("Nothing to speak yet."); return; }
    this.say(t);
  }

  say(text) {
    if (!("speechSynthesis" in window)) { this.hooks.toast("This browser can't speak text."); return; }
    // lower case, so all-caps words are read as words, not spelled out
    const u = new SpeechSynthesisUtterance(text.toLowerCase());
    u.rate = 0.95;
    speechSynthesis.speak(u);
  }

  async copy() {
    const t = this.text.trim();
    if (!t) { this.hooks.toast("Nothing to copy yet."); return; }
    try {
      await navigator.clipboard.writeText(t);
      this.hooks.toast("Copied to clipboard.");
    } catch {
      this.hooks.toast("Copy failed. Select the text and press Ctrl+C.");
    }
  }

  render() {
    const box = this.els.caption.querySelector(".text");
    if (!this.text) {
      box.innerHTML = '<span class="placeholder">Your signs will appear here</span><span class="caret"></span>';
      return;
    }
    const keep = this.fresh && this.text.endsWith(this.fresh) ? this.fresh : "";
    const head = keep ? this.text.slice(0, -keep.length) : this.text;
    box.innerHTML = esc(head) + (keep ? `<span class="fresh">${esc(keep)}</span>` : "") + '<span class="caret"></span>';
  }

  flash() {
    const b = this.els.bubble;
    b.classList.remove("flash");
    void b.offsetWidth;
    b.classList.add("flash");
  }

  // The added sign flies from the bubble into the caption.
  fly(label) {
    if (window.matchMedia("(prefers-reduced-motion: reduce)").matches) return;
    const from = this.els.glyph.getBoundingClientRect();
    const caret = this.els.caption.querySelector(".caret");
    if (!caret || !from.width) return;
    const to = caret.getBoundingClientRect();
    const el = document.createElement("div");
    el.className = "flyer";
    el.textContent = label;
    const fromSize = parseFloat(getComputedStyle(this.els.glyph).fontSize);
    const toSize = parseFloat(getComputedStyle(this.els.caption).fontSize);
    el.style.fontSize = `${fromSize}px`;
    el.style.transformOrigin = "top left";
    document.body.appendChild(el);
    const r = el.getBoundingClientRect();
    const sx = from.left + from.width / 2 - r.width / 2;
    const sy = from.top + from.height / 2 - r.height / 2;
    const scale = toSize / fromSize;
    el.animate(
      [
        { transform: `translate(${sx}px, ${sy}px) scale(1)`, opacity: 1 },
        { transform: `translate(${to.left}px, ${to.top - (r.height * scale) * 0.1}px) scale(${scale})`, opacity: 0.2 },
      ],
      { duration: 420, easing: "cubic-bezier(.3,.7,.2,1)", fill: "forwards" }
    ).onfinish = () => el.remove();
  }
}

export function esc(s) {
  return String(s).replace(/[&<>"']/g, (c) => ({ "&": "&amp;", "<": "&lt;", ">": "&gt;", '"': "&quot;", "'": "&#39;" }[c]));
}
