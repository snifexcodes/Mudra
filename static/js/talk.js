// Talk view: a phrase board and a two-way conversation.
// The signer taps a phrase, types, or signs (from Translate) and Mudra speaks
// it aloud. The other person's speech is turned into large text.

import { esc } from "./translate.js";
import { LISTEN_LANGUAGES, Listener, canListen } from "./speech.js";

const PHRASE_KEY = "mudra.phrases";
const DEFAULT_PHRASES = {
  Basics: ["Hello", "Yes", "No", "Please", "Thank you", "Sorry", "I don't understand",
    "Please say that again", "Please write it down", "I use sign language", "Please speak slowly", "Goodbye"],
  Needs: ["I need water", "I'm hungry", "I need the toilet", "I need help", "Please wait",
    "I'm okay", "Call my family", "How much is it?", "Where is it?", "I need a doctor"],
  Feelings: ["I'm happy", "I'm tired", "I'm in pain", "I feel sick", "I'm scared", "I'm cold", "I'm hot", "I'm fine"],
  Emergency: ["I can't speak. Please help me.", "Call an ambulance", "Call the police",
    "I can hear you, but I can't speak", "I am deaf. Please write it down", "I'm lost"],
};
const EMERGENCY = "Emergency";

export class Talk {
  constructor(els, settings, hooks) {
    this.els = els;
    this.settings = settings;
    this.hooks = hooks;   // { speak(text), toast(msg), spokenLang() }
    this.phrases = loadPhrases();
    this.category = Object.keys(this.phrases)[0];
    this.editing = false;
    this.log = [];        // {who: "you"|"them", text, via}
    this.interim = "";
    this.listener = new Listener({
      onText: (text, final) => this.heard(text, final),
      onState: (on) => this.setListening(on),
      onError: (msg) => { this.hooks.toast(msg); this.setListening(false); },
    });
    this.bind();
    this.renderBoard();
    this.renderLog();
  }

  bind() {
    const e = this.els;
    e.cats.addEventListener("click", (ev) => {
      const b = ev.target.closest("button[data-cat]");
      if (b) { this.category = b.dataset.cat; this.renderBoard(); }
    });
    e.board.addEventListener("click", (ev) => {
      const del = ev.target.closest("button[data-del]");
      if (del) { this.removePhrase(+del.dataset.del); return; }
      if (ev.target.closest("[data-add]")) { this.addPhrasePrompt(); return; }
      const tile = ev.target.closest("button[data-say]");
      if (tile) this.say(tile.dataset.say, "phrase");
    });
    e.edit.addEventListener("click", () => {
      this.editing = !this.editing;
      e.edit.textContent = this.editing ? "Done" : "Edit phrases";
      e.edit.setAttribute("aria-pressed", String(this.editing));
      this.renderBoard();
    });
    e.typeForm.addEventListener("submit", (ev) => {
      ev.preventDefault();
      const t = e.typeInput.value.trim();
      if (!t) return;
      this.say(t, "typed");
      e.typeInput.value = "";
    });
    e.listen.addEventListener("click", () => {
      if (this.listening) this.listener.stop();
      else this.listener.start(e.lang.value);
    });
    e.lang.innerHTML = LISTEN_LANGUAGES.map(([code, name]) => `<option value="${code}">${esc(name)}</option>`).join("");
    e.lang.addEventListener("change", () => {
      this.settings.listenLang = e.lang.value;
      this.hooks.save();
      if (this.listening) { this.listener.stop(); setTimeout(() => this.listener.start(e.lang.value), 300); }
    });
    e.clear.addEventListener("click", () => { this.log = []; this.interim = ""; this.renderLog(); });
    e.show.addEventListener("click", () => this.showBig());
    e.big.addEventListener("click", () => { e.big.hidden = true; });
    if (!canListen()) {
      e.listen.disabled = true;
      e.listenNote.textContent = "Listening works in Chrome and Edge. This browser can't turn speech into text.";
    }
  }

  // Pick the listening language for a pack (unless the user chose one).
  setDefaultLanguage(spoken) {
    const want = this.settings.listenLang || spoken || "en-IN";
    const has = LISTEN_LANGUAGES.some(([c]) => c === want);
    if (!has) this.els.lang.insertAdjacentHTML("beforeend", `<option value="${esc(want)}">${esc(want)}</option>`);
    this.els.lang.value = want;
  }

  say(text, via) {
    if (!this.hooks.speak(text)) this.hooks.toast("This browser can't speak text. The message is still shown below.");
    this.add("you", text, via);
  }

  add(who, text, via = "") {
    this.log.push({ who, text, via });
    if (this.log.length > 200) this.log.shift();
    this.renderLog();
  }

  heard(text, final) {
    if (final) {
      this.interim = "";
      if (text.trim()) this.add("them", text.trim(), "heard");
    } else {
      this.interim = text;
      this.renderLog();
    }
  }

  setListening(on) {
    this.listening = on;
    const b = this.els.listen;
    b.classList.toggle("on", on);
    b.querySelector("span").textContent = on ? "Stop listening" : "Listen to them";
    if (!on) { this.interim = ""; this.renderLog(); }
  }

  stop() { this.listener.stop(); }

  renderBoard() {
    const e = this.els;
    e.cats.innerHTML = Object.keys(this.phrases).map((c) =>
      `<button type="button" role="tab" data-cat="${esc(c)}" aria-selected="${c === this.category}" class="${c === EMERGENCY ? "sos" : ""}">${esc(c)}</button>`
    ).join("");
    const list = this.phrases[this.category] || [];
    const sos = this.category === EMERGENCY;
    e.board.innerHTML = list.map((p, i) => `
      <div class="tile-wrap">
        <button type="button" class="phrase${sos ? " sos" : ""}" data-say="${esc(p)}">${esc(p)}</button>
        ${this.editing ? `<button type="button" class="del" data-del="${i}" aria-label="Remove ${esc(p)}"><svg><use href="#i-x"/></svg></button>` : ""}
      </div>`).join("") + `<button type="button" class="phrase add" data-add>+ Add phrase</button>`;
  }

  addPhrasePrompt() {
    const e = this.els;
    const add = e.board.querySelector("[data-add]");
    const form = document.createElement("form");
    form.className = "add-phrase";
    form.innerHTML = `<input class="field" maxlength="120" aria-label="New phrase" placeholder="Type a phrase, in any language"><button class="primary" type="submit">Add</button>`;
    add.replaceWith(form);
    const input = form.querySelector("input");
    input.focus();
    form.addEventListener("submit", (ev) => {
      ev.preventDefault();
      const t = input.value.trim();
      if (t) {
        this.phrases[this.category].push(t);
        savePhrases(this.phrases);
      }
      this.renderBoard();
    });
    input.addEventListener("keydown", (ev) => { if (ev.key === "Escape") this.renderBoard(); ev.stopPropagation(); });
  }

  removePhrase(i) {
    this.phrases[this.category].splice(i, 1);
    savePhrases(this.phrases);
    this.renderBoard();
  }

  renderLog() {
    const e = this.els;
    if (!this.log.length && !this.interim) {
      e.log.innerHTML = `<p class="log-empty">Tap a phrase, type below, or sign in Translate and press Speak. Press <b>Listen to them</b> so the other person's words appear here.</p>`;
      return;
    }
    e.log.innerHTML = this.log.map((m) => `
      <div class="msg ${m.who}"><span class="who">${m.who === "you" ? "You" : "Them"}${m.via === "signed" ? ", signed" : ""}</span>${esc(m.text)}</div>`).join("")
      + (this.interim ? `<div class="msg them interim"><span class="who">Them, listening…</span>${esc(this.interim)}</div>` : "");
    e.log.scrollTop = e.log.scrollHeight;
  }

  showBig() {
    const last = [...this.log].reverse().find((m) => m.who === "you");
    const typed = this.els.typeInput.value.trim();
    const text = typed || (last && last.text);
    if (!text) { this.hooks.toast("Type or choose something to show first."); return; }
    this.els.bigText.textContent = text;
    this.els.big.hidden = false;
  }
}

function loadPhrases() {
  try {
    const saved = JSON.parse(localStorage.getItem(PHRASE_KEY) || "null");
    if (saved && typeof saved === "object" && Object.keys(saved).length) return saved;
  } catch { /* storage unavailable */ }
  return JSON.parse(JSON.stringify(DEFAULT_PHRASES));
}

function savePhrases(p) {
  try { localStorage.setItem(PHRASE_KEY, JSON.stringify(p)); } catch { /* storage unavailable */ }
}
