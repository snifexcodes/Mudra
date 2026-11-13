// Text to speech (speak for the signer) and speech to text (show what the
// other person says), using the browser's built-in speech features.

export const LISTEN_LANGUAGES = [
  ["en-IN", "English (India)"], ["hi-IN", "हिन्दी (Hindi)"], ["mr-IN", "मराठी (Marathi)"],
  ["bn-IN", "বাংলা (Bengali)"], ["ta-IN", "தமிழ் (Tamil)"], ["te-IN", "తెలుగు (Telugu)"],
  ["gu-IN", "ગુજરાતી (Gujarati)"], ["kn-IN", "ಕನ್ನಡ (Kannada)"], ["ml-IN", "മലയാളം (Malayalam)"],
  ["pa-IN", "ਪੰਜਾਬੀ (Punjabi)"], ["ur-IN", "اردو (Urdu)"], ["en-US", "English (US)"],
  ["en-GB", "English (UK)"], ["en-AU", "English (Australia)"], ["es-ES", "Español"],
  ["fr-FR", "Français"], ["de-DE", "Deutsch"], ["it-IT", "Italiano"], ["pt-BR", "Português (Brasil)"],
  ["ru-RU", "Русский"], ["ar-SA", "العربية"], ["zh-CN", "中文 (普通话)"], ["ja-JP", "日本語"],
  ["ko-KR", "한국어"], ["id-ID", "Bahasa Indonesia"], ["tr-TR", "Türkçe"], ["sw-KE", "Kiswahili"],
];

export function voices() {
  return "speechSynthesis" in window ? speechSynthesis.getVoices() : [];
}

export function speak(text, { lang = "en", voiceURI = "", rate = 1 } = {}) {
  if (!("speechSynthesis" in window)) return false;
  const t = String(text || "").trim();
  if (!t) return false;
  // Lower-case Latin text so all-caps sign names are read as words, not spelled out
  const u = new SpeechSynthesisUtterance(/[A-Z]{2,}/.test(t) ? t.toLowerCase() : t);
  const v = voices().find((x) => x.voiceURI === voiceURI);
  if (v) { u.voice = v; u.lang = v.lang; } else { u.lang = lang; }
  u.rate = rate;
  speechSynthesis.speak(u);
  return true;
}

export const canListen = () => !!(window.SpeechRecognition || window.webkitSpeechRecognition);

// Continuous listening. onText(text, isFinal) is called as words arrive.
export class Listener {
  constructor({ onText, onState, onError }) {
    this.onText = onText;
    this.onState = onState || (() => {});
    this.onError = onError || (() => {});
    this.rec = null;
    this.wanted = false;
  }

  start(lang) {
    const Rec = window.SpeechRecognition || window.webkitSpeechRecognition;
    if (!Rec) { this.onError("Listening works in Chrome and Edge. This browser can't turn speech into text."); return; }
    this.stop();
    const rec = new Rec();
    rec.lang = lang;
    rec.continuous = true;
    rec.interimResults = true;
    rec.onresult = (e) => {
      for (let i = e.resultIndex; i < e.results.length; i++) {
        const r = e.results[i];
        this.onText(r[0].transcript, r.isFinal);
      }
    };
    rec.onerror = (e) => {
      if (e.error === "no-speech" || e.error === "aborted") return;
      this.wanted = false;
      const why = {
        "not-allowed": "Microphone access is blocked. Allow it from the address bar and try again.",
        "service-not-allowed": "Microphone access is blocked. Allow it from the address bar and try again.",
        "network": "Speech recognition needs an internet connection.",
        "language-not-supported": "This browser can't listen in that language. Pick another one.",
      }[e.error] || `Listening stopped: ${e.error}`;
      this.onError(why);
    };
    rec.onend = () => {
      // browsers stop after a pause; keep going until the user presses Stop
      if (this.wanted) { try { rec.start(); return; } catch { /* fall through */ } }
      this.rec = null;
      this.onState(false);
    };
    this.rec = rec;
    this.wanted = true;
    try {
      rec.start();
      this.onState(true);
    } catch (e) {
      this.wanted = false;
      this.onError(`Couldn't start listening: ${e.message || e}`);
    }
  }

  stop() {
    this.wanted = false;
    if (this.rec) { try { this.rec.stop(); } catch { /* already stopped */ } }
  }
}
