// Fetch helpers for the Flask API. Everything about signs lives inside a pack
// (one per sign language), so most calls take the pack code first.

async function request(method, url, body, isForm = false) {
  const opts = { method, headers: {} };
  if (body !== undefined) {
    if (isForm) {
      opts.body = body;
    } else {
      opts.headers["Content-Type"] = "application/json";
      opts.body = JSON.stringify(body);
    }
  }
  let res;
  try {
    res = await fetch(url, opts);
  } catch {
    throw new Error("Can't reach the Mudra server. Is python app.py still running?");
  }
  let data = null;
  try { data = await res.json(); } catch { /* empty body */ }
  if (!res.ok) {
    const err = new Error((data && data.error) || `Server error ${res.status}`);
    err.status = res.status;
    throw err;
  }
  return data;
}

const p = (code) => `/api/packs/${encodeURIComponent(code)}`;

export const api = {
  status: () => request("GET", "/api/status"),
  packs: () => request("GET", "/api/packs"),
  createPack: (body) => request("POST", "/api/packs", body),
  deletePack: (code) => request("DELETE", p(code)),
  importPack: (file) => {
    const form = new FormData();
    form.append("file", file);
    return request("POST", "/api/packs/import", form, true);
  },
  exportUrl: (code) => `${p(code)}/export`,
  pack: (code) => request("GET", p(code)),
  predict: (code, hands, source, minConf) => request("POST", `${p(code)}/predict`, { hands, source, min_conf: minConf }),
  predictClip: (code, frames, minConf) => request("POST", `${p(code)}/predict_clip`, { frames, min_conf: minConf }),
  addSign: (code, label, kind) => request("POST", `${p(code)}/signs`, { label, kind }),
  deleteSign: (code, label) => request("DELETE", `${p(code)}/signs?label=${encodeURIComponent(label)}`),
  rename: (code, oldName, newName) => request("POST", `${p(code)}/signs/rename`, { old: oldName, new: newName }),
  addSamples: (code, label, session, frames) => request("POST", `${p(code)}/samples`, { label, session, frames }),
  addClips: (code, label, session, clips) => request("POST", `${p(code)}/samples`, { label, session, clips }),
  deleteSession: (code, session) => request("DELETE", `${p(code)}/samples?session=${encodeURIComponent(session)}`),
  train: (code) => request("POST", `${p(code)}/train`),
  trainStatus: () => request("GET", "/api/train"),
  report: (code) => request("GET", `${p(code)}/report`),
};

// Hand data in the shape the server expects.
// lm: image points in pixels (x and z scaled by width so the hand isn't squashed)
// wlm: MediaPipe world points in metres
export function packHands(result, vw, vh, max = 2) {
  const out = [];
  if (!result || !result.landmarks) return out;
  const handed = result.handedness || result.handednesses || [];
  for (let i = 0; i < Math.min(result.landmarks.length, max); i++) {
    const lm = result.landmarks[i];
    const wlm = result.worldLandmarks && result.worldLandmarks[i];
    const cat = handed[i] && handed[i][0];
    out.push({
      handed: cat ? cat.categoryName : "Right",
      score: cat ? +cat.score.toFixed(3) : 1,
      lm: lm.map((q) => [+(q.x * vw).toFixed(1), +(q.y * vh).toFixed(1), +(q.z * vw).toFixed(1)]),
      wlm: (wlm || lm).map((q) => [+q.x.toFixed(5), +q.y.toFixed(5), +q.z.toFixed(5)]),
    });
  }
  return out;
}

// Nose and both shoulders in pixels, used to tell where a sign is made.
export function packPose(poseResult, vw, vh) {
  const pl = poseResult && poseResult.landmarks && poseResult.landmarks[0];
  if (!pl || pl.length < 13) return null;
  return [0, 11, 12].map((i) => [+(pl[i].x * vw).toFixed(1), +(pl[i].y * vh).toFixed(1), +(pl[i].z * vw).toFixed(1)]);
}

// How much the main hand moved since the previous frame, in palm lengths:
// wrist travel plus half the average fingertip travel relative to the wrist.
// Matches the motion measure in core/sequence.py.
export function handMotion(prev, cur) {
  if (!prev || !cur) return 0;
  const a = prev.lm, b = cur.lm;
  const palm = Math.hypot(b[9][0] - b[0][0], b[9][1] - b[0][1]) || 1;
  const wrist = Math.hypot(b[0][0] - a[0][0], b[0][1] - a[0][1]) / palm;
  let tips = 0;
  for (const i of [4, 8, 12, 16, 20]) {
    const ax = (a[i][0] - a[0][0]), ay = (a[i][1] - a[0][1]);
    const bx = (b[i][0] - b[0][0]), by = (b[i][1] - b[0][1]);
    tips += Math.hypot(bx - ax, by - ay) / palm;
  }
  return wrist + 0.5 * (tips / 5);
}

export function newSessionId() {
  if (window.crypto && crypto.randomUUID) return crypto.randomUUID().slice(0, 12);
  return Math.random().toString(36).slice(2, 14);
}

export const isNothing = (label) => !!label && label.startsWith("NO SIGN");
