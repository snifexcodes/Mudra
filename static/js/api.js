// Small fetch helpers for the Flask API.

async function request(method, url, body) {
  const opts = { method, headers: {} };
  if (body !== undefined) {
    opts.headers["Content-Type"] = "application/json";
    opts.body = JSON.stringify(body);
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

export const api = {
  status: () => request("GET", "/api/status"),
  predict: (hands, source, minConf) => request("POST", "/api/predict", { hands, source, min_conf: minConf }),
  addSamples: (label, session, frames) => request("POST", "/api/samples", { label, session, frames }),
  deleteLabel: (label) => request("DELETE", `/api/samples?label=${encodeURIComponent(label)}`),
  deleteSession: (session) => request("DELETE", `/api/samples?session=${encodeURIComponent(session)}`),
  rename: (oldName, newName) => request("POST", "/api/signs/rename", { old: oldName, new: newName }),
  train: () => request("POST", "/api/train"),
  trainStatus: () => request("GET", "/api/train"),
  report: () => request("GET", "/api/report"),
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
      lm: lm.map((p) => [+(p.x * vw).toFixed(1), +(p.y * vh).toFixed(1), +(p.z * vw).toFixed(1)]),
      wlm: (wlm || lm).map((p) => [+p.x.toFixed(5), +p.y.toFixed(5), +p.z.toFixed(5)]),
    });
  }
  return out;
}

export function newSessionId() {
  if (window.crypto && crypto.randomUUID) return crypto.randomUUID().slice(0, 12);
  return Math.random().toString(36).slice(2, 14);
}
