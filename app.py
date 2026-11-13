"""
Mudra: real-time sign language to text, for any sign language.

Run:   python app.py
Open:  http://localhost:5050

The browser tracks hands (and shoulders, for signs made near the body) with
MediaPipe and sends the points to this server. Each sign-language pack has a
model for still hand shapes and a model for moving signs.
"""

from __future__ import annotations

import argparse
import io
import mimetypes
import threading
import time
import webbrowser
from pathlib import Path

from flask import Flask, jsonify, request, send_file, send_from_directory

from core import features, model, rules, sequence
from core.assets import CDN_BASE, MODEL_URL, POSE_MODEL_URL, VENDOR
from core.packs import DEFAULT_PACK, PackError, PackManager

ROOT = Path(__file__).resolve().parent
STATIC = ROOT / "static"

# Windows sometimes maps .js to text/plain in the registry, which makes the
# browser refuse to run the page's JavaScript modules. Set the types explicitly.
mimetypes.add_type("text/javascript", ".js")
mimetypes.add_type("text/javascript", ".mjs")
mimetypes.add_type("application/wasm", ".wasm")
mimetypes.add_type("font/woff2", ".woff2")

app = Flask(__name__, static_folder=str(STATIC), static_url_path="/static")
app.config["SEND_FILE_MAX_AGE_DEFAULT"] = 0
app.config["MAX_CONTENT_LENGTH"] = 220 * 1024 * 1024

packs = PackManager(ROOT / "packs", legacy_root=ROOT)


class TrainingJob:
    """Trains a pack's still and moving models in the background, one pack at a time."""

    def __init__(self):
        self.lock = threading.Lock()
        self.state, self.stage, self.progress, self.message = "idle", "", 0.0, ""
        self.pack, self.started, self.finished = None, None, None

    def snapshot(self):
        with self.lock:
            return {"state": self.state, "stage": self.stage, "progress": round(self.progress, 3),
                    "message": self.message, "pack": self.pack, "started": self.started,
                    "finished": self.finished}

    def start(self, pack) -> bool:
        with self.lock:
            if self.state == "running":
                return False
            self.state, self.stage, self.progress, self.message = "running", "Starting", 0.0, ""
            self.pack, self.started, self.finished = pack.code, time.time(), None
        threading.Thread(target=self._run, args=(pack,), daemon=True).start()
        return True

    def _run(self, pack):
        try:
            records = pack.data.load()
            kinds = [k for k in model.KINDS
                     if any((r.get("kind") == "clip") == (k == "moving") for r in records)]
            if not kinds:
                raise model.TrainingError("No signs recorded in this pack yet.")
            done, failed = [], []
            for i, kind in enumerate(kinds):
                share = 1 / len(kinds)

                def progress(stage, frac, i=i, share=share, kind=kind):
                    with self.lock:
                        self.stage = f"{kind.capitalize()} signs: {stage}"
                        self.progress = i * share + frac * share

                try:
                    bundle, report = model.train(records, progress=progress, kind=kind)
                    pack.recognizer(kind).save(bundle, report)
                    done.append(f"{kind} signs: {report['best_model']}, {report['accuracy'] * 100:.1f}% accuracy")
                except model.TrainingError as exc:
                    failed.append(f"{kind} signs: {exc}")
            if not done:
                raise model.TrainingError(" ".join(failed))
            msg = "Trained " + "; ".join(done) + "."
            if failed:
                msg += " Not trained: " + " ".join(failed)
            with self.lock:
                self.state, self.message = "done", msg
        except model.TrainingError as exc:
            with self.lock:
                self.state, self.message = "error", str(exc)
        except Exception as exc:  # keep the server alive, show the reason in the UI
            app.logger.exception("training failed")
            with self.lock:
                self.state, self.message = "error", f"Training failed: {exc}"
        finally:
            with self.lock:
                self.finished = time.time()


job = TrainingJob()


def asset_config():
    local_bundle = (VENDOR / "vision_bundle.mjs").exists() and (VENDOR / "wasm").is_dir()
    local_model = (VENDOR / "hand_landmarker.task").exists()
    local_pose = (VENDOR / "pose_landmarker_lite.task").exists()
    return {
        "bundle": "/static/vendor/mediapipe/vision_bundle.mjs" if local_bundle else f"{CDN_BASE}/vision_bundle.mjs",
        "wasm": "/static/vendor/mediapipe/wasm" if local_bundle else f"{CDN_BASE}/wasm",
        "model": "/static/vendor/mediapipe/hand_landmarker.task" if local_model else MODEL_URL,
        "pose_model": "/static/vendor/mediapipe/pose_landmarker_lite.task" if local_pose else POSE_MODEL_URL,
        "offline": local_bundle and local_model and local_pose,
    }


def bad_request(message, code=400):
    return jsonify({"error": message}), code


@app.errorhandler(PackError)
def pack_error(exc):
    return bad_request(str(exc), 404 if "There is no" in str(exc) else 400)


@app.get("/")
def index():
    return send_from_directory(STATIC, "index.html")


@app.get("/api/status")
def status():
    return jsonify({"assets": asset_config(), "packs": packs.list(), "default": DEFAULT_PACK,
                    "training": job.snapshot()})


# ---------- packs ----------
@app.get("/api/packs")
def list_packs():
    return jsonify({"packs": packs.list()})


@app.post("/api/packs")
def create_pack():
    b = request.get_json(silent=True) or {}
    pack = packs.create(b.get("name", ""), b.get("short", ""), b.get("region", ""),
                        b.get("spoken", "en"), bool(b.get("two_hands", True)))
    return jsonify(pack.info(detail=True)), 201


@app.post("/api/packs/import")
def import_pack():
    f = request.files.get("file")
    raw = f.read() if f else request.get_data()
    if not raw:
        return bad_request("Choose a pack .zip to import.")
    pack, added, merged = packs.import_zip(raw)
    return jsonify({"pack": pack.info(detail=True), "added": added, "merged": merged})


@app.get("/api/packs/<code>")
def get_pack(code):
    return jsonify(packs.get(code).info(detail=True))


@app.delete("/api/packs/<code>")
def delete_pack(code):
    packs.delete(code)
    return jsonify({"ok": True, "packs": packs.list()})


@app.get("/api/packs/<code>/export")
def export_pack(code):
    pack = packs.get(code)
    return send_file(io.BytesIO(pack.export_zip()), mimetype="application/zip", as_attachment=True,
                     download_name=f"mudra-{pack.code}.zip")


# ---------- recognition ----------
@app.post("/api/packs/<code>/predict")
def predict(code):
    pack = packs.get(code)
    body = request.get_json(silent=True) or {}
    source = body.get("source", "auto")
    min_conf = float(body.get("min_conf", 0.0))
    t0 = time.perf_counter()
    try:
        vec, measures = features.extract(body.get("hands") or [])
    except (ValueError, TypeError, KeyError) as exc:
        return bad_request(f"Bad landmarks: {exc}")
    if not measures:
        return jsonify({"label": None, "confidence": 0.0, "top": [], "source": None, "ms": 0})
    if pack.still.ready and source in ("auto", "model"):
        result = pack.still.predict(vec, min_conf=min_conf)
        result["source"] = "model"
    else:
        allowed = set(pack.starter_labels())
        result = rules.predict(measures, min_score=max(min_conf, 0.5), allowed=allowed) if allowed else \
            {"label": None, "confidence": 0.0, "top": []}
        result["source"] = "rules"
    result["ms"] = round((time.perf_counter() - t0) * 1000, 2)
    return jsonify(result)


@app.post("/api/packs/<code>/predict_clip")
def predict_clip(code):
    pack = packs.get(code)
    body = request.get_json(silent=True) or {}
    if not pack.moving.ready:
        return jsonify({"label": None, "confidence": 0.0, "top": [], "source": None})
    t0 = time.perf_counter()
    try:
        vec = sequence.clip_features(body.get("frames") or [])
    except (ValueError, TypeError, KeyError):
        return jsonify({"label": None, "confidence": 0.0, "top": [], "source": "moving"})
    result = pack.moving.predict(vec, min_conf=float(body.get("min_conf", 0.0)))
    result["source"] = "moving"
    result["ms"] = round((time.perf_counter() - t0) * 1000, 2)
    return jsonify(result)


# ---------- signs and recordings ----------
@app.post("/api/packs/<code>/signs")
def add_sign(code):
    pack = packs.get(code)
    b = request.get_json(silent=True) or {}
    try:
        label = pack.set_sign_kind(b.get("label", ""), b.get("kind", "still"))
    except ValueError as exc:
        return bad_request(str(exc))
    return jsonify({"label": label, "pack": pack.info(detail=True)})


@app.delete("/api/packs/<code>/signs")
def delete_sign(code):
    pack = packs.get(code)
    label = request.args.get("label", "")
    try:
        removed = pack.data.delete_label(label)
    except ValueError as exc:
        return bad_request(str(exc))
    pack.forget_sign(label.strip().upper())
    return jsonify({"removed": removed, "pack": pack.info(detail=True)})


@app.post("/api/packs/<code>/signs/rename")
def rename_sign(code):
    pack = packs.get(code)
    b = request.get_json(silent=True) or {}
    try:
        n = pack.data.rename_label(b.get("old", ""), b.get("new", ""))
    except ValueError as exc:
        return bad_request(str(exc))
    pack.rename_sign(b.get("old", "").strip().upper(), b.get("new", "").strip().upper())
    return jsonify({"renamed": n, "pack": pack.info(detail=True)})


@app.post("/api/packs/<code>/samples")
def add_samples(code):
    pack = packs.get(code)
    b = request.get_json(silent=True) or {}
    label = b.get("label", "")
    try:
        if "clips" in b:
            kind = "moving"
            pack.set_sign_kind(label, kind)
            added = pack.data.add_clips(label, b.get("session", ""), b.get("clips") or [])
        else:
            kind = "still"
            pack.set_sign_kind(label, kind)
            added = pack.data.add_recording(label, b.get("session", ""), b.get("frames") or [])
    except ValueError as exc:
        return bad_request(str(exc))
    if added == 0:
        return bad_request("No hand was visible during the recording, so nothing was saved.")
    return jsonify({"added": added, "kind": kind, "pack": pack.info(detail=True)})


@app.delete("/api/packs/<code>/samples")
def delete_samples(code):
    pack = packs.get(code)
    session = request.args.get("session")
    if not session:
        return bad_request("Say which recording to delete.")
    removed = pack.data.delete_session(session)
    return jsonify({"removed": removed, "pack": pack.info(detail=True)})


# ---------- training ----------
@app.post("/api/packs/<code>/train")
def start_training(code):
    pack = packs.get(code)
    if not job.start(pack):
        return bad_request("Training is already running.", 409)
    return jsonify(job.snapshot()), 202


@app.get("/api/train")
def training_status():
    return jsonify(job.snapshot())


@app.get("/api/packs/<code>/report")
def get_report(code):
    pack = packs.get(code)
    out = {k: pack.recognizer(k).report() for k in model.KINDS}
    if not any(out.values()):
        return bad_request("No model trained for this pack yet.", 404)
    return jsonify(out)


@app.delete("/api/packs/<code>/model")
def delete_model(code):
    pack = packs.get(code)
    kind = request.args.get("kind")
    for k in ([kind] if kind in model.KINDS else model.KINDS):
        pack.recognizer(k).clear()
    return jsonify({"ok": True})


def main():
    parser = argparse.ArgumentParser(description="Mudra sign-to-text server")
    parser.add_argument("--port", type=int, default=5050)
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--no-browser", action="store_true", help="don't open the browser")
    args = parser.parse_args()
    url = f"http://localhost:{args.port}"
    cfg = asset_config()
    print(f"\n  Mudra is running at {url}")
    print(f"  Hand tracking assets: {'offline (static/vendor)' if cfg['offline'] else 'online (CDN)'}")
    for p in packs.list():
        trained = [k for k, v in p["models"].items() if v]
        state = f"trained ({', '.join(trained)})" if trained else "not trained"
        print(f"  {p['short']:>4}  {p['name']}: {p['sign_count']} signs, {state}")
    print("  Press Ctrl+C to stop.\n")
    if not args.no_browser:
        threading.Timer(1.0, lambda: webbrowser.open(url)).start()
    app.run(host=args.host, port=args.port, debug=False, threaded=True)


if __name__ == "__main__":
    main()
