"""
Mudra: real-time sign language to text.

Run:   python app.py
Open:  http://localhost:5050

The browser tracks the hand with MediaPipe and sends 21 landmarks per hand to
this server. The server turns them into features and classifies them with the
starter rules or with the model you trained in the Train tab.
"""

from __future__ import annotations

import argparse
import mimetypes
import threading
import time
import webbrowser
from pathlib import Path

from flask import Flask, jsonify, request, send_file, send_from_directory

from core import dataset as ds
from core import features, model, rules
from core.assets import CDN_BASE, MODEL_URL, VENDOR

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
app.config["MAX_CONTENT_LENGTH"] = 32 * 1024 * 1024

data = ds.Dataset(ROOT / "data" / "samples.jsonl")
recognizer = model.Recognizer(ROOT / "models")


class TrainingJob:
    def __init__(self):
        self.lock = threading.Lock()
        self.state = "idle"
        self.stage = ""
        self.progress = 0.0
        self.message = ""
        self.started = None
        self.finished = None

    def snapshot(self):
        with self.lock:
            return {"state": self.state, "stage": self.stage, "progress": round(self.progress, 3),
                    "message": self.message, "started": self.started, "finished": self.finished}

    def _progress(self, stage, frac):
        with self.lock:
            self.stage, self.progress = stage, frac

    def start(self) -> bool:
        with self.lock:
            if self.state == "running":
                return False
            self.state, self.stage, self.progress, self.message = "running", "Starting", 0.0, ""
            self.started, self.finished = time.time(), None
        threading.Thread(target=self._run, daemon=True).start()
        return True

    def _run(self):
        try:
            bundle, report = model.train(data.load(), progress=self._progress)
            recognizer.save(bundle, report)
            msg = f"{report['best_model']} scored {report['accuracy'] * 100:.1f}% accuracy."
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
    return {
        "bundle": "/static/vendor/mediapipe/vision_bundle.mjs" if local_bundle else f"{CDN_BASE}/vision_bundle.mjs",
        "wasm": "/static/vendor/mediapipe/wasm" if local_bundle else f"{CDN_BASE}/wasm",
        "model": "/static/vendor/mediapipe/hand_landmarker.task" if local_model else MODEL_URL,
        "offline": local_bundle and local_model,
    }


def bad_request(message, code=400):
    return jsonify({"error": message}), code


@app.get("/")
def index():
    return send_from_directory(STATIC, "index.html")


@app.get("/api/status")
def status():
    return jsonify({
        "model": recognizer.info(),
        "model_error": recognizer.load_error,
        "dataset": data.summary(),
        "starter": [{"label": l, "how": rules.GUIDE[l]} for l in rules.LABELS],
        "assets": asset_config(),
        "training": job.snapshot(),
    })


@app.post("/api/predict")
def predict():
    body = request.get_json(silent=True) or {}
    hands = body.get("hands") or []
    source = body.get("source", "auto")
    min_conf = float(body.get("min_conf", 0.0))
    t0 = time.perf_counter()
    try:
        vec, measures = features.extract(hands)
    except (ValueError, TypeError, KeyError) as exc:
        return bad_request(f"Bad landmarks: {exc}")
    if not measures:
        return jsonify({"label": None, "confidence": 0.0, "top": [], "source": None, "ms": 0})

    use_model = recognizer.ready and source in ("auto", "model")
    if use_model:
        result = recognizer.predict(vec, min_conf=min_conf)
        result["source"] = "model"
    else:
        result = rules.predict(measures, min_score=max(min_conf, 0.5))
        result["source"] = "rules"
    result["ms"] = round((time.perf_counter() - t0) * 1000, 2)
    return jsonify(result)


@app.post("/api/samples")
def add_samples():
    body = request.get_json(silent=True) or {}
    try:
        added = data.add_recording(body.get("label", ""), body.get("session", ""), body.get("frames", []))
    except ValueError as exc:
        return bad_request(str(exc))
    if added == 0:
        return bad_request("No hand was visible during the recording, so nothing was saved.")
    return jsonify({"added": added, "dataset": data.summary()})


@app.delete("/api/samples")
def delete_samples():
    label = request.args.get("label")
    session = request.args.get("session")
    try:
        if label:
            removed = data.delete_label(label)
        elif session:
            removed = data.delete_session(session)
        else:
            return bad_request("Say which sign or recording to delete.")
    except ValueError as exc:
        return bad_request(str(exc))
    return jsonify({"removed": removed, "dataset": data.summary()})


@app.post("/api/signs/rename")
def rename_sign():
    body = request.get_json(silent=True) or {}
    try:
        n = data.rename_label(body.get("old", ""), body.get("new", ""))
    except ValueError as exc:
        return bad_request(str(exc))
    return jsonify({"renamed": n, "dataset": data.summary()})


@app.get("/api/samples/export")
def export_samples():
    if not data.path.exists():
        return bad_request("No samples recorded yet.", 404)
    return send_file(data.path, as_attachment=True, download_name="mudra-samples.jsonl")


@app.post("/api/train")
def start_training():
    if not job.start():
        return bad_request("Training is already running.", 409)
    return jsonify(job.snapshot()), 202


@app.get("/api/train")
def training_status():
    return jsonify(job.snapshot())


@app.get("/api/report")
def get_report():
    rep = recognizer.report()
    if rep is None:
        return bad_request("No model trained yet.", 404)
    return jsonify(rep)


@app.delete("/api/model")
def delete_model():
    recognizer.clear()
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
    if recognizer.ready:
        info = recognizer.info()
        print(f"  Trained model: {info['name']} with {len(info['labels'])} signs")
    else:
        print("  No trained model yet: using starter gestures")
    print("  Press Ctrl+C to stop.\n")
    if not args.no_browser:
        threading.Timer(1.0, lambda: webbrowser.open(url)).start()
    app.run(host=args.host, port=args.port, debug=False, threaded=True)


if __name__ == "__main__":
    main()
