"""
Desktop version of Mudra using OpenCV for the camera window.

It uses the same feature code, the same starter rules and the same trained
still-sign model (packs/<pack>/model_still.joblib) as the web app, so a model
trained in the browser works here too. Moving signs are only in the web app.

Setup (once):
    pip install -r requirements-cv.txt
    python tools/download_assets.py        # gets hand_landmarker.task

Run:
    python cv_demo.py --pack isl           # --camera 1 for another webcam

Keys:  q quit   space add a space   backspace delete   c clear   m model/rules
"""

from __future__ import annotations

import argparse
import sys
import time
from collections import Counter, deque

import cv2
import numpy as np

try:
    import mediapipe as mp
    from mediapipe.tasks import python as mp_tasks
    from mediapipe.tasks.python import vision
except ImportError:
    sys.exit("MediaPipe isn't installed. Run: pip install -r requirements-cv.txt")

from core import features, rules
from core.assets import ROOT, VENDOR
from core.packs import DEFAULT_PACK, PackError, PackManager

# finger colours as BGR: thumb, index, middle, ring, little
HUES = [(71, 181, 255), (89, 122, 255), (139, 79, 255), (255, 107, 183), (255, 140, 91)]
FINGER_OF = {}
for fi, (name, idxs) in enumerate(features.FINGERS.items()):
    for i in idxs:
        FINGER_OF[i] = fi


class HoldToAdd:
    """Same idea as the web app: smooth predictions, add a sign after holding it."""

    def __init__(self, hold=0.9, min_conf=0.7, window=0.65):
        self.hold, self.min_conf, self.window = hold, min_conf, window
        self.preds = deque()
        self.label = None
        self.since = 0.0
        self.repeat = False

    def update(self, now, label, conf):
        """Returns (shown_label, progress 0..1, label_to_add or None)."""
        self.preds.append((now, label if conf >= self.min_conf else None))
        while self.preds and now - self.preds[0][0] > self.window:
            self.preds.popleft()
        votes = Counter(l for _, l in self.preds if l)
        current = None
        if votes:
            best, n = votes.most_common(1)[0]
            if n >= max(2, int(len(self.preds) * 0.6 + 0.999)):
                current = best
        if current != self.label:
            self.label, self.since, self.repeat = current, now, False
        if not current:
            return None, 0.0, None
        need = self.hold * (1.6 if self.repeat else 1.0)
        progress = min(1.0, (now - self.since) / need)
        if progress >= 1.0:
            self.since, self.repeat = now, True
            return current, 0.0, current
        return current, progress, None

    def reset(self):
        self.preds.clear()
        self.label = None


def to_hand_dicts(result, w, h):
    hands = []
    for i, lms in enumerate(result.hand_landmarks):
        world = result.hand_world_landmarks[i] if result.hand_world_landmarks else lms
        cat = result.handedness[i][0] if result.handedness and result.handedness[i] else None
        hands.append({
            "handed": cat.category_name if cat else "Right",
            "lm": [[p.x * w, p.y * h, p.z * w] for p in lms],
            "wlm": [[p.x, p.y, p.z] for p in world],
        })
    return features.order_hands(hands)


def add_to_text(text, label):
    if label == "SPACE":
        return text if not text or text.endswith(" ") else text + " "
    if label in ("DELETE", "DEL", "BACKSPACE"):
        return text[:-1]
    if len(label) == 1:
        return text + label
    return text + (" " if text and not text.endswith(" ") else "") + label + " "


def draw_hand(img, pts):
    for a, b in features.CONNECTIONS:
        same_finger = a in FINGER_OF and FINGER_OF.get(a) == FINGER_OF.get(b)
        color = HUES[FINGER_OF[b]] if same_finger else (235, 230, 225)   # palm bones stay white
        cv2.line(img, pts[a], pts[b], color, 2, cv2.LINE_AA)
    for i, p in enumerate(pts):
        if i in features.TIPS:
            cv2.circle(img, p, 7, HUES[features.TIPS.index(i)], -1, cv2.LINE_AA)
        else:
            cv2.circle(img, p, 3, (240, 240, 240), -1, cv2.LINE_AA)


def put(img, text, org, scale=0.7, color=(240, 240, 240), thick=2):
    cv2.putText(img, text, org, cv2.FONT_HERSHEY_SIMPLEX, scale, (20, 18, 14), thick + 3, cv2.LINE_AA)
    cv2.putText(img, text, org, cv2.FONT_HERSHEY_SIMPLEX, scale, color, thick, cv2.LINE_AA)


def main():
    ap = argparse.ArgumentParser(description="Mudra OpenCV demo")
    ap.add_argument("--camera", type=int, default=0)
    ap.add_argument("--pack", default=DEFAULT_PACK, help="sign-language pack: isl, asl, bsl or your own")
    ap.add_argument("--hands", type=int, default=1, choices=(1, 2))
    ap.add_argument("--hold", type=float, default=0.9, help="seconds to hold a sign")
    ap.add_argument("--min-conf", type=float, default=0.7)
    args = ap.parse_args()

    task = VENDOR / "hand_landmarker.task"
    if not task.exists():
        sys.exit("Missing hand_landmarker.task. Run: python tools/download_assets.py")

    try:
        pack = PackManager(ROOT / "packs", legacy_root=ROOT).get(args.pack)
    except PackError as exc:
        sys.exit(str(exc))
    recognizer = pack.still
    allowed = set(pack.starter_labels())
    use_model = recognizer.ready
    print(pack.meta.get("name"), "-", f"trained model ({recognizer.info()['name']})" if use_model else "starter gestures")

    options = vision.HandLandmarkerOptions(
        base_options=mp_tasks.BaseOptions(model_asset_path=str(task)),
        running_mode=vision.RunningMode.VIDEO,
        num_hands=args.hands,
        min_hand_detection_confidence=0.6,
        min_hand_presence_confidence=0.6,
        min_tracking_confidence=0.5,
    )
    landmarker = vision.HandLandmarker.create_from_options(options)

    backend = cv2.CAP_DSHOW if sys.platform == "win32" else cv2.CAP_ANY
    cap = cv2.VideoCapture(args.camera, backend)
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)
    if not cap.isOpened():
        sys.exit(f"Couldn't open camera {args.camera}. Try --camera 1.")

    holder = HoldToAdd(args.hold, args.min_conf)
    text = ""
    last_ts = 0
    fps, frames, fps_t = 0.0, 0, time.monotonic()
    absent_since = None

    while True:
        ok, frame = cap.read()
        if not ok:
            print("Camera stopped sending frames.")
            break
        h, w = frame.shape[:2]
        ts = max(int(time.monotonic() * 1000), last_ts + 1)   # must strictly increase
        last_ts = ts
        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        result = landmarker.detect_for_video(mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb), ts)
        hands = to_hand_dicts(result, w, h)
        now = time.monotonic()

        view = cv2.flip(frame, 1)                      # mirror for display only
        view = cv2.addWeighted(view, 0.7, np.zeros_like(view), 0.3, 0)
        shown, progress, added, conf = None, 0.0, None, 0.0
        if hands:
            absent_since = None
            vec, measures = features.extract(hands)
            if use_model:
                pred = recognizer.predict(vec)
            elif allowed:
                pred = rules.predict(measures, allowed=allowed)
            else:
                pred = {"label": None, "confidence": 0.0}
            conf = pred["confidence"]
            shown, progress, added = holder.update(now, pred["label"], conf)
            if added == "NO SIGN":
                added = None
            for hand in hands:
                pts = [(int(w - x), int(y)) for x, y, _ in hand["lm"]]
                draw_hand(view, pts)
            xs = [w - p[0] for p in hands[0]["lm"]]
            ys = [p[1] for p in hands[0]["lm"]]
            bx, by = int(max(xs)) + 20, int(min(ys))
            label = shown or "?"
            put(view, label, (min(bx, w - 220), max(60, by + 40)), 1.6, (71, 181, 255) if shown else (180, 170, 160), 3)
            if shown:
                end = int(progress * 360)
                cv2.ellipse(view, (min(bx, w - 220) + 30, max(60, by + 40) - 18), (44, 44), -90, 0, end,
                            (71, 181, 255), 3, cv2.LINE_AA)
        else:
            holder.reset()
            absent_since = absent_since or now
            if text and not text.endswith(" ") and now - absent_since > 1.1:
                text += " "
        if added:
            text = add_to_text(text, added)

        frames += 1
        if now - fps_t >= 1.0:
            fps, frames, fps_t = frames / (now - fps_t), 0, now

        cv2.rectangle(view, (0, h - 90), (w, h), (48, 18, 14), -1)
        put(view, (text[-48:] if text else "Your signs will appear here") + "_", (24, h - 36), 1.1,
            (240, 240, 240) if text else (160, 150, 140), 2)
        put(view, f"{'Model' if use_model else 'Starter gestures'}  {fps:4.1f} fps   q quit  m switch  c clear",
            (16, 30), 0.6, (200, 195, 190), 1)

        cv2.imshow("Mudra (OpenCV)", view)
        key = cv2.waitKey(1) & 0xFF
        if key in (ord("q"), 27):
            break
        if key == ord(" "):
            text = add_to_text(text, "SPACE")
        elif key in (8, 127):
            text = text[:-1]
        elif key == ord("c"):
            text = ""
        elif key == ord("m"):
            if recognizer.ready:
                use_model = not use_model
            else:
                print("No trained model yet. Train one in the web app first.")

    cap.release()
    cv2.destroyAllWindows()
    landmarker.close()
    if text.strip():
        print("Transcript:", text.strip())


if __name__ == "__main__":
    main()
