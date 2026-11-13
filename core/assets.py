"""Where the MediaPipe hand-tracking files come from (CDN, or a local copy)."""

from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
VENDOR = ROOT / "static" / "vendor" / "mediapipe"
MP_VERSION = "0.10.35"
CDN_BASE = f"https://cdn.jsdelivr.net/npm/@mediapipe/tasks-vision@{MP_VERSION}"
MODEL_URL = ("https://storage.googleapis.com/mediapipe-models/hand_landmarker/"
             "hand_landmarker/float16/1/hand_landmarker.task")
POSE_MODEL_URL = ("https://storage.googleapis.com/mediapipe-models/pose_landmarker/"
                  "pose_landmarker_lite/float16/1/pose_landmarker_lite.task")
