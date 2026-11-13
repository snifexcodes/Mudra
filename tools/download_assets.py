"""
Downloads the MediaPipe hand-tracking files so Mudra works without internet
(handy for a college demo on flaky Wi-Fi). Run once from the project folder:

    python tools/download_assets.py

Files go to static/vendor/mediapipe/. The app uses them automatically when
they are present, and falls back to the CDN when they are not. The desktop
OpenCV demo (cv_demo.py) also needs hand_landmarker.task from here.
"""

from __future__ import annotations

import sys
import urllib.request
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
from core.assets import CDN_BASE, MODEL_URL, POSE_MODEL_URL, VENDOR  # noqa: E402

FILES = {
    "vision_bundle.mjs": f"{CDN_BASE}/vision_bundle.mjs",
    "wasm/vision_wasm_internal.js": f"{CDN_BASE}/wasm/vision_wasm_internal.js",
    "wasm/vision_wasm_internal.wasm": f"{CDN_BASE}/wasm/vision_wasm_internal.wasm",
    "wasm/vision_wasm_nosimd_internal.js": f"{CDN_BASE}/wasm/vision_wasm_nosimd_internal.js",
    "wasm/vision_wasm_nosimd_internal.wasm": f"{CDN_BASE}/wasm/vision_wasm_nosimd_internal.wasm",
    "hand_landmarker.task": MODEL_URL,
    "pose_landmarker_lite.task": POSE_MODEL_URL,
}


def fetch(url: str, dest: Path):
    dest.parent.mkdir(parents=True, exist_ok=True)
    tmp = dest.with_suffix(dest.suffix + ".part")
    with urllib.request.urlopen(url, timeout=60) as res, tmp.open("wb") as f:
        total = int(res.headers.get("Content-Length") or 0)
        done = 0
        while True:
            chunk = res.read(1 << 16)
            if not chunk:
                break
            f.write(chunk)
            done += len(chunk)
            if total:
                print(f"\r  {dest.name:40s} {done * 100 // total:3d}%", end="", flush=True)
    tmp.replace(dest)
    print(f"\r  {dest.name:40s} {dest.stat().st_size / 1e6:6.1f} MB")


def main():
    print(f"Saving MediaPipe files to {VENDOR}")
    failed = []
    for rel, url in FILES.items():
        dest = VENDOR / rel
        if dest.exists() and dest.stat().st_size > 0:
            print(f"  {dest.name:40s} already there")
            continue
        try:
            fetch(url, dest)
        except Exception as exc:  # network errors, proxies, etc.
            print(f"\n  Couldn't download {url}: {exc}")
            failed.append(rel)
    if failed:
        print("\nSome files are missing. Check your internet connection and run this again.")
        sys.exit(1)
    print("\nDone. Restart python app.py and it will run offline.")


if __name__ == "__main__":
    main()
