"""
Import sign-language videos into a Mudra pack as moving-sign recordings.

Works for any sign language. Put one folder per sign, with videos of that
sign inside (any depth above it is fine):

    my-dataset/
        Greetings/
            12. Thank you/
                video1.mp4
                video2.mov
            13. Hello/ ...

The sign name comes from the folder that holds the videos ("12. Thank you"
becomes THANK YOU). This matches the layout of INCLUDE, the Indian Sign
Language dataset (263 words, CC BY 4.0): https://zenodo.org/records/4010759

Setup (once):
    python -m pip install -r requirements-cv.txt
    python tools/download_assets.py           # hand and pose models

Run:
    python tools/datasets/import_videos.py --videos "D:/INCLUDE" --pack isl --top 50 --train

Each video becomes one repetition. Videos already imported are skipped, so
you can stop it and run it again later.
"""

from __future__ import annotations

import argparse
import sys
import time
from collections import defaultdict
from pathlib import Path

from common import VIDEO_EXT, label_from_folder, open_pack, pick_labels, session_id, train_moving

from core.assets import VENDOR


def find_videos(root: Path) -> dict[str, list[Path]]:
    found: dict[str, list[Path]] = defaultdict(list)
    for p in sorted(root.rglob("*")):
        if p.is_file() and p.suffix.lower() in VIDEO_EXT:
            label = label_from_folder(p.parent.name)
            if label:
                found[label].append(p)
    return found


class Extractor:
    """Runs MediaPipe hand and pose landmarkers over a video file."""

    def __init__(self, num_hands: int = 2):
        try:
            import mediapipe as mp
            from mediapipe.tasks import python as mp_tasks
            from mediapipe.tasks.python import vision
        except ImportError:
            sys.exit("MediaPipe isn't installed. Run: python -m pip install -r requirements-cv.txt")
        hand_task = VENDOR / "hand_landmarker.task"
        pose_task = VENDOR / "pose_landmarker_lite.task"
        if not hand_task.exists() or not pose_task.exists():
            sys.exit("Missing MediaPipe models. Run: python tools/download_assets.py")
        self.mp, self.vision, self.tasks = mp, vision, mp_tasks
        self.num_hands = num_hands
        self.hand_task, self.pose_task = str(hand_task), str(pose_task)

    def _new(self):
        v, t = self.vision, self.tasks
        hands = v.HandLandmarker.create_from_options(v.HandLandmarkerOptions(
            base_options=t.BaseOptions(model_asset_path=self.hand_task), running_mode=v.RunningMode.VIDEO,
            num_hands=self.num_hands, min_hand_detection_confidence=0.5,
            min_hand_presence_confidence=0.5, min_tracking_confidence=0.5))
        pose = v.PoseLandmarker.create_from_options(v.PoseLandmarkerOptions(
            base_options=t.BaseOptions(model_asset_path=self.pose_task), running_mode=v.RunningMode.VIDEO,
            num_poses=1))
        return hands, pose

    def frames(self, path: Path, fps: float, max_width: int = 960):
        import cv2
        cap = cv2.VideoCapture(str(path))
        if not cap.isOpened():
            raise OSError("can't open video")
        src_fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
        step = max(1, round(src_fps / fps))
        hands_lm, pose_lm = self._new()
        out, i = [], 0
        try:
            while True:
                ok, frame = cap.read()
                if not ok:
                    break
                if i % step == 0:
                    h, w = frame.shape[:2]
                    if w > max_width:
                        scale = max_width / w
                        frame = cv2.resize(frame, (max_width, int(h * scale)))
                        h, w = frame.shape[:2]
                    rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                    img = self.mp.Image(image_format=self.mp.ImageFormat.SRGB, data=rgb)
                    ts = int(i / src_fps * 1000)
                    out.append(to_frame(hands_lm.detect_for_video(img, ts), pose_lm.detect_for_video(img, ts),
                                        w, h, i / src_fps))
                i += 1
        finally:
            cap.release()
            hands_lm.close()
            pose_lm.close()
        return out


def to_frame(hand_result, pose_result, w: int, h: int, t: float) -> dict:
    hands = []
    for k, lms in enumerate(hand_result.hand_landmarks or []):
        world = hand_result.hand_world_landmarks[k] if hand_result.hand_world_landmarks else lms
        cat = hand_result.handedness[k][0] if hand_result.handedness and hand_result.handedness[k] else None
        hands.append({
            "handed": cat.category_name if cat else "Right",
            "lm": [[p.x * w, p.y * h, p.z * w] for p in lms],
            "wlm": [[p.x, p.y, p.z] for p in world],
        })
    pose = None
    if pose_result is not None and pose_result.pose_landmarks:
        pl = pose_result.pose_landmarks[0]
        pose = [[pl[i].x * w, pl[i].y * h, pl[i].z * w] for i in (0, 11, 12)]
    return {"t": round(t, 3), "hands": hands, "pose": pose}


def main(argv=None, extractor=None):
    ap = argparse.ArgumentParser(description="Import sign videos (one folder per sign) into a Mudra pack")
    ap.add_argument("--videos", required=True, help="dataset folder")
    ap.add_argument("--pack", default="isl", help="pack code: isl, asl, bsl or one you created")
    ap.add_argument("--top", type=int, help="only the N signs with the most videos (e.g. 50)")
    ap.add_argument("--only", help='only these signs, comma separated, e.g. "HELLO,THANK YOU"')
    ap.add_argument("--max-per-sign", type=int, default=0, help="at most this many videos per sign")
    ap.add_argument("--fps", type=float, default=15.0, help="frames per second to analyse (default 15)")
    ap.add_argument("--train", action="store_true", help="train the moving-sign model when done")
    args = ap.parse_args(argv)

    root = Path(args.videos)
    if not root.is_dir():
        sys.exit(f"Folder not found: {root}")
    found = find_videos(root)
    if not found:
        sys.exit("No videos found. Expected one folder per sign with video files inside.")
    labels = pick_labels({l: len(v) for l, v in found.items()}, args.only, args.top)
    if not labels:
        sys.exit("None of the requested signs were found.")

    pack = open_pack(args.pack)
    have = {r["session"] for r in pack.data.load()}
    clashes = [l for l in labels if pack.sign_kind(l) == "still"]
    for l in clashes:
        print(f"Skipping {l}: it's a still sign in this pack.")
    labels -= set(clashes)

    jobs = []
    for label in sorted(labels):
        vids = found[label][: args.max_per_sign or None]
        jobs += [(label, v) for v in vids]
    todo = [(l, v) for l, v in jobs if session_id(str(v.resolve())) not in have]
    print(f"{len(labels)} signs, {len(jobs)} videos ({len(jobs) - len(todo)} already imported) "
          f"into {pack.meta.get('name')}.")

    extractor = extractor or Extractor()
    added = failed = 0
    t0 = time.time()
    for n, (label, video) in enumerate(todo, 1):
        try:
            frames = extractor.frames(video, args.fps)
            kept = pack.data.add_clips(label, session_id(str(video.resolve())), [frames])
            if kept:
                pack.set_sign_kind(label, "moving")
                added += 1
            else:
                failed += 1
        except Exception as exc:  # one bad video shouldn't stop the import
            failed += 1
            print(f"\n  {video.name}: {exc}")
        rate = (time.time() - t0) / n
        print(f"\r  {n}/{len(todo)}  {label[:20]:20s}  about {rate * (len(todo) - n) / 60:4.0f} min left",
              end="", flush=True)
    print(f"\nImported {added} videos. {failed} had no visible hands or couldn't be read.")
    if args.train:
        train_moving(pack)
    else:
        print("Open Mudra, choose this pack, and press Train model in the Train tab.")


if __name__ == "__main__":
    main()
