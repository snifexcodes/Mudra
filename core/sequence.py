"""
Features for moving signs (short clips instead of single frames).

A clip is a list of frames: {"t": seconds, "hands": [...], "pose": [nose, left
shoulder, right shoulder] or None}. Pose points are image pixels, like `lm`.

Steps:
    1. Describe each frame with 105 numbers: the shape of up to two hands (48
       each), which hands are present, where each wrist is relative to the
       nose (in shoulder widths, when the body is visible), and how far the
       main wrist has moved since the sign started (in palm lengths).
    2. Trim the still part before and after the movement, so a clip recorded
       in the Train tab and a sign caught live line up the same way.
    3. Resample the moving part to 12 evenly spaced moments in time, and add
       three numbers for the whole clip: path length, duration and how much
       of the clip was movement.

12 x 105 + 3 = 1263 numbers per clip. Hand shapes are measured from image
points (not world points) so clips imported from datasets that only have
image landmarks are described the same way as live ones.
"""

from __future__ import annotations

import numpy as np

from .features import DIST_KEYS, FINGER_NAMES, TIPS, measure_hand, order_hands

SEQ_VERSION = 1
T = 12
PER_HAND = 48
PER_FRAME = PER_HAND * 2 + 2 + 4 + 2 + 1
CLIP_DIM = T * PER_FRAME + 3

# column offsets inside one frame vector
_PRESENT = PER_HAND * 2
_LOC = _PRESENT + 2          # 4: wrist A (x, y), wrist B (x, y) relative to nose
_TRAJ = _LOC + 4             # 2: wrist A displacement since the clip started
_POSE_FLAG = _TRAJ + 2

MOVE_ON = 0.8                # motion (palm lengths per second) that counts as movement
FINGER_WEIGHT = 0.5          # how much fingertip movement counts next to wrist movement
MIN_FRAMES = 3


def _hand_compact(m) -> np.ndarray:
    bends = [b for name in FINGER_NAMES for b in m.bends[name]]
    return np.concatenate([
        np.radians(bends) / np.pi,
        [m.dist[k] for k in DIST_KEYS],
        m.up,
        m.normal,
        m.local[TIPS].reshape(-1),
    ])


def _valid_pose(pose) -> bool:
    try:
        arr = np.asarray(pose, dtype=float)
        return arr.shape[0] >= 3 and arr.shape[1] >= 2 and np.all(np.isfinite(arr[:3, :2]))
    except (TypeError, ValueError, IndexError):
        return False


def frame_row(frame: dict):
    """One frame -> (vector, wrist_xy, palm_px, primary_is_left, tips_rel) or None if no hand."""
    hands = order_hands((frame or {}).get("hands") or [])[:2]
    if not hands:
        return None
    try:
        measures = [measure_hand(h, use_world=False) for h in hands]
    except (ValueError, TypeError, KeyError):
        return None
    v = np.zeros(PER_FRAME)
    v[:PER_HAND] = _hand_compact(measures[0])
    v[_PRESENT] = 1.0
    if len(measures) > 1:
        v[PER_HAND:PER_HAND * 2] = _hand_compact(measures[1])
        v[_PRESENT + 1] = 1.0

    img = [np.asarray(h["lm"], dtype=float) for h in hands]
    wrist = img[0][0, :2]
    palm = float(np.linalg.norm(img[0][9, :2] - img[0][0, :2])) or 1.0

    pose = frame.get("pose")
    if _valid_pose(pose):
        pose = np.asarray(pose, dtype=float)
        nose = pose[0, :2]
        shoulders = float(np.linalg.norm(pose[1, :2] - pose[2, :2]))
        if shoulders > 1e-6:
            v[_LOC:_LOC + 2] = np.clip((wrist - nose) / shoulders, -4, 4)
            if len(img) > 1:
                v[_LOC + 2:_LOC + 4] = np.clip((img[1][0, :2] - nose) / shoulders, -4, 4)
            v[_POSE_FLAG] = 1.0
    tips_rel = (img[0][TIPS, :2] - wrist) / palm
    return v, wrist, palm, hands[0].get("handed") == "Left", tips_rel


def _active_span(speed: np.ndarray) -> tuple[int, int]:
    moving = np.where(speed > MOVE_ON)[0]
    if len(moving) == 0:
        return 0, len(speed) - 1
    return max(0, int(moving[0]) - 1), min(len(speed) - 1, int(moving[-1]) + 1)


def clip_features(frames: list[dict]) -> np.ndarray:
    rows, times = [], []
    for f in frames or []:
        r = frame_row(f)
        if r is not None:
            rows.append(r)
            times.append(float(f.get("t", len(times) / 15.0)))
    if len(rows) < MIN_FRAMES:
        raise ValueError("clip has too few frames with a hand")

    t = np.asarray(times)
    order = np.argsort(t, kind="stable")
    t = t[order]
    rows = [rows[i] for i in order]
    vecs = np.stack([r[0] for r in rows])
    wrists = np.stack([r[1] for r in rows])
    palm_ref = float(np.median([r[2] for r in rows])) or 1.0
    mirror = sum(r[3] for r in rows) > len(rows) / 2

    tips = np.stack([r[4] for r in rows])
    dt = np.diff(t)
    dt[dt <= 1e-6] = 1 / 30
    step = np.linalg.norm(np.diff(wrists, axis=0), axis=1) / palm_ref
    finger_step = np.linalg.norm(np.diff(tips, axis=0), axis=2).mean(axis=1)
    speed = np.concatenate([[0.0], (step + FINGER_WEIGHT * finger_step) / dt])
    if len(speed) >= 3:
        speed = np.convolve(speed, np.ones(3) / 3, mode="same")
    a, b = _active_span(speed)
    if b - a + 1 < MIN_FRAMES:
        a, b = 0, len(rows) - 1

    seg = vecs[a:b + 1].copy()
    seg_t = t[a:b + 1]
    seg[:, _TRAJ:_TRAJ + 2] = np.clip((wrists[a:b + 1] - wrists[a]) / palm_ref, -15, 15)
    if mirror:   # a left-handed signer makes the mirror image of the same sign
        seg[:, [_LOC, _LOC + 2, _TRAJ]] *= -1

    span = seg_t[-1] - seg_t[0]
    if span <= 1e-6:
        resampled = np.repeat(seg[:1], T, axis=0)
    else:
        u = (seg_t - seg_t[0]) / span
        grid = np.linspace(0, 1, T)
        resampled = np.stack([np.interp(grid, u, seg[:, c]) for c in range(PER_FRAME)], axis=1)

    path = float(np.sum(step[a:b])) if b > a else 0.0
    extra = [min(path / 10.0, 3.0), min(span / 3.0, 3.0), (b - a + 1) / len(rows)]
    return np.concatenate([resampled.reshape(-1), extra])
