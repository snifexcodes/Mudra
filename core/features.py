"""
Turns raw MediaPipe hand landmarks into a fixed-length feature vector.

MediaPipe gives 21 points per hand:
    0 wrist
    1-4   thumb  (CMC, MCP, IP, TIP)
    5-8   index  (MCP, PIP, DIP, TIP)
    9-12  middle
    13-16 ring
    17-20 pinky

The same sign should give the same features no matter where the hand is in
the frame, how big it looks, or which hand is used. So each hand is:
    1. mirrored if it is a left hand (so every hand looks like a right hand)
    2. moved so the wrist is the origin
    3. rotated into the hand's own frame (y = wrist -> middle knuckle,
       x = pinky knuckle -> index knuckle)
    4. scaled so the palm length is 1

Orientation is kept separately (6 numbers) because some signs differ only
by which way the hand points.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field

import numpy as np

FEATURE_VERSION = 1

WRIST = 0
FINGERS = {
    "thumb": (1, 2, 3, 4),
    "index": (5, 6, 7, 8),
    "middle": (9, 10, 11, 12),
    "ring": (13, 14, 15, 16),
    "pinky": (17, 18, 19, 20),
}
FINGER_NAMES = list(FINGERS.keys())
TIPS = [4, 8, 12, 16, 20]

# Bones drawn by the OpenCV demo (the web UI has its own copy in render.js)
CONNECTIONS = [
    (0, 1), (1, 2), (2, 3), (3, 4),
    (0, 5), (5, 6), (6, 7), (7, 8),
    (5, 9), (9, 10), (10, 11), (11, 12),
    (9, 13), (13, 14), (14, 15), (15, 16),
    (13, 17), (0, 17), (17, 18), (18, 19), (19, 20),
]

PER_HAND_DIM = 63 + 6 + 15 + 12
FEATURE_DIM = PER_HAND_DIM * 2 + 2 + 3


def _unit(v: np.ndarray) -> np.ndarray:
    n = float(np.linalg.norm(v))
    return v / n if n > 1e-9 else v * 0.0


def _bend(a: np.ndarray, b: np.ndarray, c: np.ndarray) -> float:
    """Angle (degrees) between bone a->b and bone b->c. 0 means straight."""
    v1, v2 = _unit(b - a), _unit(c - b)
    cos = float(np.clip(np.dot(v1, v2), -1.0, 1.0))
    return math.degrees(math.acos(cos))


def _angle_between(v1: np.ndarray, v2: np.ndarray) -> float:
    cos = float(np.clip(np.dot(_unit(v1), _unit(v2)), -1.0, 1.0))
    return math.degrees(math.acos(cos))


def _as_points(points) -> np.ndarray:
    arr = np.asarray(points, dtype=float)
    if arr.shape != (21, 3):
        raise ValueError(f"expected 21x3 landmarks, got {arr.shape}")
    if not np.all(np.isfinite(arr)):
        raise ValueError("landmarks contain NaN or infinity")
    return arr


@dataclass
class HandMeasure:
    """Everything we know about one hand after normalisation."""

    handed: str
    local: np.ndarray            # 21x3, hand frame, palm length = 1
    up: np.ndarray               # hand's y-axis in camera space
    normal: np.ndarray           # palm normal in camera space
    bends: dict = field(default_factory=dict)       # finger -> (b1, b2, b3) degrees
    dist: dict = field(default_factory=dict)        # named distances, palm units
    angles: dict = field(default_factory=dict)      # named angles, degrees
    thumb_dir: np.ndarray = field(default_factory=lambda: np.zeros(3))  # camera space
    image: np.ndarray | None = None                 # 21x3 image points (pixels)

    def vector(self) -> np.ndarray:
        bend_list = [b for name in FINGER_NAMES for b in self.bends[name]]
        dist_list = [self.dist[k] for k in DIST_KEYS]
        return np.concatenate([
            self.local.reshape(-1),
            self.up,
            self.normal,
            np.radians(bend_list) / math.pi,
            np.asarray(dist_list),
        ])


DIST_KEYS = [
    "thumb_index", "thumb_middle", "thumb_ring", "thumb_pinky",
    "index_middle", "middle_ring", "ring_pinky",
    "wrist_thumb", "wrist_index", "wrist_middle", "wrist_ring", "wrist_pinky",
]


def measure_hand(hand: dict, use_world: bool = True) -> HandMeasure:
    """hand = {"handed": "Left"|"Right", "lm": 21x3 image pts, "wlm": 21x3 world pts}.

    use_world=False measures the hand from image points only. Moving-sign
    features use that, so clips imported from datasets without world
    landmarks are measured the same way as live clips.
    """
    handed = str(hand.get("handed") or "Right")
    raw = (hand.get("wlm") or hand.get("lm")) if use_world else (hand.get("lm") or hand.get("wlm"))
    if raw is None:
        raise ValueError("hand has no landmarks")
    p = _as_points(raw).copy()
    if handed == "Left":
        p[:, 0] *= -1.0          # mirror so every hand is treated as a right hand

    y_axis = p[9] - p[WRIST]
    palm = float(np.linalg.norm(y_axis))
    if palm < 1e-9:
        raise ValueError("degenerate hand (zero palm length)")
    y_axis = y_axis / palm
    across = p[5] - p[17]
    x_axis = _unit(across - np.dot(across, y_axis) * y_axis)
    z_axis = np.cross(x_axis, y_axis)
    rot = np.stack([x_axis, y_axis, z_axis])

    local = (p - p[WRIST]) @ rot.T / palm

    bends = {}
    for name, (a, b, c, d) in FINGERS.items():
        bends[name] = (
            _bend(p[WRIST], p[a], p[b]),
            _bend(p[a], p[b], p[c]),
            _bend(p[b], p[c], p[d]),
        )

    def d(i, j):
        return float(np.linalg.norm(local[i] - local[j]))

    dist = {
        "thumb_index": d(4, 8), "thumb_middle": d(4, 12),
        "thumb_ring": d(4, 16), "thumb_pinky": d(4, 20),
        "index_middle": d(8, 12), "middle_ring": d(12, 16), "ring_pinky": d(16, 20),
        "wrist_thumb": d(0, 4), "wrist_index": d(0, 8), "wrist_middle": d(0, 12),
        "wrist_ring": d(0, 16), "wrist_pinky": d(0, 20),
        # extra measures used only by the starter-gesture rules
        "thumb_tip_index_mcp": d(4, 5), "thumb_tip_middle_mcp": d(4, 9),
        "thumb_tip_pinky_mcp": d(4, 17),
    }

    def finger_dir(name):
        a, _, _, tip = FINGERS[name]
        return local[tip] - local[a]

    angles = {
        "index_pinky": _angle_between(finger_dir("index"), finger_dir("pinky")),
        "index_middle": _angle_between(finger_dir("index"), finger_dir("middle")),
        "thumb_index": _angle_between(local[4] - local[2], finger_dir("index")),
    }

    image = None
    if hand.get("lm") is not None:
        image = _as_points(hand["lm"])

    return HandMeasure(
        handed=handed, local=local, up=y_axis, normal=z_axis, bends=bends,
        dist=dist, angles=angles, thumb_dir=_unit(p[4] - p[2]), image=image,
    )


def order_hands(hands: list[dict]) -> list[dict]:
    """Primary hand first. With two hands the right hand is primary."""
    hands = [h for h in hands if h and (h.get("wlm") or h.get("lm"))]
    if len(hands) <= 1:
        return hands
    a, b = hands[0], hands[1]
    if a.get("handed") == "Left" and b.get("handed") == "Right":
        a, b = b, a
    return [a, b]


def features_from_measures(measures: list[HandMeasure]) -> np.ndarray:
    vec = np.zeros(FEATURE_DIM, dtype=float)
    if not measures:
        return vec
    vec[:PER_HAND_DIM] = measures[0].vector()
    vec[PER_HAND_DIM * 2] = 1.0
    if len(measures) > 1:
        a, b = measures[0], measures[1]
        vec[PER_HAND_DIM:PER_HAND_DIM * 2] = b.vector()
        vec[PER_HAND_DIM * 2 + 1] = 1.0
        if a.image is not None and b.image is not None:
            palm_a = float(np.linalg.norm(a.image[9, :2] - a.image[0, :2])) or 1.0
            palm_b = float(np.linalg.norm(b.image[9, :2] - b.image[0, :2]))
            off = (b.image[0, :2] - a.image[0, :2]) / palm_a
            vec[-3:-1] = np.clip(off, -10, 10)
            vec[-1] = min(palm_b / palm_a, 5.0)
    return vec


def extract(hands: list[dict]) -> tuple[np.ndarray, list[HandMeasure]]:
    """Main entry point: list of hand dicts -> (feature vector, measures)."""
    measures = [measure_hand(h) for h in order_hands(hands)[:2]]
    return features_from_measures(measures), measures
