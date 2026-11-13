"""
A tiny kinematic hand model that produces MediaPipe-shaped landmarks.

Used only by the tests, to check the feature code, the starter-gesture rules
and the training pipeline without a webcam. It is NOT used to train the real
model: real hands are messier than this.
"""

from __future__ import annotations

import math

import numpy as np

# Hand-local frame: x = thumb side, y = along the fingers, z = out of the palm.
MCP = {
    "index": np.array([0.32, 0.93, 0.0]),
    "middle": np.array([0.08, 1.00, 0.0]),
    "ring": np.array([-0.14, 0.95, 0.0]),
    "pinky": np.array([-0.34, 0.85, 0.0]),
}
BONES = {
    "index": (0.44, 0.26, 0.21),
    "middle": (0.48, 0.30, 0.23),
    "ring": (0.45, 0.28, 0.22),
    "pinky": (0.36, 0.21, 0.19),
}
SPLAY_TOGETHER = {"index": 3.0, "middle": 0.0, "ring": -3.0, "pinky": -6.0}
SPLAY_OPEN = {"index": 14.0, "middle": 2.0, "ring": -12.0, "pinky": -26.0}
CMC = np.array([0.22, 0.18, 0.02])
Z = np.array([0.0, 0.0, 1.0])

STRAIGHT = (5, 5, 4)
CURLED = (78, 100, 65)
HALF = (35, 55, 25)


def _u(v):
    return v / np.linalg.norm(v)


def finger(name, flex, splay_deg):
    s = math.radians(splay_deg)
    d0 = np.array([math.sin(s), math.cos(s), 0.0])
    pts = [MCP[name]]
    phi = 0.0
    for bone_len, f in zip(BONES[name], flex):
        phi += math.radians(f)
        d = math.cos(phi) * d0 + math.sin(phi) * Z
        pts.append(pts[-1] + bone_len * d)
    return pts  # MCP, PIP, DIP, TIP


def thumb_dirs(dirs, lengths=(0.42, 0.30, 0.25)):
    pts = [CMC]
    for d, ln in zip(dirs, lengths):
        pts.append(pts[-1] + ln * _u(np.array(d, float)))
    return pts  # CMC, MCP, IP, TIP


def thumb_to(target, bulge=0.12):
    t = np.asarray(target, float)
    side = np.array([1.0, 0.0, 0.3])
    mcp = CMC + 0.42 * (t - CMC) + bulge * side
    ip = CMC + 0.74 * (t - CMC) + 0.6 * bulge * side
    return [CMC, mcp, ip, t]


THUMB = {
    "out": [(0.75, 0.65, 0.05), (0.8, 0.6, 0.0), (0.8, 0.6, -0.02)],
    "across": [(0.3, 0.6, 0.5), (-0.6, 0.3, 0.6), (-0.9, 0.05, 0.3)],
    "side": [(0.5, 0.75, 0.3), (0.1, 1.0, 0.3), (0.0, 1.0, 0.4)],
    "up": [(0.75, 0.55, 0.2), (0.9, 0.42, 0.08), (0.92, 0.4, 0.0)],
}


def build(spec):
    fingers = {}
    for name in ("index", "middle", "ring", "pinky"):
        flex = spec.get(name, CURLED)
        splay_table = SPLAY_OPEN if spec.get("open") else SPLAY_TOGETHER
        splay = spec.get("splay", {}).get(name, splay_table[name])
        fingers[name] = finger(name, flex, splay)

    thumb = spec.get("thumb", "across")
    if thumb == "touch_index":
        t = thumb_to(fingers["index"][3] + np.array([0.03, 0.0, 0.03]))
    elif thumb == "touch_o":
        target = (fingers["index"][3] + fingers["middle"][3]) / 2 + np.array([0.04, -0.02, 0.05])
        t = thumb_to(target)
    else:
        t = thumb_dirs(THUMB[thumb])

    pts = [np.zeros(3)] + t
    for name in ("index", "middle", "ring", "pinky"):
        pts += fingers[name]
    return np.array(pts)


SPECS = {
    "A": dict(thumb="side"),
    "B": dict(thumb="across", index=STRAIGHT, middle=STRAIGHT, ring=STRAIGHT, pinky=STRAIGHT),
    "D": dict(thumb="across", index=STRAIGHT),
    "F": dict(thumb="touch_index", index=(45, 70, 35), middle=STRAIGHT, ring=STRAIGHT,
              pinky=STRAIGHT, open=True),
    "I": dict(thumb="across", pinky=STRAIGHT),
    "L": dict(thumb="out", index=STRAIGHT),
    "O": dict(thumb="touch_o", index=HALF, middle=HALF, ring=(45, 70, 35), pinky=(50, 75, 40)),
    "U": dict(thumb="across", index=STRAIGHT, middle=STRAIGHT,
              splay={"index": 1.0, "middle": -1.0}),
    "V": dict(thumb="across", index=STRAIGHT, middle=STRAIGHT,
              splay={"index": 16.0, "middle": -12.0}),
    "W": dict(thumb="across", index=STRAIGHT, middle=STRAIGHT, ring=STRAIGHT, open=True),
    "Y": dict(thumb="out", pinky=STRAIGHT, splay={"pinky": -20.0}),
    "HELLO": dict(thumb="out", index=STRAIGHT, middle=STRAIGHT, ring=STRAIGHT,
                  pinky=STRAIGHT, open=True),
    "GOOD": dict(thumb="up"),
    "I LOVE YOU": dict(thumb="out", index=STRAIGHT, pinky=STRAIGHT, open=True),
}

# Upright hand facing the camera: local x -> camera +x, local y -> camera -y
# (up), local z -> camera -z (towards the camera).
POSE_UPRIGHT = np.array([[1, 0, 0], [0, -1, 0], [0, 0, -1]], float)
# Thumbs-up pose: fingers point sideways, thumb side points up.
POSE_SIDEWAYS = np.array([[0, 1, 0], [-1, 0, 0], [0, 0, 1]], float)


def _rot(axis, deg):
    a = math.radians(deg)
    c, s = math.cos(a), math.sin(a)
    if axis == "x":
        return np.array([[1, 0, 0], [0, c, -s], [0, s, c]])
    if axis == "y":
        return np.array([[c, 0, s], [0, 1, 0], [-s, 0, c]])
    return np.array([[c, -s, 0], [s, c, 0], [0, 0, 1]])


def _vary(spec, rng, pose_noise):
    spec = dict(spec)
    if pose_noise:
        # per-sample variation in finger flexion, so each "recording" differs a bit
        for name in ("index", "middle", "ring", "pinky"):
            flex = spec.get(name, CURLED)
            spec[name] = tuple(max(0.0, f + rng.normal(0, pose_noise)) for f in flex)
    return spec


def _random_pose(rng, tilt, roll):
    R = (_rot("z", rng.uniform(-roll, roll)) @ _rot("x", rng.uniform(-tilt, tilt))
         @ _rot("y", rng.uniform(-tilt, tilt)))
    palm_m = rng.uniform(0.075, 0.095)
    scale_px = rng.uniform(1800, 4200)
    center = np.array([rng.uniform(300, 900), rng.uniform(200, 500), 0.0])
    return R, palm_m, scale_px, center


def _render(local, pose, R, palm_m, scale_px, center, handed, jitter, rng):
    cam = (R @ pose @ local.T).T
    cam += rng.normal(0, jitter, cam.shape)
    world = (cam - cam.mean(axis=0)) * palm_m
    image = cam * palm_m * scale_px + center
    if handed == "Left":
        world[:, 0] *= -1
        image[:, 0] = 1280 - image[:, 0]
    return {"handed": handed, "score": 0.97, "lm": image.tolist(), "wlm": world.tolist()}


def sample(label, rng, handed="Right", jitter=0.015, tilt=18.0, roll=22.0, pose_noise=0.0):
    """Returns a hand dict like the browser sends: handed, lm (pixels), wlm (metres)."""
    local = build(_vary(SPECS[label], rng, pose_noise))
    pose = POSE_SIDEWAYS if label == "GOOD" else POSE_UPRIGHT
    R, palm_m, scale_px, center = _random_pose(rng, tilt, roll)
    return _render(local, pose, R, palm_m, scale_px, center, handed, jitter, rng)


# ---------- moving signs ----------
# label -> (hand shape, path of the wrist in palm lengths over u = 0..1, finger motion)
MOVES = {
    "WAVE": ("HELLO", lambda u: (1.6 * math.sin(2 * math.pi * 1.5 * u), 0.0), None),
    "DOWN": ("B", lambda u: (0.0, 3.0 * u), None),
    "CIRCLE": ("D", lambda u: (1.4 * math.cos(2 * math.pi * u) - 1.4, 1.4 * math.sin(2 * math.pi * u)), None),
    "YES": ("A", lambda u: (0.0, 0.9 * math.sin(2 * math.pi * 2 * u)), None),
    "COME": ("D", lambda u: (0.0, 0.0), lambda u: 45 + 45 * math.sin(2 * math.pi * 2 * u)),
}


def clip(label, rng, handed="Right", fps=15.0, still_pad=0.3, dur=None, jitter=0.01,
         with_pose=True, pose_noise=4.0):
    """A moving sign as the browser sends it: [{t, hands, pose}], with still frames around it."""
    shape, path, finger = MOVES[label]
    dur = dur or rng.uniform(0.9, 1.4)
    amp = rng.uniform(0.85, 1.15)
    R, palm_m, scale_px, center = _random_pose(rng, 10.0, 12.0)
    palm_px = palm_m * scale_px
    nose = np.array([center[0] - 1.2 * palm_px, center[1] - 1.6 * palm_px, 0.0])
    ls = nose + np.array([-1.4 * palm_px, 1.3 * palm_px, 0.0])
    rs = nose + np.array([1.4 * palm_px, 1.3 * palm_px, 0.0])
    base_spec = _vary(SPECS[shape], rng, pose_noise)
    frames = []
    n = int((dur + 2 * still_pad) * fps)
    for k in range(n):
        t = k / fps
        u = min(1.0, max(0.0, (t - still_pad) / dur))
        dx, dy = path(u)
        spec = dict(base_spec)
        if finger is not None:
            f = finger(u) if 0 < u < 1 else 45.0
            spec["index"] = (f, f * 1.3, f * 0.7)
        local = build(spec)
        c = center + np.array([dx * amp * palm_px, dy * amp * palm_px, 0.0])
        hand = _render(local, POSE_UPRIGHT, R, palm_m, scale_px, c, handed, jitter, rng)
        pose = None
        if with_pose:
            pts = [nose.copy(), ls.copy(), rs.copy()]
            if handed == "Left":
                for p in pts:
                    p[0] = 1280 - p[0]
                pts = [pts[0], pts[2], pts[1]]
            pose = [p.tolist() for p in pts]
        frames.append({"t": round(t, 4), "hands": [hand], "pose": pose})
    return frames
