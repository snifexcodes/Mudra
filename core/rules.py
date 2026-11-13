"""
Starter gestures: a small rule-based recogniser that works before you have
trained anything.

Each gesture is described by which fingers are out and a few extra checks
(fingers spread or together, thumb touching index, thumb pointing up...).
Every check gives a fuzzy score between 0 and 1, and a gesture's score is
its weakest check. The best gesture wins if its score is high enough.

These are hand-written rules, so they are less accurate than a model trained
on your own hand. They are there so the app does something useful on first
launch, and as a baseline to compare the trained model against.
"""

from __future__ import annotations

import math

from .features import HandMeasure


def _sig(x: float) -> float:
    if x < -40:
        return 0.0
    if x > 40:
        return 1.0
    return 1.0 / (1.0 + math.exp(-x))


def finger_states(m: HandMeasure) -> dict[str, float]:
    """How extended each finger is, 0 (curled) .. 1 (straight)."""
    states = {}
    for name in ("index", "middle", "ring", "pinky"):
        mcp, pip, dip = m.bends[name]
        curl = pip + dip + 0.5 * mcp
        states[name] = _sig((85.0 - curl) / 14.0)
    # Thumb: "out" when its tip sticks out sideways past the index knuckle
    # (L, Y, open palm, thumbs up). Folded across the palm or resting against
    # the fist, the tip stays level with or inside the index knuckle.
    lateral = float(m.local[4, 0] - m.local[5, 0])
    reach = m.dist["thumb_tip_index_mcp"]
    states["thumb"] = max(_sig((lateral - 0.25) / 0.05), _sig((reach - 0.85) / 0.06))
    return states


def checks(m: HandMeasure) -> dict[str, float]:
    return {
        "spread": _sig((m.angles["index_pinky"] - 22.0) / 4.0),
        "v_spread": _sig((m.angles["index_middle"] - 13.0) / 2.5),
        "l_angle": _sig((m.angles["thumb_index"] - 38.0) / 6.0),
        "touch_index": _sig((0.32 - m.dist["thumb_index"]) / 0.05),
        "touch_middle": _sig((0.42 - m.dist["thumb_middle"]) / 0.06),
        "tips_out": _sig((m.dist["wrist_index"] - 1.05) / 0.06),
        # camera y axis points down, so "up" is negative y
        "thumb_up": _sig((-float(m.thumb_dir[1]) - 0.6) / 0.08),
        "hand_upright": _sig((-float(m.up[1]) - 0.55) / 0.08),
    }


# label, finger template (1 = out, 0 = curled, None = any), extra checks
# extra check names prefixed with "!" mean "must NOT be true"
GESTURES = [
    ("A", dict(thumb=0, index=0, middle=0, ring=0, pinky=0), ["!tips_out"]),
    ("B", dict(thumb=0, index=1, middle=1, ring=1, pinky=1), ["!spread"]),
    ("D", dict(thumb=0, index=1, middle=0, ring=0, pinky=0), ["!touch_index"]),
    ("F", dict(thumb=None, index=None, middle=1, ring=1, pinky=1), ["touch_index"]),
    ("I", dict(thumb=0, index=0, middle=0, ring=0, pinky=1), []),
    ("L", dict(thumb=1, index=1, middle=0, ring=0, pinky=0), ["l_angle"]),
    ("O", dict(thumb=None, index=None, middle=None, ring=0, pinky=0),
        ["touch_index", "touch_middle", "tips_out"]),
    ("U", dict(thumb=0, index=1, middle=1, ring=0, pinky=0), ["!v_spread"]),
    ("V", dict(thumb=0, index=1, middle=1, ring=0, pinky=0), ["v_spread"]),
    ("W", dict(thumb=0, index=1, middle=1, ring=1, pinky=0), []),
    ("Y", dict(thumb=1, index=0, middle=0, ring=0, pinky=1), []),
    ("HELLO", dict(thumb=1, index=1, middle=1, ring=1, pinky=1), ["spread"]),
    ("GOOD", dict(thumb=1, index=0, middle=0, ring=0, pinky=0), ["thumb_up", "!hand_upright"]),
    ("I LOVE YOU", dict(thumb=1, index=1, middle=0, ring=0, pinky=1), []),
]

# Shown in the web UI's gesture guide
GUIDE = {
    "A": "Fist, thumb resting against the side of your index finger",
    "B": "Four fingers straight up and together, thumb folded across the palm",
    "D": "Index finger up, other fingers and thumb curled",
    "F": "Thumb and index tips touching, other three fingers up",
    "I": "Only the little finger up",
    "L": "Thumb and index out in an L shape",
    "O": "All fingertips curved to meet the thumb, like an O",
    "U": "Index and middle up, pressed together",
    "V": "Index and middle up, spread apart",
    "W": "Index, middle and ring up",
    "Y": "Thumb and little finger out, others curled",
    "HELLO": "Open palm, all five fingers spread",
    "GOOD": "Thumbs up",
    "I LOVE YOU": "Thumb, index and little finger out",
}

LABELS = [g[0] for g in GESTURES]


def score_all(m: HandMeasure) -> list[tuple[str, float]]:
    states = finger_states(m)
    extra = checks(m)
    scored = []
    for label, template, needs in GESTURES:
        parts = []
        for finger, want in template.items():
            if want is None:
                continue
            s = states[finger]
            parts.append(s if want == 1 else 1.0 - s)
        for need in needs:
            if need.startswith("!"):
                parts.append(1.0 - extra[need[1:]])
            else:
                parts.append(extra[need])
        scored.append((label, min(parts) if parts else 0.0))
    scored.sort(key=lambda t: t[1], reverse=True)
    return scored


def predict(measures: list[HandMeasure], min_score: float = 0.5, allowed=None) -> dict:
    """Uses the primary hand only. `allowed` limits which gestures can win."""
    if not measures:
        return {"label": None, "confidence": 0.0, "top": []}
    scored = score_all(measures[0])
    if allowed is not None:
        scored = [s for s in scored if s[0] in allowed]
        if not scored:
            return {"label": None, "confidence": 0.0, "top": []}
    best_label, best = scored[0]
    top = [{"label": l, "confidence": round(s, 3)} for l, s in scored[:3]]
    if best < min_score:
        return {"label": None, "confidence": round(best, 3), "top": top}
    return {"label": best_label, "confidence": round(best, 3), "top": top}
