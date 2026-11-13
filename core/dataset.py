"""
Stores recorded training samples as JSON lines (one frame per line).

Raw landmarks are stored, not feature vectors, so the feature code can change
later and the model can simply be retrained from the same recordings.

Each line:
    {"label": "HELLO", "session": "a1b2c3", "ts": 1731500000.0, "hands": [...]}

A "session" is one press of the Record button. Frames from the same session
look almost identical, so the evaluation keeps whole sessions together when
it splits train and test data (see model.py).
"""

from __future__ import annotations

import json
import re
import threading
import time
from collections import defaultdict
from pathlib import Path

MAX_LABEL_LEN = 24
_lock = threading.Lock()


def clean_label(label: str) -> str:
    text = re.sub(r"\s+", " ", str(label or "")).strip().upper()
    if not text:
        raise ValueError("Sign name can't be empty.")
    if len(text) > MAX_LABEL_LEN:
        raise ValueError(f"Keep sign names under {MAX_LABEL_LEN} characters.")
    if any(ch in text for ch in '<>"\\'):
        raise ValueError("Sign names can't contain < > \" or \\.")
    return text


def _valid_hand(hand) -> bool:
    try:
        pts = hand.get("wlm") or hand.get("lm")
        return len(pts) == 21 and all(len(p) == 3 for p in pts)
    except (AttributeError, TypeError):
        return False


class Dataset:
    def __init__(self, path: Path):
        self.path = Path(path)
        self.path.parent.mkdir(parents=True, exist_ok=True)

    def add_recording(self, label: str, session: str, frames: list[dict]) -> int:
        label = clean_label(label)
        session = re.sub(r"[^A-Za-z0-9_-]", "", str(session or ""))[:40] or f"s{int(time.time() * 1000)}"
        lines = []
        now = time.time()
        for frame in frames or []:
            hands = [h for h in (frame or {}).get("hands", []) if _valid_hand(h)]
            if not hands:
                continue
            slim = [
                {
                    "handed": h.get("handed", "Right"),
                    "lm": [[round(float(v), 2) for v in p] for p in h["lm"]],
                    "wlm": [[round(float(v), 5) for v in p] for p in (h.get("wlm") or h["lm"])],
                }
                for h in hands[:2]
            ]
            lines.append(json.dumps({"label": label, "session": session, "ts": now, "hands": slim}))
        if lines:
            with _lock, self.path.open("a", encoding="utf-8") as f:
                f.write("\n".join(lines) + "\n")
        return len(lines)

    def load(self) -> list[dict]:
        if not self.path.exists():
            return []
        out = []
        with _lock, self.path.open("r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                try:
                    rec = json.loads(line)
                except json.JSONDecodeError:
                    continue
                if rec.get("label") and rec.get("hands"):
                    out.append(rec)
        return out

    def summary(self) -> dict:
        counts: dict[str, int] = defaultdict(int)
        sessions: dict[str, list[str]] = defaultdict(list)
        last_session = None
        last_ts = -1.0
        for rec in self.load():
            label = rec["label"]
            counts[label] += 1
            if rec["session"] not in sessions[label]:
                sessions[label].append(rec["session"])
            if rec.get("ts", 0) >= last_ts:
                last_ts = rec.get("ts", 0)
                last_session = {"label": label, "session": rec["session"]}
        signs = [
            {"label": k, "samples": counts[k], "sessions": len(sessions[k])}
            for k in sorted(counts)
        ]
        return {
            "signs": signs,
            "total": sum(counts.values()),
            "last": last_session,
        }

    def _rewrite(self, keep) -> int:
        records = self.load()
        kept = [r for r in records if keep(r)]
        with _lock:
            tmp = self.path.with_suffix(".tmp")
            with tmp.open("w", encoding="utf-8") as f:
                for r in kept:
                    f.write(json.dumps(r) + "\n")
            tmp.replace(self.path)
        return len(records) - len(kept)

    def delete_label(self, label: str) -> int:
        label = clean_label(label)
        return self._rewrite(lambda r: r["label"] != label)

    def delete_session(self, session: str) -> int:
        return self._rewrite(lambda r: r["session"] != session)

    def rename_label(self, old: str, new: str) -> int:
        old, new = clean_label(old), clean_label(new)
        records = self.load()
        n = 0
        for r in records:
            if r["label"] == old:
                r["label"] = new
                n += 1
        with _lock:
            tmp = self.path.with_suffix(".tmp")
            with tmp.open("w", encoding="utf-8") as f:
                for r in records:
                    f.write(json.dumps(r) + "\n")
            tmp.replace(self.path)
        return n
