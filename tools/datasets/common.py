"""Shared helpers for the dataset importers."""

from __future__ import annotations

import hashlib
import re
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from core.dataset import clean_label  # noqa: E402
from core.packs import PackError, PackManager  # noqa: E402

VIDEO_EXT = {".mp4", ".mov", ".avi", ".mkv", ".webm", ".m4v"}


def label_from_folder(name: str) -> str | None:
    """'12. Thank you' -> 'THANK YOU'. Returns None if nothing usable is left."""
    text = re.sub(r"^\s*\d+\s*[.)_\-:]*\s*", "", str(name))
    text = re.sub(r"[_]+", " ", text).strip()
    try:
        return clean_label(text)
    except ValueError:
        return None


def session_id(*parts: str) -> str:
    return "imp-" + hashlib.sha1("|".join(parts).encode("utf-8")).hexdigest()[:12]


def open_pack(code: str):
    try:
        return PackManager(ROOT / "packs", legacy_root=ROOT).get(code)
    except PackError as exc:
        sys.exit(f"{exc} Create it in the app first, or use isl, asl or bsl.")


def pick_labels(counts: dict[str, int], only: str | None, top: int | None) -> set[str]:
    if only:
        wanted = set()
        for part in only.split(","):
            label = label_from_folder(part)
            if label:
                wanted.add(label)
        return {l for l in counts if l in wanted}
    labels = sorted(counts, key=lambda l: (-counts[l], l))
    return set(labels[:top] if top else labels)


def train_moving(pack):
    from core import model
    print("\nTraining the moving-sign model (this can take a while for big datasets)...")
    try:
        bundle, report = model.train(pack.data.load(), kind="moving",
                                     progress=lambda stage, f: print(f"  {f * 100:5.1f}%  {stage}"))
    except model.TrainingError as exc:
        print(f"Couldn't train: {exc}")
        return
    pack.moving.save(bundle, report)
    print(f"Done. {report['best_model']}: {report['accuracy'] * 100:.1f}% accuracy, "
          f"macro F1 {report['macro_f1'] * 100:.1f}% on {report['dataset']['signs']} signs.")
