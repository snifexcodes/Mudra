"""
Sign-language packs.

Every sign language is its own pack, in packs/<code>/:
    pack.json            name, region, spoken language, which signs move
    samples.jsonl        recordings (see dataset.py)
    model_still.joblib   model for hand shapes held still     (+ report_still.json)
    model_moving.joblib  model for signs that move             (+ report_moving.json)

Indian (ISL), American (ASL) and British (BSL) Sign Language are built in.
Anyone can add another sign language, and packs can be exported as a .zip
and imported on another computer, so recordings can be shared and pooled.
"""

from __future__ import annotations

import io
import json
import re
import shutil
import threading
import zipfile
from pathlib import Path

from . import rules
from .dataset import Dataset, clean_label
from .model import Recognizer

BUILTIN = {
    "isl": {
        "name": "Indian Sign Language", "short": "ISL", "region": "India",
        "spoken": "en-IN", "two_hands": True, "starter": ["HELLO", "GOOD"],
        "about": "Fingerspelling uses both hands. Most word signs move.",
    },
    "asl": {
        "name": "American Sign Language", "short": "ASL", "region": "USA and Canada",
        "spoken": "en-US", "two_hands": False, "starter": "all",
        "about": "One-handed fingerspelling. J and Z move; the other letters are held still.",
    },
    "bsl": {
        "name": "British Sign Language", "short": "BSL", "region": "United Kingdom",
        "spoken": "en-GB", "two_hands": True, "starter": ["HELLO", "GOOD"],
        "about": "Fingerspelling uses both hands. Most word signs move.",
    },
}
DEFAULT_PACK = "isl"
NO_SIGN = "NO SIGN"          # recorded "nothing" class: recognised but never added to the text
MAX_IMPORT_BYTES = 200 * 1024 * 1024


class PackError(ValueError):
    pass


def slug(text: str) -> str:
    s = re.sub(r"[^a-z0-9]+", "-", str(text or "").lower()).strip("-")
    return s[:16]


class Pack:
    def __init__(self, root: Path, code: str, meta: dict | None = None):
        self.code = code
        self.dir = Path(root) / code
        self.dir.mkdir(parents=True, exist_ok=True)
        self.meta_path = self.dir / "pack.json"
        self._lock = threading.Lock()
        if meta is not None:
            self.meta = meta
            self.save_meta()
        else:
            self.meta = json.loads(self.meta_path.read_text(encoding="utf-8"))
        self.meta.setdefault("signs", {})
        self.data = Dataset(self.dir / "samples.jsonl")
        self.still = Recognizer(self.dir, "still")
        self.moving = Recognizer(self.dir, "moving")

    def save_meta(self):
        with self._lock:
            tmp = self.meta_path.with_suffix(".tmp")
            tmp.write_text(json.dumps(self.meta, indent=2, ensure_ascii=False), encoding="utf-8")
            tmp.replace(self.meta_path)

    @property
    def builtin(self) -> bool:
        return self.code in BUILTIN

    def starter_labels(self) -> list[str]:
        s = self.meta.get("starter", [])
        if s == "all":
            return list(rules.LABELS)
        return [l for l in rules.LABELS if l in (s or [])]

    def sign_kind(self, label: str) -> str | None:
        return self.meta["signs"].get(label)

    def set_sign_kind(self, label: str, kind: str):
        label = clean_label(label)
        if kind not in ("still", "moving"):
            raise PackError("A sign is either still or moving.")
        recorded = {s["label"]: s["kind"] for s in self.data.summary()["signs"]}
        if label in recorded and recorded[label] != kind:
            raise PackError(f"{label} already has {recorded[label]} recordings. Delete them first to change its type.")
        self.meta["signs"][label] = kind
        self.save_meta()
        return label

    def forget_sign(self, label: str):
        self.meta["signs"].pop(label, None)
        self.save_meta()

    def rename_sign(self, old: str, new: str):
        if old in self.meta["signs"]:
            self.meta["signs"][new] = self.meta["signs"].pop(old)
            self.save_meta()

    def recognizer(self, kind: str) -> Recognizer:
        return self.moving if kind == "moving" else self.still

    def info(self, detail: bool = False) -> dict:
        out = {
            "code": self.code,
            "name": self.meta.get("name", self.code.upper()),
            "short": self.meta.get("short", self.code.upper()),
            "region": self.meta.get("region", ""),
            "spoken": self.meta.get("spoken", "en"),
            "two_hands": bool(self.meta.get("two_hands")),
            "about": self.meta.get("about", ""),
            "builtin": self.builtin,
            "models": {"still": self.still.info(), "moving": self.moving.info()},
            "model_errors": {k: r.load_error for k, r in (("still", self.still), ("moving", self.moving)) if r.load_error},
        }
        if detail:
            summary = self.data.summary()
            kinds = dict(self.meta["signs"])
            for s in summary["signs"]:
                kinds[s["label"]] = s["kind"]
            out["dataset"] = summary
            out["sign_kinds"] = kinds
            out["starter"] = [{"label": l, "how": rules.GUIDE[l]} for l in self.starter_labels()]
        else:
            summary = self.data.summary()
            out["sign_count"] = len(summary["signs"])
            out["samples"] = summary["total"]
        return out

    def export_zip(self) -> bytes:
        buf = io.BytesIO()
        with zipfile.ZipFile(buf, "w", zipfile.ZIP_DEFLATED) as z:
            meta = {k: v for k, v in self.meta.items()}
            meta["code"] = self.code
            meta["format"] = "mudra-pack/1"
            z.writestr("pack.json", json.dumps(meta, indent=2, ensure_ascii=False))
            if self.data.path.exists():
                z.write(self.data.path, "samples.jsonl")
        return buf.getvalue()


class PackManager:
    def __init__(self, root: Path, legacy_root: Path | None = None):
        self.root = Path(root)
        self.root.mkdir(parents=True, exist_ok=True)
        self._lock = threading.Lock()
        self.packs: dict[str, Pack] = {}
        for code, meta in BUILTIN.items():
            d = self.root / code
            if (d / "pack.json").exists():
                self.packs[code] = Pack(self.root, code)
            else:
                self.packs[code] = Pack(self.root, code, {**meta, "signs": {}})
        for d in sorted(self.root.iterdir()):
            if d.is_dir() and d.name not in self.packs and (d / "pack.json").exists():
                try:
                    self.packs[d.name] = Pack(self.root, d.name)
                except (OSError, json.JSONDecodeError):
                    continue
        if legacy_root is not None:
            self._migrate_legacy(Path(legacy_root))

    def _migrate_legacy(self, legacy: Path):
        """Version 1 kept one dataset in data/ and one model in models/: move them into ISL."""
        old_samples = legacy / "data" / "samples.jsonl"
        target = self.packs[DEFAULT_PACK]
        if old_samples.exists() and not target.data.path.exists():
            shutil.move(str(old_samples), str(target.data.path))
            for name, new in (("model.joblib", "model_still.joblib"), ("report.json", "report_still.json")):
                src = legacy / "models" / name
                if src.exists():
                    shutil.move(str(src), str(target.dir / new))
            self.packs[DEFAULT_PACK] = Pack(self.root, DEFAULT_PACK)

    def get(self, code: str) -> Pack:
        pack = self.packs.get(str(code or "").lower())
        if pack is None:
            raise PackError(f"There is no sign-language pack called {code!r}.")
        return pack

    def list(self) -> list[dict]:
        order = list(BUILTIN) + sorted(c for c in self.packs if c not in BUILTIN)
        return [self.packs[c].info() for c in order]

    def create(self, name: str, short: str = "", region: str = "", spoken: str = "en",
               two_hands: bool = True, code: str = "") -> Pack:
        name = re.sub(r"\s+", " ", str(name or "")).strip()
        if not name:
            raise PackError("Give the sign language a name.")
        if len(name) > 60:
            raise PackError("Keep the name under 60 characters.")
        short = re.sub(r"\s+", "", str(short or "")).upper()[:8] or "".join(w[0] for w in name.split()[:4]).upper()
        base = slug(code or short or name) or "pack"
        with self._lock:
            code = base
            n = 2
            while code in self.packs:
                code = f"{base}-{n}"
                n += 1
            meta = {
                "name": name, "short": short, "region": str(region or "")[:60],
                "spoken": re.sub(r"[^A-Za-z0-9-]", "", str(spoken or "en"))[:20] or "en",
                "two_hands": bool(two_hands), "starter": [], "about": "", "signs": {},
            }
            pack = Pack(self.root, code, meta)
            self.packs[code] = pack
        return pack

    def delete(self, code: str):
        pack = self.get(code)
        if pack.builtin:
            raise PackError("Built-in packs can't be deleted. Delete their signs instead.")
        with self._lock:
            self.packs.pop(pack.code, None)
            shutil.rmtree(pack.dir, ignore_errors=True)

    def import_zip(self, raw: bytes) -> tuple[Pack, int, bool]:
        """Returns (pack, records added, merged into an existing pack?)."""
        if len(raw) > MAX_IMPORT_BYTES:
            raise PackError("That file is too big to import (over 200 MB).")
        try:
            z = zipfile.ZipFile(io.BytesIO(raw))
        except zipfile.BadZipFile:
            raise PackError("That isn't a Mudra pack. Choose a .zip exported from Mudra.")
        with z:
            names = set(z.namelist())
            if "pack.json" not in names:
                raise PackError("That zip has no pack.json, so it isn't a Mudra pack.")
            try:
                meta = json.loads(z.read("pack.json").decode("utf-8"))
            except (UnicodeDecodeError, json.JSONDecodeError):
                raise PackError("The pack.json inside that zip is damaged.")
            records = []
            if "samples.jsonl" in names:
                for line in z.read("samples.jsonl").decode("utf-8", errors="replace").splitlines():
                    line = line.strip()
                    if not line:
                        continue
                    try:
                        records.append(json.loads(line))
                    except json.JSONDecodeError:
                        continue
        code = slug(meta.get("code") or meta.get("short") or meta.get("name"))
        merged = code in self.packs
        if merged:
            pack = self.packs[code]
        else:
            pack = self.create(
                meta.get("name") or code.upper() or "Imported pack", meta.get("short", ""),
                meta.get("region", ""), meta.get("spoken", "en"), bool(meta.get("two_hands", True)), code,
            )
        for label, kind in (meta.get("signs") or {}).items():
            try:
                if kind in ("still", "moving") and pack.sign_kind(clean_label(label)) is None:
                    pack.meta["signs"][clean_label(label)] = kind
            except ValueError:
                continue
        pack.save_meta()
        existing = {s["label"]: s["kind"] for s in pack.data.summary()["signs"]}

        def fits(r):
            try:
                label = clean_label(r.get("label"))
            except (ValueError, AttributeError):
                return False
            kind = "moving" if r.get("kind") == "clip" else "still"
            return existing.get(label, pack.sign_kind(label) or kind) == kind

        added = pack.data.append_records([r for r in records if isinstance(r, dict) and fits(r)])
        return pack, added, merged
