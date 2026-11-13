"""Dataset importers, tested with synthetic data (no real dataset or MediaPipe needed)."""

import sys
from pathlib import Path

import numpy as np
import pytest

from core import packs
from tests import synth

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "tools" / "datasets"))
import common  # noqa: E402
import import_kaggle_asl  # noqa: E402
import import_videos  # noqa: E402


def test_label_from_folder():
    assert common.label_from_folder("12. Thank you") == "THANK YOU"
    assert common.label_from_folder("3_good_morning") == "GOOD MORNING"
    assert common.label_from_folder("Hello") == "HELLO"
    assert common.label_from_folder("42.") is None


def test_pick_labels():
    counts = {"A": 5, "B": 9, "C": 1}
    assert common.pick_labels(counts, None, 2) == {"B", "A"}
    assert common.pick_labels(counts, "a, c, z", None) == {"A", "C"}


class FakeExtractor:
    def __init__(self):
        self.rng = np.random.default_rng(0)
        self.calls = 0

    def frames(self, path, fps):
        self.calls += 1
        move = {"WAVE HELLO": "WAVE", "GO DOWN": "DOWN"}[common.label_from_folder(path.parent.name)]
        return synth.clip(move, self.rng)


def test_import_videos(tmp_path, monkeypatch):
    ds_root = tmp_path / "INCLUDE"
    for folder in ("Greetings/1. Wave hello", "Directions/7. Go down"):
        d = ds_root / folder
        d.mkdir(parents=True)
        for i in range(6):
            (d / f"clip{i}.MOV").write_bytes(b"")
    (ds_root / "Greetings" / "notes.txt").write_text("ignore me")
    found = import_videos.find_videos(ds_root)
    assert {k: len(v) for k, v in found.items()} == {"WAVE HELLO": 6, "GO DOWN": 6}

    pm = packs.PackManager(tmp_path / "packs")
    monkeypatch.setattr(import_videos, "open_pack", lambda code: pm.get(code))
    fake = FakeExtractor()
    import_videos.main(["--videos", str(ds_root), "--pack", "isl", "--train"], extractor=fake)
    isl = pm.get("isl")
    kinds = {s["label"]: (s["kind"], s["samples"]) for s in isl.data.summary()["signs"]}
    assert kinds == {"WAVE HELLO": ("moving", 6), "GO DOWN": ("moving", 6)}
    assert isl.moving.ready

    import_videos.main(["--videos", str(ds_root), "--pack", "isl"], extractor=fake)
    assert fake.calls == 12                           # second run skips imported videos


def _kaggle_dataset(root: Path, rng):
    import pandas as pd
    rows = []
    (root / "train_landmark_files").mkdir(parents=True)
    seq = 1000
    for sign, move in (("wave", "WAVE"), ("down", "DOWN"), ("yes", "YES")):
        for signer in (16069, 2044):
            for _ in range(5):
                frames = synth.clip(move, rng, handed="Right")
                recs = []
                for i, f in enumerate(frames):
                    lm = np.asarray(f["hands"][0]["lm"])
                    for j, p in enumerate(lm):
                        recs.append((i, "right_hand", j, p[0] / 1000, p[1] / 1000, p[2] / 1000))
                    for j in range(21):
                        recs.append((i, "left_hand", j, np.nan, np.nan, np.nan))
                    pose = {0: f["pose"][0], 11: f["pose"][1], 12: f["pose"][2]}
                    for j in range(33):
                        x, y, z = pose.get(j, (np.nan, np.nan, np.nan))
                        recs.append((i, "pose", j, x / 1000, y / 1000, z / 1000))
                df = pd.DataFrame(recs, columns=["frame", "type", "landmark_index", "x", "y", "z"])
                rel = f"train_landmark_files/{signer}/{seq}.parquet"
                (root / rel).parent.mkdir(parents=True, exist_ok=True)
                df.to_parquet(root / rel)
                rows.append((rel, signer, seq, sign))
                seq += 1
    pd.DataFrame(rows, columns=["path", "participant_id", "sequence_id", "sign"]).to_csv(root / "train.csv", index=False)


def test_import_kaggle_asl(tmp_path, monkeypatch):
    pytest.importorskip("pyarrow")
    root = tmp_path / "asl-signs"
    _kaggle_dataset(root, np.random.default_rng(3))
    pm = packs.PackManager(tmp_path / "packs")
    monkeypatch.setattr(import_kaggle_asl, "open_pack", lambda code: pm.get(code))
    import_kaggle_asl.main(["--data", str(root), "--top", "3", "--train"])
    asl = pm.get("asl")
    summary = {s["label"]: (s["samples"], s["sessions"]) for s in asl.data.summary()["signs"]}
    assert summary == {"WAVE": (10, 2), "DOWN": (10, 2), "YES": (10, 2)}
    assert asl.moving.ready
    assert asl.moving.report()["evaluation"]["method"] == "grouped"   # held out by signer
    import_kaggle_asl.main(["--data", str(root)])
    assert asl.data.summary()["total"] == 30                            # nothing imported twice
