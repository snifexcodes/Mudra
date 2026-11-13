"""
Run with:  python -m pytest -q
Uses synthetic hands from tests/synth.py, so no webcam is needed.
"""

import io
import time
import uuid

import numpy as np
import pytest

from core import dataset as ds
from core import features, model, packs, rules, sequence
from tests import synth


def rng(seed=0):
    return np.random.default_rng(seed)


def test_feature_shape():
    vec, ms = features.extract([synth.sample("L", rng())])
    assert vec.shape == (features.FEATURE_DIM,)
    assert len(ms) == 1
    assert vec[features.PER_HAND_DIM * 2] == 1.0       # primary hand present
    assert vec[features.PER_HAND_DIM * 2 + 1] == 0.0   # no second hand


def test_translation_and_scale_invariance():
    h = synth.sample("V", rng(1), jitter=0)
    moved = dict(h)
    moved["wlm"] = (np.asarray(h["wlm"]) * 1.7 + np.array([0.3, -0.2, 0.1])).tolist()
    a, _ = features.extract([h])
    b, _ = features.extract([moved])
    assert np.allclose(a[:features.PER_HAND_DIM], b[:features.PER_HAND_DIM], atol=1e-6)


def test_left_hand_matches_mirrored_right_hand():
    right = synth.sample("Y", rng(2), jitter=0)
    left = dict(right)
    w = np.asarray(right["wlm"]).copy()
    w[:, 0] *= -1
    left.update(handed="Left", wlm=w.tolist())
    a, _ = features.extract([right])
    b, _ = features.extract([left])
    assert np.allclose(a[:features.PER_HAND_DIM], b[:features.PER_HAND_DIM], atol=1e-6)


def test_two_hands_fill_second_slot_with_right_hand_first():
    left = synth.sample("B", rng(3), handed="Left")
    right = synth.sample("D", rng(4), handed="Right")
    vec, ms = features.extract([left, right])
    assert ms[0].handed == "Right"
    assert vec[features.PER_HAND_DIM * 2 + 1] == 1.0
    assert np.any(vec[features.PER_HAND_DIM:features.PER_HAND_DIM * 2] != 0)


def test_rejects_bad_landmarks():
    with pytest.raises(ValueError):
        features.extract([{"handed": "Right", "lm": [[0, 0, 0]] * 20}])


def test_starter_rules_on_synthetic_hands():
    r = rng(5)
    total = hits = 0
    for label in synth.SPECS:
        for handed in ("Right", "Left"):
            for _ in range(25):
                _, ms = features.extract([synth.sample(label, r, handed=handed, pose_noise=5)])
                hits += rules.predict(ms)["label"] == label
                total += 1
    assert hits / total > 0.95, hits / total


def test_every_starter_gesture_has_a_guide_entry():
    assert set(rules.LABELS) == set(rules.GUIDE)


def make_records(labels, sessions=3, frames=40, seed=7):
    r = rng(seed)
    out = []
    for label in labels:
        for s in range(sessions):
            sid = uuid.uuid4().hex[:8]
            handed = "Left" if s == 1 else "Right"
            for _ in range(frames):
                out.append({"label": label, "session": sid,
                            "hands": [synth.sample(label, r, handed=handed, pose_noise=6)]})
    return out


def test_dataset_roundtrip(tmp_path):
    d = ds.Dataset(tmp_path / "samples.jsonl")
    frames = [{"hands": [synth.sample("L", rng(8))]} for _ in range(5)] + [{"hands": []}]
    assert d.add_recording(" hello  there ", "abc", frames) == 5
    s = d.summary()
    assert s["signs"] == [{"label": "HELLO THERE", "samples": 5, "sessions": 1, "kind": "still"}]
    assert s["last"]["session"] == "abc"
    assert d.rename_label("hello there", "HI") == 5
    assert d.delete_session("abc") == 5
    assert d.summary()["total"] == 0
    with pytest.raises(ValueError):
        ds.clean_label("   ")


def test_training_pipeline_grouped(tmp_path):
    labels = ["A", "B", "L", "V", "Y", "HELLO"]
    records = make_records(labels)
    bundle, report = model.train(records)
    assert report["evaluation"]["method"] == "grouped"
    assert report["labels"] == sorted(labels)
    assert report["accuracy"] > 0.9
    assert len(report["confusion"]) == len(labels)
    assert {m["name"] for m in report["models"]} == set(model.candidates())
    assert report["baseline_rules"]["signs"] == sorted(labels)

    rec = model.Recognizer(tmp_path)
    rec.save(bundle, report)
    rec2 = model.Recognizer(tmp_path)          # reload from disk
    assert rec2.ready
    vec, _ = features.extract([synth.sample("V", rng(99))])
    assert rec2.predict(vec)["label"] == "V"


def test_training_warns_with_single_recording():
    records = make_records(["A", "B"], sessions=1, frames=30)
    _, report = model.train(records)
    assert report["evaluation"]["method"] == "frames"
    assert report["warnings"]


def test_training_rejects_too_little_data():
    with pytest.raises(model.TrainingError):
        model.train(make_records(["A"], sessions=2, frames=30))
    with pytest.raises(model.TrainingError):
        model.train(make_records(["A", "B"], sessions=1, frames=5))


# ---------- moving signs ----------
def make_clips(labels, sessions=2, reps=6, seed=21):
    r = rng(seed)
    out = []
    for label in labels:
        for s in range(sessions):
            sid = uuid.uuid4().hex[:8]
            for i in range(reps):
                handed = "Left" if (s + i) % 3 == 0 else "Right"
                out.append({"label": label, "session": sid, "kind": "clip",
                            "frames": synth.clip(label, r, handed=handed, with_pose=(i % 4 != 0))})
    return out


def test_clip_features_shape_and_invariances():
    v = sequence.clip_features(synth.clip("WAVE", rng(1)))
    assert v.shape == (sequence.CLIP_DIM,)
    a = sequence.clip_features(synth.clip("DOWN", rng(2), still_pad=0.2, jitter=0))
    b = sequence.clip_features(synth.clip("DOWN", rng(2), still_pad=1.0, jitter=0))
    assert np.abs(a - b).mean() < 0.02            # still frames around the sign are trimmed
    left = sequence.clip_features(synth.clip("CIRCLE", rng(3), handed="Left", jitter=0))
    right = sequence.clip_features(synth.clip("CIRCLE", rng(3), handed="Right", jitter=0))
    assert np.allclose(left, right, atol=1e-6)    # left-handed signing = mirror image
    with pytest.raises(ValueError):
        sequence.clip_features([{"t": 0, "hands": []}])


def test_moving_model_trains_and_predicts(tmp_path):
    records = make_clips(list(synth.MOVES))
    records += make_records(["A", "B"], sessions=2, frames=25)   # still records are ignored
    bundle, report = model.train(records, kind="moving")
    assert report["kind"] == "moving" and report["evaluation"]["method"] == "grouped"
    assert report["labels"] == sorted(synth.MOVES)
    assert report["accuracy"] > 0.85, report["accuracy"]
    rec = model.Recognizer(tmp_path, "moving")
    rec.save(bundle, report)
    assert model.Recognizer(tmp_path, "moving").ready
    hits = sum(rec.predict(sequence.clip_features(synth.clip(l, rng(70 + i))))["label"] == l
               for i, l in enumerate(list(synth.MOVES) * 2))
    assert hits >= 9


def test_moving_model_needs_repetitions():
    with pytest.raises(model.TrainingError):
        model.train(make_clips(["WAVE", "DOWN"], sessions=1, reps=3), kind="moving")


# ---------- packs ----------
def test_pack_manager_builtins_custom_and_zip_roundtrip(tmp_path):
    pm = packs.PackManager(tmp_path / "packs")
    assert [p["code"] for p in pm.list()] == ["isl", "asl", "bsl"]
    assert pm.get("asl").starter_labels() == rules.LABELS
    assert pm.get("isl").starter_labels() == ["HELLO", "GOOD"]
    lsf = pm.create("Langue des signes française", "LSF", "France", "fr-FR", True)
    assert lsf.code == "lsf" and pm.create("Other LSF", "LSF").code == "lsf-2"
    lsf.data.add_recording("A", "s1", [{"hands": [synth.sample("A", rng(1))]} for _ in range(5)])
    lsf.data.add_clips("MERCI", "s2", [synth.clip("WAVE", rng(2))])
    lsf.set_sign_kind("BONJOUR", "moving")
    with pytest.raises(packs.PackError):
        lsf.set_sign_kind("A", "moving")              # A already has still recordings
    raw = lsf.export_zip()

    other = packs.PackManager(tmp_path / "elsewhere")
    pack, added, merged = other.import_zip(raw)
    assert pack.code == "lsf" and added == 6 and not merged
    assert pack.meta["name"] == "Langue des signes française"
    assert pack.meta["signs"]["BONJOUR"] == "moving"
    kinds = {s["label"]: s["kind"] for s in pack.data.summary()["signs"]}
    assert kinds == {"A": "still", "MERCI": "moving"}
    _, again, merged = other.import_zip(raw)           # same sessions are not added twice
    assert merged and again == 0
    with pytest.raises(packs.PackError):
        other.import_zip(b"not a zip")
    with pytest.raises(packs.PackError):
        other.delete("isl")
    other.delete("lsf")
    assert "lsf" not in [p["code"] for p in other.list()]


def test_legacy_data_moves_into_isl(tmp_path):
    (tmp_path / "data").mkdir()
    d = ds.Dataset(tmp_path / "data" / "samples.jsonl")
    d.add_recording("L", "old", [{"hands": [synth.sample("L", rng(3))]} for _ in range(4)])
    pm = packs.PackManager(tmp_path / "packs", legacy_root=tmp_path)
    assert pm.get("isl").data.summary()["total"] == 4
    assert not (tmp_path / "data" / "samples.jsonl").exists()


# ---------- API ----------
@pytest.fixture
def client(tmp_path, monkeypatch):
    import app as app_module
    monkeypatch.setattr(app_module, "packs", packs.PackManager(tmp_path / "packs"))
    monkeypatch.setattr(app_module, "job", app_module.TrainingJob())
    app_module.app.config["TESTING"] = True
    return app_module.app.test_client(), app_module


def wait_for_training(c):
    for _ in range(900):
        snap = c.get("/api/train").get_json()
        if snap["state"] != "running":
            return snap
        time.sleep(0.1)
    raise AssertionError("training did not finish")


def test_api_end_to_end(client):
    c, mod = client
    st = c.get("/api/status").get_json()
    assert [p["code"] for p in st["packs"]] == ["isl", "asl", "bsl"] and st["default"] == "isl"
    asl = c.get("/api/packs/asl").get_json()
    assert len(asl["starter"]) == len(rules.LABELS) and asl["dataset"]["total"] == 0

    res = c.post("/api/packs/asl/predict", json={"hands": [synth.sample("L", rng(10))]}).get_json()
    assert res["source"] == "rules" and res["label"] == "L"
    res = c.post("/api/packs/isl/predict", json={"hands": [synth.sample("L", rng(10))]}).get_json()
    assert res["label"] is None                         # L is not an ISL starter gesture
    assert c.post("/api/packs/asl/predict", json={"hands": [{"lm": [[1, 2]]}]}).status_code == 400
    assert c.get("/api/packs/xyz").status_code == 404
    assert c.post("/api/packs/asl/samples", json={"label": "X", "frames": [{"hands": []}]}).status_code == 400

    r = rng(11)
    for label in ("L", "Y", "B"):
        for s in range(2):
            frames = [{"hands": [synth.sample(label, r, pose_noise=6)]} for _ in range(25)]
            assert c.post("/api/packs/asl/samples",
                          json={"label": label, "session": f"{label}{s}", "frames": frames}).status_code == 200
    for label in ("WAVE", "DOWN", "YES"):
        for s in range(2):
            clips = [synth.clip(label, r) for _ in range(5)]
            out = c.post("/api/packs/asl/samples", json={"label": label, "session": f"m{label}{s}", "clips": clips})
            assert out.status_code == 200 and out.get_json()["kind"] == "moving"
    assert c.post("/api/packs/asl/samples", json={"label": "L", "clips": [synth.clip("WAVE", r)]}).status_code == 400

    assert c.post("/api/packs/asl/signs", json={"label": "j", "kind": "moving"}).get_json()["label"] == "J"
    assert c.post("/api/packs/asl/train").status_code == 202
    assert c.post("/api/packs/asl/train").status_code in (202, 409)
    snap = wait_for_training(c)
    assert snap["state"] == "done", snap
    assert "still signs" in snap["message"] and "moving signs" in snap["message"]

    rep = c.get("/api/packs/asl/report").get_json()
    assert rep["still"]["labels"] == ["B", "L", "Y"] and rep["moving"]["labels"] == ["DOWN", "WAVE", "YES"]
    res = c.post("/api/packs/asl/predict", json={"hands": [synth.sample("Y", rng(12))]}).get_json()
    assert res["source"] == "model" and res["label"] == "Y"
    res = c.post("/api/packs/asl/predict_clip", json={"frames": synth.clip("DOWN", rng(13))}).get_json()
    assert res["source"] == "moving" and res["label"] == "DOWN"
    res = c.post("/api/packs/isl/predict_clip", json={"frames": synth.clip("DOWN", rng(13))}).get_json()
    assert res["label"] is None                         # ISL has no moving model

    exported = c.get("/api/packs/asl/export")
    assert exported.status_code == 200 and exported.mimetype == "application/zip"
    created = c.post("/api/packs", json={"name": "Auslan", "short": "AUSLAN", "spoken": "en-AU"})
    assert created.status_code == 201 and created.get_json()["code"] == "auslan"
    imp = c.post("/api/packs/import", data={"file": (io.BytesIO(exported.data), "mudra-asl.zip")},
                 content_type="multipart/form-data").get_json()
    assert imp["merged"] and imp["added"] == 0

    out = c.delete("/api/packs/asl/signs?label=B").get_json()
    assert "B" not in out["pack"]["sign_kinds"]
    assert c.delete("/api/packs/asl/model").get_json()["ok"]
    assert c.get("/api/packs/asl/report").status_code == 404
    assert c.delete("/api/packs/auslan").get_json()["ok"]
