"""
Run with:  python -m pytest -q
Uses synthetic hands from tests/synth.py, so no webcam is needed.
"""

import time
import uuid

import numpy as np
import pytest

from core import dataset as ds
from core import features, model, rules
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
    assert s["signs"] == [{"label": "HELLO THERE", "samples": 5, "sessions": 1}]
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


@pytest.fixture
def client(tmp_path, monkeypatch):
    import app as app_module
    monkeypatch.setattr(app_module, "data", ds.Dataset(tmp_path / "data" / "samples.jsonl"))
    monkeypatch.setattr(app_module, "recognizer", model.Recognizer(tmp_path / "models"))
    monkeypatch.setattr(app_module, "job", app_module.TrainingJob())
    app_module.app.config["TESTING"] = True
    return app_module.app.test_client(), app_module


def test_api_end_to_end(client):
    c, mod = client
    st = c.get("/api/status").get_json()
    assert st["model"] is None and st["dataset"]["total"] == 0
    assert len(st["starter"]) == len(rules.LABELS)

    res = c.post("/api/predict", json={"hands": [synth.sample("L", rng(10))]}).get_json()
    assert res["source"] == "rules" and res["label"] == "L"

    assert c.post("/api/predict", json={"hands": [{"lm": [[1, 2]]}]}).status_code == 400
    assert c.post("/api/samples", json={"label": "X", "frames": [{"hands": []}]}).status_code == 400

    r = rng(11)
    for label in ("L", "Y", "B"):
        for s in range(2):
            frames = [{"hands": [synth.sample(label, r, pose_noise=6)]} for _ in range(25)]
            resp = c.post("/api/samples", json={"label": label, "session": f"{label}{s}", "frames": frames})
            assert resp.status_code == 200

    assert c.post("/api/train").status_code == 202
    for _ in range(600):
        snap = c.get("/api/train").get_json()
        if snap["state"] != "running":
            break
        time.sleep(0.1)
    assert snap["state"] == "done", snap

    rep = c.get("/api/report").get_json()
    assert rep["labels"] == ["B", "L", "Y"]
    res = c.post("/api/predict", json={"hands": [synth.sample("Y", rng(12))]}).get_json()
    assert res["source"] == "model" and res["label"] == "Y"
    res = c.post("/api/predict", json={"hands": [synth.sample("Y", rng(12))], "source": "rules"}).get_json()
    assert res["source"] == "rules"

    out = c.delete("/api/samples?label=B").get_json()
    assert [s["label"] for s in out["dataset"]["signs"]] == ["L", "Y"]
    assert c.delete("/api/model").get_json()["ok"]
    assert c.get("/api/report").status_code == 404
