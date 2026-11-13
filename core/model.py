"""
Training, evaluation and live prediction.

Four classic classifiers are compared on the same recordings:
    k-Nearest Neighbours, SVM (RBF kernel), Random Forest, Neural network (MLP)

Evaluation avoids an easy mistake: frames from one recording are almost
identical, so a random train/test split puts near-copies of the test frames
into training and reports inflated accuracy. Instead, whole recordings are
held out (StratifiedGroupKFold), which measures how well the model handles a
recording it has never seen. If a sign has only one recording, that is not
possible and the report says the accuracy is likely optimistic.

The model with the best macro F1 score is retrained on all data and saved.
"""

from __future__ import annotations

import json
import threading
import time
from collections import Counter
from pathlib import Path

import joblib
import numpy as np
from sklearn.base import clone
from sklearn.calibration import CalibratedClassifierCV
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, confusion_matrix, f1_score, precision_recall_fscore_support
from sklearn.model_selection import StratifiedGroupKFold, StratifiedKFold, cross_val_predict
from sklearn.neighbors import KNeighborsClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC

from . import features, rules

MIN_SAMPLES_PER_SIGN = 20
MIN_SIGNS = 2


def candidates() -> dict:
    return {
        "k-Nearest Neighbours": make_pipeline(
            StandardScaler(), KNeighborsClassifier(n_neighbors=5, weights="distance")
        ),
        # Platt scaling turns SVM scores into probabilities (needed for confidence)
        "SVM (RBF kernel)": make_pipeline(
            StandardScaler(),
            CalibratedClassifierCV(SVC(kernel="rbf", C=10, gamma="scale"), method="sigmoid",
                                   cv=3, ensemble=False),
        ),
        "Random Forest": RandomForestClassifier(n_estimators=150, n_jobs=-1, random_state=42),
        "Neural network (MLP)": make_pipeline(
            StandardScaler(),
            MLPClassifier(hidden_layer_sizes=(128, 64), alpha=1e-3, max_iter=400,
                          early_stopping=True, n_iter_no_change=15, random_state=42),
        ),
    }


class TrainingError(Exception):
    pass


def build_matrix(records: list[dict]):
    X, y, groups, measures = [], [], [], []
    skipped = 0
    for rec in records:
        try:
            vec, ms = features.extract(rec["hands"])
        except (ValueError, KeyError, TypeError):
            skipped += 1
            continue
        X.append(vec)
        y.append(rec["label"])
        groups.append(rec["session"])
        measures.append(ms)
    return np.asarray(X), np.asarray(y), np.asarray(groups), measures, skipped


def _latency_ms(est, x_one: np.ndarray, repeats: int = 30) -> float:
    est.predict_proba(x_one)  # warm up
    t0 = time.perf_counter()
    for _ in range(repeats):
        est.predict_proba(x_one)
    return (time.perf_counter() - t0) / repeats * 1000.0


def train(records: list[dict], progress=lambda stage, frac: None) -> tuple[dict, dict]:
    """Returns (bundle, report). Raises TrainingError with a user-facing message."""
    progress("Preparing features", 0.02)
    X, y, groups, measures, skipped = build_matrix(records)
    if len(X) == 0:
        raise TrainingError("No samples yet. Record a few signs first.")

    counts = Counter(y.tolist())
    labels = sorted(counts)
    if len(labels) < MIN_SIGNS:
        raise TrainingError("Record at least two different signs to train a model.")
    short = [l for l in labels if counts[l] < MIN_SAMPLES_PER_SIGN]
    if short:
        raise TrainingError(
            f"Each sign needs at least {MIN_SAMPLES_PER_SIGN} samples. Record more for: {', '.join(short)}."
        )

    sessions_per_label = {l: len(set(groups[y == l].tolist())) for l in labels}
    min_sessions = min(sessions_per_label.values())
    warnings = []
    if min_sessions >= 2:
        n_splits = min(5, min_sessions)
        cv = StratifiedGroupKFold(n_splits=n_splits, shuffle=True, random_state=42)
        method = "grouped"
        method_text = (
            f"{n_splits}-fold cross-validation, holding out whole recordings. "
            "Every frame is scored by a model that never saw its recording."
        )
    else:
        n_splits = min(5, min(counts.values()))
        cv = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=42)
        method = "frames"
        single = [l for l in labels if sessions_per_label[l] < 2]
        method_text = f"{n_splits}-fold cross-validation over individual frames."
        warnings.append(
            "Some signs have only one recording ("
            + ", ".join(single)
            + "), so test frames look almost identical to training frames and the "
            "accuracy is likely optimistic. Record each sign at least twice for an honest score."
        )

    results = []
    fitted = {}
    cands = candidates()
    n = len(cands)
    for i, (name, est) in enumerate(cands.items()):
        base = 0.05 + 0.85 * i / n
        progress(f"Evaluating {name}", base)
        try:
            t0 = time.perf_counter()
            y_pred = cross_val_predict(clone(est), X, y, groups=groups if method == "grouped" else None, cv=cv)
            cv_seconds = time.perf_counter() - t0

            progress(f"Training {name} on all samples", base + 0.85 / n * 0.7)
            final = clone(est)
            t1 = time.perf_counter()
            final.fit(X, y)
            fit_seconds = time.perf_counter() - t1
            if "n_jobs" in final.get_params():
                final.set_params(n_jobs=1)  # parallel overhead slows single predictions
            latency = _latency_ms(final, X[:1])
        except Exception as exc:  # one model failing (e.g. too few samples) shouldn't stop the others
            results.append({"name": name, "error": str(exc)})
            continue
        fitted[name] = final
        results.append({
            "name": name,
            "accuracy": float(accuracy_score(y, y_pred)),
            "macro_f1": float(f1_score(y, y_pred, average="macro", zero_division=0)),
            "fit_seconds": round(fit_seconds, 3),
            "cv_seconds": round(cv_seconds, 3),
            "latency_ms": round(latency, 3),
            "_pred": y_pred,
        })

    scored = [r for r in results if "error" not in r]
    if not scored:
        raise TrainingError("Every model failed to train. " + "; ".join(r["error"] for r in results))
    best = max(scored, key=lambda r: (round(r["macro_f1"], 4), round(r["accuracy"], 4), -r["latency_ms"]))

    progress("Writing report", 0.95)
    y_best = best["_pred"]
    cm = confusion_matrix(y, y_best, labels=labels)
    p, r, f, s = precision_recall_fscore_support(y, y_best, labels=labels, zero_division=0)
    per_class = [
        {"label": l, "precision": float(p[i]), "recall": float(r[i]), "f1": float(f[i]),
         "support": int(s[i]), "sessions": sessions_per_label[l]}
        for i, l in enumerate(labels)
    ]

    # Baseline: how the hand-written starter rules do on signs they know
    known = [i for i, l in enumerate(y) if l in rules.LABELS]
    baseline = None
    if known:
        hits = sum(1 for i in known if rules.predict(measures[i])["label"] == y[i])
        baseline = {
            "accuracy": hits / len(known),
            "samples": len(known),
            "signs": sorted({str(y[i]) for i in known}),
        }

    for res in results:
        res.pop("_pred", None)

    trained_at = time.strftime("%Y-%m-%d %H:%M:%S")
    two_hand = int(np.sum(X[:, features.PER_HAND_DIM * 2 + 1] > 0))
    report = {
        "trained_at": trained_at,
        "best_model": best["name"],
        "accuracy": best["accuracy"],
        "macro_f1": best["macro_f1"],
        "models": results,
        "labels": labels,
        "confusion": cm.tolist(),
        "per_class": per_class,
        "evaluation": {"method": method, "text": method_text, "folds": n_splits},
        "dataset": {
            "samples": int(len(X)),
            "signs": len(labels),
            "recordings": int(len(set(groups.tolist()))),
            "two_hand_samples": two_hand,
            "skipped": skipped,
        },
        "baseline_rules": baseline,
        "warnings": warnings,
        "feature_version": features.FEATURE_VERSION,
        "feature_dim": features.FEATURE_DIM,
    }
    bundle = {
        "estimator": fitted[best["name"]],
        "name": best["name"],
        "labels": labels,
        "feature_version": features.FEATURE_VERSION,
        "trained_at": trained_at,
        "accuracy": best["accuracy"],
    }
    progress("Done", 1.0)
    return bundle, report


class Recognizer:
    """Holds the current trained model and answers predictions thread-safely."""

    def __init__(self, model_dir: Path):
        self.model_dir = Path(model_dir)
        self.model_dir.mkdir(parents=True, exist_ok=True)
        self.model_path = self.model_dir / "model.joblib"
        self.report_path = self.model_dir / "report.json"
        self._bundle = None
        self._lock = threading.Lock()
        self.load_error = None
        self.load()

    def load(self):
        self.load_error = None
        if not self.model_path.exists():
            self._bundle = None
            return
        try:
            bundle = joblib.load(self.model_path)
        except Exception as exc:  # corrupt file or sklearn version change
            self._bundle = None
            self.load_error = f"Saved model couldn't be loaded ({exc.__class__.__name__}). Train again."
            return
        if bundle.get("feature_version") != features.FEATURE_VERSION:
            self._bundle = None
            self.load_error = "Saved model was made with older feature code. Train again."
            return
        with self._lock:
            self._bundle = bundle

    def save(self, bundle: dict, report: dict):
        joblib.dump(bundle, self.model_path)
        self.report_path.write_text(json.dumps(report, indent=2), encoding="utf-8")
        with self._lock:
            self._bundle = bundle
        self.load_error = None

    def clear(self):
        with self._lock:
            self._bundle = None
        for p in (self.model_path, self.report_path):
            if p.exists():
                p.unlink()

    @property
    def ready(self) -> bool:
        return self._bundle is not None

    def info(self) -> dict | None:
        b = self._bundle
        if not b:
            return None
        return {"name": b["name"], "labels": b["labels"], "trained_at": b["trained_at"],
                "accuracy": b.get("accuracy")}

    def report(self) -> dict | None:
        if not self.report_path.exists():
            return None
        try:
            return json.loads(self.report_path.read_text(encoding="utf-8"))
        except json.JSONDecodeError:
            return None

    def predict(self, vec: np.ndarray, min_conf: float = 0.0) -> dict:
        with self._lock:
            b = self._bundle
        if b is None:
            return {"label": None, "confidence": 0.0, "top": []}
        est = b["estimator"]
        probs = est.predict_proba(vec.reshape(1, -1))[0]
        classes = list(est.classes_)
        order = np.argsort(probs)[::-1][:3]
        top = [{"label": str(classes[i]), "confidence": round(float(probs[i]), 3)} for i in order]
        best = top[0]
        if best["confidence"] < min_conf:
            return {"label": None, "confidence": best["confidence"], "top": top}
        return {"label": best["label"], "confidence": best["confidence"], "top": top}
