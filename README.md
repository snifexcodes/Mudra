# Mudra: real-time sign language to text

Mudra watches your hand through a webcam and turns hand signs into text, live,
in a browser on your own computer. You can teach it new signs, and it trains and
compares four machine-learning models on your recordings.

- **Translate:** full-screen camera, a hand skeleton with fingertip light trails,
  and a sign bubble that follows your hand. Hold a sign until the ring fills and
  it's added to the captions at the bottom. Captions can be spoken aloud.
- **Train:** add signs (A–Z, words, anything), record yourself making each one,
  and press Train model.
- **Results:** accuracy, a comparison of four models, a confusion matrix and
  per-sign scores. Print it or save it as a PDF for your project report.

It works on first launch with 14 starter gestures (A B D F I L O U V W Y,
HELLO, GOOD, I LOVE YOU). These are hand-written rules, so they act as a baseline
until you train your own model.

---

## Quick start

You need Python 3.10 or newer, a webcam, and Chrome, Edge or Firefox.

**Windows:** double-click `run.bat`.

**macOS / Linux:** run `./run.sh`.

Either one creates a virtual environment, installs the four small dependencies,
starts the server and opens **http://localhost:5050**.

To do it by hand:

1. `python -m venv .venv`
2. Activate it: `.venv\Scripts\activate` (Windows) or `source .venv/bin/activate`
3. `pip install -r requirements.txt`
4. `python app.py`
5. Open http://localhost:5050 and press **Start camera**.

Always open it as `localhost`. Browsers only allow the camera on `localhost` or HTTPS,
so a LAN address like `192.168.x.x` won't get camera access.

### Offline mode (recommended before a demo)

Hand tracking loads from a CDN by default. To run without internet:

```
python tools/download_assets.py
```

This saves the MediaPipe files (about 30 MB) to `static/vendor/mediapipe/`, and the
app uses them automatically from then on.

---

## Using it

### Translate
1. Raise one hand into view. The bubble shows the sign it sees and how sure it is.
2. Hold the sign steady. When the ring fills (0.9 s by default), the sign is added.
3. Keep holding to add it again (for double letters like the LL in HELLO).
4. Lower your hand to add a space after a fingerspelled word.
5. Keyboard: **Space** adds a space, **Backspace** deletes, **Enter** speaks the text.
   Keys **1**, **2** and **3** switch tabs.

Single letters join into words (H, I → HI). Multi-letter signs like HELLO are
added as whole words.

### Train your own signs
1. Open **Train**. Type a sign name or use Quick add (A–Z, starter set, common words).
2. Pick a sign and press **Record** (or **R**). After a 3-second countdown it captures
   100 frames (about 7 seconds). Move your hand slightly while recording: closer,
   farther, a little tilted.
3. Record each sign **2–3 times**. Training needs at least two signs with
   20 samples each, and the score is only honest with two or more recordings per sign
   (see Evaluation below).
4. Press **Train model**. It takes from a few seconds to about a minute. Translate then
   switches to your model automatically.
5. To control the captions by hand, train signs named `SPACE` and `DELETE`.

J and Z are left out of the alphabet preset because they need movement, and Mudra
classifies one frame at a time.

### Results
Shows the best model's accuracy, all four models side by side, the confusion matrix
and precision/recall per sign. **Print or save as PDF** gives a clean white version
for your report.

### Settings
Hold time, minimum confidence, auto-space, speak words aloud, two-handed signs,
light trails, mirror, and camera choice. All saved in the browser.

---

## How it works

```
 Browser                                           Python (Flask, app.py)
 ───────────────────────────────────────           ───────────────────────────────
 Webcam ─▶ MediaPipe Hand Landmarker (WASM)
            21 points per hand, ~30 fps
                     │  POST /api/predict  ──────▶ core/features.py  → 197 numbers
                     │  (points only, no video)            │
                     │                                     ├─▶ core/rules.py  (starter gestures)
                     │                                     └─▶ core/model.py  (trained model)
 Smoothing + hold-to-add ◀── label + confidence ───────────┘
 Captions, speech

 Train tab:  POST /api/samples ─▶ data/samples.jsonl (raw landmarks)
             POST /api/train   ─▶ compare 4 models ─▶ models/model.joblib + report.json
```

### 1. Hand tracking
Google's MediaPipe Hand Landmarker runs in the browser through WebAssembly. For each
hand it gives 21 landmarks (wrist, and 4 joints for each finger) in image coordinates
and in real-world 3D coordinates (metres). Only these numbers go to the server. The video
never leaves the browser.

### 2. Features (`core/features.py`)
The same sign has to look the same wherever the hand is, however big it appears,
and whichever hand is used. Each hand is:

1. **mirrored** if it's a left hand, so every hand is treated as a right hand
2. **translated** so the wrist is the origin
3. **rotated** into the hand's own frame (y = wrist → middle knuckle,
   x = little-finger knuckle → index knuckle)
4. **scaled** so the palm length is 1

From that, each hand gives 96 numbers: 63 normalised coordinates, 6 for orientation
(which way the hand points), 15 joint bend angles and 12 fingertip distances. With
two slots for two hands, plus hand presence and their relative position, the
vector has **197 features**.

### 3. Starter gestures (`core/rules.py`)
A fuzzy rule-based recogniser. Each finger gets an "extended" score from 0 to 1 based
on its joint angles. Each gesture is a template (which fingers are out) plus a few checks
(fingers spread, thumb touching index, thumb pointing up). A gesture's score is its weakest
check, and the best gesture wins.

### 4. Models (`core/model.py`)
Four scikit-learn classifiers are trained and compared on the same data:

| Model | Why it's included |
|---|---|
| k-Nearest Neighbours | Simplest baseline: "which recorded samples is this closest to?" |
| SVM (RBF kernel) | Strong on small, clean datasets; Platt scaling gives probabilities |
| Random Forest | Many decision trees voting; robust and needs little tuning |
| Neural network (MLP) | Two hidden layers (128, 64); fastest at prediction time |

The model with the best **macro F1** is saved and used live. Macro F1 averages the F1
of each sign, so a sign with few samples counts as much as one with many.

### 5. Evaluation (and the leak it avoids)
Frames from one recording are nearly identical. A random train/test split would put
near-copies of the test frames in the training set and report inflated accuracy.
Mudra holds out **whole recordings** instead (`StratifiedGroupKFold`): every frame is
scored by a model that never saw the recording it came from. If any sign has only one
recording, it falls back to a frame-level split and the report warns that the
score is optimistic.

The Results page also scores the starter rules on the same signs (where they overlap),
which gives a baseline to compare the trained model against.

### 6. Live smoothing
Predictions arrive about 16 times a second. Mudra takes a majority vote over the last
0.65 s, ignores anything under the confidence threshold, and only adds a sign after
it has been held for the hold time. This stops flicker between similar signs from
typing junk.

---

## Desktop version with OpenCV (optional)

`cv_demo.py` does the same thing in an OpenCV window, using the same features, rules
and trained model. A model trained in the browser works here too.

1. `pip install -r requirements-cv.txt`
2. `python tools/download_assets.py` (it needs `hand_landmarker.task`)
3. `python cv_demo.py` (add `--camera 1` for a second webcam, `--hands 2` for two hands)

Keys: **q** quit, **space**, **backspace**, **c** clear, **m** switch model/rules.
If pip can't find `mediapipe` for your Python version, use Python 3.11 or 3.12 for
this demo.

---

## Project structure

```
mudra/
├── app.py                 Flask server and REST API
├── cv_demo.py             Optional OpenCV desktop demo
├── core/
│   ├── features.py        Landmarks → 197-number feature vector
│   ├── rules.py           Starter gestures (fuzzy rules)
│   ├── model.py           Training, evaluation, live prediction
│   ├── dataset.py         Sample storage (JSON lines)
│   └── assets.py          MediaPipe file locations
├── static/
│   ├── index.html
│   ├── css/app.css
│   ├── js/                main, tracker, render, translate, studio, results, api
│   ├── fonts/             Atkinson Hyperlegible Next (bundled)
│   └── vendor/            Offline MediaPipe files (after download_assets.py)
├── data/samples.jsonl     Your recordings (created when you record)
├── models/                model.joblib and report.json (created when you train)
├── tools/download_assets.py
├── tests/                 pytest suite using a synthetic hand model
├── run.bat / run.sh
└── requirements*.txt
```

### API

| Method | Path | Does |
|---|---|---|
| GET | `/api/status` | Model info, dataset summary, starter gestures, asset URLs |
| POST | `/api/predict` | `{hands, source}` → `{label, confidence, top, source}` |
| POST | `/api/samples` | Save a recording `{label, session, frames}` |
| DELETE | `/api/samples?label=X` or `?session=Y` | Delete a sign or one recording |
| POST | `/api/signs/rename` | `{old, new}` |
| GET | `/api/samples/export` | Download all recordings |
| POST / GET | `/api/train` | Start training / check progress |
| GET | `/api/report` | Latest evaluation report |
| DELETE | `/api/model` | Remove the trained model |

### Tests

```
pip install -r requirements-dev.txt
python -m pytest -q
```

The tests use a small kinematic hand model (`tests/synth.py`) to generate landmark data
for each starter gesture. They cover feature invariance (position, scale, left/right
hand), the rules, the dataset store, the full training pipeline and the API. The
synthetic hands are only for testing, never for training the real model.

---

## Talking points for your viva

1. **Why landmarks instead of raw images?** 21 points are about 60 numbers instead of
   about 900,000 pixels. Training is fast, needs little data, and ignores lighting,
   skin tone and background.
2. **Why normalise?** So position, distance from the camera and left vs right hand don't
   change the features. The tests check this invariance.
3. **Why four models?** To justify the choice with numbers, not guesswork. The best one
   by macro F1 is picked automatically.
4. **Why grouped cross-validation?** Frames from one recording are near-duplicates. A
   random split leaks them into training and inflates accuracy.
5. **Why smoothing and hold-to-add?** Single-frame predictions flicker. A majority vote
   plus a hold time turns them into deliberate input.
6. **Privacy:** video never leaves the browser, and the server only receives coordinates.

### Limitations and future scope
- One frame at a time, so signs that need **movement** (J, Z and many ASL/ISL words)
  aren't supported. Next step: a sequence model (LSTM, GRU or a small Transformer) over 30–60
  frames of landmarks.
- Two-handed signs are supported in the features, but they need training data. The
  starter rules are one-handed.
- No facial expressions or body position, which carry grammar in real sign languages.
  MediaPipe Holistic could add them.
- Trained on one person, it works best for that person. Recordings from several people
  improve it for everyone.
- A word-level language model could suggest corrections and complete words.

---

## Troubleshooting

| Problem | Fix |
|---|---|
| "Camera access is blocked" | Click the camera icon in the address bar, allow it, reload |
| "Camera is busy" | Close Zoom, Meet, Teams, OBS or another browser tab using it |
| Hand tracking won't load | Check your internet, or run `python tools/download_assets.py` |
| Page is blank or stuck | Hard-refresh with Ctrl+F5 and check the terminal running `app.py` for errors |
| Port 5050 in use | `python app.py --port 8000` |
| Signs flicker or add too fast | Raise **Hold a sign to add it** or **Minimum confidence** in Settings |
| Starter gestures misread | Good light, hand 40–80 cm from the camera, palm facing the camera. Better: train your own |
| "Saved model couldn't be loaded" | scikit-learn version changed. Press Train model again |

## Credits

- Hand tracking: [MediaPipe Hand Landmarker](https://ai.google.dev/edge/mediapipe/solutions/vision/hand_landmarker) (Apache 2.0)
- Machine learning: [scikit-learn](https://scikit-learn.org)
- Typeface: [Atkinson Hyperlegible Next](https://www.brailleinstitute.org/freefont/) by the Braille Institute (SIL Open Font License)
