# Mudra: sign language to text and speech, for any sign language

Mudra watches your hands through a webcam and turns signs into text and speech,
live, in a browser on your own computer. It is built for people who sign or can't
speak, so they can talk with people who don't know sign language.

- **Translate:** full-screen camera, a hand skeleton with fingertip light trails,
  and a sign bubble that follows your hand. Hold a still sign until the ring
  fills, or make a moving sign and pause: it's added to the captions and can be
  spoken aloud.
- **Talk:** tap ready-made phrases ("I need water", "Call an ambulance"), or type,
  and Mudra speaks them. Press **Listen** and the other person's speech appears
  as large text, so it works as a two-way conversation. Any spoken language the
  browser supports.
- **Train:** add signs, record yourself, and train. Still signs (hand shapes) and
  moving signs (signs with motion, one or two hands) each get their own model.
- **Results:** accuracy, a comparison of four models, a confusion matrix and
  per-sign scores. Print it or save it as a PDF for a report.
- **Sign-language packs:** Indian (ISL), American (ASL) and British (BSL) Sign
  Language are built in, and you can add any other sign language. Packs can be
  exported and imported, so a class, a family or a community can pool recordings.

### What Mudra can and can't do

There are hundreds of sign languages, and the official ISL dictionary alone has
over 10,000 terms. No app recognises a whole sign language, and Mudra doesn't
claim to. It recognises the signs **you** record (or import from a dataset).
Starter gestures that work with no training: 14 for ASL (A B D F I L O U V W Y,
HELLO, GOOD, I LOVE YOU), and HELLO and GOOD for the other packs. These are
hand-written rules, so treat them as a baseline. The Talk tab works with no
training at all.

---

## Quick start

You need Python 3.10 or newer, a webcam, and Chrome or Edge (Firefox works too,
but can't listen in the Talk tab).

**Windows:** double-click `run.bat`. **macOS / Linux:** run `./run.sh`.

Either one creates a virtual environment, installs four small dependencies,
starts the server and opens **http://localhost:5050**.

To do it by hand:

1. `python -m venv .venv`
2. Activate it: `.venv\Scripts\activate` (Windows) or `source .venv/bin/activate`
3. `python -m pip install -r requirements.txt`
4. `python app.py`
5. Open http://localhost:5050 and press **Start camera**.

Use `localhost`, not a LAN address: browsers only allow the camera there or on HTTPS.

**Windows blocks run.bat or pip.exe?** Files from a downloaded zip can be blocked by
Smart App Control. Clone with git instead (`git clone https://github.com/snifexcodes/Mudra.git`),
or run `python -m pip install -r requirements.txt` and `python app.py` yourself.

### Offline mode (recommended before a demo)

Hand tracking loads from a CDN by default. To run without internet:

```
python tools/download_assets.py
```

This saves the MediaPipe hand and pose models and runtime (about 35 MB) to
`static/vendor/mediapipe/`. The Talk tab's Listen button still needs internet,
because Chrome and Edge do speech recognition online.

---

## Using it

### Choose a sign language
Click the language chip at the top right (it shows **ISL** at first). Pick ISL,
ASL, BSL, or **Add a sign language** with its name, short name and spoken language
(for example `ja-JP`, `fr-FR`, `hi-IN`). Everything in Train, Translate and
Results belongs to the chosen pack.

### Translate
1. Raise your hand into view. The bubble shows what it sees and how sure it is.
2. **Still sign:** hold it until the ring fills (0.9 s by default). Keep holding to
   add it again (the LL in HELLO).
3. **Moving sign:** make it once, then pause. The ring spins while Mudra reads the
   movement, then adds the sign.
4. Lower your hand to add a space after a fingerspelled word.
5. Keyboard: **Space**, **Backspace**, **Enter** speaks the text and adds it to the
   Talk conversation. Keys **1–4** switch tabs.

### Talk
1. Tap a phrase to speak it. Groups: Basics, Needs, Feelings, Emergency.
2. **Edit phrases** to remove phrases or add your own, in any language.
3. Type in the box and press **Speak**.
4. **Listen to them** turns the other person's speech into large text. Choose the
   language they speak next to the button. (Chrome and Edge send the audio to
   Google or Microsoft for this.)
5. **Show big** fills the screen with your last message, to show someone in a noisy place.

### Train your own signs
1. Open **Train**. Type a sign name, choose **Still** (a hand shape you hold) or
   **Moving** (a sign with motion), and press Add. Or use **Quick add a set of signs**.
2. Pick a sign and press **Record** (or **R**).
   - Still: after a 3-second countdown it captures about 7 seconds. Move your hand a
     little: closer, farther, tilted.
   - Moving: it asks for 10 repetitions. Make the sign once each time it says
     **Sign now**, and rest your hands in between.
3. Record each sign in **two or more sessions** (different times, or different people).
   The accuracy score is only honest when it can test on a session it didn't train on.
4. Add **Nothing (rest and random moves)** and record yourself resting, scratching,
   reaching. Mudra learns to ignore those (`NO SIGN` is never added to the text).
5. Press **Train model**. It trains the still model, the moving model, or both.
6. Train signs named `SPACE` and `DELETE` to edit the captions by hand.

Minimums: two signs of a kind; 20 samples per still sign; 5 repetitions per moving sign.

### Share a pack
In the language drawer, **Export this pack** saves a `.zip` with the pack's settings
and recordings. **Import a pack** on another computer adds those recordings. If the
pack already exists there, they're merged (recordings already present are skipped).
Train again after importing.

### Results
Switch between **Still signs** and **Moving signs**. **Print or save as PDF** gives
a clean white page for a report.

---

## Pretrained word sets from public datasets

Recording every sign yourself takes time, so Mudra can import public datasets.
These run on your computer (big downloads, then some hours of processing).

```
python -m pip install -r requirements-datasets.txt
python tools/download_assets.py
```

**Any sign language, from videos** (`tools/datasets/import_videos.py`): put one folder
per sign with its videos inside. The folder name is the sign (`12. Thank you` becomes
`THANK YOU`). This is the layout of **INCLUDE**, the Indian Sign Language dataset
(263 words, CC BY 4.0, [zenodo.org/records/4010759](https://zenodo.org/records/4010759)):

```
python tools/datasets/import_videos.py --videos "D:/INCLUDE" --pack isl --top 50 --train
```

The same script works for your own videos or another sign language's dataset.

**ASL, from landmarks** (`tools/datasets/import_kaggle_asl.py`): the Google Isolated
Sign Language Recognition data on Kaggle (250 ASL signs by 21 signers;
[kaggle.com/competitions/asl-signs](https://www.kaggle.com/competitions/asl-signs/data),
needs a Kaggle account and accepting the competition rules; check they allow your use):

```
python tools/datasets/import_kaggle_asl.py --data "D:/asl-signs" --top 50 --train
```

Each signer counts as one recording, so the score measures how well the model handles
**signers it has never seen**. That dataset doesn't store the video size; if the model
works poorly on your live signing, re-import with `--aspect 0.5625` or `--swap-hands`.

Start with `--top 50`. Both importers skip videos or sequences they've already
imported, so you can stop and resume. Without `--train`, train from the app.

There's no freely downloadable BSL dataset like these (BOBSL, the largest, needs an
application), so the BSL pack starts from your own recordings.

---

## How it works

```
 Browser                                          Python (Flask, app.py)
 ─────────────────────────────────────────        ──────────────────────────────────
 Webcam ─▶ MediaPipe Hand Landmarker (WASM)
           + Pose Landmarker (nose, shoulders)
                │ still: POST /predict ─────────▶ features.py  → 197 numbers → still model
                │ moving: POST /predict_clip ───▶ sequence.py  → 1263 numbers → moving model
 Smoothing, hold-to-add,                          (per pack, in packs/<code>/)
 movement detection ◀── label + confidence ───────┘
 Captions, speech, Talk
```

### 1. Tracking
Google's MediaPipe runs in the browser through WebAssembly: 21 landmarks per hand,
plus the nose and shoulders from the pose model so moving signs know where they're
made (chin, chest, beside the head). Only these numbers reach the server; video never
leaves the browser.

### 2. Still-sign features (`core/features.py`)
Each hand is mirrored if it's a left hand, moved so the wrist is the origin, rotated
into the hand's own frame and scaled to palm length 1. Then: 63 coordinates,
6 orientation values, 15 joint bend angles, 12 fingertip distances = 96 per hand.
Two hand slots plus presence and relative position: **197 features**.

### 3. Moving-sign features (`core/sequence.py`)
1. Each frame is described by 105 numbers: both hand shapes (48 each), which hands are
   present, each wrist's position relative to the nose in shoulder widths, and how far
   the hand has travelled since the sign started.
2. The still part before and after the movement is trimmed, so a recorded repetition
   and a live sign line up the same way.
3. The movement is resampled to 12 evenly spaced moments, plus path length, duration
   and how much of the clip moved: **1263 features**.

Left-handed signing is mirrored, so it matches right-handed signing.

### 4. Starter gestures (`core/rules.py`)
Fuzzy rules: each finger gets an "extended" score from its joint angles, and each
gesture is a template plus a few checks (spread, touching, thumb up). Its score is its
weakest check.

### 5. Models (`core/model.py`)
For each pack and each kind (still, moving), four scikit-learn classifiers are
compared: k-Nearest Neighbours, SVM (RBF kernel, Platt-scaled), Random Forest and a
neural network (MLP, 128 and 64 units). The best by **macro F1** is saved and used live.

### 6. Evaluation (and the leak it avoids)
Frames from one recording are nearly identical. A random split would put near-copies of
the test data into training and inflate the score. Mudra holds out **whole recordings**
(`StratifiedGroupKFold`). With imported Kaggle data, a recording is a signer, so it tests
on unseen people. If a sign has only one recording, the report says so.

### 7. Live recognition
- Still signs: about 16 predictions a second, a majority vote over 0.65 s, and
  hold-to-add.
- Moving signs: the browser measures how fast the hand (wrist and fingertips) moves.
  When it speeds up, Mudra collects frames; when it stops, the clip goes to the moving
  model once. The hand shape a sign ends in isn't added again as a still sign.

---

## Desktop version with OpenCV (optional)

`cv_demo.py` runs still-sign recognition in an OpenCV window with the same features,
rules and still model. Moving signs and Talk are web-only.

1. `python -m pip install -r requirements-cv.txt`
2. `python tools/download_assets.py`
3. `python cv_demo.py --pack isl` (`--camera 1`, `--hands 2`)

Keys: **q** quit, **space**, **backspace**, **c** clear, **m** model/rules.

---

## Project structure

```
mudra/
├── app.py                    Flask server and REST API
├── cv_demo.py                Optional OpenCV desktop demo
├── core/
│   ├── features.py           Still signs: landmarks → 197 features
│   ├── sequence.py           Moving signs: clip → 1263 features
│   ├── rules.py              Starter gestures (fuzzy rules)
│   ├── model.py              Training, evaluation, prediction (still and moving)
│   ├── dataset.py            Recordings (JSON lines)
│   ├── packs.py              Sign-language packs, export/import
│   └── assets.py             MediaPipe file locations
├── static/
│   ├── index.html, css/app.css
│   ├── js/                   main, tracker, render, translate, talk, speech, studio, results, api
│   ├── fonts/                Atkinson Hyperlegible Next (bundled)
│   └── vendor/               Offline MediaPipe files (after download_assets.py)
├── packs/<code>/             pack.json, samples.jsonl, models (created as you use it)
├── tools/
│   ├── download_assets.py
│   └── datasets/             import_videos.py, import_kaggle_asl.py
├── tests/                    pytest suite with a synthetic hand model
└── run.bat, run.sh, requirements*.txt
```

Recordings from version 1 (`data/samples.jsonl`) are moved into the ISL pack the first
time you start this version.

### API

| Method | Path | Does |
|---|---|---|
| GET | `/api/status` | Packs, asset URLs, training state |
| GET / POST | `/api/packs` | List packs / add a sign language |
| POST | `/api/packs/import` | Import a pack `.zip` (merges if it exists) |
| GET / DELETE | `/api/packs/<code>` | Pack details / delete a custom pack |
| GET | `/api/packs/<code>/export` | Download the pack as `.zip` |
| POST | `/api/packs/<code>/predict` | Still sign from `{hands}` |
| POST | `/api/packs/<code>/predict_clip` | Moving sign from `{frames}` |
| POST / DELETE | `/api/packs/<code>/signs` | Add a sign with its kind / delete a sign |
| POST | `/api/packs/<code>/signs/rename` | `{old, new}` |
| POST / DELETE | `/api/packs/<code>/samples` | Save `{frames}` or `{clips}` / delete a recording |
| POST | `/api/packs/<code>/train` | Train the pack's models |
| GET | `/api/train` | Training progress |
| GET | `/api/packs/<code>/report` | `{still, moving}` evaluation reports |
| DELETE | `/api/packs/<code>/model` | Remove trained models |

### Tests

```
python -m pip install -r requirements-dev.txt
python -m pytest -q
```

A small kinematic hand model (`tests/synth.py`) generates still gestures and moving signs
(wave, circle, down, nod, beckon). The tests cover feature invariance (position, scale,
left/right hand, padding around a movement), the rules, packs and export/import, both
models, both dataset importers and the whole API. Synthetic hands are only for testing.

---

## Talking points for a viva

1. **Landmarks instead of pixels:** about 60 numbers instead of 900,000 pixels, so it
   trains fast on little data and ignores lighting, skin tone and background.
2. **Normalisation:** position, distance, rotation and left/right hand don't change the
   features; the tests prove it.
3. **Still vs moving:** one frame is enough for a hand shape; motion needs a clip. Mudra
   detects movement, trims it and resamples it to a fixed length so classic models can
   compare clips.
4. **Body position:** the pose model gives nose and shoulders, so signs made at the chin
   and at the chest differ even with the same hand shape.
5. **Four models, picked by numbers:** macro F1 on held-out recordings.
6. **No data leak:** whole recordings (or signers) are held out.
7. **Rejecting non-signs:** a recorded `NO SIGN` class, plus a confidence threshold.
8. **Accessibility beyond recognition:** a phrase board, text-to-speech and speech-to-text
   make it usable for a real conversation today.
9. **Privacy:** video stays in the browser; only coordinates go to a server on the same
   computer.

### Limitations and future scope
- A whole sign language can't be covered; accuracy depends on how many signs and
  signers you record or import.
- Facial expressions and mouth shapes carry grammar in sign languages and aren't
  used yet (MediaPipe's face landmarker could add them).
- Signs are recognised one at a time, not as continuous sentences with grammar.
  A sequence-to-sequence model (Transformer) trained on sentence datasets is the next step.
- The moving model resamples to 12 moments; an LSTM or temporal CNN could capture finer
  timing with more data.

---

## Troubleshooting

| Problem | Fix |
|---|---|
| Camera access is blocked | Click the camera icon in the address bar, allow it, reload |
| Camera is busy | Close Zoom, Meet, Teams, OBS or another tab using it |
| Hand tracking won't load | Check your internet, or run `python tools/download_assets.py` |
| Windows blocks run.bat or pip.exe | Use `git clone` instead of the zip, or run `python -m pip …` and `python app.py` |
| Port 5050 in use | `python app.py --port 8000` |
| Signs flicker or add too fast | Raise **Hold a sign to add it** or **Minimum confidence** in Settings |
| Moving signs aren't picked up | Make the sign clearly, then pause for half a second; record more repetitions |
| Random movements become signs | Record the **Nothing** set, then train again |
| Listen doesn't work | Use Chrome or Edge, allow the microphone, and stay online |
| Saved model couldn't be loaded | scikit-learn version changed: press Train model again |

## Credits

- Hand and pose tracking: [MediaPipe](https://ai.google.dev/edge/mediapipe/solutions/vision/hand_landmarker) (Apache 2.0)
- Machine learning: [scikit-learn](https://scikit-learn.org)
- Typeface: [Atkinson Hyperlegible Next](https://www.brailleinstitute.org/freefont/) by the Braille Institute (SIL Open Font License)
- Datasets you can import: [INCLUDE](https://zenodo.org/records/4010759) (AI4Bharat, CC BY 4.0), [Google ASL Signs](https://www.kaggle.com/competitions/asl-signs) (Kaggle)
