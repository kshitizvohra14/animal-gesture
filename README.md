# Pet Gesture / Emotion Reader — Prototype

A small end-to-end prototype: browser webcam → Flask backend → MobileNetV2
classifier → live emotion/gesture readout, styled as a "field log."

## What's here

```
app.py            Flask server: /predict (frame -> emotion+motion+gesture), /predict_audio (mic -> vocalization)
model.py          Emotion model architecture + loading logic (MobileNetV2, trained per-frame)
motion.py         Still/Walking/Running from frame-differencing — heuristic fallback, no training needed
pose_motion.py    Skeleton-based motion (paw velocity) + posture (sit/stand/lie) from YOLO dog keypoints
wellness_alerts.py  Heuristic welfare flags (limping, collapse, stillness, low tail carriage) from pose history
train_pose.py     Fine-tunes a YOLO pose model on Ultralytics' Dog-Pose dataset (24 keypoints)
dog_detector.py   General COCO-pretrained dog-presence gate (runs before emotion/motion/pose)
audio.py          Barking/Growling/Whimpering from mic audio, via pretrained YAMNet — no training needed
fusion.py         Weighted cue-matching engine combining emotion (hard-gated) + motion + posture + vocalization
train.py          Fine-tunes the emotion model on a real labeled dataset
prepare_data.py   Splits a downloaded class-folder dataset into train/val
templates/
  index.html      Webcam + mic capture UI (vanilla JS, no build step)
requirements.txt
data/             (empty — put your dataset here, see below)
models/           (empty — trained emotion weights land here)
```

## How the three signals combine

- **Emotion** (Happy/Relaxed/Sad/Angry) — trained CNN, per video frame.
- **Motion** (Still/Walking/Running) — no model at all: just measures how much
  each frame changed from the last one. Crude but effective, and a natural slot
  to replace later with a proper temporal/pose-based model (see the literature
  review — ASBAR, PoseR) once you have labeled video clips.
- **Vocalization** (Barking/Growling/Whimpering/Quiet) — uses YAMNet, a
  pretrained general sound-event model, filtered down to the handful of
  AudioSet classes that are dog vocalizations. No training required; it
  downloads automatically (~15MB) the first time `audio.py` runs.
- **Fusion** — `fusion.py` holds a small rule table (e.g. Happy + Barking →
  "Excited / play solicitation bark"). This is hand-written on purpose: there's
  no existing dataset that maps (emotion, motion, sound) triples to named
  gestures. Once you've logged and labeled some real combinations from your
  own dog/cat, replace the rule table with a small trained classifier
  (3 inputs → 1 gesture label) for something more rigorous.

## Quick start (demo mode, no training required)

This runs the full pipeline immediately so you can see the architecture working,
using an **untrained** classification head — predictions won't mean anything yet,
but the webcam capture, backend inference call, and live UI all function.

```bash
cd pet-gesture-prototype
pip install -r requirements.txt
python app.py
```

Open `http://localhost:5000`, click **Start reading**, and allow both webcam
and microphone access. You'll see a banner confirming the emotion model is
in demo mode — motion and vocalization work immediately since they don't
need training.

## Making it real: training on actual data

1. Pick a dataset (see options and links in the docstring at the top of `train.py`) —
   easiest starting point is the Kaggle "doggos_emotion_recognition" set (~2000 images,
   4 classes).
2. Reorganize the images into:
   ```
   data/train/<class_name>/*.jpg
   data/val/<class_name>/*.jpg
   ```
   Class folder names become your labels — check they match (or update) `CLASSES` in `model.py`.
3. Train:
   ```bash
   python train.py --epochs 15 --data_dir data
   ```
   This writes `models/dog_emotion.h5`.
4. Restart `python app.py` — it auto-loads the trained weights and the demo-mode
   banner disappears.

## Upgrading motion: skeleton-based instead of pixel-diff

By default motion uses raw frame-differencing (crude but zero-setup). To upgrade to
real skeletal tracking:

```bash
pip install ultralytics
python train_pose.py --epochs 100
```

This fine-tunes a YOLO pose model on Ultralytics' Dog-Pose dataset (6,773 training
images, 24 keypoints per dog — paws, knees, ears, tail, withers, etc.) and saves the
result to `models/dog_pose_best.pt`. Restart `app.py`: it detects the trained weights
automatically and switches from pixel-diff motion to pose-based motion, and adds a new
**posture** signal (Sitting / Standing / Lying) derived from joint geometry — something
pixel-differencing structurally cannot produce, since it only sees *change*, not *body
configuration*. If no trained pose model is found, the app cleanly falls back to the
original `motion.py` heuristic — nothing breaks either way.

Note: `train_pose.py` defaults to `yolo11n-pose.pt` as the base checkpoint. If your
installed `ultralytics` version supports YOLO26, pass `--base_model yolo26n-pose.pt`
for a newer backbone.

## Welfare alerts (only active once the pose model is trained)

`wellness_alerts.py` watches the same keypoint stream over time (not single frames —
these patterns only fire on *sustained* signals, to avoid one noisy frame triggering
a false alarm) and flags four things:

- **Possible limping** — a specific paw never reaches ground level relative to the
  other three over a full walking/running window. A healthy gait cycles all four paws
  through ground contact; one paw staying persistently elevated is the classic visual
  pattern of non-weight-bearing lameness.
- **Possible collapse** — a sudden Standing/Walking → Lying transition that skips
  Sitting. Dogs normally sit before lying down; a fast drop straight to Lying while
  still moving is atypical and flagged as a higher-severity ("critical") alert.
- **Prolonged stillness** — no movement detected for an extended stretch (a lethargy
  watch, not a diagnosis of anything specific).
- **Low/tucked tail carriage** — tail held low for a sustained period, which *can*
  correlate with anxiety or discomfort but is also just breed anatomy or a resting
  posture for many dogs.

**These are pattern flags, not veterinary diagnoses.** They're simple geometric heuristics
on 2D keypoints from one camera angle — camera angle, occlusion, breed anatomy (e.g.
naturally low-set tails), and senior dogs' normal reduced gait range can all trigger a
false positive. The UI shows this disclaimer alongside every alert; keep it there if you
extend this. Treat every alert as "worth watching," and see a vet for anything persistent.

**Adding your own alert:** follow the pattern in `wellness_alerts.py` — track whatever raw
signal you need in `WellnessTracker.__init__`, update it each frame in `update()`, and only
append an alert once you've seen a *sustained* pattern, not a single frame. Ideas that fit
this same architecture: repetitive head-shaking/scratching (high-frequency small-amplitude
ear keypoint motion), reduced range of motion over a session (knee/elbow angle variability
trending down), or asymmetric weight-bearing while standing still (not just mid-gait).

## Important OOD / Human-Rejection Gate

The dog emotion classifier is **not** a dog-vs-human classifier. Without a
separate gate, a human image can still receive a high softmax score for
Happy/Relaxed/Sad/Angry because a softmax classifier must choose one of its
known classes.

This version fixes that failure mode:

1. YOLO detects a COCO `dog` first.
2. If no dog is detected above the confidence threshold, inference stops.
3. Only the detected dog crop is passed to the emotion and pose models.
4. Video frames without a dog are excluded from fusion and reset temporal state.
5. If the YOLO detector cannot load, the system **fails closed** instead of
   assuming every frame is a dog.

Default dog-detection threshold: `0.55`. You can change it with
`DOG_DETECTOR_MIN_CONFIDENCE`.

## Known limitations of this prototype

- **Motion (pixel-diff fallback) is a heuristic, not a trained model.** Frame-differencing
  thresholds (`motion.py`) are tuned by eye and will need adjusting for your camera/lighting.
  It also can't tell "walking toward camera" from "something else moved in frame" — train
  `train_pose.py` to replace it with skeleton-based motion, which doesn't have this problem.
- **Posture is geometry, not a trained classifier.** `classify_posture()` in `pose_motion.py`
  uses fixed thresholds on joint positions (withers height, leg extension) — reasonable
  starting rules, but not validated against real labeled sit/stand/lie examples. Expect to
  need to tune the thresholds for your own dog's proportions and camera angle.
- **Pose model needs a visible, mostly-unoccluded dog.** Like any keypoint detector, accuracy
  drops when the dog is partially out of frame, facing away, or overlapping with furniture/other
  objects — the Dog-Pose dataset is varied but still mostly clear, single-dog shots.
- **Vocalization detection assumes the dog is the only sound source.** YAMNet
  classifies whatever's loudest in the ~2.5s window; background noise/other
  voices will interfere. Works best in a quiet room with the mic close to the dog.
- **Fusion is a weighted cue-matching engine, not a learned model.** Emotion is a hard
  gate (a gesture requires a specific emotion to even be considered), and cues are scored
  as matched-weight / total-weight with unmeasured cues counting as unmatched, never
  skipped — but it's still hand-designed, not trained on real behavior data. Treat the
  "gesture" output as a hypothesis to check against what you actually observe.
- **The dog-presence gate uses a separate general-purpose detector.** `dog_detector.py`
  runs a COCO-pretrained YOLO (not fine-tuned on anything project-specific) before the
  emotion/motion/pose pipeline runs at all, specifically so a human in frame doesn't get
  scored as a dog. If `ultralytics` can't download its weights (offline environment), this
  gate silently disables itself and every frame proceeds unguarded — check the startup
  console output for a warning if that matters for your demo.
- **Emotion confidence gating (MIN_CONFIDENCE=0.85, MIN_MARGIN=0.25) will reject more
  frames than before.** This is intentional — a weak, ambiguous, or out-of-distribution
  reading now shows as "Uncertain" rather than confidently (and wrongly) picking a class.
  Expect fewer but more trustworthy emotion readings; tune the constants in `model.py`
  once you have real validation data showing where the right cutoff is for your dataset.
- **State is per-browser-session now, not global**, so multiple people testing the app
  from different browsers/tabs no longer share (and corrupt) each other's motion history,
  vocalization state, and gesture-smoothing window.
- **Face/muzzle-centric emotion datasets.** Most public dog-emotion datasets are
  close-up face shots, so accuracy will drop if the whole dog's body is in frame —
  consider adding a detector to crop the dog/face first if you run into this.
- **4 broad emotion classes.** Swap `CLASSES` in `model.py` (and your data folders)
  if you want a different taxonomy (e.g. DEBIw's aggression/anxiety/contentment/fear).


## Human-rejection dog gate

The visual pipeline has a mandatory two-stage architecture:

`camera/upload -> COCO YOLO dog detector -> dog crop -> emotion/pose -> fusion`

The emotion model is **not** a dog-vs-human classifier. It is a four-class dog
emotion model, so it must never receive an arbitrary human frame directly.

### Required dog detector

Place `yolo11n.pt` beside `app.py`, or set:

```text
DOG_DETECTOR_WEIGHTS=C:\\path\\to\\yolo11n.pt
```

Ultralytics will attempt to resolve the official `yolo11n.pt` asset when the
standard filename is used and the environment has network access.

### Verify before starting FastAPI

```bash
python test_dog_detector.py human.jpg
python test_dog_detector.py dog.jpg
```

A human image should print `present: False`. A dog image should print
`present: True` with a confidence score.

The backend also exposes:

```text
GET /dog_gate_status
```

The frontend cannot bypass the gate; older `skip_dog_gate` client fields are
accepted for compatibility but ignored by the server.
