"""
media_test.py — run REAL dog media (images, video clips, or your webcam)
through your trained emotion model + fusion.py and print one line per result.

Put this file next to fusion.py. Examples (Windows cmd / PowerShell):

  # 1) Sanity check: does my preprocessing match training? (expect ~your val accuracy)
  python media_test.py --model emotion_model.keras --labeled-dir "D:\\dataset\\val"

  # 2) One image or a folder of images
  python media_test.py --model emotion_model.keras --input test_media\\images

  # 3) A video clip -> one line per second + annotated output video
  python media_test.py --model emotion_model.keras --input test_media\\videos\\dog_run.mp4 --save out.mp4

  # 4) Live webcam (press q to quit)
  python media_test.py --model emotion_model.keras --input 0 --show

Which signal comes from where:
  * EMOTION      -> your Keras model, on the image / video frame (real).
  * MOTION       -> estimated from frame-to-frame change in a video (rough,
                    assumes a mostly static camera). Override with --motion.
                    For a single image there is no motion, so it defaults to Still.
  * VOCALIZATION -> set with --vocal (Barking/Growling/Howling/Whimpering/Quiet).
                    Swap in your own audio model where marked  >>> HOOK <<<  below.
  * POSTURE      -> optional, --posture Sitting|Standing|Lying.
"""

import argparse
import json
import sys
from collections import Counter, deque
from pathlib import Path

import cv2
import numpy as np

from fusion import fuse

IMG_EXT = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}
VID_EXT = {".mp4", ".avi", ".mov", ".mkv", ".webm", ".m4v"}
ANALYSIS_HZ = 5  # emotion + motion are re-estimated this many times per second


# ----------------------------------------------------------------------
# Model helpers
# ----------------------------------------------------------------------
def load_model(path):
    import tensorflow as tf  # imported lazily so --help works without TF
    return tf.keras.models.load_model(path)


def model_input_size(model):
    shape = model.input_shape
    if isinstance(shape, list):
        shape = shape[0]
    h, w = shape[1], shape[2]
    return (int(w) if w else 224, int(h) if h else 224)  # cv2 wants (width, height)


def preprocess(bgr, size, mode):
    rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
    rgb = cv2.resize(rgb, size).astype("float32")
    if mode == "mobilenet":      # tf.keras.applications.mobilenet_v2.preprocess_input -> [-1, 1]
        rgb = rgb / 127.5 - 1.0
    elif mode == "rescale":      # ImageDataGenerator(rescale=1./255) -> [0, 1]
        rgb = rgb / 255.0
    # mode == "none": model has preprocessing layers inside it
    return rgb[None, ...]


def predict_probs(model, bgr, size, mode):
    out = np.asarray(model.predict(preprocess(bgr, size, mode), verbose=0)[0], dtype="float64")
    if not np.isclose(out.sum(), 1.0, atol=1e-3) or (out < 0).any():  # logits -> softmax
        e = np.exp(out - out.max())
        out = e / e.sum()
    return out


def resolve_classes(args, model, model_path):
    if args.classes:
        classes = args.classes
    else:
        side = Path(model_path).with_name("class_names.json")
        if side.exists():
            classes = json.loads(side.read_text())
        else:
            # flow_from_directory / image_dataset_from_directory sort folder names alphabetically
            classes = ["angry", "happy", "relaxed", "sad"]
            print("WARNING: no --classes given, assuming alphabetical order:", classes,
                  "\n         Pass --classes in the SAME order as training if this is wrong.\n")
    n_out = int(model.output_shape[-1])
    if len(classes) != n_out:
        sys.exit(f"ERROR: model has {n_out} outputs but {len(classes)} class names: {classes}")
    return classes


# ----------------------------------------------------------------------
# Motion estimate from video (rough stand-in; replace with your own if you have one)
# ----------------------------------------------------------------------
class MotionEstimator:
    def __init__(self, still_thr, run_thr, window=ANALYSIS_HZ):
        self.still_thr, self.run_thr = still_thr, run_thr
        self.prev = None
        self.vals = deque(maxlen=window)

    def update(self, bgr):
        g = cv2.cvtColor(cv2.resize(bgr, (160, 90)), cv2.COLOR_BGR2GRAY).astype("float32")
        if self.prev is not None:
            self.vals.append(float(np.mean(np.abs(g - self.prev))))
        self.prev = g

    def label(self):
        if not self.vals:
            return "Still"
        m = float(np.mean(self.vals))
        if m < self.still_thr:
            return "Still"
        return "Walking" if m < self.run_thr else "Running"


# ----------------------------------------------------------------------
# One fusion call -> one line
# ----------------------------------------------------------------------
def fuse_once(probs, classes, motion, vocal, posture, min_conf):
    top = int(np.argmax(probs))
    label, conf = classes[top], float(probs[top])
    if conf < min_conf:
        label = "Uncertain"  # emotion confidence gate failed
    # >>> HOOK <<< replace `vocal` with your own audio model's label for this moment if you have one
    pose = {"posture": posture} if posture else None
    return fuse(label, conf, motion, vocal, pose_cues=pose)


def annotate(frame, line):
    h, w = frame.shape[:2]
    scale = 0.6
    while cv2.getTextSize(line, cv2.FONT_HERSHEY_SIMPLEX, scale, 2)[0][0] > w - 20 and scale > 0.25:
        scale -= 0.05
    cv2.rectangle(frame, (0, 0), (w, 34), (0, 0, 0), -1)
    cv2.putText(frame, line, (10, 24), cv2.FONT_HERSHEY_SIMPLEX, scale, (255, 255, 255), 2, cv2.LINE_AA)
    return frame


# ----------------------------------------------------------------------
# Modes
# ----------------------------------------------------------------------
def run_labeled_dir(args, model, classes, size):
    """Accuracy on a folder of class-named subfolders: confirms preprocessing/class order match training."""
    root = Path(args.labeled_dir)
    total = correct = 0
    per_class = {c: [0, 0] for c in classes}
    for sub in sorted(p for p in root.iterdir() if p.is_dir()):
        if sub.name not in classes:
            print(f"skipping folder '{sub.name}' (not in classes {classes})")
            continue
        files = [f for f in sorted(sub.iterdir()) if f.suffix.lower() in IMG_EXT][: args.max_per_class]
        for f in files:
            img = cv2.imread(str(f))
            if img is None:
                continue
            pred = classes[int(np.argmax(predict_probs(model, img, size, args.preprocess)))]
            per_class[sub.name][1] += 1
            per_class[sub.name][0] += int(pred == sub.name)
            total += 1
            correct += int(pred == sub.name)
    for c, (ok, n) in per_class.items():
        if n:
            print(f"  {c:<12} {ok}/{n}  ({ok / n:.0%})")
    if total:
        print(f"\nOverall: {correct}/{total} = {correct / total:.1%}  (preprocess={args.preprocess})")
        print("If this is far below your validation accuracy, try --preprocess rescale / none, or fix --classes order.")
    else:
        print("No images found. Expected subfolders named after the classes.")


def run_images(args, model, classes, size, paths):
    out_dir = Path(args.save) if args.save else None
    if out_dir:
        out_dir.mkdir(parents=True, exist_ok=True)
    for p in paths:
        img = cv2.imread(str(p))
        if img is None:
            print(f"{p.name}: could not read image")
            continue
        probs = predict_probs(model, img, size, args.preprocess)
        res = fuse_once(probs, classes, args.motion or "Still", args.vocal, args.posture, args.min_conf)
        print(f"{p.name}: {res['line']}")
        if out_dir:
            cv2.imwrite(str(out_dir / p.name), annotate(img.copy(), res["line"]))


def run_video(args, model, classes, size):
    src = int(args.input) if args.input.isdigit() else args.input
    cap = cv2.VideoCapture(src)
    if not cap.isOpened():
        sys.exit(f"ERROR: cannot open {args.input}")
    fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
    step = max(1, int(round(fps / ANALYSIS_HZ)))
    motion_est = MotionEstimator(args.still_thr, args.run_thr)
    recent = deque(maxlen=ANALYSIS_HZ)  # ~1 s of emotion probabilities, averaged to stop flicker
    writer = None
    line, gestures, frame_i = "Dog gesture: (warming up)", Counter(), 0

    while True:
        ok, frame = cap.read()
        if not ok:
            break
        if frame_i % step == 0:
            motion_est.update(frame)
            recent.append(predict_probs(model, frame, size, args.preprocess))
            res = fuse_once(np.mean(recent, axis=0), classes, args.motion or motion_est.label(),
                            args.vocal, args.posture, args.min_conf)
            line = res["line"]
            gestures[res["gesture"] if res["matched_rule"] else "(no confident match)"] += 1
            if frame_i % (step * ANALYSIS_HZ) == 0:  # print about once per second
                print(f"t={frame_i / fps:5.1f}s  {line}")
        annotated = annotate(frame, line)
        if args.save:
            if writer is None:
                h, w = annotated.shape[:2]
                writer = cv2.VideoWriter(args.save, cv2.VideoWriter_fourcc(*"mp4v"), fps, (w, h))
            writer.write(annotated)
        if args.show:
            cv2.imshow("dog gesture", annotated)
            if cv2.waitKey(1) & 0xFF == ord("q"):
                break
        frame_i += 1

    cap.release()
    if writer:
        writer.release()
        print(f"\nSaved annotated video: {args.save}")
    if args.show:
        cv2.destroyAllWindows()
    if gestures:
        top, n = gestures.most_common(1)[0]
        print(f"Most frequent reading: {top}  ({n}/{sum(gestures.values())} analysis steps)")


def main():
    ap = argparse.ArgumentParser(description="Run dog media through emotion model + fusion.py")
    ap.add_argument("--model", required=True, help="path to your saved Keras model (.keras / .h5)")
    ap.add_argument("--input", help="image, folder of images, video file, or webcam index like 0")
    ap.add_argument("--labeled-dir", help="folder with one subfolder per class, to check accuracy")
    ap.add_argument("--classes", nargs="+", help="class names in training order")
    ap.add_argument("--preprocess", choices=["mobilenet", "rescale", "none"], default="mobilenet",
                    help="must match train.py (default: mobilenet [-1,1])")
    ap.add_argument("--motion", choices=["Still", "Walking", "Running"], help="override estimated motion")
    ap.add_argument("--vocal", default="Quiet", choices=["Barking", "Growling", "Howling", "Whimpering", "Quiet"],
                    help="vocalization label for this media (default Quiet)")
    ap.add_argument("--posture", choices=["Sitting", "Standing", "Lying"])
    ap.add_argument("--min-conf", type=float, default=0.0,
                    help="emotion confidence below this is treated as 'Uncertain' (default off)")
    ap.add_argument("--still-thr", type=float, default=1.5, help="motion score below this = Still")
    ap.add_argument("--run-thr", type=float, default=6.0, help="motion score above this = Running")
    ap.add_argument("--max-per-class", type=int, default=100)
    ap.add_argument("--save", help="output video path (video/webcam) or output folder (images)")
    ap.add_argument("--show", action="store_true", help="show a live window (needs opencv-python, not -headless)")
    args = ap.parse_args()

    model = load_model(args.model)
    classes = resolve_classes(args, model, args.model)
    size = model_input_size(model)
    print(f"Model input {size[0]}x{size[1]}, classes: {classes}, preprocess: {args.preprocess}\n")

    if args.labeled_dir:
        return run_labeled_dir(args, model, classes, size)
    if not args.input:
        sys.exit("Give --input (image/folder/video/webcam index) or --labeled-dir.")

    if args.input.isdigit() or Path(args.input).suffix.lower() in VID_EXT:
        return run_video(args, model, classes, size)
    p = Path(args.input)
    paths = sorted(f for f in p.iterdir() if f.suffix.lower() in IMG_EXT) if p.is_dir() else [p]
    if not paths:
        sys.exit(f"No images found in {p}")
    run_images(args, model, classes, size, paths)


if __name__ == "__main__":
    main()
