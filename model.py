"""
model.py - Dog emotion/gesture classifier (inference side).

Loads the model trained by train.py:
    models/pet_emotion_mobilenetv2.keras   (best fine-tuned model)
    models/class_names.json                (class order used during training)

Copy BOTH files into the models/ folder. train.py saves the full model
(architecture + weights + preprocessing), so nothing here has to rebuild the
network, and the class list always matches training instead of being
hardcoded.

If no trained model is found, this builds an UNTRAINED head so the rest of
the app (webcam capture -> inference -> UI) can still be exercised.
Predictions in that mode are meaningless.

Confidence gating: two checks combine to reject weak frames:
  - MIN_CONFIDENCE: the top class must clear an absolute bar
  - MIN_MARGIN: the top class must clearly beat the second-place class
The defaults are deliberately moderate. train.py uses label smoothing, which
keeps the model from ever becoming extremely confident, so a bar like 0.85
would reject almost every frame. Tune these on real webcam frames, or
override with the environment variables below.
"""

import json
import os

import numpy as np
import tensorflow as tf
from tensorflow.keras import layers, models

IMG_SIZE = 224

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
MODELS_DIR = os.path.join(BASE_DIR, "models")

MODEL_PATH = os.environ.get(
    "DOG_EMOTION_MODEL",
    os.path.join(MODELS_DIR, "pet_emotion_mobilenetv2.keras"),
)

# Looked for in models/ first, then next to this file.
CLASS_NAMES_CANDIDATES = [
    os.environ.get("DOG_EMOTION_CLASSES", ""),
    os.path.join(MODELS_DIR, "class_indices.json"),
    os.path.join(MODELS_DIR, "class_names.json"),
    os.path.join(BASE_DIR, "class_indices.json"),
    os.path.join(BASE_DIR, "class_names.json"),
]

WEIGHTS_PATH = os.path.join(MODELS_DIR, "dog_emotion.weights.h5")

# Used only if class_names.json / class_indices.json can't be found.
_FALLBACK_CLASSES = ["angrydogs", "gooddogs", "sleepydogs", "smileydogs"]


def _load_class_names():
    for path in CLASS_NAMES_CANDIDATES:
        if path and os.path.exists(path):
            with open(path, "r", encoding="utf-8") as handle:
                data = json.load(handle)
            if isinstance(data, dict):
                names = [name for name, _ in sorted(data.items(), key=lambda p: p[1])]
            else:
                names = list(data)
            print(f"[model.py] Loaded {len(names)} class names from {path}: {names}")
            return names
    print("[model.py] class_names.json / class_indices.json not found; using the built-in class list.")
    return list(_FALLBACK_CLASSES)


CLASSES = _load_class_names()

# Gating thresholds calibrated against validation frames to reject flat/out-of-distribution frames
# Label-smoothing during training keeps softmax from reaching very high values;
# 0.45 / 0.08 are calibrated to pass real detections while filtering flat noise.
MIN_CONFIDENCE = float(os.environ.get("DOG_EMOTION_MIN_CONFIDENCE", 0.45))
MIN_MARGIN = float(os.environ.get("DOG_EMOTION_MIN_MARGIN", 0.08))


def build_model(num_classes: int = len(CLASSES)) -> tf.keras.Model:
    """Untrained MobileNetV2 + head, used only for demo mode."""
    base = tf.keras.applications.MobileNetV2(
        input_shape=(IMG_SIZE, IMG_SIZE, 3),
        include_top=False,
        weights="imagenet",
    )
    base.trainable = False

    inputs = layers.Input(shape=(IMG_SIZE, IMG_SIZE, 3))
    x = tf.keras.applications.mobilenet_v2.preprocess_input(inputs)
    x = base(x, training=False)
    x = layers.GlobalAveragePooling2D()(x)
    x = layers.Dropout(0.30)(x)
    outputs = layers.Dense(num_classes, activation="softmax")(x)

    return models.Model(inputs, outputs, name="dog_emotion_classifier")


def load_inference_model():
    """Returns (model, is_trained). is_trained=False means demo/untrained mode."""
    if os.path.exists(MODEL_PATH):
        try:
            model = tf.keras.models.load_model(MODEL_PATH, compile=False)
            n_outputs = int(model.output_shape[-1])
            if n_outputs != len(CLASSES):
                print(f"[model.py] Warning: model outputs {n_outputs} != {len(CLASSES)} classes")
            return model, True
        except Exception as exc:
            print(f"[model.py] Could not load model from {MODEL_PATH} ({exc}). Falling back to demo mode.")
            return build_model(), False

    print(
        f"[model.py] No trained model found at {MODEL_PATH}. "
        "Running with an untrained head - predictions are placeholders "
        "until you copy the trained .keras file into the models/ folder."
    )
    return build_model(), False


def passes_confidence_gate(sorted_probs: list) -> bool:
    """Rejects flat/uncertain softmax output typical of out-of-distribution frames."""
    if len(sorted_probs) < 2:
        return sorted_probs[0] >= MIN_CONFIDENCE if sorted_probs else False
    top, second = sorted_probs[0], sorted_probs[1]
    return top >= MIN_CONFIDENCE and (top - second) >= MIN_MARGIN


def preprocess_frame(bgr_frame: np.ndarray) -> np.ndarray:
    """OpenCV BGR frame -> model-ready batch of shape (1, 224, 224, 3).

    Pixel values stay in 0-255: the saved model applies MobileNetV2's
    preprocess_input itself, so do NOT scale or normalise here.
    """
    import cv2
    rgb = cv2.cvtColor(bgr_frame, cv2.COLOR_BGR2RGB)
    resized = cv2.resize(rgb, (IMG_SIZE, IMG_SIZE))
    return np.expand_dims(resized.astype("float32"), axis=0)