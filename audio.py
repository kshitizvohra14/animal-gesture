"""
audio.py — Detects dog vocalizations (bark / growl / whimper / howl) from
a short raw-audio buffer using YAMNet, a general-purpose sound-event model
pretrained on AudioSet (521 classes). We don't train our own bark classifier
here — YAMNet already includes dog vocalization classes out of the box, so
this is transfer learning at inference time: run the pretrained model, then
keep only the classes we care about.

Expects mono float32 audio at 16kHz (see prepare instructions in app.py /
index.html — the browser sends raw PCM, not a compressed file, so no
audio-decoding library like ffmpeg/pydub is needed here).
"""

import csv
import time
import traceback
import numpy as np
import tensorflow as tf
# tensorflow_hub is imported lazily in _load to allow test stubs without requiring the package.

_yamnet_model = None
_class_names = None
_load_failed = False  # set on failure, but retried after _RETRY_COOLDOWN_SECONDS
_last_load_attempt = 0.0
_last_load_error = None

# How long to wait before retrying a failed load. YAMNet's first load
# downloads ~15MB from https://tfhub.dev, which fails if that request
# happens during a transient network hiccup (or before the network/proxy is
# up on the host). Previously a single failed attempt set _load_failed
# permanently for the process lifetime, so vocalization silently returned
# "Quiet"/unavailable forever afterward, even once the network was fine
# again -- indistinguishable from a real (but misleading) "model unavailable"
# state until the server was manually restarted. Retrying periodically
# instead means a transient failure self-heals.
_RETRY_COOLDOWN_SECONDS = 60

# AudioSet class names (from YAMNet's class map) relevant to dog vocalization.
# Anything not in this set is treated as "Quiet" for this prototype, since we
# only care about dog sounds, not the full 521-class taxonomy.
DOG_SOUND_MAP = {
    "Bark": "Barking",
    "Bow-wow": "Barking",
    "Yip": "Barking",
    "Howl": "Howling",
    "Growling": "Growling",
    "Whimper (dog)": "Whimpering",
    "Whimper": "Whimpering",
}

# A dog-class score below this is treated as "not actually detected", even
# if it happened to be the top-ranked class among the ones we're filtering
# for. Named constant (not buried in classify logic) so it's easy to find
# and tune after listening to real false-positive/false-negative examples.
# 0.15 is the practical floor — values below this are almost always ambient
# room noise bleeding into the nearest dog-sound bucket rather than a real
# vocalization. Raise further (e.g. 0.25) if false positives persist.
MIN_VOCAL_SCORE = 0.10


def _load():
    """Loads YAMNet once. On failure, retries after _RETRY_COOLDOWN_SECONDS
    instead of failing permanently -- see classify_vocalization()."""
    global _yamnet_model, _class_names, _load_failed, _last_load_attempt, _last_load_error
    if _yamnet_model is not None:
        return
    if _load_failed and (time.time() - _last_load_attempt) < _RETRY_COOLDOWN_SECONDS:
        return

    _last_load_attempt = time.time()
    try:
        print("[audio.py] loading YAMNet (downloads ~15MB from tfhub.dev on first success)...")
        # Import tensorflow_hub lazily to avoid ImportError when the package is not installed.
        try:
            import tensorflow_hub as hub
        except ImportError as hub_exc:
            raise ImportError(
                "tensorflow_hub is not installed -- run: pip install tensorflow-hub"
            ) from hub_exc
        _yamnet_model = hub.load("https://tfhub.dev/google/yamnet/1")
        class_map_path = _yamnet_model.class_map_path().numpy().decode("utf-8")
        with tf.io.gfile.GFile(class_map_path) as f:
            reader = csv.DictReader(f)
            _class_names = [row["display_name"] for row in reader]
        _load_failed = False
        _last_load_error = None
        print("[audio.py] YAMNet loaded successfully.")
    except Exception as exc:
        # Print the full traceback, not just str(exc) -- "failed to load"
        # alone doesn't tell you whether it's a missing package, a blocked
        # network request to tfhub.dev, or something else entirely, and
        # those need different fixes.
        traceback.print_exc()
        print(f"[audio.py] YAMNet failed to load: {exc}")
        _yamnet_model = None
        _load_failed = True
        _last_load_error = str(exc)


def get_status() -> dict:
    """Diagnostic snapshot -- e.g. for a /health endpoint or manual check."""
    return {
        "loaded": _yamnet_model is not None,
        "load_failed": _load_failed,
        "last_error": _last_load_error,
        "seconds_until_retry": (
            max(0.0, _RETRY_COOLDOWN_SECONDS - (time.time() - _last_load_attempt))
            if _load_failed else 0.0
        ),
    }


def classify_vocalization(waveform: np.ndarray, top_k: int = 5) -> dict:
    """
    waveform: 1-D float32 numpy array, mono, sampled at 16kHz, values in [-1, 1].
    Returns {'label': str, 'confidence': float, 'raw_top_class': str, 'available': bool}.
    """
    _load()
    if _load_failed or _yamnet_model is None:
        return {"label": "Quiet", "confidence": 0.0, "raw_top_class": "model unavailable", "available": False}

    scores, _, _ = _yamnet_model(waveform)
    scores_np = scores.numpy()  # shape: (num_patches, 521) -- one row per ~0.96s patch

    # Max over time, not mean: a short bark inside an otherwise quiet 2.5s
    # window gets diluted below threshold by averaging across patches that
    # contain no bark at all. Max-pooling preserves the peak instead.
    peak_scores = np.max(scores_np, axis=0)

    # Directly scan dog vocalization classes across all AudioSet classes
    dog_idx = [i for i, n in enumerate(_class_names) if n in DOG_SOUND_MAP]
    if dog_idx:
        best_dog_idx = max(dog_idx, key=lambda i: peak_scores[i])
        best_dog_score = float(peak_scores[best_dog_idx])
        if best_dog_score >= MIN_VOCAL_SCORE:
            best_dog_class = _class_names[best_dog_idx]
            return {
                "label": DOG_SOUND_MAP[best_dog_class],
                "confidence": best_dog_score,
                "raw_top_class": best_dog_class,
                "available": True,
            }

    top_idx = int(np.argmax(peak_scores))
    return {
        "label": "Quiet",
        "confidence": float(peak_scores[top_idx]),
        "raw_top_class": _class_names[top_idx],
        "available": True,
    }


def decode_pcm16_base64(raw_bytes: bytes) -> np.ndarray:
    """Client sends 16-bit PCM samples; convert to float32 in [-1, 1]."""
    int16 = np.frombuffer(raw_bytes, dtype=np.int16)
    return (int16.astype(np.float32) / 32768.0)