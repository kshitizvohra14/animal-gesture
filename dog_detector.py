"""
dog_detector.py — hard dog-presence gate.

The emotion classifier is dog-only. Therefore every visual inference path must
first pass through this detector. A frame/image that does not contain a
confident COCO "dog" detection is rejected and NEVER reaches the dog emotion
classifier.
"""

import os
from pathlib import Path
from typing import Optional

from ultralytics import YOLO

COCO_DOG_CLASS_ID = 16
MIN_DETECTION_CONFIDENCE = float(
    os.environ.get("DOG_DETECTOR_MIN_CONFIDENCE", "0.55")
)

BASE_DIR = Path(__file__).resolve().parent
DEFAULT_WEIGHTS = BASE_DIR / "yolo11n.pt"

_detector: Optional[YOLO] = None
_detector_error: Optional[str] = None


def _candidate_paths():
    configured = os.environ.get("DOG_DETECTOR_WEIGHTS", "").strip()
    if configured:
        yield Path(configured).expanduser().resolve()
    yield DEFAULT_WEIGHTS


def load_dog_detector():
    """Load the COCO detector. Never silently fail open."""
    global _detector, _detector_error

    if _detector is not None:
        return _detector

    last_error = None

    for candidate in _candidate_paths():
        try:
            # Ultralytics can resolve/download the official model asset when a
            # standard model name such as yolo11n.pt is supplied.
            _detector = YOLO(str(candidate) if candidate.exists() else candidate.name)
            print(f"[dog_detector] READY: {_detector.ckpt_path if hasattr(_detector, 'ckpt_path') else candidate}")
            _detector_error = None
            return _detector
        except Exception as exc:
            last_error = exc
            print(f"[dog_detector] Could not load {candidate}: {exc}")

    _detector_error = str(last_error) if last_error else "unknown detector load error"
    _detector = None
    print("[dog_detector] DOG GATE DISABLED because the detector could not be loaded.")
    print("[dog_detector] Put yolo11n.pt beside app.py or set DOG_DETECTOR_WEIGHTS.")
    return None


def detect_dog(frame_bgr, conf_threshold: float = MIN_DETECTION_CONFIDENCE) -> dict:
    """Return the best COCO dog detection; detector failure = no dog."""
    detector = load_dog_detector()

    if detector is None:
        return {
            "present": False,
            "confidence": 0.0,
            "bbox": None,
            "available": False,
            "error": _detector_error,
        }

    try:
        results = detector(
            frame_bgr,
            imgsz=640,
            conf=conf_threshold,
            classes=[COCO_DOG_CLASS_ID],
            verbose=False,
        )
    except Exception as exc:
        return {
            "present": False,
            "confidence": 0.0,
            "bbox": None,
            "available": True,
            "error": str(exc),
        }

    if not results or results[0].boxes is None or len(results[0].boxes) == 0:
        return {
            "present": False,
            "confidence": 0.0,
            "bbox": None,
            "available": True,
            "error": None,
        }

    boxes = results[0].boxes
    best_idx = int(boxes.conf.argmax())
    confidence = float(boxes.conf[best_idx])

    if confidence < conf_threshold:
        return {
            "present": False,
            "confidence": confidence,
            "bbox": None,
            "available": True,
            "error": None,
        }

    x1, y1, x2, y2 = boxes.xyxy[best_idx].tolist()
    return {
        "present": True,
        "confidence": confidence,
        "bbox": (x1, y1, x2, y2),
        "available": True,
        "error": None,
    }


def crop_dog(frame_bgr, detection: dict, padding: float = 0.10):
    """Crop the detected dog with small context padding."""
    if not detection.get("present") or not detection.get("bbox"):
        return None, None

    h, w = frame_bgr.shape[:2]
    x1, y1, x2, y2 = map(float, detection["bbox"])
    bw = max(1.0, x2 - x1)
    bh = max(1.0, y2 - y1)

    x1 = max(0, int(x1 - bw * padding))
    y1 = max(0, int(y1 - bh * padding))
    x2 = min(w, int(x2 + bw * padding))
    y2 = min(h, int(y2 + bh * padding))

    if x2 <= x1 or y2 <= y1:
        return None, None

    return frame_bgr[y1:y2, x1:x2].copy(), (x1, y1)