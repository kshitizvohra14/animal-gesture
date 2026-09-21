import os
from ultralytics import YOLO

COCO_DOG_CLASS_ID = 16
MIN_DETECTION_CONFIDENCE = 0.45

_detector = None


def load_dog_detector():
    global _detector
    if _detector is None:
        weights_path = os.environ.get("DOG_DETECTOR_WEIGHTS", "yolo11n.pt")
        try:
            _detector = YOLO(weights_path)
        except Exception as exc:
            print(f"[dog_detector] Could not load YOLO weights ({weights_path}): {exc}")
            _detector = None
    return _detector


def detect_dog(frame_bgr, conf_threshold: float = MIN_DETECTION_CONFIDENCE) -> dict:
    """
    Returns {"present": bool, "confidence": float, "bbox": (x1,y1,x2,y2) or None}.
    Picks the highest-confidence COCO "dog" detection in frame, if any.
    """
    detector = load_dog_detector()
    if detector is None:
        # Fail gracefully: skip dog presence check
        return {"present": True, "confidence": 0.0, "bbox": None}

    results = detector(frame_bgr, verbose=False, classes=[COCO_DOG_CLASS_ID])
    if not results or results[0].boxes is None or len(results[0].boxes) == 0:
        return {"present": False, "confidence": 0.0, "bbox": None}

    boxes = results[0].boxes
    best_idx = int(boxes.conf.argmax())
    confidence = float(boxes.conf[best_idx])

    if confidence < conf_threshold:
        return {"present": False, "confidence": confidence, "bbox": None}

    x1, y1, x2, y2 = boxes.xyxy[best_idx].tolist()
    return {"present": True, "confidence": confidence, "bbox": (x1, y1, x2, y2)}
