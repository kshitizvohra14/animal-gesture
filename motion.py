"""
motion.py

Lightweight motion-state classifier: Still / Walking / Running, based on
frame-to-frame pixel differencing from a webcam. No dataset or model
needed -- this is a heuristic that works out of the box.

(See the project roadmap for the planned upgrade: replacing this with a
YOLO dog-pose model and computing motion from keypoint velocity instead
of raw pixel change, which will be more robust to lighting/camera shake.)

Usage:
    python motion.py                 # opens webcam, prints live motion state
"""

import argparse
from collections import deque

import cv2
import numpy as np


def to_gray_small(frame: np.ndarray, size=(160, 120)) -> np.ndarray:
    """Downsamples and blurs a frame for lightweight motion differencing."""
    resized = cv2.resize(frame, size)
    gray = cv2.cvtColor(resized, cv2.COLOR_BGR2GRAY)
    blurred = cv2.GaussianBlur(gray, (15, 15), 0)
    return blurred


def classify_motion(prev_gray: np.ndarray, curr_gray: np.ndarray,
                    still_thresh: float = 0.01, run_thresh: float = 0.06) -> dict:
    """Classifies motion between two consecutive small grayscale frames."""
    if prev_gray is None or curr_gray is None:
        return {"label": "Still", "score": 0.0}

    diff = cv2.absdiff(prev_gray, curr_gray)
    _, thresh = cv2.threshold(diff, 25, 255, cv2.THRESH_BINARY)
    moving_ratio = float(np.count_nonzero(thresh) / thresh.size)

    if moving_ratio < still_thresh:
        label = "Still"
    elif moving_ratio < run_thresh:
        label = "Walking"
    else:
        label = "Running"

    return {"label": label, "score": round(moving_ratio, 3)}


class MotionClassifier:
    """Classifies motion state from a rolling window of frame-difference ratios."""

    def __init__(self, window_size: int = 10, still_thresh: float = 0.01, run_thresh: float = 0.06):
        self.window = deque(maxlen=window_size)
        self.still_thresh = still_thresh
        self.run_thresh = run_thresh
        self._prev_gray = None

    def update(self, frame: np.ndarray) -> str:
        """Feed one BGR frame, get back the current motion state."""
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        gray = cv2.GaussianBlur(gray, (21, 21), 0)

        if self._prev_gray is None:
            self._prev_gray = gray
            return "Still"

        diff = cv2.absdiff(self._prev_gray, gray)
        _, thresh = cv2.threshold(diff, 25, 255, cv2.THRESH_BINARY)
        moving_ratio = np.count_nonzero(thresh) / thresh.size

        self._prev_gray = gray
        self.window.append(moving_ratio)
        avg_ratio = sum(self.window) / len(self.window)

        if avg_ratio < self.still_thresh:
            return "Still"
        elif avg_ratio < self.run_thresh:
            return "Walking"
        else:
            return "Running"


def main():
    ap = argparse.ArgumentParser(description="Live motion-state classifier from webcam")
    ap.add_argument("--camera", type=int, default=0, help="Camera index")
    ap.add_argument("--show", action="store_true", help="Show the video window with overlay")
    args = ap.parse_args()

    cap = cv2.VideoCapture(args.camera)
    if not cap.isOpened():
        raise SystemExit(f"Could not open camera index {args.camera}")

    clf = MotionClassifier()
    print("Running motion classifier. Press 'q' in the video window (or Ctrl+C) to quit.")

    try:
        while True:
            ok, frame = cap.read()
            if not ok:
                break

            state = clf.update(frame)

            if args.show:
                cv2.putText(frame, f"Motion: {state}", (10, 30),
                            cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
                cv2.imshow("motion.py", frame)
                if cv2.waitKey(1) & 0xFF == ord("q"):
                    break
            else:
                print(state)
    except KeyboardInterrupt:
        pass
    finally:
        cap.release()
        cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
