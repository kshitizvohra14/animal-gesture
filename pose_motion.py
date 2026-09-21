"""
pose_motion.py

Dog pose + motion classification using YOLO pose keypoints.

Features:
    - Motion: Still / Walking / Running
    - Posture: Standing / Sitting / Lying
    - Tail: Low / Normal
    - Temporal smoothing
    - Motion/posture consistency checks
    - Confidence-aware keypoints
    - Prevents impossible/noisy combinations such as:
          Running + Sitting
          Running + Lying
          Walking + Lying

The system separates:
    1. Raw pose estimation
    2. Raw motion estimation
    3. Temporal smoothing
    4. State consistency
"""

import os
from collections import deque

import numpy as np


# ============================================================
# MODEL
# ============================================================

MODEL_PATH = os.path.join(
    os.path.dirname(__file__),
    "models",
    "dog_pose_best.pt"
)


# ============================================================
# KEYPOINTS
# ============================================================

KEYPOINT_NAMES = [
    "front_left_paw",
    "front_left_knee",
    "front_left_elbow",

    "rear_left_paw",
    "rear_left_knee",
    "rear_left_elbow",

    "front_right_paw",
    "front_right_knee",
    "front_right_elbow",

    "rear_right_paw",
    "rear_right_knee",
    "rear_right_elbow",

    "tail_start",
    "tail_end",

    "left_ear_base",
    "right_ear_base",

    "nose",
    "chin",

    "left_ear_tip",
    "right_ear_tip",

    "left_eye",
    "right_eye",

    "withers",
    "throat",
]


KP = {
    name: i
    for i, name in enumerate(KEYPOINT_NAMES)
}


PAW_INDICES = [
    KP["front_left_paw"],
    KP["rear_left_paw"],
    KP["front_right_paw"],
    KP["rear_right_paw"],
]


# ============================================================
# MOTION THRESHOLDS
# ============================================================

# Normalized displacement per frame.
#
# These should eventually be calibrated using your own videos.

STILL_MAX = 0.008
WALKING_MAX = 0.025


# ============================================================
# TEMPORAL SETTINGS
# ============================================================

MOTION_HISTORY = 5
POSTURE_HISTORY = 7


# ============================================================
# MODEL LOADING
# ============================================================

def load_pose_model():
    """
    Load trained YOLO pose model.

    Returns:
        YOLO model
        or None if model doesn't exist.
    """

    if not os.path.exists(MODEL_PATH):

        print(
            f"[pose_motion] No pose model found at:\n"
            f"{MODEL_PATH}\n"
            f"Falling back to original motion system."
        )

        return None

    from ultralytics import YOLO

    print(
        f"[pose_motion] Loading pose model:\n"
        f"{MODEL_PATH}"
    )

    return YOLO(MODEL_PATH)


# ============================================================
# KEYPOINT EXTRACTION
# ============================================================

def extract_keypoints(pose_model, bgr_frame):
    """
    Run YOLO pose detection.

    Returns:

        keypoints_xy
            Shape: (24, 2)

        bbox_diagonal
            Bounding box diagonal

        pose_confidence
            Detection confidence

    or:

        (None, None, 0.0)
    """

    results = pose_model(
        bgr_frame,
        verbose=False
    )

    if (
        not results
        or results[0].keypoints is None
        or len(results[0].keypoints) == 0
    ):
        return None, None, 0.0


    result = results[0]

    boxes = result.boxes


    if boxes is not None and len(boxes) > 0:

        best_idx = int(
            boxes.conf.argmax()
        )

        pose_confidence = float(
            boxes.conf[best_idx].item()
        )

    else:

        best_idx = 0
        pose_confidence = 0.0


    kpts = (
        result
        .keypoints
        .xy[best_idx]
        .cpu()
        .numpy()
    )


    # --------------------------------------------------------
    # Bounding box
    # --------------------------------------------------------

    if boxes is not None and len(boxes) > 0:

        x1, y1, x2, y2 = (
            boxes
            .xyxy[best_idx]
            .cpu()
            .numpy()
        )

        diagonal = float(
            np.hypot(
                x2 - x1,
                y2 - y1
            )
        )

    else:

        diagonal = float(
            np.hypot(
                bgr_frame.shape[1],
                bgr_frame.shape[0]
            )
        )


    return (
        kpts,
        diagonal,
        pose_confidence
    )


# ============================================================
# MOTION
# ============================================================

def classify_motion_from_pose(
    prev_kpts,
    curr_kpts,
    bbox_diagonal
):
    """
    Estimate motion from paw displacement.

    Returns:

        {
            "label": "Still" / "Walking" / "Running",
            "score": float
        }
    """

    if (
        prev_kpts is None
        or curr_kpts is None
        or bbox_diagonal is None
        or bbox_diagonal <= 0
    ):
        return {
            "label": "Still",
            "score": 0.0
        }


    displacements = []


    for idx in PAW_INDICES:

        if (
            idx >= len(prev_kpts)
            or idx >= len(curr_kpts)
        ):
            continue


        prev = prev_kpts[idx]
        curr = curr_kpts[idx]


        # Ignore invalid points
        if (
            np.all(prev == 0)
            or np.all(curr == 0)
        ):
            continue


        distance = np.linalg.norm(
            curr - prev
        )

        displacements.append(
            distance
        )


    if not displacements:

        return {
            "label": "Still",
            "score": 0.0
        }


    score = (
        float(np.mean(displacements))
        / bbox_diagonal
    )


    if score < STILL_MAX:

        label = "Still"

    elif score < WALKING_MAX:

        label = "Walking"

    else:

        label = "Running"


    return {
        "label": label,
        "score": round(score, 4)
    }


# ============================================================
# POSTURE
# ============================================================

def classify_posture(
    kpts,
    bbox_diagonal
):
    """
    Estimate posture using body geometry.

    Returns:

        Standing
        Sitting
        Lying
        Unknown
    """

    if (
        kpts is None
        or bbox_diagonal is None
        or bbox_diagonal <= 0
    ):
        return {
            "label": "Unknown",
            "confidence": 0.0
        }


    try:

        withers_y = (
            kpts[KP["withers"]][1]
        )

        rear_knee_y = np.mean([
            kpts[KP["rear_left_knee"]][1],
            kpts[KP["rear_right_knee"]][1]
        ])

        rear_paw_y = np.mean([
            kpts[KP["rear_left_paw"]][1],
            kpts[KP["rear_right_paw"]][1]
        ])

        front_paw_y = np.mean([
            kpts[KP["front_left_paw"]][1],
            kpts[KP["front_right_paw"]][1]
        ])

    except (IndexError, KeyError):

        return {
            "label": "Unknown",
            "confidence": 0.0
        }


    # --------------------------------------------------------
    # Geometry
    # --------------------------------------------------------

    withers_to_ground = (
        rear_paw_y - withers_y
    ) / bbox_diagonal


    rear_leg_extension = (
        rear_paw_y - rear_knee_y
    ) / bbox_diagonal


    front_rear_paw_gap = (
        abs(front_paw_y - rear_paw_y)
        / bbox_diagonal
    )


    # --------------------------------------------------------
    # Lying
    # --------------------------------------------------------

    if withers_to_ground < 0.15:

        return {
            "label": "Lying",
            "confidence": 0.7
        }


    # --------------------------------------------------------
    # Sitting
    # --------------------------------------------------------

    if (
        rear_leg_extension < 0.08
        and front_rear_paw_gap < 0.10
    ):

        return {
            "label": "Sitting",
            "confidence": 0.65
        }


    # --------------------------------------------------------
    # Standing
    # --------------------------------------------------------

    return {
        "label": "Standing",
        "confidence": 0.65
    }


# ============================================================
# TAIL
# ============================================================

def compute_tail_low(
    kpts,
    bbox_diagonal
):
    """
    Tail is considered low if both tail points
    are below the withers.
    """

    if (
        kpts is None
        or bbox_diagonal is None
        or bbox_diagonal <= 0
    ):
        return False


    try:

        withers_y = (
            kpts[KP["withers"]][1]
        )

        tail_start_y = (
            kpts[KP["tail_start"]][1]
        )

        tail_end_y = (
            kpts[KP["tail_end"]][1]
        )

    except (IndexError, KeyError):

        return False


    return (
        tail_start_y > withers_y
        and
        tail_end_y > withers_y
    )


# ============================================================
# STATE CONSISTENCY
# ============================================================

def enforce_motion_posture_consistency(
    motion,
    posture
):
    """
    Prevent physically inconsistent states.

    Examples:

        Running + Sitting
        Running + Lying
        Walking + Lying

    Motion has priority because it is based on
    temporal information.
    """

    motion_label = motion["label"]
    posture_label = posture["label"]


    # --------------------------------------------------------
    # RUNNING
    # --------------------------------------------------------

    if motion_label == "Running":

        if posture_label in (
            "Sitting",
            "Lying"
        ):

            posture_label = "Standing"

            posture["confidence"] *= 0.5


    # --------------------------------------------------------
    # WALKING
    # --------------------------------------------------------

    elif motion_label == "Walking":

        if posture_label == "Lying":

            posture_label = "Standing"

            posture["confidence"] *= 0.5


    posture["label"] = posture_label

    return posture


# ============================================================
# TEMPORAL CLASSIFIER
# ============================================================

class PoseStateTracker:

    """
    Maintains temporal history so that one bad frame
    doesn't immediately change the dog's state.
    """

    def __init__(
        self,
        motion_history=MOTION_HISTORY,
        posture_history=POSTURE_HISTORY
    ):

        self.motion_history = deque(
            maxlen=motion_history
        )

        self.posture_history = deque(
            maxlen=posture_history
        )

        self.prev_kpts = None


    # --------------------------------------------------------
    # Majority vote
    # --------------------------------------------------------

    @staticmethod
    def majority_vote(history):

        if not history:

            return "Unknown"

        counts = {}

        for item in history:

            counts[item] = (
                counts.get(item, 0)
                + 1
            )

        return max(
            counts,
            key=counts.get
        )


    # --------------------------------------------------------
    # Update
    # --------------------------------------------------------

    def update(
        self,
        kpts,
        bbox_diagonal
    ):

        # --------------------------------------------
        # Motion
        # --------------------------------------------

        motion = classify_motion_from_pose(
            self.prev_kpts,
            kpts,
            bbox_diagonal
        )


        # --------------------------------------------
        # Posture
        # --------------------------------------------

        posture = classify_posture(
            kpts,
            bbox_diagonal
        )


        # --------------------------------------------
        # Store history
        # --------------------------------------------

        self.motion_history.append(
            motion["label"]
        )

        self.posture_history.append(
            posture["label"]
        )


        # --------------------------------------------
        # Smooth
        # --------------------------------------------

        smooth_motion = self.majority_vote(
            self.motion_history
        )

        smooth_posture = self.majority_vote(
            self.posture_history
        )


        # --------------------------------------------
        # Replace raw labels
        # --------------------------------------------

        motion["label"] = smooth_motion

        posture["label"] = smooth_posture


        # --------------------------------------------
        # Consistency
        # --------------------------------------------

        posture = enforce_motion_posture_consistency(
            motion,
            posture
        )


        # --------------------------------------------
        # Save previous frame
        # --------------------------------------------

        if kpts is not None:

            self.prev_kpts = kpts.copy()


        return {
            "motion": motion,
            "posture": posture,
            "tail_low": compute_tail_low(
                kpts,
                bbox_diagonal
            )
        }


    # --------------------------------------------------------
    # Reset
    # --------------------------------------------------------

    def reset(self):

        self.motion_history.clear()
        self.posture_history.clear()

        self.prev_kpts = None