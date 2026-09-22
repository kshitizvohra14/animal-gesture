"""
pose_motion.py

Dog-pose features for Paw Fussion.

This module does four jobs:
    1. Extract and validate dog pose keypoints.
    2. Estimate motion from normalized paw movement.
    3. Classify posture.
    4. Derive temporal behavior cues such as tail wagging.

Important design choice:
    The emotion model must NOT decide whether an image contains a dog.
    dog_detector.py should gate the frame first. This module assumes the
    incoming crop/frame has already passed that dog-presence gate.
"""

import math
import os
from collections import deque

import numpy as np


# ============================================================
# MODEL
# ============================================================

MODEL_PATH = os.path.join(
    os.path.dirname(__file__),
    "models",
    "dog_pose_best.pt",
)

MIN_POSE_CONFIDENCE = 0.45
MIN_VALID_KEYPOINTS = 8
KEYPOINT_CONFIDENCE = 0.25


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

KP = {name: i for i, name in enumerate(KEYPOINT_NAMES)}

PAW_INDICES = [
    KP["front_left_paw"],
    KP["rear_left_paw"],
    KP["front_right_paw"],
    KP["rear_right_paw"],
]

FRONT_PAW_INDICES = [
    KP["front_left_paw"],
    KP["front_right_paw"],
]

STABLE_BODY_INDICES = [
    KP["withers"],
    KP["throat"],
    KP["nose"],
    KP["chin"],
    KP["left_eye"],
    KP["right_eye"],
]


# ============================================================
# MOTION THRESHOLDS
# ============================================================

# Motion is measured in pose-space normalized by body length, which makes
# it much less sensitive to crop translation/scale than raw pixel motion.
STILL_MAX = 0.025
WALKING_MAX = 0.070


# ============================================================
# TEMPORAL SETTINGS
# ============================================================

MOTION_HISTORY = 5
POSTURE_HISTORY = 7
TAIL_HISTORY = 12

TAIL_DEADBAND = 0.06
TAIL_WAG_MIN_AMPLITUDE = 0.22
TAIL_WAG_MIN_SIGN_CHANGES = 2
TAIL_MOVE_THRESHOLD = 0.08


# ============================================================
# GENERIC HELPERS
# ============================================================

def _valid_point(point) -> bool:
    try:
        x, y = float(point[0]), float(point[1])
    except (TypeError, ValueError, IndexError):
        return False
    return np.isfinite(x) and np.isfinite(y) and not (abs(x) < 1e-6 and abs(y) < 1e-6)


def _mean_valid(points):
    valid = [np.asarray(p, dtype=np.float32) for p in points if _valid_point(p)]
    return np.mean(valid, axis=0) if valid else None


def _distance(a, b) -> float:
    if not (_valid_point(a) and _valid_point(b)):
        return 0.0
    return float(np.linalg.norm(np.asarray(a, dtype=np.float32) - np.asarray(b, dtype=np.float32)))


def _body_scale(kpts) -> float:
    """Use withers->nose distance, falling back to a body bounding span."""
    withers = kpts[KP["withers"]]
    nose = kpts[KP["nose"]]
    scale = _distance(withers, nose)
    if scale >= 5.0:
        return scale

    valid = [np.asarray(p, dtype=np.float32) for p in kpts if _valid_point(p)]
    if len(valid) >= 2:
        arr = np.vstack(valid)
        span = np.ptp(arr, axis=0)
        scale = float(np.hypot(span[0], span[1]))
    return max(scale, 1.0)


def _normalize_keypoints(kpts):
    """Translation/scale-normalized pose for temporal comparison."""
    if kpts is None or len(kpts) < len(KEYPOINT_NAMES):
        return None

    anchor = np.asarray(kpts[KP["withers"]], dtype=np.float32)
    if not _valid_point(anchor):
        anchor = _mean_valid([
            kpts[KP["withers"]],
            kpts[KP["throat"]],
        ])
    if anchor is None:
        return None

    scale = _body_scale(kpts)
    out = np.zeros_like(np.asarray(kpts, dtype=np.float32))
    for i, point in enumerate(kpts):
        if _valid_point(point):
            out[i] = (np.asarray(point, dtype=np.float32) - anchor) / scale
    return out


# ============================================================
# MODEL LOADING
# ============================================================

def load_pose_model():
    """Load the trained dog pose model, or return None if unavailable."""
    if not os.path.exists(MODEL_PATH):
        print(
            f"[pose_motion] No pose model found at:\n{MODEL_PATH}\n"
            "Pose features will be unavailable."
        )
        return None

    from ultralytics import YOLO

    print(f"[pose_motion] Loading pose model:\n{MODEL_PATH}")
    return YOLO(MODEL_PATH)


# ============================================================
# KEYPOINT EXTRACTION
# ============================================================

def extract_keypoints(pose_model, bgr_frame):
    """
    Run YOLO pose inference and return:
        (keypoints_xy, bbox_diagonal, pose_confidence)

    Weak detections are rejected. Low-confidence keypoints are zeroed so
    downstream geometry/motion code does not treat hallucinated points as real.
    """
    if pose_model is None or bgr_frame is None:
        return None, None, 0.0

    try:
        results = pose_model(bgr_frame, verbose=False)
    except Exception as exc:
        print(f"[pose_motion] pose inference failed: {exc}")
        return None, None, 0.0

    if not results:
        return None, None, 0.0

    result = results[0]
    boxes = result.boxes
    keypoints = result.keypoints

    if keypoints is None or len(keypoints) == 0:
        return None, None, 0.0

    count = len(keypoints)
    best_idx = 0
    best_conf = 0.0

    if boxes is not None and len(boxes) == count:
        box_conf = boxes.conf.detach().cpu().numpy().astype(float)
        best_idx = int(np.argmax(box_conf))
        best_conf = float(box_conf[best_idx])
    elif boxes is not None and len(boxes) > 0:
        box_conf = boxes.conf.detach().cpu().numpy().astype(float)
        best_idx = min(int(np.argmax(box_conf)), count - 1)
        best_conf = float(box_conf[best_idx])

    if best_conf > 0 and best_conf < MIN_POSE_CONFIDENCE:
        return None, None, best_conf

    kpts = keypoints.xy[best_idx].detach().cpu().numpy().astype(np.float32)

    # Mask keypoints using per-point confidence when the model provides it.
    kp_conf = getattr(keypoints, "conf", None)
    if kp_conf is not None:
        conf_row = kp_conf[best_idx].detach().cpu().numpy().astype(float)
        kpts[conf_row < KEYPOINT_CONFIDENCE] = 0.0

    valid_count = sum(_valid_point(p) for p in kpts)
    if valid_count < MIN_VALID_KEYPOINTS:
        return None, None, best_conf

    if boxes is not None and len(boxes) > best_idx:
        x1, y1, x2, y2 = boxes.xyxy[best_idx].detach().cpu().numpy().astype(float)
        diagonal = float(np.hypot(x2 - x1, y2 - y1))
    else:
        h, w = bgr_frame.shape[:2]
        diagonal = float(np.hypot(w, h))

    return kpts, max(diagonal, 1.0), best_conf


# ============================================================
# MOTION
# ============================================================

def classify_motion_from_pose(prev_kpts, curr_kpts, bbox_diagonal=None):
    """Estimate Still / Walking / Running from normalized paw movement."""
    if prev_kpts is None or curr_kpts is None:
        return {"label": "Still", "score": 0.0, "confidence": 0.0}

    prev_norm = _normalize_keypoints(prev_kpts)
    curr_norm = _normalize_keypoints(curr_kpts)
    if prev_norm is None or curr_norm is None:
        return {"label": "Still", "score": 0.0, "confidence": 0.0}

    displacements = []
    for idx in PAW_INDICES:
        if idx >= len(prev_norm) or idx >= len(curr_norm):
            continue
        if not (_valid_point(prev_kpts[idx]) and _valid_point(curr_kpts[idx])):
            continue
        displacements.append(float(np.linalg.norm(curr_norm[idx] - prev_norm[idx])))

    if len(displacements) < 2:
        return {"label": "Still", "score": 0.0, "confidence": 0.0}

    # Median is more robust than mean when one paw keypoint jumps.
    score = float(np.median(displacements))

    if score < STILL_MAX:
        label = "Still"
    elif score < WALKING_MAX:
        label = "Walking"
    else:
        label = "Running"

    confidence = min(1.0, max(0.0, (score - STILL_MAX) / max(WALKING_MAX - STILL_MAX, 1e-6)))
    if label == "Running":
        confidence = min(1.0, 0.65 + score / 0.5)

    return {
        "label": label,
        "score": round(score, 4),
        "confidence": round(confidence, 3),
    }


# ============================================================
# POSTURE
# ============================================================

def classify_posture(kpts, bbox_diagonal):
    """Estimate Standing / Sitting / Lying using normalized body geometry."""
    if kpts is None or bbox_diagonal is None or bbox_diagonal <= 0:
        return {"label": "Unknown", "confidence": 0.0}

    required = [
        "withers",
        "rear_left_knee",
        "rear_right_knee",
        "rear_left_paw",
        "rear_right_paw",
        "front_left_paw",
        "front_right_paw",
    ]
    if not all(_valid_point(kpts[KP[name]]) for name in required):
        return {"label": "Unknown", "confidence": 0.0}

    withers_y = float(kpts[KP["withers"]][1])
    rear_knee_y = float(np.mean([
        kpts[KP["rear_left_knee"]][1],
        kpts[KP["rear_right_knee"]][1],
    ]))
    rear_paw_y = float(np.mean([
        kpts[KP["rear_left_paw"]][1],
        kpts[KP["rear_right_paw"]][1],
    ]))
    front_paw_y = float(np.mean([
        kpts[KP["front_left_paw"]][1],
        kpts[KP["front_right_paw"]][1],
    ]))

    body_height = (rear_paw_y - withers_y) / bbox_diagonal
    hind_extension = (rear_paw_y - rear_knee_y) / bbox_diagonal
    front_rear_gap = abs(front_paw_y - rear_paw_y) / bbox_diagonal

    # Small vertical body height: body is close to the ground.
    if body_height < 0.16:
        return {"label": "Lying", "confidence": 0.78}

    # Hind legs folded and front/rear paws relatively close in image height.
    if hind_extension < 0.085 and front_rear_gap < 0.13:
        return {"label": "Sitting", "confidence": 0.70}

    return {"label": "Standing", "confidence": 0.68}


# ============================================================
# TAIL POSITION / MOVEMENT
# ============================================================

def _tail_lateral_signal(kpts):
    """
    Return a signed tail-side signal in roughly [-1, 1].

    It measures the tail vector's sideways component relative to the dog's
    head/body axis. Translation and scale therefore have little influence.
    """
    needed = ["withers", "nose", "tail_start", "tail_end"]
    if not all(_valid_point(kpts[KP[name]]) for name in needed):
        return None

    body = np.asarray(kpts[KP["nose"]], dtype=np.float32) - np.asarray(kpts[KP["withers"]], dtype=np.float32)
    tail = np.asarray(kpts[KP["tail_end"]], dtype=np.float32) - np.asarray(kpts[KP["tail_start"]], dtype=np.float32)

    body_len = float(np.linalg.norm(body))
    tail_len = float(np.linalg.norm(tail))
    if body_len < 5.0 or tail_len < 5.0:
        return None

    body_unit = body / body_len
    side_unit = np.array([-body_unit[1], body_unit[0]], dtype=np.float32)
    lateral = float(np.dot(tail, side_unit) / tail_len)
    return float(np.clip(lateral, -1.0, 1.0))


def compute_tail_low(kpts, bbox_diagonal):
    """Backward-compatible low/tucked-tail boolean."""
    features = compute_tail_features(kpts, bbox_diagonal, history=None)
    return bool(features["low"])


def compute_tail_position(kpts, bbox_diagonal):
    """Return Low / Neutral / Raised with a simple geometry confidence."""
    if kpts is None or bbox_diagonal is None or bbox_diagonal <= 0:
        return {"label": "Unknown", "confidence": 0.0}

    needed = ["withers", "tail_start", "tail_end"]
    if not all(_valid_point(kpts[KP[name]]) for name in needed):
        return {"label": "Unknown", "confidence": 0.0}

    withers_y = float(kpts[KP["withers"]][1])
    tail_y = float(np.mean([
        kpts[KP["tail_start"]][1],
        kpts[KP["tail_end"]][1],
    ]))
    rel = (tail_y - withers_y) / bbox_diagonal

    if rel > 0.10:
        return {"label": "Low", "confidence": 0.76}
    if rel < -0.08:
        return {"label": "Raised", "confidence": 0.70}
    return {"label": "Neutral", "confidence": 0.65}


def compute_tail_features(kpts, bbox_diagonal, history=None):
    """Compute tail position, movement and wagging from a temporal history."""
    position = compute_tail_position(kpts, bbox_diagonal)
    signal = _tail_lateral_signal(kpts) if kpts is not None else None

    moving = False
    wagging = False
    wag_score = 0.0
    movement_score = 0.0
    sign_changes = 0
    amplitude = 0.0

    if history is not None and signal is not None:
        history.append(signal)
        values = [float(v) for v in history if v is not None]

        if len(values) >= 3:
            diffs = np.abs(np.diff(values))
            movement_score = float(np.mean(diffs))
            moving = movement_score >= TAIL_MOVE_THRESHOLD

        if len(values) >= 6:
            amplitude = float(max(values) - min(values))
            signs = []
            for value in values:
                if value > TAIL_DEADBAND:
                    signs.append(1)
                elif value < -TAIL_DEADBAND:
                    signs.append(-1)

            for a, b in zip(signs, signs[1:]):
                if a != b:
                    sign_changes += 1

            amplitude_score = min(1.0, amplitude / max(TAIL_WAG_MIN_AMPLITUDE, 1e-6))
            change_score = min(1.0, sign_changes / 4.0)
            wag_score = 0.55 * amplitude_score + 0.45 * change_score
            wagging = (
                amplitude >= TAIL_WAG_MIN_AMPLITUDE
                and sign_changes >= TAIL_WAG_MIN_SIGN_CHANGES
                and movement_score >= TAIL_MOVE_THRESHOLD
            )

    return {
        "position": position["label"],
        "position_confidence": position["confidence"],
        "low": position["label"] == "Low",
        "raised": position["label"] == "Raised",
        "moving": moving,
        "wagging": wagging,
        "movement_score": round(movement_score, 4),
        "wag_score": round(wag_score, 3),
        "amplitude": round(amplitude, 3),
        "direction_changes": int(sign_changes),
    }


# ============================================================
# EARS / HEAD / PAW CUES
# ============================================================

def classify_ears(kpts):
    if kpts is None:
        return {"label": "Unknown", "confidence": 0.0}

    names = [
        ("left_ear_base", "left_ear_tip"),
        ("right_ear_base", "right_ear_tip"),
    ]
    if not all(_valid_point(kpts[KP[a]]) and _valid_point(kpts[KP[b]]) for a, b in names):
        return {"label": "Unknown", "confidence": 0.0}

    withers = np.asarray(kpts[KP["withers"]], dtype=np.float32)
    nose = np.asarray(kpts[KP["nose"]], dtype=np.float32)
    body = nose - withers
    body_len = float(np.linalg.norm(body))
    if body_len < 5.0:
        return {"label": "Unknown", "confidence": 0.0}

    body /= body_len
    forward_scores = []
    for base_name, tip_name in names:
        vec = np.asarray(kpts[KP[tip_name]], dtype=np.float32) - np.asarray(kpts[KP[base_name]], dtype=np.float32)
        length = float(np.linalg.norm(vec))
        if length < 3.0:
            continue
        forward_scores.append(float(np.dot(vec / length, body)))

    if len(forward_scores) < 2:
        return {"label": "Unknown", "confidence": 0.0}

    score = float(np.mean(forward_scores))
    if score > 0.28:
        label = "Forward"
    elif score < -0.28:
        label = "Back"
    else:
        label = "Neutral"

    return {"label": label, "confidence": round(min(1.0, 0.5 + abs(score) * 0.5), 3)}


def classify_head_direction(kpts):
    if kpts is None:
        return {"label": "Unknown", "confidence": 0.0}

    required = ["left_eye", "right_eye", "nose", "withers"]
    if not all(_valid_point(kpts[KP[n]]) for n in required):
        return {"label": "Unknown", "confidence": 0.0}

    eye_mid = np.mean([
        kpts[KP["left_eye"]],
        kpts[KP["right_eye"]],
    ], axis=0)
    nose = np.asarray(kpts[KP["nose"]], dtype=np.float32)
    withers = np.asarray(kpts[KP["withers"]], dtype=np.float32)
    body = nose - withers
    body_len = float(np.linalg.norm(body))
    if body_len < 5.0:
        return {"label": "Unknown", "confidence": 0.0}

    body_unit = body / body_len
    lateral_unit = np.array([-body_unit[1], body_unit[0]], dtype=np.float32)
    head_vec = nose - eye_mid
    head_len = float(np.linalg.norm(head_vec))
    if head_len < 3.0:
        return {"label": "Center", "confidence": 0.35}

    lateral = float(np.dot(head_vec / head_len, lateral_unit))
    if lateral > 0.25:
        label = "Left"
    elif lateral < -0.25:
        label = "Right"
    else:
        label = "Center"

    return {"label": label, "confidence": round(min(1.0, 0.45 + abs(lateral) * 0.5), 3)}


def detect_raised_paw(kpts, bbox_diagonal):
    if kpts is None or bbox_diagonal is None or bbox_diagonal <= 0:
        return {"label": "None", "confidence": 0.0}

    candidates = []
    for paw_name, elbow_name in [
        ("front_left_paw", "front_left_elbow"),
        ("front_right_paw", "front_right_elbow"),
    ]:
        paw = kpts[KP[paw_name]]
        elbow = kpts[KP[elbow_name]]
        if _valid_point(paw) and _valid_point(elbow):
            lift = (float(elbow[1]) - float(paw[1])) / bbox_diagonal
            candidates.append((paw_name, lift))

    if not candidates:
        return {"label": "None", "confidence": 0.0}

    paw_name, lift = max(candidates, key=lambda item: item[1])
    if lift < 0.045:
        return {"label": "None", "confidence": 0.35}

    side = "Left" if "left" in paw_name else "Right"
    confidence = min(1.0, 0.55 + lift * 2.5)
    return {"label": f"{side} front paw raised", "confidence": round(confidence, 3)}


# ============================================================
# CONSISTENCY
# ============================================================

def enforce_motion_posture_consistency(motion, posture):
    """Prevent impossible motion/posture combinations."""
    motion_label = motion.get("label", "Unknown")
    posture_label = posture.get("label", "Unknown")

    if motion_label == "Running" and posture_label in {"Sitting", "Lying"}:
        posture = dict(posture)
        posture["label"] = "Standing"
        posture["confidence"] = round(float(posture.get("confidence", 0.0)) * 0.5, 3)
    elif motion_label == "Walking" and posture_label == "Lying":
        posture = dict(posture)
        posture["label"] = "Standing"
        posture["confidence"] = round(float(posture.get("confidence", 0.0)) * 0.5, 3)

    return posture


# ============================================================
# TEMPORAL TRACKER
# ============================================================

class PoseStateTracker:
    """Keeps pose history per camera/session and derives stable behavior cues."""

    def __init__(
        self,
        motion_history=MOTION_HISTORY,
        posture_history=POSTURE_HISTORY,
        tail_history=TAIL_HISTORY,
    ):
        self.motion_history = deque(maxlen=motion_history)
        self.posture_history = deque(maxlen=posture_history)
        self.tail_history = deque(maxlen=tail_history)
        self.prev_kpts = None

    @staticmethod
    def majority_vote(history):
        if not history:
            return "Unknown"
        values = list(history)
        counts = {item: values.count(item) for item in set(values)}
        return max(counts, key=counts.get)

    def update(self, kpts, bbox_diagonal):
        """Process one pose frame and return all pose-derived features."""
        if kpts is None:
            self.reset()
            return {
                "motion": {"label": "Unknown", "score": 0.0, "confidence": 0.0},
                "posture": {"label": "Unknown", "confidence": 0.0},
                "tail": {
                    "position": "Unknown",
                    "position_confidence": 0.0,
                    "low": False,
                    "raised": False,
                    "moving": False,
                    "wagging": False,
                    "movement_score": 0.0,
                    "wag_score": 0.0,
                    "amplitude": 0.0,
                    "direction_changes": 0,
                },
                "ears": {"label": "Unknown", "confidence": 0.0},
                "head": {"label": "Unknown", "confidence": 0.0},
                "paw": {"label": "None", "confidence": 0.0},
            }

        motion = classify_motion_from_pose(self.prev_kpts, kpts, bbox_diagonal)
        posture = classify_posture(kpts, bbox_diagonal)
        posture = enforce_motion_posture_consistency(motion, posture)
        tail = compute_tail_features(kpts, bbox_diagonal, self.tail_history)
        ears = classify_ears(kpts)
        head = classify_head_direction(kpts)
        paw = detect_raised_paw(kpts, bbox_diagonal)

        self.motion_history.append(motion["label"])
        self.posture_history.append(posture["label"])

        smooth_motion = self.majority_vote(self.motion_history)
        smooth_posture = self.majority_vote(self.posture_history)

        motion = dict(motion)
        posture = dict(posture)
        motion["raw_label"] = motion["label"]
        posture["raw_label"] = posture["label"]
        motion["label"] = smooth_motion
        posture["label"] = smooth_posture
        posture = enforce_motion_posture_consistency(motion, posture)

        self.prev_kpts = np.asarray(kpts, dtype=np.float32).copy()

        return {
            "motion": motion,
            "posture": posture,
            "tail": tail,
            "ears": ears,
            "head": head,
            "paw": paw,
        }

    def reset(self):
        self.motion_history.clear()
        self.posture_history.clear()
        self.tail_history.clear()
        self.prev_kpts = None


# Backward-compatible alias for code that wants a simple state object.
PoseTracker = PoseStateTracker
