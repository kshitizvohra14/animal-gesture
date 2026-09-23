"""
pose_motion.py — robust pose/motion layer for Paw Fussion.

Keeps the public functions used by app.py/fusion.py:
    load_pose_model
    extract_keypoints
    classify_motion_from_pose
    classify_posture
    compute_tail_low
    compute_tail_position
    compute_tail_features
    classify_ears
    classify_head_direction
    detect_raised_paw
    enforce_motion_posture_consistency
    PoseStateTracker / PoseTracker
"""

import os
from collections import deque
import numpy as np

MODEL_PATH = os.path.join(os.path.dirname(__file__), "models", "dog_pose_best.pt")

MIN_POSE_CONFIDENCE = 0.45
MIN_VALID_KEYPOINTS = 8
KEYPOINT_CONFIDENCE = 0.25

KEYPOINT_NAMES = [
    "front_left_paw", "front_left_knee", "front_left_elbow",
    "rear_left_paw", "rear_left_knee", "rear_left_elbow",
    "front_right_paw", "front_right_knee", "front_right_elbow",
    "rear_right_paw", "rear_right_knee", "rear_right_elbow",
    "tail_start", "tail_end", "left_ear_base", "right_ear_base",
    "nose", "chin", "left_ear_tip", "right_ear_tip",
    "left_eye", "right_eye", "withers", "throat",
]
KP = {n: i for i, n in enumerate(KEYPOINT_NAMES)}

PAW_INDICES = [KP["front_left_paw"], KP["rear_left_paw"],
               KP["front_right_paw"], KP["rear_right_paw"]]

# Motion thresholds are deliberately conservative because pose detectors
# have frame-to-frame jitter even when the dog is stationary.
STILL_MAX = 0.035
WALKING_MAX = 0.085
MOTION_HISTORY = 5
POSTURE_HISTORY = 5

TAIL_HISTORY = 12
TAIL_DEADBAND = 0.06
TAIL_WAG_MIN_AMPLITUDE = 0.22
TAIL_WAG_MIN_SIGN_CHANGES = 2
TAIL_MOVE_THRESHOLD = 0.08


def _valid_point(p):
    try:
        x, y = float(p[0]), float(p[1])
    except (TypeError, ValueError, IndexError):
        return False
    return np.isfinite(x) and np.isfinite(y) and not (abs(x) < 1e-6 and abs(y) < 1e-6)


def _distance(a, b):
    if not (_valid_point(a) and _valid_point(b)):
        return 0.0
    return float(np.linalg.norm(np.asarray(a, dtype=np.float32) -
                                np.asarray(b, dtype=np.float32)))


def _mean_valid(points):
    vals = [np.asarray(p, dtype=np.float32) for p in points if _valid_point(p)]
    return np.mean(vals, axis=0) if vals else None


def _body_scale(kpts):
    scale = _distance(kpts[KP["withers"]], kpts[KP["nose"]])
    if scale >= 5:
        return scale
    vals = [np.asarray(p, dtype=np.float32) for p in kpts if _valid_point(p)]
    if len(vals) >= 2:
        arr = np.vstack(vals)
        span = np.ptp(arr, axis=0)
        return max(float(np.hypot(span[0], span[1])), 1.0)
    return 1.0


def _normalize_keypoints(kpts):
    if kpts is None or len(kpts) < len(KEYPOINT_NAMES):
        return None

    anchor = kpts[KP["withers"]]
    if not _valid_point(anchor):
        anchor = _mean_valid([kpts[KP["withers"]], kpts[KP["throat"]]])
    if anchor is None:
        return None

    scale = _body_scale(kpts)
    arr = np.asarray(kpts, dtype=np.float32)
    out = np.zeros_like(arr)
    for i, p in enumerate(arr):
        if _valid_point(p):
            out[i] = (p - anchor) / scale
    return out


def load_pose_model():
    if not os.path.exists(MODEL_PATH):
        print(f"[pose_motion] No pose model found at: {MODEL_PATH}")
        return None
    from ultralytics import YOLO
    print(f"[pose_motion] Loading pose model:\n{MODEL_PATH}")
    return YOLO(MODEL_PATH)


def extract_keypoints(pose_model, bgr_frame):
    """Return (keypoints, bbox_diagonal, pose_confidence)."""
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

    if boxes is not None and len(boxes) > 0:
        confs = boxes.conf.detach().cpu().numpy().astype(float)
        best_idx = int(np.argmax(confs))
        best_idx = min(best_idx, count - 1)
        best_conf = float(confs[best_idx])

    if best_conf > 0 and best_conf < MIN_POSE_CONFIDENCE:
        return None, None, best_conf

    kpts = keypoints.xy[best_idx].detach().cpu().numpy().astype(np.float32)

    kp_conf = getattr(keypoints, "conf", None)
    if kp_conf is not None:
        conf_row = kp_conf[best_idx].detach().cpu().numpy().astype(float)
        kpts[conf_row < KEYPOINT_CONFIDENCE] = 0.0

    if sum(_valid_point(p) for p in kpts) < MIN_VALID_KEYPOINTS:
        return None, None, best_conf

    if boxes is not None and len(boxes) > best_idx:
        x1, y1, x2, y2 = boxes.xyxy[best_idx].detach().cpu().numpy().astype(float)
        diagonal = float(np.hypot(x2 - x1, y2 - y1))
    else:
        h, w = bgr_frame.shape[:2]
        diagonal = float(np.hypot(w, h))

    return kpts, max(diagonal, 1.0), best_conf


def _motion_score(prev_kpts, curr_kpts):
    prev = _normalize_keypoints(prev_kpts)
    curr = _normalize_keypoints(curr_kpts)
    if prev is None or curr is None:
        return None

    values = []
    for idx in PAW_INDICES:
        if _valid_point(prev_kpts[idx]) and _valid_point(curr_kpts[idx]):
            values.append(float(np.linalg.norm(curr[idx] - prev[idx])))

    if len(values) < 2:
        return None

    # Median rejects one badly jittering paw.
    return float(np.median(values))


def classify_motion_from_pose(prev_kpts, curr_kpts, bbox_diagonal=None):
    """
    Conservative instantaneous motion estimate.

    Important: first frame is Still/Unknown rather than Walking.
    Small pose jitter is explicitly treated as Still.
    """
    if prev_kpts is None or curr_kpts is None:
        return {"label": "Still", "score": 0.0, "confidence": 0.0}

    score = _motion_score(prev_kpts, curr_kpts)
    if score is None:
        return {"label": "Still", "score": 0.0, "confidence": 0.0}

    if score <= STILL_MAX:
        label = "Still"
        confidence = min(1.0, (STILL_MAX - score) / STILL_MAX + 0.55)
    elif score <= WALKING_MAX:
        label = "Walking"
        confidence = min(1.0, max(0.0, (score - STILL_MAX) /
                                  (WALKING_MAX - STILL_MAX)))
    else:
        label = "Running"
        confidence = min(1.0, 0.70 + (score - WALKING_MAX) / 0.20)

    return {
        "label": label,
        "score": round(score, 4),
        "confidence": round(float(confidence), 3),
    }


def _valid_names(kpts, names, minimum=None):
    valid = [n for n in names if _valid_point(kpts[KP[n]])]
    return len(valid) >= (minimum if minimum is not None else len(names))


def classify_posture(kpts, bbox_diagonal):
    """
    Tolerant posture classifier.

    It no longer requires all seven leg points. It uses the strongest
    available geometry and returns Unknown only when the pose genuinely
    lacks enough information.
    """
    if kpts is None or not bbox_diagonal or bbox_diagonal <= 0:
        return {"label": "Unknown", "confidence": 0.0}

    # Core anchor. Without withers there is no reliable body-relative geometry.
    if not _valid_point(kpts[KP["withers"]]):
        return {"label": "Unknown", "confidence": 0.0}

    w = np.asarray(kpts[KP["withers"]], dtype=np.float32)
    front_paws = [kpts[KP[n]] for n in ("front_left_paw", "front_right_paw")
                  if _valid_point(kpts[KP[n]])]
    rear_paws = [kpts[KP[n]] for n in ("rear_left_paw", "rear_right_paw")
                 if _valid_point(kpts[KP[n]])]
    rear_knees = [kpts[KP[n]] for n in ("rear_left_knee", "rear_right_knee")
                  if _valid_point(kpts[KP[n]])]

    # Body height: median paw distance below withers.
    all_paws = front_paws + rear_paws
    if len(all_paws) >= 2:
        paw_y = float(np.median([p[1] for p in all_paws]))
        body_height = (paw_y - w[1]) / bbox_diagonal
    else:
        body_height = None

    # Lying: low body geometry, but avoid calling an upright dog lying just
    # because one paw is missing.
    if body_height is not None and body_height < 0.16 and len(all_paws) >= 2:
        return {"label": "Lying", "confidence": 0.78}

    # Sitting: use folded hind-leg geometry. One rear side is enough when
    # front paws are also visible, but confidence is reduced.
    sitting_votes = []
    for knee, paw in zip(
        [kpts[KP["rear_left_knee"]], kpts[KP["rear_right_knee"]]],
        [kpts[KP["rear_left_paw"]], kpts[KP["rear_right_paw"]]],
    ):
        if _valid_point(knee) and _valid_point(paw):
            folded = abs(float(paw[1]) - float(knee[1])) / bbox_diagonal
            sitting_votes.append(folded < 0.10)

    if sitting_votes and any(sitting_votes):
        if len(front_paws) >= 1:
            return {"label": "Sitting", "confidence": 0.74 if len(sitting_votes) == 2 else 0.62}

    # A dog with multiple reliable paws and normal body height is standing.
    if len(all_paws) >= 2:
        return {"label": "Standing", "confidence": 0.72}

    # Front paws + withers + nose provide enough evidence for an upright body
    # in many side-view frames.
    if len(front_paws) >= 2 and _valid_point(kpts[KP["nose"]]):
        return {"label": "Standing", "confidence": 0.58}

    return {"label": "Unknown", "confidence": 0.0}


def _tail_lateral_signal(kpts):
    names = ("withers", "nose", "tail_start", "tail_end")
    if not all(_valid_point(kpts[KP[n]]) for n in names):
        return None

    body = np.asarray(kpts[KP["nose"]], dtype=np.float32) - np.asarray(
        kpts[KP["withers"]], dtype=np.float32)
    tail = np.asarray(kpts[KP["tail_end"]], dtype=np.float32) - np.asarray(
        kpts[KP["tail_start"]], dtype=np.float32)

    bl, tl = float(np.linalg.norm(body)), float(np.linalg.norm(tail))
    if bl < 5 or tl < 5:
        return None

    body /= bl
    side = np.array([-body[1], body[0]], dtype=np.float32)
    return float(np.clip(np.dot(tail, side) / tl, -1, 1))


def compute_tail_position(kpts, bbox_diagonal):
    if kpts is None or not bbox_diagonal or bbox_diagonal <= 0:
        return {"label": "Unknown", "confidence": 0.0}

    needed = ("withers", "tail_start", "tail_end")
    if not all(_valid_point(kpts[KP[n]]) for n in needed):
        return {"label": "Unknown", "confidence": 0.0}

    wy = float(kpts[KP["withers"]][1])
    ty = float(np.mean([kpts[KP["tail_start"]][1],
                        kpts[KP["tail_end"]][1]]))
    rel = (ty - wy) / bbox_diagonal

    if rel > 0.10:
        return {"label": "Low", "confidence": 0.76}
    if rel < -0.08:
        return {"label": "Raised", "confidence": 0.70}
    return {"label": "Neutral", "confidence": 0.65}


def compute_tail_low(kpts, bbox_diagonal):
    return compute_tail_position(kpts, bbox_diagonal)["label"] == "Low"


def compute_tail_features(kpts, bbox_diagonal, history=None):
    position = compute_tail_position(kpts, bbox_diagonal)
    signal = _tail_lateral_signal(kpts) if kpts is not None else None

    moving = False
    wagging = False
    movement_score = 0.0
    wag_score = 0.0
    amplitude = 0.0
    sign_changes = 0

    if history is not None and signal is not None:
        history.append(signal)
        vals = list(history)

        if len(vals) >= 3:
            movement_score = float(np.mean(np.abs(np.diff(vals))))
            moving = movement_score >= TAIL_MOVE_THRESHOLD

        if len(vals) >= 6:
            amplitude = float(max(vals) - min(vals))
            signs = []
            for v in vals:
                if v > TAIL_DEADBAND:
                    signs.append(1)
                elif v < -TAIL_DEADBAND:
                    signs.append(-1)

            sign_changes = sum(a != b for a, b in zip(signs, signs[1:]))
            amp_score = min(1.0, amplitude / TAIL_WAG_MIN_AMPLITUDE)
            change_score = min(1.0, sign_changes / 4.0)
            wag_score = 0.55 * amp_score + 0.45 * change_score
            wagging = (
                amplitude >= TAIL_WAG_MIN_AMPLITUDE and
                sign_changes >= TAIL_WAG_MIN_SIGN_CHANGES and
                movement_score >= TAIL_MOVE_THRESHOLD
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


def classify_ears(kpts):
    if kpts is None:
        return {"label": "Unknown", "confidence": 0.0}

    pairs = [("left_ear_base", "left_ear_tip"),
             ("right_ear_base", "right_ear_tip")]
    vals = []
    for base, tip in pairs:
        if _valid_point(kpts[KP[base]]) and _valid_point(kpts[KP[tip]]):
            vals.append((np.asarray(kpts[KP[tip]]) -
                         np.asarray(kpts[KP[base]])))

    if len(vals) < 2 or not _valid_point(kpts[KP["withers"]]) or \
       not _valid_point(kpts[KP["nose"]]):
        return {"label": "Unknown", "confidence": 0.0}

    body = np.asarray(kpts[KP["nose"]]) - np.asarray(kpts[KP["withers"]])
    bl = np.linalg.norm(body)
    if bl < 5:
        return {"label": "Unknown", "confidence": 0.0}
    body /= bl

    scores = [float(np.dot(v / max(np.linalg.norm(v), 1e-6), body)) for v in vals]
    score = float(np.mean(scores))
    label = "Forward" if score > 0.28 else "Back" if score < -0.28 else "Neutral"
    return {"label": label, "confidence": round(min(1.0, 0.5 + abs(score) * 0.5), 3)}


def classify_head_direction(kpts):
    needed = ("left_eye", "right_eye", "nose", "withers")
    if kpts is None or not all(_valid_point(kpts[KP[n]]) for n in needed):
        return {"label": "Unknown", "confidence": 0.0}

    eyes = np.mean([kpts[KP["left_eye"]], kpts[KP["right_eye"]]], axis=0)
    nose = np.asarray(kpts[KP["nose"]], dtype=np.float32)
    body = nose - np.asarray(kpts[KP["withers"]], dtype=np.float32)
    bl = np.linalg.norm(body)
    hv = nose - eyes
    hl = np.linalg.norm(hv)
    if bl < 5 or hl < 3:
        return {"label": "Center", "confidence": 0.35}

    body /= bl
    lateral_axis = np.array([-body[1], body[0]], dtype=np.float32)
    lateral = float(np.dot(hv / hl, lateral_axis))
    label = "Left" if lateral > 0.25 else "Right" if lateral < -0.25 else "Center"
    return {"label": label, "confidence": round(min(1.0, 0.45 + abs(lateral) * 0.5), 3)}


def detect_raised_paw(kpts, bbox_diagonal):
    if kpts is None or not bbox_diagonal or bbox_diagonal <= 0:
        return {"label": "None", "confidence": 0.0}

    candidates = []
    for paw_name, elbow_name in [
        ("front_left_paw", "front_left_elbow"),
        ("front_right_paw", "front_right_elbow"),
    ]:
        paw, elbow = kpts[KP[paw_name]], kpts[KP[elbow_name]]
        if _valid_point(paw) and _valid_point(elbow):
            lift = (float(elbow[1]) - float(paw[1])) / bbox_diagonal
            candidates.append((paw_name, lift))

    if not candidates:
        return {"label": "None", "confidence": 0.0}

    paw_name, lift = max(candidates, key=lambda x: x[1])
    if lift < 0.045:
        return {"label": "None", "confidence": 0.35}

    side = "Left" if "left" in paw_name else "Right"
    return {"label": f"{side} front paw raised",
            "confidence": round(min(1.0, 0.55 + lift * 2.5), 3)}


def enforce_motion_posture_consistency(motion, posture):
    # Do not turn a static sitting/lying dog into Standing merely because
    # one noisy frame was called Walking. Only Running is strong enough to
    # contradict a static posture.
    if motion.get("label") == "Running" and posture.get("label") in {"Sitting", "Lying"}:
        p = dict(posture)
        p["label"] = "Standing"
        p["confidence"] = round(float(p.get("confidence", 0)) * 0.5, 3)
        return p
    return posture


class PoseStateTracker:
    """
    Stateful tracker for live/video use.

    Motion is smoothed over several frames and posture is majority-voted.
    Missing pose resets the temporal chain so a reappearing dog cannot create
    a fake large motion spike.
    """
    def __init__(self, motion_history=MOTION_HISTORY,
                 posture_history=POSTURE_HISTORY,
                 tail_history=TAIL_HISTORY):
        self.motion_history = deque(maxlen=motion_history)
        self.posture_history = deque(maxlen=posture_history)
        self.tail_history = deque(maxlen=tail_history)
        self.prev_kpts = None

    @staticmethod
    def majority_vote(history):
        if not history:
            return "Unknown"
        counts = {}
        for value in history:
            counts[value] = counts.get(value, 0) + 1
        return max(counts, key=counts.get)

    def update(self, kpts, bbox_diagonal):
        if kpts is None:
            self.reset()
            return {
                "motion": {"label": "Unknown", "score": 0.0, "confidence": 0.0},
                "posture": {"label": "Unknown", "confidence": 0.0},
                "tail": compute_tail_features(None, None, None),
                "ears": {"label": "Unknown", "confidence": 0.0},
                "head": {"label": "Unknown", "confidence": 0.0},
                "paw": {"label": "None", "confidence": 0.0},
            }

        raw_motion = classify_motion_from_pose(self.prev_kpts, kpts, bbox_diagonal)
        raw_posture = classify_posture(kpts, bbox_diagonal)

        self.motion_history.append(raw_motion["label"])
        self.posture_history.append(raw_posture["label"])

        smooth_motion = self.majority_vote(self.motion_history)
        smooth_posture = self.majority_vote(self.posture_history)

        motion = dict(raw_motion)
        posture = dict(raw_posture)
        motion["raw_label"] = raw_motion["label"]
        posture["raw_label"] = raw_posture["label"]
        motion["label"] = smooth_motion
        posture["label"] = smooth_posture

        posture = enforce_motion_posture_consistency(motion, posture)

        tail = compute_tail_features(kpts, bbox_diagonal, self.tail_history)
        ears = classify_ears(kpts)
        head = classify_head_direction(kpts)
        paw = detect_raised_paw(kpts, bbox_diagonal)

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


PoseTracker = PoseStateTracker
