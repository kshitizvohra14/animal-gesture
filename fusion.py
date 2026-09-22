"""
fusion.py

Multimodal gesture fusion for Paw Fussion.

Inputs:
    - emotion
    - motion
    - vocalization
    - posture
    - tail position / wagging
    - ears
    - head direction
    - raised paw

The fusion engine does NOT claim that a single cue proves an emotion.
Instead it combines the available signals and reports which behavior rule
matched best.
"""

from dataclasses import dataclass, field
from typing import Any


# ============================================================
# CUE REGISTRY
# ============================================================

POSE_CUES = {
    "still": "Motion is Still",
    "walking": "Motion is Walking",
    "running": "Motion is Running",
    "sitting": "Posture is Sitting",
    "standing": "Posture is Standing",
    "lying": "Posture is Lying",
    "tail_low": "Tail position is Low",
    "tail_raised": "Tail position is Raised",
    "tail_moving": "Tail is moving",
    "tail_wagging": "Tail shows repeated side-to-side motion",
    "ears_forward": "Ears are oriented forward",
    "ears_back": "Ears are oriented backward",
    "head_left": "Head is turned left",
    "head_right": "Head is turned right",
    "paw_raised": "A front paw is raised",
    "barking": "Vocalization is Barking",
    "growling": "Vocalization is Growling",
    "howling": "Vocalization is Howling",
    "whimpering": "Vocalization is Whimpering",
    "quiet": "No recent vocalization was detected",
}

EMOTION_CANONICAL = {
    "smileydogs": "Happy",
    "gooddogs": "Relaxed",
    "sleepydogs": "Sad",
    "angrydogs": "Angry",
    "happy": "Happy",
    "relaxed": "Relaxed",
    "sad": "Sad",
    "angry": "Angry",
    "uncertain": "Uncertain",
    "unknown": "Uncertain",
}


def _canonicalize(label: str) -> str:
    if label is None:
        return "Uncertain"
    text = str(label).strip()
    return EMOTION_CANONICAL.get(text.lower(), text)


def _clamp01(value) -> float:
    try:
        return max(0.0, min(1.0, float(value)))
    except (TypeError, ValueError):
        return 0.0


def _cue_active(value: Any):
    """Return (is_active, confidence) for bool/number/dict cue values."""
    if isinstance(value, dict):
        active = bool(value.get("active", value.get("value", False)))
        confidence = _clamp01(value.get("confidence", 1.0 if active else 0.0))
        return active, confidence

    if isinstance(value, bool):
        return value, 1.0 if value else 0.0

    if isinstance(value, (int, float)):
        confidence = _clamp01(value)
        return confidence >= 0.5, confidence

    if isinstance(value, str):
        return value not in {"", "Unknown", "None", "False"}, 1.0

    return False, 0.0


@dataclass
class Gesture:
    name: str
    required_emotion: str
    cues: dict = field(default_factory=dict)
    min_match: float = 0.60
    required_emotion_canonical: str = field(init=False)

    def __post_init__(self):
        self.required_emotion_canonical = _canonicalize(self.required_emotion)


@dataclass
class SignalFrame:
    emotion_label: str
    emotion_conf: float
    motion_label: str
    vocalization_label: str
    vocal_conf: float = None
    pose_cues: dict = None


# ============================================================
# BEHAVIOR RULES
# ============================================================

GESTURES = [
    # Happy
    Gesture("Excited play", "Happy", {"running": 1.0, "tail_wagging": 1.2, "paw_raised": 0.4}),
    Gesture("Playful walk", "Happy", {"walking": 1.0, "tail_wagging": 1.1}),
    Gesture("Happy greeting", "Happy", {"standing": 0.6, "tail_wagging": 1.2, "paw_raised": 0.5}),
    Gesture("Happy greeting bark", "Happy", {"standing": 0.5, "barking": 1.2, "tail_wagging": 0.8}),
    Gesture("Cheerful and relaxed", "Happy", {"sitting": 0.8, "tail_moving": 0.6, "quiet": 0.5}),

    # Relaxed
    Gesture("Friendly relaxed greeting", "Relaxed", {"standing": 0.5, "tail_wagging": 1.2, "quiet": 0.4}),
    Gesture("Casual walk", "Relaxed", {"walking": 1.0, "tail_moving": 0.7}),
    Gesture("Resting calmly", "Relaxed", {"lying": 1.1, "still": 0.9, "quiet": 0.5}),
    Gesture("Alert but calm", "Relaxed", {"sitting": 0.8, "ears_forward": 0.7, "quiet": 0.5}),

    # Sad
    Gesture("Distress whimper", "Sad", {"whimpering": 1.2, "still": 0.7, "tail_low": 0.7}),
    Gesture("Withdrawn / low energy", "Sad", {"lying": 1.0, "tail_low": 0.9, "still": 0.8}),
    Gesture("Anxious pacing", "Sad", {"walking": 1.0, "tail_low": 1.1}),

    # Angry
    Gesture("Alert / warning bark", "Angry", {"barking": 1.2, "standing": 0.5, "ears_forward": 0.7}),
    Gesture("Defensive growl", "Angry", {"growling": 1.2, "tail_raised": 0.7, "standing": 0.4}),
    Gesture("Tense / on guard", "Angry", {"standing": 0.8, "tail_raised": 0.8, "ears_forward": 0.8}),

    # Uncertain emotion
    Gesture("Attentive / observing", "Uncertain", {"standing": 1.0, "ears_forward": 0.8}),
    Gesture("Resting / observing", "Uncertain", {"lying": 1.0, "still": 0.7}),
]

_undocumented = sorted({cue for gesture in GESTURES for cue in gesture.cues} - set(POSE_CUES))
if _undocumented:
    raise ValueError(f"fusion.py: undocumented cues: {_undocumented}")


# ============================================================
# ENGINE
# ============================================================

class FusionEngine:
    def __init__(self, threshold: float = 0.60):
        self.threshold = float(threshold)

    def _active_cues(self, motion_label, vocalization_label, pose_cues):
        active = {}

        motion_map = {"Still": "still", "Walking": "walking", "Running": "running"}
        if motion_label in motion_map:
            active[motion_map[motion_label]] = 1.0

        vocal_map = {
            "Barking": "barking",
            "Growling": "growling",
            "Howling": "howling",
            "Whimpering": "whimpering",
            "Quiet": "quiet",
        }
        if vocalization_label in vocal_map:
            cue = vocal_map[vocalization_label]
            confidence = 1.0
            if isinstance(pose_cues, dict):
                confidence = _clamp01(pose_cues.get("vocal_conf", 1.0))
            active[cue] = confidence

        pose_cues = pose_cues or {}

        posture_map = {
            "Sitting": "sitting",
            "Standing": "standing",
            "Lying": "lying",
        }
        posture = pose_cues.get("posture")
        if posture in posture_map:
            active[posture_map[posture]] = _clamp01(pose_cues.get("posture_confidence", 1.0))

        # Boolean / score-based cues from pose_motion.py.
        cue_aliases = {
            "tail_low": "tail_low",
            "tail_raised": "tail_raised",
            "tail_moving": "tail_moving",
            "tail_wagging": "tail_wagging",
            "ears_forward": "ears_forward",
            "ears_back": "ears_back",
            "head_left": "head_left",
            "head_right": "head_right",
            "paw_raised": "paw_raised",
        }

        for key, cue_name in cue_aliases.items():
            if key not in pose_cues:
                continue
            is_active, confidence = _cue_active(pose_cues[key])
            if is_active:
                active[cue_name] = confidence

        # More convenient structured inputs from PoseStateTracker.
        tail = pose_cues.get("tail")
        if isinstance(tail, dict):
            if tail.get("low"):
                active["tail_low"] = _clamp01(tail.get("position_confidence", 1.0))
            if tail.get("raised"):
                active["tail_raised"] = _clamp01(tail.get("position_confidence", 1.0))
            if tail.get("moving"):
                active["tail_moving"] = max(_clamp01(tail.get("movement_score", 0.0) * 4.0), 0.5)
            if tail.get("wagging"):
                active["tail_wagging"] = max(_clamp01(tail.get("wag_score", 0.0)), 0.5)

        ears = pose_cues.get("ears")
        if isinstance(ears, dict):
            label = ears.get("label")
            if label == "Forward":
                active["ears_forward"] = _clamp01(ears.get("confidence", 1.0))
            elif label == "Back":
                active["ears_back"] = _clamp01(ears.get("confidence", 1.0))

        head = pose_cues.get("head")
        if isinstance(head, dict):
            label = head.get("label")
            if label == "Left":
                active["head_left"] = _clamp01(head.get("confidence", 1.0))
            elif label == "Right":
                active["head_right"] = _clamp01(head.get("confidence", 1.0))

        paw = pose_cues.get("paw")
        if isinstance(paw, dict):
            label = str(paw.get("label", "None"))
            if label and label != "None":
                active["paw_raised"] = _clamp01(paw.get("confidence", 1.0))

        return active

    def _score_gesture(self, gesture: Gesture, emotion, active):
        if gesture.required_emotion_canonical != emotion:
            return 0.0, []

        if not gesture.cues:
            return 0.0, []

        total = sum(float(weight) for weight in gesture.cues.values())
        matched = 0.0
        matched_cues = []

        for cue, weight in gesture.cues.items():
            confidence = float(active.get(cue, 0.0))
            if confidence > 0:
                matched += float(weight) * confidence
                matched_cues.append({"cue": cue, "confidence": round(confidence, 3)})

        return (matched / total if total > 0 else 0.0), matched_cues

    def select(self, emotion_label, emotion_conf, motion_label,
               vocalization_label, vocal_conf=None, pose_cues=None):
        emotion = _canonicalize(emotion_label)
        emotion_conf = _clamp01(emotion_conf)
        active = self._active_cues(motion_label, vocalization_label, {
            **(pose_cues or {}),
            "vocal_conf": vocal_conf if vocal_conf is not None else 1.0,
        })

        scored = []
        for gesture in GESTURES:
            match_score, matched_cues = self._score_gesture(gesture, emotion, active)
            scored.append((gesture, match_score, matched_cues))

        best_gesture, best_match, matched_cues = max(
            scored,
            key=lambda item: item[1],
            default=(None, 0.0, []),
        )

        # Emotion confidence is a reliability factor, not a replacement for
        # the cue-match score. When emotion is Uncertain, it remains possible
        # to produce an observation-style reading from pose cues.
        if best_gesture is None or best_match < max(self.threshold, best_gesture.min_match):
            fallback = f"{emotion} | {motion_label} | {vocalization_label}"
            return {
                "gesture": fallback,
                "matched_rule": False,
                "score": 0.0,
                "match_score": 0.0,
                "confidence": emotion_conf,
                "matched_cues": [],
                "active_cues": sorted(active.keys()),
                "line": format_line(fallback, False, 0.0, emotion,
                                     motion_label, vocalization_label, emotion_conf),
            }

        # Only the signals actually used by this rule contribute to the final
        # reliability score.
        reliability = [emotion_conf]
        for item in matched_cues:
            cue = item["cue"]
            if cue in {"barking", "growling", "howling", "whimpering", "quiet"} and vocal_conf is not None:
                reliability.append(_clamp01(vocal_conf))
            else:
                reliability.append(_clamp01(item["confidence"]))

        mean_reliability = sum(reliability) / max(1, len(reliability))
        final_score = round(best_match * mean_reliability, 3)

        return {
            "gesture": best_gesture.name,
            "matched_rule": True,
            "score": final_score,
            "match_score": round(best_match, 3),
            "confidence": round(mean_reliability, 3),
            "matched_cues": matched_cues,
            "active_cues": sorted(active.keys()),
            "line": format_line(best_gesture.name, True, final_score, emotion,
                                motion_label, vocalization_label, emotion_conf),
        }


_engine = FusionEngine(threshold=0.60)


def format_line(gesture, matched, score, emotion, motion, vocalization, emotion_conf=None):
    emo = f"{emotion} {emotion_conf:.0%}" if emotion_conf is not None else emotion
    signals = f"{emo} | {motion} | {vocalization}"
    if matched:
        return " ".join(f"Dog gesture: {gesture} (score {score:.2f}) | {signals}".split())
    return " ".join(f"Dog gesture: no confident match | {signals}".split())


def fuse(emotion_label: str, emotion_conf: float, motion_label: str,
         vocalization_label: str, vocal_conf: float = None,
         pose_cues: dict = None) -> dict:
    return _engine.select(
        emotion_label,
        emotion_conf,
        motion_label,
        vocalization_label,
        vocal_conf,
        pose_cues,
    )


def fuse_line(emotion_label: str, emotion_conf: float, motion_label: str,
              vocalization_label: str, vocal_conf: float = None,
              pose_cues: dict = None) -> str:
    return fuse(
        emotion_label,
        emotion_conf,
        motion_label,
        vocalization_label,
        vocal_conf,
        pose_cues,
    )["line"]


def fuse_frame(frame: SignalFrame) -> dict:
    return fuse(
        frame.emotion_label,
        frame.emotion_conf,
        frame.motion_label,
        frame.vocalization_label,
        frame.vocal_conf,
        frame.pose_cues,
    )


def fuse_frame_line(frame: SignalFrame) -> str:
    return fuse_frame(frame)["line"]


if __name__ == "__main__":
    tests = [
        ("Happy", 0.87, "Running", "Barking",
         {"tail": {"wagging": True, "wag_score": 0.88}, "posture": "Standing"}),
        ("Happy", 0.84, "Walking", "Quiet",
         {"tail": {"wagging": True, "wag_score": 0.78}, "posture": "Standing"}),
        ("Relaxed", 0.81, "Still", "Quiet",
         {"posture": "Lying"}),
        ("Angry", 0.82, "Still", "Growling",
         {"posture": "Standing", "tail": {"raised": True, "position_confidence": 0.75}}),
    ]
    for args in tests:
        print(fuse_line(*args[:-1], pose_cues=args[-1]))
