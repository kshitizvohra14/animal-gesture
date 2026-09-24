"""
fusion.py — Multimodal behavior fusion for Paw Fussion.

The engine combines:
    - facial emotion
    - motion
    - posture
    - vocalization
    - tail position / movement / wagging
    - ears
    - head direction
    - raised paw

Important design:
    • Emotion is evidence, NOT a hard gate.
    • Positive and contradictory cues are both considered.
    • Cue confidence is propagated instead of artificially boosted.
    • Partial matches are penalized when important evidence is missing.
    • The result explains which signals supported or contradicted the rule.
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


EMOTIONS = {
    "Happy",
    "Relaxed",
    "Sad",
    "Angry",
    "Uncertain",
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
    """Normalize emotion labels without inventing new classes."""
    if label is None:
        return "Uncertain"

    text = str(label).strip()

    return EMOTION_CANONICAL.get(
        text.lower(),
        text if text in EMOTIONS else "Uncertain",
    )


def _clamp01(value) -> float:
    try:
        return max(0.0, min(1.0, float(value)))
    except (TypeError, ValueError):
        return 0.0


def _cue_active(value: Any):
    """
    Convert bool/number/string/dict cue values to:
        (active, confidence)
    """

    if isinstance(value, dict):
        # Support both:
        # {"active": True, "confidence": 0.8}
        # {"value": True, "confidence": 0.8}
        # {"wagging": True, ...} is handled by the structured
        # tail parser separately.
        active = bool(
            value.get(
                "active",
                value.get("value", False),
            )
        )

        confidence = _clamp01(
            value.get(
                "confidence",
                1.0 if active else 0.0,
            )
        )

        return active, confidence

    if isinstance(value, bool):
        return value, 1.0 if value else 0.0

    if isinstance(value, (int, float)):
        confidence = _clamp01(value)
        return confidence > 0.0, confidence

    if isinstance(value, str):
        active = value.strip().lower() not in {
            "",
            "unknown",
            "none",
            "false",
            "no",
        }
        return active, 1.0 if active else 0.0

    return False, 0.0


# ============================================================
# RULES
# ============================================================

@dataclass
class BehaviorRule:
    """
    A behavior hypothesis.

    positive:
        Evidence that supports the behavior.

    negative:
        Evidence that contradicts the behavior.

    emotion_weights:
        Emotion-specific evidence.

    min_evidence:
        Minimum number of distinct positive cues required.
        This prevents one strong cue from creating a confident
        behavior by itself.
    """

    name: str

    positive: dict[str, float] = field(default_factory=dict)
    negative: dict[str, float] = field(default_factory=dict)

    emotion_weights: dict[str, float] = field(default_factory=dict)

    min_evidence: int = 2
    threshold: float = 0.58

    # Backward-compatible field.
    required_emotion: str = "Uncertain"

    def __post_init__(self):
        self.required_emotion = _canonicalize(
            self.required_emotion
        )

        normalized = {}

        for key, value in self.emotion_weights.items():
            normalized[_canonicalize(key)] = float(value)

        self.emotion_weights = normalized


# ============================================================
# BEHAVIOR RULES
# ============================================================
#
# These rules are intentionally conservative.
# A behavior needs multiple pieces of evidence and contradictory
# cues reduce the score.
#
# The system does NOT claim that these rules prove an emotion.
# They are behavior interpretations based on available signals.
# ============================================================

GESTURES = [

    # --------------------------------------------------------
    # HAPPY
    # --------------------------------------------------------

    BehaviorRule(
        name="Excited play",
        required_emotion="Happy",
        emotion_weights={"Happy": 0.9},
        positive={
            "running": 1.0,
            "tail_wagging": 1.0,
            "paw_raised": 0.35,
        },
        negative={
            "whimpering": 1.1,
            "tail_low": 0.8,
        },
        min_evidence=2,
        threshold=0.62,
    ),

    BehaviorRule(
        name="Playful walk",
        required_emotion="Happy",
        emotion_weights={"Happy": 0.8},
        positive={
            "walking": 1.0,
            "tail_wagging": 1.0,
        },
        negative={
            "tail_low": 0.6,
            "whimpering": 0.9,
        },
        min_evidence=2,
        threshold=0.62,
    ),

    BehaviorRule(
        name="Happy greeting",
        required_emotion="Happy",
        emotion_weights={"Happy": 0.8},
        positive={
            "standing": 0.7,
            "tail_wagging": 1.0,
            "paw_raised": 0.5,
        },
        negative={
            "growling": 1.0,
            "tail_low": 0.7,
        },
        min_evidence=2,
        threshold=0.60,
    ),

    BehaviorRule(
        name="Happy greeting bark",
        required_emotion="Happy",
        emotion_weights={"Happy": 0.8},
        positive={
            "standing": 0.5,
            "barking": 1.0,
            "tail_wagging": 0.8,
        },
        negative={
            "growling": 0.8,
            "tail_low": 0.7,
        },
        min_evidence=2,
        threshold=0.60,
    ),

    BehaviorRule(
        name="Cheerful and relaxed",
        required_emotion="Happy",
        emotion_weights={"Happy": 0.7},
        positive={
            "sitting": 0.8,
            "tail_moving": 0.7,
            "quiet": 0.5,
        },
        negative={
            "running": 0.9,
            "growling": 0.9,
            "whimpering": 0.9,
        },
        min_evidence=2,
        threshold=0.60,
    ),

    # --------------------------------------------------------
    # RELAXED
    # --------------------------------------------------------

    BehaviorRule(
        name="Friendly relaxed greeting",
        required_emotion="Relaxed",
        emotion_weights={"Relaxed": 0.9},
        positive={
            "standing": 0.5,
            "tail_wagging": 1.0,
            "quiet": 0.5,
        },
        negative={
            "growling": 1.0,
            "whimpering": 0.8,
            "running": 0.5,
        },
        min_evidence=2,
        threshold=0.60,
    ),

    BehaviorRule(
        name="Casual walk",
        required_emotion="Relaxed",
        emotion_weights={"Relaxed": 0.8},
        positive={
            "walking": 1.0,
            "tail_moving": 0.7,
        },
        negative={
            "running": 0.8,
            "growling": 0.8,
            "whimpering": 0.8,
        },
        min_evidence=2,
        threshold=0.60,
    ),

    BehaviorRule(
        name="Resting calmly",
        required_emotion="Relaxed",
        emotion_weights={"Relaxed": 0.9},
        positive={
            "lying": 1.1,
            "still": 0.9,
            "quiet": 0.5,
        },
        negative={
            "running": 1.0,
            "walking": 0.5,
            "barking": 0.8,
            "growling": 0.8,
        },
        min_evidence=2,
        threshold=0.62,
    ),

    BehaviorRule(
        name="Alert but calm",
        required_emotion="Relaxed",
        emotion_weights={"Relaxed": 0.6},
        positive={
            "sitting": 0.8,
            "ears_forward": 0.9,
        },
        negative={
            "growling": 0.8,
            "barking": 0.5,
            "whimpering": 0.7,
        },
        min_evidence=2,
        threshold=0.60,
    ),

    # --------------------------------------------------------
    # SAD / DISTRESS
    # --------------------------------------------------------

    BehaviorRule(
        name="Distress whimper",
        required_emotion="Sad",
        emotion_weights={"Sad": 0.8},
        positive={
            "whimpering": 1.2,
            "tail_low": 0.8,
            "still": 0.6,
        },
        negative={
            "tail_wagging": 0.8,
            "running": 0.8,
        },
        min_evidence=2,
        threshold=0.62,
    ),

    BehaviorRule(
        name="Withdrawn / low energy",
        required_emotion="Sad",
        emotion_weights={"Sad": 0.9},
        positive={
            "lying": 1.0,
            "tail_low": 0.9,
            "still": 0.8,
        },
        negative={
            "running": 1.0,
            "tail_wagging": 0.8,
        },
        min_evidence=2,
        threshold=0.62,
    ),

    BehaviorRule(
        name="Anxious pacing",
        required_emotion="Sad",
        emotion_weights={"Sad": 0.7},
        positive={
            "walking": 1.0,
            "tail_low": 1.0,
        },
        negative={
            "tail_wagging": 0.8,
        },
        min_evidence=2,
        threshold=0.62,
    ),

    # --------------------------------------------------------
    # ANGRY / WARNING
    # --------------------------------------------------------

    BehaviorRule(
        name="Alert / warning bark",
        required_emotion="Angry",
        emotion_weights={"Angry": 0.8, "Uncertain": 0.5},
        positive={
            "barking": 1.5,
            "standing": 0.6,
            "ears_forward": 0.6,
        },
        negative={
            "whimpering": 0.8,
            "tail_wagging": 0.7,
            "lying": 0.6,
        },
        min_evidence=1,
        threshold=0.48,
    ),

    BehaviorRule(
        name="Defensive growl",
        required_emotion="Angry",
        emotion_weights={"Angry": 0.9, "Uncertain": 0.5},
        positive={
            "growling": 1.5,
            "standing": 0.6,
            "still": 0.4,
            "ears_forward": 0.5,
            "tail_raised": 0.6,
        },
        negative={
            "whimpering": 0.8,
            "tail_wagging": 0.9,
            "lying": 0.8,
        },
        min_evidence=1,
        threshold=0.48,
    ),

    BehaviorRule(
        name="Tense / on guard",
        required_emotion="Angry",
        emotion_weights={"Angry": 0.8},
        positive={
            "standing": 0.8,
            "tail_raised": 0.8,
            "ears_forward": 0.8,
        },
        negative={
            "tail_wagging": 0.6,
            "lying": 0.7,
        },
        min_evidence=2,
        threshold=0.62,
    ),

    # --------------------------------------------------------
    # UNCERTAIN / OBSERVATION
    # --------------------------------------------------------

    BehaviorRule(
        name="Attentive / observing",
        required_emotion="Uncertain",
        emotion_weights={"Uncertain": 0.4},
        positive={
            "standing": 1.0,
            "ears_forward": 0.9,
        },
        negative={
            "lying": 0.5,
            "growling": 0.5,
            "whimpering": 0.5,
        },
        min_evidence=2,
        threshold=0.60,
    ),

    BehaviorRule(
        name="Resting / observing",
        required_emotion="Uncertain",
        emotion_weights={"Uncertain": 0.3},
        positive={
            "lying": 1.0,
            "still": 0.7,
        },
        negative={
            "running": 0.9,
            "barking": 0.6,
        },
        min_evidence=2,
        threshold=0.60,
    ),
]


# Validate every rule cue.
_documented = set(POSE_CUES)

for _rule in GESTURES:
    undocumented = (
        set(_rule.positive)
        | set(_rule.negative)
    ) - _documented

    if undocumented:
        raise ValueError(
            f"fusion.py: undocumented cues in "
            f"{_rule.name}: {sorted(undocumented)}"
        )


# ============================================================
# DATA STRUCTURES
# ============================================================

@dataclass
class SignalFrame:
    emotion_label: str
    emotion_conf: float

    motion_label: str
    vocalization_label: str

    vocal_conf: float = None
    pose_cues: dict = None


# ============================================================
# FUSION ENGINE
# ============================================================

class FusionEngine:

    def __init__(
        self,
        threshold: float = 0.60,
        contradiction_penalty: float = 0.35,
    ):
        self.threshold = float(threshold)
        self.contradiction_penalty = float(
            contradiction_penalty
        )

    # --------------------------------------------------------
    # Active cues
    # --------------------------------------------------------

    def _active_cues(
        self,
        motion_label,
        vocalization_label,
        vocal_conf,
        pose_cues,
    ):
        active = {}

        # -----------------------------
        # Motion
        # -----------------------------

        motion_map = {
            "Still": "still",
            "Walking": "walking",
            "Running": "running",
        }

        motion_key = motion_map.get(
            str(motion_label)
        )

        if motion_key:
            active[motion_key] = 1.0

        # -----------------------------
        # Vocalization
        # -----------------------------

        vocal_map = {
            "Barking": "barking",
            "Growling": "growling",
            "Howling": "howling",
            "Whimpering": "whimpering",
            "Quiet": "quiet",
        }

        vocal_key = vocal_map.get(
            str(vocalization_label)
        )

        if vocal_key:
            active[vocal_key] = (
                _clamp01(
                    vocal_conf
                    if vocal_conf is not None
                    else 1.0
                )
            )

        # -----------------------------
        # Pose
        # -----------------------------

        pose_cues = pose_cues or {}

        posture_map = {
            "Sitting": "sitting",
            "Standing": "standing",
            "Lying": "lying",
        }

        posture = pose_cues.get("posture")

        if posture in posture_map:
            confidence = pose_cues.get(
                "posture_confidence",
                1.0,
            )

            active[
                posture_map[posture]
            ] = _clamp01(confidence)

        # -----------------------------
        # Direct boolean/score cues
        # -----------------------------

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

            is_active, confidence = _cue_active(
                pose_cues[key]
            )

            if is_active:
                active[cue_name] = confidence

        # -----------------------------
        # Structured tail data
        # -----------------------------

        tail = pose_cues.get("tail")

        if isinstance(tail, dict):

            position_conf = _clamp01(
                tail.get(
                    "position_confidence",
                    0.0,
                )
            )

            if bool(tail.get("low")):
                active["tail_low"] = position_conf

            if bool(tail.get("raised")):
                active["tail_raised"] = position_conf

            # IMPORTANT:
            # Do NOT force a minimum confidence of 0.5.
            # Use the actual tail signal.
            if bool(tail.get("moving")):

                movement_score = _clamp01(
                    tail.get(
                        "movement_score",
                        0.0,
                    )
                    / 0.12
                )

                active["tail_moving"] = (
                    movement_score
                )

            if bool(tail.get("wagging")):

                wag_score = _clamp01(
                    tail.get(
                        "wag_score",
                        0.0,
                    )
                )

                active["tail_wagging"] = (
                    wag_score
                )

        # -----------------------------
        # Structured ears
        # -----------------------------

        ears = pose_cues.get("ears")

        if isinstance(ears, dict):

            label = str(
                ears.get("label", "")
            )

            confidence = _clamp01(
                ears.get("confidence", 0.0)
            )

            if label == "Forward":
                active["ears_forward"] = confidence

            elif label == "Back":
                active["ears_back"] = confidence

        # -----------------------------
        # Structured head
        # -----------------------------

        head = pose_cues.get("head")

        if isinstance(head, dict):

            label = str(
                head.get("label", "")
            )

            confidence = _clamp01(
                head.get("confidence", 0.0)
            )

            if label == "Left":
                active["head_left"] = confidence

            elif label == "Right":
                active["head_right"] = confidence

        # -----------------------------
        # Structured paw
        # -----------------------------

        paw = pose_cues.get("paw")

        if isinstance(paw, dict):

            label = str(
                paw.get("label", "None")
            )

            confidence = _clamp01(
                paw.get("confidence", 0.0)
            )

            if (
                label
                and label.lower() != "none"
                and label.lower() != "unknown"
            ):
                active["paw_raised"] = confidence

        return {
            key: value
            for key, value in active.items()
            if value > 0.0
        }

    # --------------------------------------------------------
    # Score emotion evidence
    # --------------------------------------------------------

    def _emotion_score(
        self,
        rule: BehaviorRule,
        emotion: str,
        emotion_conf: float,
    ):
        """
        Emotion contributes as weighted evidence.

        It does NOT block a rule when the emotion differs.
        """

        weight = rule.emotion_weights.get(
            emotion,
            0.0,
        )

        if weight <= 0:
            return 0.0, None

        confidence = _clamp01(
            emotion_conf
        )

        return (
            weight * confidence,
            {
                "cue": f"emotion:{emotion}",
                "confidence": round(
                    confidence,
                    3,
                ),
            },
        )

    # --------------------------------------------------------
    # Score one behavior
    # --------------------------------------------------------

    def _score_gesture(
        self,
        rule: BehaviorRule,
        emotion: str,
        emotion_conf: float,
        active: dict,
    ):
        positive_total = (
            sum(
                float(v)
                for v in rule.positive.values()
            )
            + sum(
                float(v)
                for v in rule.emotion_weights.values()
            )
        )

        if positive_total <= 0:
            return {
                "score": 0.0,
                "positive_score": 0.0,
                "negative_score": 0.0,
                "evidence_count": 0,
                "matched_cues": [],
                "contradictions": [],
            }

        positive_weighted = 0.0
        matched_cues = []

        # -----------------------------
        # Emotion evidence
        # -----------------------------

        emotion_contribution, emotion_item = (
            self._emotion_score(
                rule,
                emotion,
                emotion_conf,
            )
        )

        positive_weighted += (
            emotion_contribution
        )

        if emotion_item:
            matched_cues.append(
                emotion_item
            )

        # -----------------------------
        # Positive pose/audio evidence
        # -----------------------------

        for cue, weight in rule.positive.items():

            confidence = _clamp01(
                active.get(cue, 0.0)
            )

            if confidence <= 0:
                continue

            contribution = (
                float(weight) * confidence
            )

            positive_weighted += contribution

            matched_cues.append(
                {
                    "cue": cue,
                    "confidence": round(
                        confidence,
                        3,
                    ),
                }
            )

        positive_score = (
            positive_weighted
            / positive_total
        )

        # -----------------------------
        # Contradictory evidence
        # -----------------------------

        negative_total = sum(
            float(v)
            for v in rule.negative.values()
        )

        negative_weighted = 0.0
        contradictions = []

        for cue, weight in rule.negative.items():

            confidence = _clamp01(
                active.get(cue, 0.0)
            )

            if confidence <= 0:
                continue

            negative_weighted += (
                float(weight) * confidence
            )

            contradictions.append(
                {
                    "cue": cue,
                    "confidence": round(
                        confidence,
                        3,
                    ),
                }
            )

        negative_score = (
            negative_weighted
            / negative_total
            if negative_total > 0
            else 0.0
        )

        # -----------------------------
        # Minimum evidence
        # -----------------------------

        evidence_count = len(
            matched_cues
        )

        # Emotion counts as evidence only when
        # the rule actually expects that emotion.
        pose_evidence_count = sum(
            1
            for item in matched_cues
            if not str(
                item["cue"]
            ).startswith("emotion:")
        )

        if (
            pose_evidence_count
            < rule.min_evidence
        ):
            evidence_factor = (
                pose_evidence_count
                / float(
                    max(
                        1,
                        rule.min_evidence,
                    )
                )
            )
        else:
            evidence_factor = 1.0

        # -----------------------------
        # Final match
        # -----------------------------

        final_score = (
            positive_score
            * evidence_factor
            * (
                1.0
                - self.contradiction_penalty
                * negative_score
            )
        )

        final_score = _clamp01(
            final_score
        )

        return {
            "score": final_score,
            "positive_score": _clamp01(
                positive_score
            ),
            "negative_score": _clamp01(
                negative_score
            ),
            "evidence_count": evidence_count,
            "pose_evidence_count": pose_evidence_count,
            "matched_cues": matched_cues,
            "contradictions": contradictions,
        }

    # --------------------------------------------------------
    # Select best behavior
    # --------------------------------------------------------

    def select(
        self,
        emotion_label,
        emotion_conf,
        motion_label,
        vocalization_label,
        vocal_conf=None,
        pose_cues=None,
    ):
        emotion = _canonicalize(
            emotion_label
        )

        emotion_conf = _clamp01(
            emotion_conf
        )

        active = self._active_cues(
            motion_label=motion_label,
            vocalization_label=vocalization_label,
            vocal_conf=vocal_conf,
            pose_cues=pose_cues,
        )

        scored = []

        for rule in GESTURES:

            result = self._score_gesture(
                rule=rule,
                emotion=emotion,
                emotion_conf=emotion_conf,
                active=active,
            )

            scored.append(
                (
                    rule,
                    result,
                )
            )

        # Highest evidence score.
        best_rule, best_result = max(
            scored,
            key=lambda item: item[1]["score"],
            default=(None, None),
        )

        if best_rule is None:
            return self._fallback(
                emotion,
                emotion_conf,
                motion_label,
                vocalization_label,
                active,
            )

        best_score = float(
            best_result["score"]
        )

        # A rule must pass both the global threshold
        # and its own threshold.
        required_threshold = max(
            self.threshold,
            best_rule.threshold,
        )

        if best_score < required_threshold:
            return self._fallback(
                emotion,
                emotion_conf,
                motion_label,
                vocalization_label,
                active,
                best_rule=best_rule,
                best_result=best_result,
            )

        # Final confidence is based on:
        #   • rule match
        #   • emotion reliability
        #   • evidence quality
        #
        # Do not multiply by every cue confidence again;
        # the cue confidences already participate in best_score.
        final_confidence = _clamp01(
            0.70 * best_score
            + 0.30 * emotion_conf
        )

        return {
            "gesture": best_rule.name,
            "matched_rule": True,
            "matched_emotion": best_rule.required_emotion,

            "score": round(
                final_confidence,
                3,
            ),

            "match_score": round(
                best_score,
                3,
            ),

            "confidence": round(
                final_confidence,
                3,
            ),

            "positive_score": round(
                best_result["positive_score"],
                3,
            ),

            "negative_score": round(
                best_result["negative_score"],
                3,
            ),

            "matched_cues": best_result[
                "matched_cues"
            ],

            "contradictions": best_result[
                "contradictions"
            ],

            "active_cues": sorted(
                active.keys()
            ),

            "line": format_line(
                best_rule.name,
                True,
                final_confidence,
                emotion,
                motion_label,
                vocalization_label,
                emotion_conf,
                matched_cues=best_result[
                    "matched_cues"
                ],
                contradictions=best_result[
                    "contradictions"
                ],
            ),
        }

    # --------------------------------------------------------
    # Fallback
    # --------------------------------------------------------

    def _fallback(
        self,
        emotion,
        emotion_conf,
        motion_label,
        vocalization_label,
        active,
        best_rule=None,
        best_result=None,
    ):
        """
        Return an observation instead of forcing an unreliable
        behavior label.
        """

        return {
            "gesture": (
                f"{emotion} | "
                f"{motion_label} | "
                f"{vocalization_label}"
            ),

            "matched_rule": False,
            "matched_emotion": (
                best_rule.required_emotion
                if best_rule
                else None
            ),

            "score": 0.0,
            "match_score": (
                round(
                    best_result["score"],
                    3,
                )
                if best_result
                else 0.0
            ),

            "confidence": round(
                emotion_conf,
                3,
            ),

            "positive_score": (
                round(
                    best_result[
                        "positive_score"
                    ],
                    3,
                )
                if best_result
                else 0.0
            ),

            "negative_score": (
                round(
                    best_result[
                        "negative_score"
                    ],
                    3,
                )
                if best_result
                else 0.0
            ),

            "matched_cues": (
                best_result[
                    "matched_cues"
                ]
                if best_result
                else []
            ),

            "contradictions": (
                best_result[
                    "contradictions"
                ]
                if best_result
                else []
            ),

            "active_cues": sorted(
                active.keys()
            ),

            "line": format_line(
                None,
                False,
                0.0,
                emotion,
                motion_label,
                vocalization_label,
                emotion_conf,
                matched_cues=(
                    best_result[
                        "matched_cues"
                    ]
                    if best_result
                    else []
                ),
                contradictions=(
                    best_result[
                        "contradictions"
                    ]
                    if best_result
                    else []
                ),
            ),
        }


# ============================================================
# FORMATTING
# ============================================================

def _cue_display_name(cue):
    if cue.startswith("emotion:"):
        return cue.replace(
            "emotion:",
            "Emotion = ",
        )

    return POSE_CUES.get(
        cue,
        cue.replace("_", " ").title(),
    )


def format_line(
    gesture,
    matched,
    score,
    emotion,
    motion,
    vocalization,
    emotion_conf=None,
    matched_cues=None,
    contradictions=None,
):
    """
    Human-readable one-line output.
    """

    emo = (
        f"{emotion} "
        f"{emotion_conf:.0%}"
        if emotion_conf is not None
        else emotion
    )

    signals = (
        f"{emo} | "
        f"{motion} | "
        f"{vocalization}"
    )

    if not matched:
        return (
            " ".join(
                f"Dog gesture: no confident match | "
                f"{signals}"
            .split())
        )

    base = (
        f"Dog gesture: {gesture} "
        f"(confidence {score:.0%}) | "
        f"{signals}"
    )

    matched_cues = matched_cues or []
    contradictions = contradictions or []

    if matched_cues:
        evidence = ", ".join(
            _cue_display_name(
                item["cue"]
            )
            for item in matched_cues
            if item.get("confidence", 0) >= 0.35
        )

        if evidence:
            base += f" | Evidence: {evidence}"

    if contradictions:
        conflict = ", ".join(
            _cue_display_name(
                item["cue"]
            )
            for item in contradictions
            if item.get("confidence", 0) >= 0.35
        )

        if conflict:
            base += f" | Conflict: {conflict}"

    return " ".join(base.split())


# ============================================================
# PUBLIC API
# ============================================================

_engine = FusionEngine(
    threshold=0.60,
    contradiction_penalty=0.35,
)


def resolve_final_emotion(
    canonical_emotion,
    emotion_conf,
    emotion_is_confident,
    fused,
    motion_label,
    posture_label,
):
    """
    Correct the displayed emotion using strong, unambiguous behavioral
    counter-evidence, instead of blindly trusting a single-frame,
    appearance-only image classifier.

    This closes a gap where fuse()/GestureEngine.select() computed a
    gesture and contradiction evidence (e.g. tail_wagging contradicting a
    "Sad" rule), but nothing ever fed that back into the emotion label the
    UI actually shows -- so a dog standing/walking with an actively
    wagging tail could still be displayed as "Sad" forever, no matter how
    much pose evidence disagreed with the CNN's guess for that one frame.

    Returns (label, confidence, overridden: bool, reason: str | None).
    """
    base_label = canonical_emotion if emotion_is_confident else "Uncertain"
    base_conf = float(emotion_conf)

    if not fused:
        return base_label, base_conf, False, None

    active_cues = set(fused.get("active_cues") or [])

    # A dog can't plausibly be read as Sad/Angry in a frame where its tail
    # is actively wagging AND it's standing, sitting, walking or running --
    # and isn't simultaneously whimpering or growling (which would be
    # genuine corroborating distress/aggression evidence, not a
    # contradiction).
    strong_positive_behavior = (
        "tail_wagging" in active_cues
        and (
            motion_label in ("Walking", "Running")
            or posture_label in ("Standing", "Sitting")
        )
        and "whimpering" not in active_cues
        and "growling" not in active_cues
    )

    if strong_positive_behavior and base_label in ("Sad", "Angry"):
        matched_emotion = fused.get("matched_emotion")
        if fused.get("matched_rule") and matched_emotion in ("Happy", "Relaxed"):
            return (
                matched_emotion,
                float(fused.get("confidence", base_conf)),
                True,
                f"overridden by matched behavior '{fused.get('gesture')}'",
            )
        # No single positive gesture rule cleared its threshold, but the
        # raw label is still clearly contradicted by tail-wagging + active
        # posture -- showing a confidently wrong Sad/Angry is worse than
        # admitting uncertainty.
        return (
            "Uncertain",
            base_conf,
            True,
            "raw emotion contradicted by tail-wagging + active posture/motion",
        )

    return base_label, base_conf, False, None


def fuse(
    emotion_label: str,
    emotion_conf: float,
    motion_label: str,
    vocalization_label: str,
    vocal_conf: float = None,
    pose_cues: dict = None,
) -> dict:

    return _engine.select(
        emotion_label=emotion_label,
        emotion_conf=emotion_conf,
        motion_label=motion_label,
        vocalization_label=vocalization_label,
        vocal_conf=vocal_conf,
        pose_cues=pose_cues,
    )


def fuse_line(
    emotion_label: str,
    emotion_conf: float,
    motion_label: str,
    vocalization_label: str,
    vocal_conf: float = None,
    pose_cues: dict = None,
) -> str:

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


# ============================================================
# TESTS
# ============================================================

if __name__ == "__main__":

    tests = [

        # Strong happy/play signal.
        (
            "Happy",
            0.87,
            "Running",
            "Quiet",
            0.70,
            {
                "posture": "Standing",
                "posture_confidence": 0.82,
                "tail": {
                    "wagging": True,
                    "wag_score": 0.88,
                    "moving": True,
                    "movement_score": 0.11,
                },
                "paw": {
                    "label": "Raised",
                    "confidence": 0.70,
                },
            },
        ),

        # Happy walk.
        (
            "Happy",
            0.84,
            "Walking",
            "Quiet",
            0.75,
            {
                "posture": "Standing",
                "posture_confidence": 0.80,
                "tail": {
                    "wagging": True,
                    "wag_score": 0.78,
                    "moving": True,
                    "movement_score": 0.08,
                },
            },
        ),

        # Relaxed resting.
        (
            "Relaxed",
            0.81,
            "Still",
            "Quiet",
            0.90,
            {
                "posture": "Lying",
                "posture_confidence": 0.88,
            },
        ),

        # Angry warning.
        (
            "Angry",
            0.82,
            "Still",
            "Growling",
            0.91,
            {
                "posture": "Standing",
                "posture_confidence": 0.80,
                "tail": {
                    "raised": True,
                    "position_confidence": 0.75,
                },
                "ears": {
                    "label": "Forward",
                    "confidence": 0.82,
                },
            },
        ),

        # Deliberately contradictory case.
        # This should NOT confidently become Excited play.
        (
            "Happy",
            0.52,
            "Running",
            "Whimpering",
            0.94,
            {
                "posture": "Standing",
                "posture_confidence": 0.82,
                "tail": {
                    "low": True,
                    "position_confidence": 0.84,
                    "wagging": False,
                    "moving": False,
                },
            },
        ),
    ]

    for index, args in enumerate(
        tests,
        start=1,
    ):
        result = fuse(*args)

        print(
            f"\n--- TEST {index} ---"
        )

        print(
            result["line"]
        )

        print(
            "match_score:",
            result["match_score"]
        )

        print(
            "confidence:",
            result["confidence"]
        )

        print(
            "matched_cues:",
            result["matched_cues"]
        )

        print(
            "contradictions:",
            result["contradictions"]
        )