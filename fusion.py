"""
fusion.py — Combines emotion, motion, vocalization, and pose-derived cues
(posture, tail carriage, gait) into a single "gesture" description, using
a weighted cue-matching engine rather than a flat rule table.

Design principles:

1. Emotion is a HARD GATE, not just another weighted cue. Each gesture
   requires a specific emotion; if the observed emotion doesn't match, the
   gesture scores 0 and can never win — this is what stops an angry,
   barking, running dog from ever being read as a playful gesture, and
   what guarantees different emotions produce different top gestures
   rather than collapsing onto whichever gesture has the loosest cues.

2. Every cue a gesture can require is registered in POSE_CUES up front.
   Gestures are built FROM that registry, not the other way around — so
   it's structurally impossible to reference an undocumented cue.

3. Only cues we can ACTUALLY measure from the current pipeline are used.
   Cues like "hackles raised" or a true play-bow posture (front legs down,
   rear up) aren't measured by anything in this project, so they're never
   defined as gesture requirements — an unmeasurable cue left in a
   gesture's requirements would always read as "not matched," silently
   capping that gesture's score rather than ever letting it win by luck.

4. The reported score is a genuine match fraction (matched cue weight /
   total required weight), scaled by how confident the underlying signals
   actually were. It is not a deep-model probability and is not labeled
   as one, and it can't be inflated by cues that were never measured.

5. threshold = 0.6 — a fusion result below this reports the plain
   composite description instead of forcing a weak winner via tie-break.

6. EMOTION_CANONICAL decouples this file from whatever exact class-name
   strings model.py's trained classifier happens to use (dataset folder
   names like "smileydogs"/"gooddogs" are common in public dog-emotion
   sets, and differ across datasets). Gestures are always written against
   the canonical Happy/Relaxed/Sad/Angry/Uncertain space; raw labels are
   normalized through this map once, not scattered as string comparisons
   throughout the scoring logic.

7. Every result carries a "line" key: the whole prediction as ONE
   plain-ASCII line (no newlines), e.g.
       Dog gesture: Excited play chase (score 0.82) | Happy 87% | Running | Barking
   Use fuse_line() / fuse_frame_line() if you only want that string.
"""

from dataclasses import dataclass, field


# ----------------------------------------------------------------------
# Cue registry. Every cue referenced by any gesture below MUST appear
# here — this file's own consistency (checked at import time) depends on
# gestures never reaching for a cue this dict doesn't define.
# ----------------------------------------------------------------------
POSE_CUES = {
    "still": "Motion signal reports Still",
    "walking": "Motion signal reports Walking",
    "running": "Motion signal reports Running",
    "sitting": "Posture signal reports Sitting",
    "standing": "Posture signal reports Standing",
    "lying": "Posture signal reports Lying",
    "tail_low": "Tail held low/tucked, from pose keypoints",
    "barking": "Vocalization signal reports Barking",
    "growling": "Vocalization signal reports Growling",
    "howling": "Vocalization signal reports Howling",
    "whimpering": "Vocalization signal reports Whimpering",
    "quiet": "Vocalization signal reports Quiet / no vocalization",
}

# Raw class-name strings (however your trained model/dataset spells them)
# -> canonical emotion categories that GESTURES is written against. Add
# entries here if you train on a dataset with different folder names —
# nothing else in this file needs to change.
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
}


def _canonicalize(label: str) -> str:
    return EMOTION_CANONICAL.get(label.lower(), label)


@dataclass
class Gesture:
    name: str
    required_emotion: str  # hard gate -- must canonically match the observed emotion
    cues: dict = field(default_factory=dict)  # {cue_name: weight}, cue_name must be in POSE_CUES
    required_emotion_canonical: str = field(init=False)

    def __post_init__(self):
        # Normalize once at definition time, not on every scoring call --
        # this is looked up ~14 times per /predict request otherwise.
        self.required_emotion_canonical = _canonicalize(self.required_emotion)


@dataclass
class SignalFrame:
    """Optional structured container for one frame of signals -- use this
    or the positional fuse() arguments, whichever reads better at the call
    site. See fuse_frame() below."""
    emotion_label: str
    emotion_conf: float
    motion_label: str
    vocalization_label: str
    vocal_conf: float = None
    pose_cues: dict = None


GESTURES = [
    # --- Happy ---
    Gesture("Excited play chase", "Happy", {"running": 1.0, "barking": 1.0}),
    Gesture("Playful run", "Happy", {"running": 1.0, "quiet": 0.5}),
    Gesture("Cheerful walk", "Happy", {"walking": 1.0}),
    Gesture("Happy greeting bark", "Happy", {"standing": 0.5, "barking": 1.0}),
    Gesture("Content and relaxed", "Happy", {"sitting": 1.0, "quiet": 1.0}),

    # --- Relaxed ---
    Gesture("Casual walk", "Relaxed", {"walking": 1.0}),
    Gesture("Resting calmly", "Relaxed", {"lying": 1.0, "quiet": 1.0}),
    Gesture("Alert but calm", "Relaxed", {"sitting": 1.0, "quiet": 1.0}),

    # --- Sad ---
    Gesture("Distress whimper", "Sad", {"whimpering": 1.0, "still": 0.5}),
    Gesture("Withdrawn / low mood", "Sad", {"lying": 1.0, "quiet": 1.0}),
    Gesture("Anxious pacing", "Sad", {"walking": 1.0, "tail_low": 1.0}),

    # --- Angry ---
    Gesture("Alert / warning bark", "Angry", {"barking": 1.0, "standing": 0.5}),
    Gesture("Defensive growl", "Angry", {"growling": 1.0}),
    Gesture("Tense / on guard", "Angry", {"still": 1.0, "standing": 1.0}),

    # --- Uncertain (emotion confidence gate failed) -- still gives a
    #     meaningful reading from motion/posture/vocalization alone,
    #     instead of always falling through to the generic composite. ---
    Gesture("Attentive / observing", "Uncertain", {"standing": 1.0, "quiet": 0.5}),
    Gesture("Resting / observing", "Uncertain", {"lying": 1.0, "quiet": 1.0}),
]

# Any cue used above that isn't registered in POSE_CUES is a bug in this
# file, not in the caller -- fail loudly at import time.
_undocumented = sorted({cue for g in GESTURES for cue in g.cues} - set(POSE_CUES))
if _undocumented:
    raise ValueError(f"fusion.py: cues used by gestures but missing from POSE_CUES: {_undocumented}")


def format_line(gesture: str, matched: bool, score: float, emotion: str,
                motion: str, vocalization: str, emotion_conf: float = None) -> str:
    """
    Renders one fusion result as a SINGLE plain-ASCII line, e.g.

        Dog gesture: Excited play chase (score 0.82) | Happy 87% | Running | Barking
        Dog gesture: no confident match | Happy 87% | Running | Quiet

    ASCII-only on purpose so print() never fails on a Windows cp1252 console.
    """
    emo = f"{emotion} {emotion_conf:.0%}" if emotion_conf is not None else f"{emotion}"
    signals = f"{emo} | {motion} | {vocalization}"
    if matched:
        line = f"Dog gesture: {gesture} (score {score:.2f}) | {signals}"
    else:
        line = f"Dog gesture: no confident match | {signals}"
    # Guarantee "one line" even if a label ever arrives with a newline in it.
    return " ".join(line.split())


class FusionEngine:
    def __init__(self, threshold: float = 0.6):
        self.threshold = threshold  # a match below this is reported as a plain composite, not forced to win

    def _active_cues(self, motion_label, vocalization_label, pose_cues) -> set:
        """Builds the set of currently-true cue names from the raw signals."""
        active = set()

        motion_map = {"Still": "still", "Walking": "walking", "Running": "running"}
        if motion_label in motion_map:
            active.add(motion_map[motion_label])

        vocal_map = {"Barking": "barking", "Growling": "growling", "Howling": "howling",
                     "Whimpering": "whimpering", "Quiet": "quiet"}
        if vocalization_label in vocal_map:
            active.add(vocal_map[vocalization_label])

        pose_cues = pose_cues or {}
        posture = pose_cues.get("posture")
        posture_map = {"Sitting": "sitting", "Standing": "standing", "Lying": "lying"}
        if posture in posture_map:
            active.add(posture_map[posture])

        if pose_cues.get("tail_low"):
            active.add("tail_low")

        return active

    def _score_gesture(self, gesture: Gesture, emotion_canonical: str, active_cues: set) -> float:
        """
        Returns a match fraction in [0, 1]. A cue not present in active_cues
        counts as NOT matched (0 contribution) -- never skipped, so a
        gesture can't win by having most of its required cues simply
        unmeasured. Emotion mismatch is a hard 0.
        """
        if gesture.required_emotion_canonical != emotion_canonical:
            return 0.0
        if not gesture.cues:
            return 0.0

        total_weight = sum(gesture.cues.values())
        matched_weight = sum(w for cue, w in gesture.cues.items() if cue in active_cues)
        return matched_weight / total_weight if total_weight > 0 else 0.0

    def select(self, emotion_label, emotion_conf, motion_label, vocalization_label,
               vocal_conf=None, pose_cues=None) -> dict:
        emotion_canonical = _canonicalize(emotion_label)
        active_cues = self._active_cues(motion_label, vocalization_label, pose_cues)

        scored = [(g, self._score_gesture(g, emotion_canonical, active_cues)) for g in GESTURES]
        best_gesture, best_match = max(scored, key=lambda pair: pair[1], default=(None, 0.0))

        def _result(gesture_name, matched, score):
            return {
                "gesture": gesture_name,
                "matched_rule": matched,
                "score": score,
                "line": format_line(gesture_name, matched, score, emotion_canonical,
                                    motion_label, vocalization_label, emotion_conf),
            }

        if best_gesture is None or best_match < self.threshold:
            return _result(f"{emotion_canonical} · {motion_label} · {vocalization_label}", False, 0.0)

        # Blend in the real confidences of whichever signals actually
        # contributed a matched cue -- never a hardcoded stand-in.
        confidences = [c for c in [emotion_conf] if c is not None]
        used_vocal = any(cue in active_cues and cue in ("barking", "growling", "howling", "whimpering", "quiet")
                          for cue in best_gesture.cues)
        if used_vocal and vocal_conf is not None:
            confidences.append(vocal_conf)
        blended_confidence = sum(confidences) / len(confidences) if confidences else 1.0

        final_score = round(best_match * blended_confidence, 3)

        return _result(best_gesture.name, True, final_score)


# Module-level engine instance -- callers reference this directly
# (fusion._engine.threshold, etc.) rather than constructing their own.
_engine = FusionEngine(threshold=0.6)


def fuse(emotion_label: str, emotion_conf: float, motion_label: str,
         vocalization_label: str, vocal_conf: float = None, pose_cues: dict = None) -> dict:
    """
    pose_cues: optional dict, e.g. {"posture": "Standing", "tail_low": False}.
    Pass whatever you have -- missing keys just mean those cues are never
    "active" and can't contribute to any gesture's score.

    Returns {"gesture", "matched_rule", "score", "line"}; "line" is the
    whole prediction as one string.
    """
    return _engine.select(emotion_label, emotion_conf, motion_label,
                           vocalization_label, vocal_conf, pose_cues)


def fuse_line(emotion_label: str, emotion_conf: float, motion_label: str,
              vocalization_label: str, vocal_conf: float = None, pose_cues: dict = None) -> str:
    """Same as fuse(), but returns ONLY the single-line string."""
    return fuse(emotion_label, emotion_conf, motion_label,
                vocalization_label, vocal_conf, pose_cues)["line"]


def fuse_frame(frame: SignalFrame) -> dict:
    """Same as fuse(), but takes a SignalFrame instead of positional args --
    convenient when you're already assembling one for logging/serialization."""
    return fuse(frame.emotion_label, frame.emotion_conf, frame.motion_label,
                frame.vocalization_label, frame.vocal_conf, frame.pose_cues)


def fuse_frame_line(frame: SignalFrame) -> str:
    """Same as fuse_frame(), but returns ONLY the single-line string."""
    return fuse_frame(frame)["line"]


if __name__ == "__main__":
    # Quick smoke demo: python fusion.py
    print(fuse_line("happy", 0.87, "Running", "Barking"))
    print(fuse_line("angry", 0.74, "Still", "Growling", vocal_conf=0.9,
                    pose_cues={"posture": "Standing"}))
    print(fuse_line("sad", 0.55, "Walking", "Quiet"))