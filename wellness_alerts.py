"""
wellness_alerts.py — Heuristic welfare flags derived from pose keypoint
history over time: possible limping, sudden collapse, prolonged
stillness, and low tail carriage.

IMPORTANT: these are pattern flags, not diagnoses. They use rough
geometric heuristics on 2D keypoints from a single camera angle — not
validated against real veterinary ground truth. Breed anatomy (e.g.
naturally low-set tails), camera angle, occlusion, and senior dogs'
naturally reduced gait range can all trigger false positives. Always
surface these as "worth watching," never as a medical claim, and point
the user to a vet for anything persistent or concerning.

To add a new alert: track whatever raw signal you need in
WellnessTracker.__init__, update it in update(), and only append an
alert once you've seen a *sustained* pattern (never a single frame).
"""

from collections import deque
import numpy as np

from pose_motion import KP, PAW_INDICES

PAW_LABELS = {
    KP["front_left_paw"]: "front-left leg",
    KP["rear_left_paw"]: "rear-left leg",
    KP["front_right_paw"]: "front-right leg",
    KP["rear_right_paw"]: "rear-right leg",
}

LOW_TAIL_RATIO = 0.15  # tail_end below tail_start by more than this fraction of bbox diagonal


class WellnessTracker:
    def __init__(self, window: int = 20, gait_deficit_threshold: float = 0.06):
        """window: number of recent /predict calls considered (roughly
        window * ~0.9s at the frontend's polling rate)."""
        self.window = window
        self.gait_deficit_threshold = gait_deficit_threshold

        self.paw_deficit_history = {idx: deque(maxlen=window) for idx in PAW_INDICES}
        self.gait_frames_seen = 0
        self.motion_history = deque(maxlen=window)
        self.tail_drop_history = deque(maxlen=window)
        self.posture_history = deque(maxlen=3)

    def update(self, kpts, bbox_diagonal, motion_label: str, posture_label: str = None) -> list:
        alerts = []
        self.motion_history.append(motion_label)

        if posture_label is not None:
            alerts.extend(self._check_collapse(posture_label, motion_label))

        alerts.extend(self._check_prolonged_stillness())  # motion-only, no keypoints needed

        if kpts is None or bbox_diagonal in (None, 0):
            return alerts

        alerts.extend(self._check_gait(kpts, bbox_diagonal, motion_label))
        self._track_tail(kpts, bbox_diagonal)
        alerts.extend(self._check_tail_carriage())
        return alerts

    # --- possible limping: a paw that never reaches ground level while moving ---
    def _check_gait(self, kpts, bbox_diagonal, motion_label: str) -> list:
        if motion_label not in ("Walking", "Running"):
            self.gait_frames_seen = 0
            for buf in self.paw_deficit_history.values():
                buf.clear()
            return []

        paw_ys = {idx: float(kpts[idx][1]) for idx in PAW_INDICES if idx < len(kpts)}
        if len(paw_ys) < 4:
            return []

        ground_y = max(paw_ys.values())  # largest y = lowest in image = closest to ground
        for idx, y in paw_ys.items():
            deficit = (ground_y - y) / bbox_diagonal  # ~0 if this paw is the grounded one
            self.paw_deficit_history[idx].append(deficit)
        self.gait_frames_seen = min(self.gait_frames_seen + 1, self.window)

        if self.gait_frames_seen < self.window:
            return []  # not enough data yet to judge a full gait cycle

        for idx, buf in self.paw_deficit_history.items():
            if len(buf) == self.window and min(buf) > self.gait_deficit_threshold:
                leg = PAW_LABELS.get(idx, "a leg")
                return [{
                    "type": "possible_limping",
                    "severity": "watch",
                    "message": f"Possible limping: {leg} hasn't touched the ground over the "
                               f"last {self.window} frames of movement.",
                }]
        return []

    # --- sudden collapse: standing/moving -> lying without sitting first ---
    def _check_collapse(self, posture_label: str, motion_label: str) -> list:
        self.posture_history.append(posture_label)
        if len(self.posture_history) < 2:
            return []
        prev = self.posture_history[-2]
        curr = self.posture_history[-1]
        if curr == "Lying" and prev == "Standing" and motion_label != "Still":
            return [{
                "type": "possible_collapse",
                "severity": "critical",
                "message": "Sudden change detected: went from standing/moving to lying down "
                           "very quickly, without sitting first. Worth checking on your dog.",
            }]
        return []

    # --- prolonged stillness: lethargy watch, not a diagnosis ---
    def _check_prolonged_stillness(self) -> list:
        if len(self.motion_history) == self.window and all(m == "Still" for m in self.motion_history):
            return [{
                "type": "prolonged_stillness",
                "severity": "info",
                "message": "No movement detected for a sustained period.",
            }]
        return []

    # --- low/tucked tail carriage sustained over time ---
    def _track_tail(self, kpts, bbox_diagonal):
        try:
            tail_start = kpts[KP["tail_start"]]
            tail_end = kpts[KP["tail_end"]]
            tail_drop = float(tail_end[1] - tail_start[1]) / bbox_diagonal
            self.tail_drop_history.append(tail_drop)
        except (IndexError, KeyError):
            pass

    def _check_tail_carriage(self) -> list:
        if len(self.tail_drop_history) == self.window:
            avg_drop = float(np.mean(self.tail_drop_history))
            if avg_drop > LOW_TAIL_RATIO:
                return [{
                    "type": "low_tail_carriage",
                    "severity": "watch",
                    "message": "Tail has been held low for a sustained period — sometimes "
                               "associated with anxiety or discomfort, but also just a breed "
                               "trait or a resting posture.",
                }]
        return []
