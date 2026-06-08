"""
gesture_classifier.py - Rule-based human gesture and posture recognition.

The classifier uses simple geometric rules over MediaPipe landmarks and joint
angles. This is easy to explain in a report and robust enough for a webcam demo
without requiring a training dataset.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np

from kinematics import (
    LEFT_ANKLE,
    LEFT_ELBOW,
    LEFT_HIP,
    LEFT_KNEE,
    LEFT_SHOULDER,
    LEFT_WRIST,
    NOSE,
    RIGHT_ANKLE,
    RIGHT_ELBOW,
    RIGHT_HIP,
    RIGHT_KNEE,
    RIGHT_SHOULDER,
    RIGHT_WRIST,
    compute_all_angles,
)
from utils import Point2D, Point3D, clamp01, mean_visibility, min_visibility


@dataclass(frozen=True)
class GestureResult:
    """Classification result for one frame."""

    label: str
    confidence: float
    details: Dict[str, float]

    def to_dict(self) -> Dict:
        return asdict(self)


class GestureClassifier:
    """
    Recognize common gestures/postures from MediaPipe Pose landmarks.

    Supported labels:
    - STANDING
    - SITTING
    - LEFT_HAND_UP
    - RIGHT_HAND_UP
    - BOTH_HANDS_UP
    - BENDING_FORWARD
    - UNKNOWN
    - NO_PERSON
    """

    def __init__(self, visibility_threshold: float = 0.35):
        self.visibility_threshold = visibility_threshold

    def classify(
        self,
        detection: Optional[Dict],
        angles: Optional[Dict[str, float]] = None,
    ) -> GestureResult:
        """Classify the current pose from a legacy detection dict."""
        if detection is None:
            return GestureResult("NO_PERSON", 0.0, {"reason": 0.0})

        landmarks_2d = detection.get("landmarks_2d", [])
        landmarks_3d = detection.get("landmarks_3d", [])
        visibility = detection.get("visibility", [])
        if len(landmarks_3d) < 33 or len(landmarks_2d) < 33:
            return GestureResult("UNKNOWN", 0.0, {"reason": 0.0})

        angles = angles or compute_all_angles(landmarks_3d, visibility, vis_threshold=0.25)
        scores = self._score_all(landmarks_2d, landmarks_3d, visibility, angles)

        priority = [
            "BOTH_HANDS_UP",
            "LEFT_HAND_UP",
            "RIGHT_HAND_UP",
            "SITTING",
            "BENDING_FORWARD",
            "STANDING",
        ]
        best_label = "UNKNOWN"
        best_score = 0.0
        for label in priority:
            score = scores.get(label, 0.0)
            if score > best_score:
                best_label = label
                best_score = score

        if best_score < 0.42:
            return GestureResult("UNKNOWN", best_score, scores)
        return GestureResult(best_label, best_score, scores)

    def _score_all(
        self,
        landmarks_2d: Sequence[Point2D],
        landmarks_3d: Sequence[Point3D],
        visibility: Sequence[float],
        angles: Dict[str, float],
    ) -> Dict[str, float]:
        """Calculate confidence scores for every supported label."""
        pts2 = np.asarray(landmarks_2d, dtype=float)
        pts3 = np.asarray(landmarks_3d, dtype=float)

        left_arm_vis = min_visibility(visibility, [LEFT_SHOULDER, LEFT_ELBOW, LEFT_WRIST])
        right_arm_vis = min_visibility(visibility, [RIGHT_SHOULDER, RIGHT_ELBOW, RIGHT_WRIST])
        leg_vis = mean_visibility(visibility, [LEFT_HIP, RIGHT_HIP, LEFT_KNEE, RIGHT_KNEE, LEFT_ANKLE, RIGHT_ANKLE])
        torso_vis = mean_visibility(visibility, [NOSE, LEFT_SHOULDER, RIGHT_SHOULDER, LEFT_HIP, RIGHT_HIP])

        left_hand_up = self._hand_up_score(pts2, LEFT_SHOULDER, LEFT_WRIST) * left_arm_vis
        right_hand_up = self._hand_up_score(pts2, RIGHT_SHOULDER, RIGHT_WRIST) * right_arm_vis
        both_hands_up = min(left_hand_up, right_hand_up)

        left_knee = angles.get("LEFT_KNEE", 0.0)
        right_knee = angles.get("RIGHT_KNEE", 0.0)
        left_hip_angle = angles.get("LEFT_HIP", 0.0)
        right_hip_angle = angles.get("RIGHT_HIP", 0.0)

        knee_extension = self._range_score((left_knee + right_knee) / 2.0, 150.0, 180.0)
        hip_extension = self._range_score((left_hip_angle + right_hip_angle) / 2.0, 145.0, 180.0)
        standing = min(knee_extension, hip_extension) * leg_vis

        knee_bent = 1.0 - self._range_score((left_knee + right_knee) / 2.0, 145.0, 180.0)
        hip_bent = 1.0 - self._range_score((left_hip_angle + right_hip_angle) / 2.0, 145.0, 180.0)
        sitting = max(knee_bent * 0.75 + hip_bent * 0.25, 0.0) * leg_vis

        torso_angle = self._torso_angle_from_vertical(pts3)
        head_drop = self._head_drop_score(pts2)
        bending = max(
            self._range_score(torso_angle, 25.0, 75.0),
            head_drop,
        ) * torso_vis

        return {
            "STANDING": clamp01(standing),
            "SITTING": clamp01(sitting),
            "LEFT_HAND_UP": clamp01(left_hand_up),
            "RIGHT_HAND_UP": clamp01(right_hand_up),
            "BOTH_HANDS_UP": clamp01(both_hands_up),
            "BENDING_FORWARD": clamp01(bending),
            "torso_angle": float(torso_angle),
            "avg_knee_angle": float((left_knee + right_knee) / 2.0),
            "avg_hip_angle": float((left_hip_angle + right_hip_angle) / 2.0),
        }

    @staticmethod
    def _range_score(value: float, low: float, high: float) -> float:
        """Map value to [0,1] between low and high."""
        if high <= low:
            return 0.0
        return clamp01((value - low) / (high - low))

    @staticmethod
    def _hand_up_score(pts2: np.ndarray, shoulder_idx: int, wrist_idx: int) -> float:
        """Score whether a wrist is above its shoulder in image coordinates."""
        shoulder_y = pts2[shoulder_idx][1]
        wrist_y = pts2[wrist_idx][1]
        margin_px = max(24.0, abs(pts2[LEFT_SHOULDER][1] - pts2[LEFT_HIP][1]) * 0.12)
        return clamp01((shoulder_y - wrist_y) / margin_px)

    @staticmethod
    def _torso_angle_from_vertical(pts3: np.ndarray) -> float:
        """Estimate torso lean angle against the vertical axis."""
        hip_mid = (pts3[LEFT_HIP] + pts3[RIGHT_HIP]) / 2.0
        shoulder_mid = (pts3[LEFT_SHOULDER] + pts3[RIGHT_SHOULDER]) / 2.0
        torso = shoulder_mid - hip_mid
        norm = np.linalg.norm(torso)
        if norm < 1e-8:
            return 0.0
        vertical = np.array([0.0, -1.0, 0.0])
        cos_value = np.dot(torso / norm, vertical)
        return float(np.degrees(np.arccos(np.clip(abs(cos_value), -1.0, 1.0))))

    @staticmethod
    def _head_drop_score(pts2: np.ndarray) -> float:
        """Score a front-view bend where the nose drops toward the shoulders."""
        shoulder_y = (pts2[LEFT_SHOULDER][1] + pts2[RIGHT_SHOULDER][1]) / 2.0
        hip_y = (pts2[LEFT_HIP][1] + pts2[RIGHT_HIP][1]) / 2.0
        torso_px = max(40.0, abs(hip_y - shoulder_y))
        nose_y = pts2[NOSE][1]
        return clamp01((nose_y - shoulder_y) / (torso_px * 0.35))


def classify_gesture(
    detection: Optional[Dict],
    angles: Optional[Dict[str, float]] = None,
) -> GestureResult:
    """Convenience function for one-off classification."""
    return GestureClassifier().classify(detection, angles)
