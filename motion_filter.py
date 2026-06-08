"""
motion_filter.py - Temporal filters for pose landmarks.

MediaPipe landmarks can jitter between frames, especially with webcam input.
This module keeps the filter independent from detection so the pipeline can
switch smoothing on/off without changing the detector.
"""

from typing import Dict, List, Optional, Tuple

import numpy as np


Point3D = Tuple[float, float, float]


class LandmarkSmoother:
    """
    Exponential moving average smoother for 3D pose landmarks.

    alpha controls how much of the newest frame is used:
    - 1.0 disables smoothing.
    - lower values produce smoother but more delayed motion.
    """

    def __init__(self, alpha: float = 0.55, reset_after_misses: int = 10):
        if not 0.0 < alpha <= 1.0:
            raise ValueError("alpha must be in the range (0, 1].")
        self.alpha = alpha
        self.reset_after_misses = reset_after_misses
        self._prev: Optional[np.ndarray] = None
        self._misses = 0

    def reset(self) -> None:
        """Clear the internal filter state."""
        self._prev = None
        self._misses = 0

    def apply(self, detection: Optional[Dict]) -> Optional[Dict]:
        """
        Return a copy of detection with smoothed landmarks_3d.

        A run of missing detections resets the filter so the next detected pose
        does not interpolate from stale body coordinates.
        """
        if detection is None:
            self._misses += 1
            if self._misses >= self.reset_after_misses:
                self.reset()
            return None

        pts = np.asarray(detection["landmarks_3d"], dtype=float)
        if self._prev is None or self._prev.shape != pts.shape:
            smoothed = pts
        else:
            smoothed = self.alpha * pts + (1.0 - self.alpha) * self._prev

        self._prev = smoothed.copy()
        self._misses = 0

        result = dict(detection)
        result["landmarks_3d"] = [tuple(map(float, row)) for row in smoothed]
        return result


def hip_centered_landmarks(
    landmarks_3d: List[Point3D],
    left_hip: int = 23,
    right_hip: int = 24,
) -> List[Point3D]:
    """Return landmarks translated so the hip midpoint is the origin."""
    pts = np.asarray(landmarks_3d, dtype=float)
    center = (pts[left_hip] + pts[right_hip]) / 2.0
    shifted = pts - center
    return [tuple(map(float, row)) for row in shifted]
