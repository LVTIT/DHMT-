"""
visualizer_2d.py - 2D skeleton overlay on OpenCV frames.

Draws skeleton connections with color-coded body parts,
overlays joint angles as text, and shows FPS counter.
"""

import cv2
import time
import numpy as np
from typing import Dict, List, Tuple, Optional

from kinematics import (
    build_skeleton_segments,
    ANGLE_JOINT_INDEX,
)


# ======================================================================
# Color scheme (BGR) for each body part group
# ======================================================================
SEGMENT_COLORS: Dict[str, Tuple[int, int, int]] = {
    "torso":     (255, 255, 255),   # White
    "left_arm":  (255, 200, 50),    # Cyan-blue
    "right_arm": (50, 200, 255),    # Orange
    "left_leg":  (100, 255, 100),   # Green
    "right_leg": (100, 100, 255),   # Red
    "face":      (200, 200, 200),   # Light gray
}

JOINT_COLOR = (0, 255, 255)          # Yellow dots for joints
ANGLE_TEXT_COLOR = (0, 255, 128)     # Green text for angles
FPS_TEXT_COLOR = (0, 255, 0)         # Green FPS counter
LOW_VIS_COLOR = (128, 128, 128)      # Gray for low-visibility segments


class Visualizer2D:
    """
    Draws 2D skeleton overlay on OpenCV frames.

    Features:
    - Color-coded skeleton connections (arms/legs/torso)
    - Joint angle text overlay at each computed joint
    - FPS counter (top-left)
    - Faded drawing for low-visibility landmarks
    """

    def __init__(self):
        self._segments = build_skeleton_segments()
        self._prev_time = time.time()
        self._fps = 0.0
        self._fps_smooth = 0.0  # Exponential moving average

    def draw(
        self,
        frame: np.ndarray,
        detection: Optional[Dict],
        angles: Optional[Dict[str, float]] = None,
        vis_threshold: float = 0.5,
    ) -> np.ndarray:
        """
        Draw skeleton overlay on frame (in-place modification).

        Parameters
        ----------
        frame : np.ndarray
            BGR frame from OpenCV.
        detection : dict | None
            Detection result from PoseDetector.detect().
            Contains landmarks_2d, landmarks_3d, visibility.
        angles : dict | None
            {joint_name: angle_degrees} from compute_all_angles().
        vis_threshold : float
            Minimum visibility to draw a landmark at full opacity.

        Returns
        -------
        np.ndarray
            The modified frame (same reference as input).
        """
        # Update FPS
        self._update_fps()

        # Draw FPS counter
        self._draw_fps(frame)

        if detection is None:
            # Draw "No person detected" message
            cv2.putText(
                frame, "No person detected",
                (10, 60), cv2.FONT_HERSHEY_SIMPLEX,
                0.7, (0, 0, 255), 2,
            )
            return frame

        landmarks_2d = detection["landmarks_2d"]
        visibility = detection["visibility"]

        # Draw skeleton segments
        self._draw_segments(frame, landmarks_2d, visibility, vis_threshold)

        # Draw joint dots
        self._draw_joints(frame, landmarks_2d, visibility, vis_threshold)

        # Draw angle text overlay
        if angles:
            self._draw_angles(frame, landmarks_2d, angles, visibility, vis_threshold)

        return frame

    # ------------------------------------------------------------------
    # Private methods
    # ------------------------------------------------------------------
    def _update_fps(self) -> None:
        """Calculate FPS with exponential moving average."""
        now = time.time()
        dt = now - self._prev_time
        self._prev_time = now

        if dt > 0:
            instant_fps = 1.0 / dt
            alpha = 0.1  # Smoothing factor
            self._fps_smooth = alpha * instant_fps + (1 - alpha) * self._fps_smooth
            self._fps = self._fps_smooth

    def _draw_fps(self, frame: np.ndarray) -> None:
        """Draw FPS counter on top-left corner."""
        text = f"FPS: {self._fps:.1f}"
        cv2.putText(
            frame, text,
            (10, 30), cv2.FONT_HERSHEY_SIMPLEX,
            0.8, FPS_TEXT_COLOR, 2, cv2.LINE_AA,
        )

    def _draw_segments(
        self,
        frame: np.ndarray,
        landmarks_2d: List[Tuple[int, int]],
        visibility: List[float],
        vis_threshold: float,
    ) -> None:
        """Draw skeleton connection lines with body-part colors."""
        for group_name, connections in self._segments.items():
            color = SEGMENT_COLORS.get(group_name, (255, 255, 255))

            for idx1, idx2 in connections:
                # Skip if either landmark is out of range
                if idx1 >= len(landmarks_2d) or idx2 >= len(landmarks_2d):
                    continue

                vis1 = visibility[idx1]
                vis2 = visibility[idx2]

                # Use gray color for low-visibility connections
                draw_color = color if (vis1 >= vis_threshold and vis2 >= vis_threshold) else LOW_VIS_COLOR
                thickness = 3 if (vis1 >= vis_threshold and vis2 >= vis_threshold) else 1

                pt1 = landmarks_2d[idx1]
                pt2 = landmarks_2d[idx2]
                cv2.line(frame, pt1, pt2, draw_color, thickness, cv2.LINE_AA)

    def _draw_joints(
        self,
        frame: np.ndarray,
        landmarks_2d: List[Tuple[int, int]],
        visibility: List[float],
        vis_threshold: float,
    ) -> None:
        """Draw circles at joint positions."""
        # Only draw key joints (not all 33)
        key_joints = [0, 11, 12, 13, 14, 15, 16, 23, 24, 25, 26, 27, 28]
        for idx in key_joints:
            if idx >= len(landmarks_2d):
                continue

            vis = visibility[idx]
            if vis >= vis_threshold:
                cv2.circle(frame, landmarks_2d[idx], 5, JOINT_COLOR, -1, cv2.LINE_AA)
            else:
                cv2.circle(frame, landmarks_2d[idx], 3, LOW_VIS_COLOR, -1, cv2.LINE_AA)

    def _draw_angles(
        self,
        frame: np.ndarray,
        landmarks_2d: List[Tuple[int, int]],
        angles: Dict[str, float],
        visibility: List[float],
        vis_threshold: float,
    ) -> None:
        """Draw angle values as text near each joint."""
        for angle_name, angle_deg in angles.items():
            joint_idx = ANGLE_JOINT_INDEX.get(angle_name)
            if joint_idx is None or joint_idx >= len(landmarks_2d):
                continue

            # Only draw if joint is visible
            if visibility[joint_idx] < vis_threshold:
                continue

            px, py = landmarks_2d[joint_idx]
            text = f"{angle_deg:.0f} deg"

            # Offset text slightly so it doesn't overlap the joint dot
            offset_x, offset_y = 10, -10
            cv2.putText(
                frame, text,
                (px + offset_x, py + offset_y),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.45, ANGLE_TEXT_COLOR, 1, cv2.LINE_AA,
            )
