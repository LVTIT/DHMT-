"""
detector.py - MediaPipe Pose Landmarker wrapper.

Detect 33 skeleton landmarks using the new MediaPipe Tasks API (>=0.10.14).
Returns 2D pixel coordinates, 3D world coordinates, and visibility per landmark.
"""

import os
import mediapipe as mp
import numpy as np
import cv2
from typing import Optional, Dict, List, Tuple

# ======================================================================
# JOINT NAME MAPPING  -  MediaPipe Pose 33 landmarks
# https://developers.google.com/mediapipe/solutions/vision/pose_landmarker
# ======================================================================
JOINT_NAMES: Dict[int, str] = {
    0:  "NOSE",
    1:  "LEFT_EYE_INNER",
    2:  "LEFT_EYE",
    3:  "LEFT_EYE_OUTER",
    4:  "RIGHT_EYE_INNER",
    5:  "RIGHT_EYE",
    6:  "RIGHT_EYE_OUTER",
    7:  "LEFT_EAR",
    8:  "RIGHT_EAR",
    9:  "MOUTH_LEFT",
    10: "MOUTH_RIGHT",
    11: "LEFT_SHOULDER",
    12: "RIGHT_SHOULDER",
    13: "LEFT_ELBOW",
    14: "RIGHT_ELBOW",
    15: "LEFT_WRIST",
    16: "RIGHT_WRIST",
    17: "LEFT_PINKY",
    18: "RIGHT_PINKY",
    19: "LEFT_INDEX",
    20: "RIGHT_INDEX",
    21: "LEFT_THUMB",
    22: "RIGHT_THUMB",
    23: "LEFT_HIP",
    24: "RIGHT_HIP",
    25: "LEFT_KNEE",
    26: "RIGHT_KNEE",
    27: "LEFT_ANKLE",
    28: "RIGHT_ANKLE",
    29: "LEFT_HEEL",
    30: "RIGHT_HEEL",
    31: "LEFT_FOOT_INDEX",
    32: "RIGHT_FOOT_INDEX",
}

# Reverse lookup: name -> index
JOINT_INDEX: Dict[str, int] = {v: k for k, v in JOINT_NAMES.items()}

# Default model path (relative to this file)
_DEFAULT_MODEL_PATH = os.path.join(
    os.path.dirname(os.path.abspath(__file__)), "pose_landmarker.task"
)


class PoseDetector:
    """
    MediaPipe Pose Landmarker wrapper using Tasks API.

    Parameters
    ----------
    model_path : str
        Path to the .task model file.
    num_poses : int
        Maximum number of poses to detect (default: 1).
    min_detection_confidence : float
        Minimum confidence for pose detection (0-1).
    min_tracking_confidence : float
        Minimum confidence for tracking between frames (0-1).
    running_mode : str
        "VIDEO" for frame-by-frame processing with timestamp tracking.
        "IMAGE" for single-image processing (no tracking).
    """

    def __init__(
        self,
        model_path: str = _DEFAULT_MODEL_PATH,
        num_poses: int = 1,
        min_detection_confidence: float = 0.5,
        min_tracking_confidence: float = 0.5,
        running_mode: str = "VIDEO",
    ):
        BaseOptions = mp.tasks.BaseOptions
        PoseLandmarker = mp.tasks.vision.PoseLandmarker
        PoseLandmarkerOptions = mp.tasks.vision.PoseLandmarkerOptions
        RunningMode = mp.tasks.vision.RunningMode

        mode = RunningMode.VIDEO if running_mode == "VIDEO" else RunningMode.IMAGE

        options = PoseLandmarkerOptions(
            base_options=BaseOptions(model_asset_path=model_path),
            running_mode=mode,
            num_poses=num_poses,
            min_pose_detection_confidence=min_detection_confidence,
            min_tracking_confidence=min_tracking_confidence,
        )

        self._landmarker = PoseLandmarker.create_from_options(options)
        self._running_mode = mode
        self._timestamp_ms = 0

    # ------------------------------------------------------------------
    # Core detection
    # ------------------------------------------------------------------
    def detect(self, frame: np.ndarray) -> Optional[Dict]:
        """
        Detect pose landmarks from a BGR frame.

        Parameters
        ----------
        frame : np.ndarray
            BGR frame from OpenCV.

        Returns
        -------
        dict | None
            None if no person detected.
            Otherwise:
            {
                "landmarks_2d": [(x_px, y_px), ...],   # 33 points, pixel coords
                "landmarks_3d": [(x, y, z), ...],       # 33 points, 3D world coords
                "visibility":   [float, ...],           # 33 values 0-1
            }
        """
        h, w = frame.shape[:2]

        # Convert BGR -> RGB and wrap in mp.Image
        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb)

        # Detect based on running mode
        RunningMode = mp.tasks.vision.RunningMode
        if self._running_mode == RunningMode.VIDEO:
            self._timestamp_ms += 33  # ~30fps increment
            results = self._landmarker.detect_for_video(mp_image, self._timestamp_ms)
        else:
            results = self._landmarker.detect(mp_image)

        # Check if any poses were found
        if not results.pose_landmarks or len(results.pose_landmarks) == 0:
            return None

        # Take the first detected pose
        pose_lm = results.pose_landmarks[0]

        landmarks_2d: List[Tuple[int, int]] = []
        landmarks_3d: List[Tuple[float, float, float]] = []
        visibility: List[float] = []

        # 2D: normalized (0-1) -> pixel coordinates
        for lm in pose_lm:
            px = int(lm.x * w)
            py = int(lm.y * h)
            landmarks_2d.append((px, py))
            visibility.append(lm.visibility if lm.visibility else 0.0)

        # 3D: world landmarks (metres, hip-centred)
        if results.pose_world_landmarks and len(results.pose_world_landmarks) > 0:
            world_lm = results.pose_world_landmarks[0]
            for lm in world_lm:
                landmarks_3d.append((lm.x, lm.y, lm.z))
        else:
            # Fallback: use normalized coords + z
            for lm in pose_lm:
                landmarks_3d.append((lm.x, lm.y, lm.z if lm.z else 0.0))

        return {
            "landmarks_2d": landmarks_2d,
            "landmarks_3d": landmarks_3d,
            "visibility": visibility,
        }

    # ------------------------------------------------------------------
    # Cleanup
    # ------------------------------------------------------------------
    def close(self) -> None:
        """Release MediaPipe resources."""
        self._landmarker.close()

    def __enter__(self) -> "PoseDetector":
        return self

    def __exit__(self, exc_type, exc_val, exc_tb) -> None:
        self.close()

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------
    @staticmethod
    def get_joint_name(index: int) -> str:
        """Get joint name by index."""
        return JOINT_NAMES.get(index, f"UNKNOWN_{index}")

    @staticmethod
    def get_joint_index(name: str) -> int:
        """Get index by joint name."""
        return JOINT_INDEX[name]

    @staticmethod
    def filter_by_visibility(
        detection: Dict,
        threshold: float = 0.5,
    ) -> Dict[int, bool]:
        """
        Return dict {index: True/False} indicating which landmarks
        are reliable (visibility >= threshold).
        """
        return {
            i: (v >= threshold)
            for i, v in enumerate(detection["visibility"])
        }
