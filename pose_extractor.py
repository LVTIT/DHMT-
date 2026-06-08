"""
pose_extractor.py - MediaPipe Pose wrapper for 33 landmark extraction.

This module is the public pose-extraction layer used by both the FastAPI
runtime and a small webcam smoke test. It converts MediaPipe output into
plain Python dataclasses so the rest of the project can stay framework-light.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Dict, List, Optional, Tuple

import cv2
import numpy as np

from detector import PoseDetector
from kinematics import compute_all_angles
from utils import JOINT_NAMES, Point3D, now_ms
from visualizer_2d import Visualizer2D


@dataclass(frozen=True)
class Joint:
    """One MediaPipe Pose landmark with pixel and world coordinates."""

    index: int
    name: str
    x: float
    y: float
    z: float
    pixel_x: int
    pixel_y: int
    visibility: float

    def to_dict(self) -> Dict:
        """Return a JSON-friendly representation."""
        return asdict(self)


@dataclass(frozen=True)
class PoseFrame:
    """Pose extraction result for one video frame."""

    detected: bool
    timestamp_ms: float
    frame_width: int
    frame_height: int
    joints: List[Joint]

    @property
    def landmarks_2d(self) -> List[Tuple[int, int]]:
        return [(joint.pixel_x, joint.pixel_y) for joint in self.joints]

    @property
    def landmarks_3d(self) -> List[Point3D]:
        return [(joint.x, joint.y, joint.z) for joint in self.joints]

    @property
    def visibility(self) -> List[float]:
        return [joint.visibility for joint in self.joints]

    def to_detection(self) -> Optional[Dict]:
        """Return the legacy dict shape used by existing project modules."""
        if not self.detected:
            return None
        return {
            "landmarks_2d": self.landmarks_2d,
            "landmarks_3d": self.landmarks_3d,
            "visibility": self.visibility,
            "joints": [joint.to_dict() for joint in self.joints],
            "timestamp_ms": self.timestamp_ms,
            "frame_width": self.frame_width,
            "frame_height": self.frame_height,
        }

    def to_dict(self) -> Dict:
        """Return a JSON-friendly frame payload."""
        return {
            "detected": self.detected,
            "timestamp_ms": self.timestamp_ms,
            "frame_width": self.frame_width,
            "frame_height": self.frame_height,
            "joints": [joint.to_dict() for joint in self.joints],
        }


class PoseExtractor:
    """
    Extract 33 2D/3D human-pose landmarks from OpenCV BGR frames.

    Parameters mirror the existing PoseDetector wrapper so older modules can
    keep working while new code uses the clearer PoseFrame/Joint model.
    """

    def __init__(
        self,
        model_path: Optional[str] = None,
        running_mode: str = "VIDEO",
        min_detection_confidence: float = 0.5,
        min_tracking_confidence: float = 0.5,
    ):
        kwargs = {
            "running_mode": running_mode,
            "min_detection_confidence": min_detection_confidence,
            "min_tracking_confidence": min_tracking_confidence,
        }
        if model_path:
            kwargs["model_path"] = model_path
        self._detector = PoseDetector(**kwargs)

    def extract(self, frame: np.ndarray) -> PoseFrame:
        """
        Extract pose landmarks from one BGR frame.

        If no person is detected, the returned PoseFrame has detected=False and
        an empty joint list. Callers can still display frame size and timestamp.
        """
        height, width = frame.shape[:2]
        detection = self._detector.detect(frame)
        timestamp = now_ms()
        if detection is None:
            return PoseFrame(
                detected=False,
                timestamp_ms=timestamp,
                frame_width=width,
                frame_height=height,
                joints=[],
            )

        joints = []
        for idx, ((px, py), (x, y, z), visibility) in enumerate(zip(
            detection["landmarks_2d"],
            detection["landmarks_3d"],
            detection["visibility"],
        )):
            joints.append(Joint(
                index=idx,
                name=JOINT_NAMES.get(idx, f"UNKNOWN_{idx}"),
                x=float(x),
                y=float(y),
                z=float(z),
                pixel_x=int(px),
                pixel_y=int(py),
                visibility=float(visibility),
            ))

        return PoseFrame(
            detected=True,
            timestamp_ms=timestamp,
            frame_width=width,
            frame_height=height,
            joints=joints,
        )

    def close(self) -> None:
        """Release MediaPipe resources."""
        self._detector.close()

    def __enter__(self) -> "PoseExtractor":
        return self

    def __exit__(self, exc_type, exc_val, exc_tb) -> None:
        self.close()


def run_webcam_test(camera_index: int = 0) -> None:
    """
    Open a webcam and display a 2D MediaPipe overlay.

    This is intentionally small: it gives the team a quick way to validate
    MediaPipe and camera access before running the full FastAPI app.
    """
    cap = cv2.VideoCapture(camera_index)
    if not cap.isOpened():
        raise RuntimeError(f"Cannot open camera index {camera_index}")

    visualizer = Visualizer2D()
    extractor = PoseExtractor()
    try:
        while True:
            ok, frame = cap.read()
            if not ok:
                break
            pose_frame = extractor.extract(frame)
            detection = pose_frame.to_detection()
            angles = compute_all_angles(
                detection["landmarks_3d"],
                detection["visibility"],
                vis_threshold=0.3,
            ) if detection else {}
            visualizer.draw(frame, detection, angles)
            cv2.imshow("Pose Extractor Test", frame)
            if cv2.waitKey(1) & 0xFF == ord("q"):
                break
    finally:
        extractor.close()
        cap.release()
        cv2.destroyAllWindows()


if __name__ == "__main__":
    run_webcam_test()
