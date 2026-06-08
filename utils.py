"""
utils.py - Shared math and serialization helpers for motion capture.

The project keeps small, reusable functions here so pose extraction,
gesture classification, kinematics, and streaming all speak the same data
shape without duplicating vector math.
"""

from __future__ import annotations

import base64
import time
from typing import Dict, Iterable, List, Optional, Sequence, Tuple, Union

import cv2
import numpy as np


Point2D = Tuple[float, float]
Point3D = Tuple[float, float, float]


JOINT_NAMES: Dict[int, str] = {
    0: "NOSE",
    1: "LEFT_EYE_INNER",
    2: "LEFT_EYE",
    3: "LEFT_EYE_OUTER",
    4: "RIGHT_EYE_INNER",
    5: "RIGHT_EYE",
    6: "RIGHT_EYE_OUTER",
    7: "LEFT_EAR",
    8: "RIGHT_EAR",
    9: "MOUTH_LEFT",
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

JOINT_INDEX: Dict[str, int] = {name: idx for idx, name in JOINT_NAMES.items()}


def now_ms() -> float:
    """Return a monotonic timestamp in milliseconds."""
    return time.perf_counter() * 1000.0


def parse_source(value: Union[str, int, None]) -> Union[int, str]:
    """Parse a camera index or file path from user/API input."""
    if value is None:
        return 0
    if isinstance(value, int):
        return value
    text = str(value).strip()
    if text.isdigit():
        return int(text)
    return text


def clamp01(value: float) -> float:
    """Clamp a numeric value to the range [0, 1]."""
    return float(max(0.0, min(1.0, value)))


def angle_degrees(a: Sequence[float], b: Sequence[float], c: Sequence[float]) -> float:
    """Calculate the angle ABC in degrees."""
    a_v = np.asarray(a, dtype=float)
    b_v = np.asarray(b, dtype=float)
    c_v = np.asarray(c, dtype=float)
    ba = a_v - b_v
    bc = c_v - b_v
    norm_ba = np.linalg.norm(ba)
    norm_bc = np.linalg.norm(bc)
    if norm_ba < 1e-8 or norm_bc < 1e-8:
        return 0.0
    cos_value = np.dot(ba, bc) / (norm_ba * norm_bc)
    return float(np.degrees(np.arccos(np.clip(cos_value, -1.0, 1.0))))


def mean_visibility(visibility: Optional[Sequence[float]], indices: Iterable[int]) -> float:
    """Return the average visibility for selected landmark indices."""
    if not visibility:
        return 0.0
    values = [float(visibility[idx]) for idx in indices if idx < len(visibility)]
    if not values:
        return 0.0
    return clamp01(float(np.mean(values)))


def min_visibility(visibility: Optional[Sequence[float]], indices: Iterable[int]) -> float:
    """Return the weakest visibility among selected landmark indices."""
    if not visibility:
        return 0.0
    values = [float(visibility[idx]) for idx in indices if idx < len(visibility)]
    if not values:
        return 0.0
    return clamp01(min(values))


def hip_center(landmarks_3d: Sequence[Point3D]) -> Point3D:
    """Compute the midpoint between left and right hips."""
    pts = np.asarray(landmarks_3d, dtype=float)
    if len(pts) <= JOINT_INDEX["RIGHT_HIP"]:
        return (0.0, 0.0, 0.0)
    center = (pts[JOINT_INDEX["LEFT_HIP"]] + pts[JOINT_INDEX["RIGHT_HIP"]]) / 2.0
    return tuple(map(float, center))


def normalize_landmarks(
    landmarks_3d: Sequence[Point3D],
    scale_to_shoulders: bool = True,
) -> List[Point3D]:
    """Translate landmarks to the hip center and optionally normalize scale."""
    pts = np.asarray(landmarks_3d, dtype=float)
    if len(pts) == 0:
        return []
    center = np.asarray(hip_center(landmarks_3d), dtype=float)
    shifted = pts - center
    if scale_to_shoulders and len(pts) > JOINT_INDEX["RIGHT_SHOULDER"]:
        shoulder_width = np.linalg.norm(
            pts[JOINT_INDEX["LEFT_SHOULDER"]] - pts[JOINT_INDEX["RIGHT_SHOULDER"]]
        )
        if shoulder_width > 1e-8:
            shifted = shifted / shoulder_width
    return [tuple(map(float, row)) for row in shifted]


def landmarks_delta(
    current: Optional[Sequence[Point3D]],
    previous: Optional[Sequence[Point3D]],
    visibility: Optional[Sequence[float]] = None,
    vis_threshold: float = 0.25,
) -> float:
    """
    Return average per-joint displacement between two landmark frames.

    Low-visibility joints are ignored so brief occlusions do not trigger
    unnecessary WebSocket traffic.
    """
    if current is None or previous is None:
        return float("inf")
    if len(current) != len(previous) or len(current) == 0:
        return float("inf")

    curr = np.asarray(current, dtype=float)
    prev = np.asarray(previous, dtype=float)
    mask = np.ones(len(curr), dtype=bool)
    if visibility:
        vis = np.asarray(visibility, dtype=float)
        mask = vis[: len(curr)] >= vis_threshold
    if not np.any(mask):
        return 0.0
    return float(np.mean(np.linalg.norm(curr[mask] - prev[mask], axis=1)))


def encode_jpeg_bytes(frame: np.ndarray, quality: int = 82) -> bytes:
    """Encode a BGR frame to JPEG bytes for MJPEG streaming."""
    quality = int(max(1, min(100, quality)))
    ok, buffer = cv2.imencode(".jpg", frame, [cv2.IMWRITE_JPEG_QUALITY, quality])
    if not ok:
        return b""
    return buffer.tobytes()


def encode_jpeg_base64(frame: np.ndarray, quality: int = 82) -> str:
    """Encode a BGR frame as a base64 data URL."""
    raw = encode_jpeg_bytes(frame, quality=quality)
    if not raw:
        return ""
    return "data:image/jpeg;base64," + base64.b64encode(raw).decode("ascii")


def landmarks_to_dicts(
    landmarks_3d: Sequence[Point3D],
    visibility: Optional[Sequence[float]] = None,
) -> List[Dict[str, float]]:
    """Serialize landmarks to JSON-friendly dictionaries."""
    payload = []
    for idx, (x, y, z) in enumerate(landmarks_3d):
        payload.append({
            "index": idx,
            "name": JOINT_NAMES.get(idx, f"UNKNOWN_{idx}"),
            "x": float(x),
            "y": float(y),
            "z": float(z),
            "visibility": float(visibility[idx]) if visibility and idx < len(visibility) else 0.0,
        })
    return payload
