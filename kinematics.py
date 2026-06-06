"""
kinematics.py - Joint angle calculation and skeleton segment definitions.

Provides functions to:
- Calculate angle between three 3D points
- Compute all major joint angles from landmarks
- Define skeleton bone connections for visualization
"""

import numpy as np
from typing import Dict, List, Tuple

# ======================================================================
# Joint index constants (from MediaPipe Pose)
# ======================================================================
# Upper body
NOSE = 0
LEFT_SHOULDER = 11
RIGHT_SHOULDER = 12
LEFT_ELBOW = 13
RIGHT_ELBOW = 14
LEFT_WRIST = 15
RIGHT_WRIST = 16

# Lower body
LEFT_HIP = 23
RIGHT_HIP = 24
LEFT_KNEE = 25
RIGHT_KNEE = 26
LEFT_ANKLE = 27
RIGHT_ANKLE = 28

# Extremities
LEFT_HEEL = 29
RIGHT_HEEL = 30
LEFT_FOOT_INDEX = 31
RIGHT_FOOT_INDEX = 32

# Face
LEFT_EAR = 7
RIGHT_EAR = 8

# ======================================================================
# Angle definitions: (point_a, vertex_b, point_c, name)
# Angle is measured at vertex_b between vectors b->a and b->c
# ======================================================================
ANGLE_DEFINITIONS: List[Tuple[int, int, int, str]] = [
    # Elbows: shoulder -> elbow -> wrist
    (LEFT_SHOULDER,  LEFT_ELBOW,  LEFT_WRIST,  "LEFT_ELBOW"),
    (RIGHT_SHOULDER, RIGHT_ELBOW, RIGHT_WRIST, "RIGHT_ELBOW"),

    # Shoulders: elbow -> shoulder -> hip
    (LEFT_ELBOW,  LEFT_SHOULDER,  LEFT_HIP,  "LEFT_SHOULDER"),
    (RIGHT_ELBOW, RIGHT_SHOULDER, RIGHT_HIP, "RIGHT_SHOULDER"),

    # Hips: shoulder -> hip -> knee
    (LEFT_SHOULDER,  LEFT_HIP,  LEFT_KNEE,  "LEFT_HIP"),
    (RIGHT_SHOULDER, RIGHT_HIP, RIGHT_KNEE, "RIGHT_HIP"),

    # Knees: hip -> knee -> ankle
    (LEFT_HIP,  LEFT_KNEE,  LEFT_ANKLE,  "LEFT_KNEE"),
    (RIGHT_HIP, RIGHT_KNEE, RIGHT_ANKLE, "RIGHT_KNEE"),
]

# Map angle name -> vertex joint index (for 2D overlay positioning)
ANGLE_JOINT_INDEX: Dict[str, int] = {
    defn[3]: defn[1] for defn in ANGLE_DEFINITIONS
}


def angle_between(
    a: np.ndarray,
    b: np.ndarray,
    c: np.ndarray,
) -> float:
    """
    Calculate the angle at point b formed by vectors b->a and b->c.

    Parameters
    ----------
    a : np.ndarray
        First endpoint (shape: (2,) or (3,)).
    b : np.ndarray
        Vertex point where angle is measured.
    c : np.ndarray
        Second endpoint.

    Returns
    -------
    float
        Angle in degrees [0, 180].
    """
    ba = a - b
    bc = c - b

    # Handle zero-length vectors
    norm_ba = np.linalg.norm(ba)
    norm_bc = np.linalg.norm(bc)
    if norm_ba < 1e-8 or norm_bc < 1e-8:
        return 0.0

    cos_angle = np.dot(ba, bc) / (norm_ba * norm_bc)
    # Clamp to [-1, 1] to avoid numerical errors with arccos
    cos_angle = np.clip(cos_angle, -1.0, 1.0)

    return float(np.degrees(np.arccos(cos_angle)))


def compute_all_angles(
    landmarks_3d: List[Tuple[float, float, float]],
    visibility: List[float] = None,
    vis_threshold: float = 0.5,
) -> Dict[str, float]:
    """
    Compute angles for all major joints.

    Parameters
    ----------
    landmarks_3d : list of (x, y, z)
        33 landmark 3D coordinates.
    visibility : list of float, optional
        Visibility scores per landmark. If provided, joints with
        any participating landmark below vis_threshold are skipped.
    vis_threshold : float
        Minimum visibility to consider a landmark reliable.

    Returns
    -------
    dict
        {joint_name: angle_degrees} for all computable joints.
        Missing joints (due to low visibility) are excluded.
    """
    pts = np.array(landmarks_3d)
    angles: Dict[str, float] = {}

    for idx_a, idx_b, idx_c, name in ANGLE_DEFINITIONS:
        # Check visibility if provided
        if visibility is not None:
            if (visibility[idx_a] < vis_threshold or
                visibility[idx_b] < vis_threshold or
                visibility[idx_c] < vis_threshold):
                continue

        a = pts[idx_a]
        b = pts[idx_b]
        c = pts[idx_c]

        angles[name] = angle_between(a, b, c)

    return angles


def build_skeleton_segments() -> Dict[str, List[Tuple[int, int]]]:
    """
    Define skeleton bone connections grouped by body part.

    Returns
    -------
    dict
        {
            "torso":     [(idx1, idx2), ...],
            "left_arm":  [...],
            "right_arm": [...],
            "left_leg":  [...],
            "right_leg": [...],
            "face":      [...],
        }
    """
    return {
        "torso": [
            (LEFT_SHOULDER, RIGHT_SHOULDER),
            (LEFT_SHOULDER, LEFT_HIP),
            (RIGHT_SHOULDER, RIGHT_HIP),
            (LEFT_HIP, RIGHT_HIP),
        ],
        "left_arm": [
            (LEFT_SHOULDER, LEFT_ELBOW),
            (LEFT_ELBOW, LEFT_WRIST),
        ],
        "right_arm": [
            (RIGHT_SHOULDER, RIGHT_ELBOW),
            (RIGHT_ELBOW, RIGHT_WRIST),
        ],
        "left_leg": [
            (LEFT_HIP, LEFT_KNEE),
            (LEFT_KNEE, LEFT_ANKLE),
            (LEFT_ANKLE, LEFT_HEEL),
            (LEFT_ANKLE, LEFT_FOOT_INDEX),
        ],
        "right_leg": [
            (RIGHT_HIP, RIGHT_KNEE),
            (RIGHT_KNEE, RIGHT_ANKLE),
            (RIGHT_ANKLE, RIGHT_HEEL),
            (RIGHT_ANKLE, RIGHT_FOOT_INDEX),
        ],
        "face": [
            (NOSE, LEFT_SHOULDER),
            (NOSE, RIGHT_SHOULDER),
        ],
    }


def get_all_segments_flat() -> List[Tuple[int, int]]:
    """Return all skeleton segments as a flat list of (idx1, idx2) pairs."""
    segments = build_skeleton_segments()
    flat = []
    for group in segments.values():
        flat.extend(group)
    return flat
