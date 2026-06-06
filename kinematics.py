"""
kinematics.py - Joint angle calculation and skeleton segment definitions.

Provides functions to:
- Calculate angle between three 3D points
- Compute all major joint angles from landmarks
- Define skeleton bone connections for visualization
"""

import numpy as np
from typing import Dict, List, Optional, Tuple

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

# Skeleton chains used by FK/IK helpers.
FK_CHAINS: Dict[str, List[int]] = {
    "left_arm": [LEFT_SHOULDER, LEFT_ELBOW, LEFT_WRIST],
    "right_arm": [RIGHT_SHOULDER, RIGHT_ELBOW, RIGHT_WRIST],
    "left_leg": [LEFT_HIP, LEFT_KNEE, LEFT_ANKLE],
    "right_leg": [RIGHT_HIP, RIGHT_KNEE, RIGHT_ANKLE],
    "spine": [LEFT_HIP, LEFT_SHOULDER, NOSE],
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


def compute_bone_lengths(
    landmarks_3d: List[Tuple[float, float, float]],
) -> Dict[str, float]:
    """Compute 3D bone lengths for every visible skeleton segment."""
    pts = np.array(landmarks_3d, dtype=float)
    lengths: Dict[str, float] = {}
    for group_name, segments in build_skeleton_segments().items():
        for idx1, idx2 in segments:
            key = f"{group_name}_{idx1}_{idx2}"
            lengths[key] = float(np.linalg.norm(pts[idx2] - pts[idx1]))
    return lengths


def compute_segment_directions(
    landmarks_3d: List[Tuple[float, float, float]],
) -> Dict[str, Tuple[float, float, float]]:
    """
    Compute normalized direction vectors for every skeleton segment.

    These vectors are a compact forward-kinematics representation: each child
    joint can be reconstructed from its parent, the bone length, and direction.
    """
    pts = np.array(landmarks_3d, dtype=float)
    directions: Dict[str, Tuple[float, float, float]] = {}
    for group_name, segments in build_skeleton_segments().items():
        for idx1, idx2 in segments:
            vec = pts[idx2] - pts[idx1]
            length = np.linalg.norm(vec)
            if length < 1e-8:
                unit = np.zeros(3)
            else:
                unit = vec / length
            directions[f"{group_name}_{idx1}_{idx2}"] = tuple(map(float, unit))
    return directions


def compute_fk_summary(
    landmarks_3d: List[Tuple[float, float, float]],
) -> Dict[str, Dict]:
    """
    Build a readable FK summary for report/demo purposes.

    Each chain stores the root joint, child joints, bone lengths, and segment
    directions in MediaPipe world coordinates.
    """
    pts = np.array(landmarks_3d, dtype=float)
    summary: Dict[str, Dict] = {}

    for chain_name, chain in FK_CHAINS.items():
        bones = []
        for parent, child in zip(chain, chain[1:]):
            vec = pts[child] - pts[parent]
            length = float(np.linalg.norm(vec))
            direction = np.zeros(3) if length < 1e-8 else vec / length
            bones.append({
                "parent": parent,
                "child": child,
                "length": length,
                "direction": tuple(map(float, direction)),
            })
        summary[chain_name] = {
            "root": chain[0],
            "joints": chain,
            "bones": bones,
        }

    return summary


def solve_two_bone_ik(
    root: Tuple[float, float, float],
    mid: Tuple[float, float, float],
    end: Tuple[float, float, float],
    target: Tuple[float, float, float],
    pole: Optional[Tuple[float, float, float]] = None,
) -> Tuple[Tuple[float, float, float], Tuple[float, float, float], Tuple[float, float, float]]:
    """
    Solve a simple two-bone IK chain while preserving original bone lengths.

    The returned tuple is (new_root, new_mid, new_end). It is useful for a small
    IK demonstration on arm or leg chains without changing the realtime detector.
    """
    root_v = np.array(root, dtype=float)
    mid_v = np.array(mid, dtype=float)
    end_v = np.array(end, dtype=float)
    target_v = np.array(target, dtype=float)

    len_a = np.linalg.norm(mid_v - root_v)
    len_b = np.linalg.norm(end_v - mid_v)
    if len_a < 1e-8 or len_b < 1e-8:
        return tuple(root_v), tuple(mid_v), tuple(end_v)

    to_target = target_v - root_v
    distance = np.linalg.norm(to_target)
    if distance < 1e-8:
        direction = np.array([1.0, 0.0, 0.0])
    else:
        direction = to_target / distance

    max_reach = len_a + len_b - 1e-8
    min_reach = abs(len_a - len_b) + 1e-8
    clamped_distance = float(np.clip(distance, min_reach, max_reach))

    if pole is None:
        pole_v = mid_v - root_v
        pole_v = pole_v - np.dot(pole_v, direction) * direction
        if np.linalg.norm(pole_v) < 1e-8:
            pole_v = np.array([0.0, 1.0, 0.0])
    else:
        pole_v = np.array(pole, dtype=float) - root_v
        pole_v = pole_v - np.dot(pole_v, direction) * direction

    pole_len = np.linalg.norm(pole_v)
    if pole_len < 1e-8:
        pole_v = np.array([0.0, 1.0, 0.0])
        pole_len = 1.0
    pole_dir = pole_v / pole_len

    along = (len_a * len_a - len_b * len_b + clamped_distance * clamped_distance) / (2.0 * clamped_distance)
    height_sq = max(len_a * len_a - along * along, 0.0)
    height = np.sqrt(height_sq)

    new_mid = root_v + direction * along + pole_dir * height
    new_end = root_v + direction * clamped_distance
    return tuple(map(float, root_v)), tuple(map(float, new_mid)), tuple(map(float, new_end))


def demo_ik_targets(
    landmarks_3d: List[Tuple[float, float, float]],
    offset: Tuple[float, float, float] = (0.12, 0.0, 0.0),
) -> Dict[str, List[Tuple[float, float, float]]]:
    """
    Produce small IK demo chains by moving wrists/ankles by offset.

    This keeps the realtime pipeline honest: detection remains MediaPipe-based,
    while IK is shown as an additional kinematics feature.
    """
    pts = np.array(landmarks_3d, dtype=float)
    offset_v = np.array(offset, dtype=float)
    chains = {
        "left_arm": (LEFT_SHOULDER, LEFT_ELBOW, LEFT_WRIST),
        "right_arm": (RIGHT_SHOULDER, RIGHT_ELBOW, RIGHT_WRIST),
        "left_leg": (LEFT_HIP, LEFT_KNEE, LEFT_ANKLE),
        "right_leg": (RIGHT_HIP, RIGHT_KNEE, RIGHT_ANKLE),
    }
    solved: Dict[str, List[Tuple[float, float, float]]] = {}
    for name, (root_idx, mid_idx, end_idx) in chains.items():
        target = pts[end_idx] + offset_v
        solved[name] = list(solve_two_bone_ik(
            pts[root_idx],
            pts[mid_idx],
            pts[end_idx],
            target,
            pole=pts[mid_idx],
        ))
    return solved


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
