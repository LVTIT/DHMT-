"""
bvh_exporter.py - Export motion capture data to BVH format.

Custom BVH writer that converts MediaPipe 3D landmarks to standard
BVH skeleton hierarchy with Euler rotation channels.
Compatible with Blender and other 3D software.
"""

import numpy as np
from typing import List, Tuple, Optional


# ======================================================================
# BVH Skeleton Hierarchy Definition
# Maps MediaPipe landmark indices to a BVH-compatible skeleton tree.
# ======================================================================

# MediaPipe indices used in BVH skeleton
_MP = {
    "Hips": (23, 24),         # Midpoint of left/right hip
    "Spine": (23, 24, 11, 12),  # Midpoint hips -> midpoint shoulders
    "Spine1": (11, 12),       # Midpoint shoulders
    "Neck": (0,),             # Nose area
    "Head": (0,),             # Nose
    "LeftShoulder": (11,),
    "LeftArm": (11,),
    "LeftForeArm": (13,),
    "LeftHand": (15,),
    "RightShoulder": (12,),
    "RightArm": (12,),
    "RightForeArm": (14,),
    "RightHand": (16,),
    "LeftUpLeg": (23,),
    "LeftLeg": (25,),
    "LeftFoot": (27,),
    "RightUpLeg": (24,),
    "RightLeg": (26,),
    "RightFoot": (28,),
}

# BVH hierarchy tree: (joint_name, [children])
BVH_HIERARCHY = (
    "Hips", [
        ("Spine", [
            ("Spine1", [
                ("Neck", [
                    ("Head", [])
                ]),
                ("LeftShoulder", [
                    ("LeftArm", [
                        ("LeftForeArm", [
                            ("LeftHand", [])
                        ])
                    ])
                ]),
                ("RightShoulder", [
                    ("RightArm", [
                        ("RightForeArm", [
                            ("RightHand", [])
                        ])
                    ])
                ]),
            ])
        ]),
        ("LeftUpLeg", [
            ("LeftLeg", [
                ("LeftFoot", [])
            ])
        ]),
        ("RightUpLeg", [
            ("RightLeg", [
                ("RightFoot", [])
            ])
        ]),
    ]
)


def _get_joint_position(name: str, landmarks_3d: np.ndarray) -> np.ndarray:
    """Get 3D position of a BVH joint from MediaPipe landmarks."""
    indices = _MP[name]
    if len(indices) == 1:
        return landmarks_3d[indices[0]]
    elif len(indices) == 2:
        return (landmarks_3d[indices[0]] + landmarks_3d[indices[1]]) / 2.0
    elif len(indices) == 4:
        # Spine: between hip midpoint and shoulder midpoint
        hip_mid = (landmarks_3d[indices[0]] + landmarks_3d[indices[1]]) / 2.0
        sho_mid = (landmarks_3d[indices[2]] + landmarks_3d[indices[3]]) / 2.0
        return (hip_mid + sho_mid) / 2.0
    return np.zeros(3)


def _rotation_matrix_from_vectors(v_from: np.ndarray, v_to: np.ndarray) -> np.ndarray:
    """Compute rotation matrix that rotates v_from to v_to."""
    v_from = v_from / (np.linalg.norm(v_from) + 1e-8)
    v_to = v_to / (np.linalg.norm(v_to) + 1e-8)
    cross = np.cross(v_from, v_to)
    dot = np.dot(v_from, v_to)
    if np.linalg.norm(cross) < 1e-8:
        if dot > 0:
            return np.eye(3)
        else:
            # 180 degree rotation
            perp = np.array([1, 0, 0]) if abs(v_from[0]) < 0.9 else np.array([0, 1, 0])
            perp = perp - np.dot(perp, v_from) * v_from
            perp = perp / (np.linalg.norm(perp) + 1e-8)
            return 2 * np.outer(perp, perp) - np.eye(3)

    skew = np.array([
        [0, -cross[2], cross[1]],
        [cross[2], 0, -cross[0]],
        [-cross[1], cross[0], 0]
    ])
    R = np.eye(3) + skew + skew @ skew * (1.0 / (1.0 + dot))
    return R


def _rotation_to_euler_zxy(R: np.ndarray) -> Tuple[float, float, float]:
    """
    Extract ZXY Euler angles from rotation matrix.
    Returns (z_deg, x_deg, y_deg).
    """
    # ZXY: R = Rz * Rx * Ry
    x = np.arcsin(np.clip(R[2, 1], -1.0, 1.0))
    if abs(np.cos(x)) > 1e-6:
        y = np.arctan2(-R[2, 0], R[2, 2])
        z = np.arctan2(-R[0, 1], R[1, 1])
    else:
        y = 0.0
        z = np.arctan2(R[1, 0], R[0, 0])
    return (np.degrees(z), np.degrees(x), np.degrees(y))


class BVHExporter:
    """
    Records motion capture frames and exports to BVH format.

    Usage:
        exporter = BVHExporter(fps=30.0)
        exporter.start_recording()
        for each frame:
            exporter.add_frame(landmarks_3d)
        exporter.save("output.bvh")
    """

    def __init__(self, fps: float = 30.0):
        """
        Parameters
        ----------
        fps : float
            Frame rate for the BVH animation.
        """
        self._fps = fps
        self._frames: List[np.ndarray] = []
        self._recording = False
        self._rest_pose: Optional[np.ndarray] = None

        # Build joint order from hierarchy
        self._joint_order: List[str] = []
        self._parent_map: dict = {}
        self._build_joint_order(BVH_HIERARCHY[0], BVH_HIERARCHY[1], parent=None)

    def _build_joint_order(self, name: str, children: list, parent: Optional[str]):
        """Recursively build joint traversal order."""
        self._joint_order.append(name)
        self._parent_map[name] = parent
        for child_name, child_children in children:
            self._build_joint_order(child_name, child_children, parent=name)

    @property
    def is_recording(self) -> bool:
        return self._recording

    @property
    def frame_count(self) -> int:
        return len(self._frames)

    def start_recording(self) -> None:
        """Start recording frames. Clears any previous data."""
        self._frames = []
        self._rest_pose = None
        self._recording = True

    def stop_recording(self) -> None:
        """Stop recording frames."""
        self._recording = False

    def add_frame(self, landmarks_3d: List[Tuple[float, float, float]]) -> None:
        """
        Add a frame of 3D landmarks to the recording.

        Parameters
        ----------
        landmarks_3d : list of (x, y, z)
            33 MediaPipe 3D world landmarks.
        """
        if not self._recording:
            return

        pts = np.array(landmarks_3d)
        # Store first frame as rest pose
        if self._rest_pose is None:
            self._rest_pose = pts.copy()
        self._frames.append(pts)

    def save(self, filepath: str) -> None:
        """
        Save recorded frames as a BVH file.

        Parameters
        ----------
        filepath : str
            Output .bvh file path.
        """
        if len(self._frames) == 0:
            print("[BVH] No frames recorded, nothing to save.")
            return

        lines: List[str] = []

        # --- HIERARCHY section ---
        lines.append("HIERARCHY")
        self._write_hierarchy(lines, BVH_HIERARCHY[0], BVH_HIERARCHY[1],
                              self._frames[0], indent=0, is_root=True)

        # --- MOTION section ---
        lines.append("MOTION")
        lines.append(f"Frames: {len(self._frames)}")
        lines.append(f"Frame Time: {1.0 / self._fps:.6f}")

        for frame_pts in self._frames:
            frame_data = self._compute_frame_data(frame_pts)
            lines.append(" ".join(f"{v:.4f}" for v in frame_data))

        with open(filepath, "w") as f:
            f.write("\n".join(lines) + "\n")

        print(f"[BVH] Saved {len(self._frames)} frames to {filepath}")

    # ------------------------------------------------------------------
    # Private: hierarchy writing
    # ------------------------------------------------------------------
    def _write_hierarchy(self, lines, name, children, rest_pts,
                         indent, is_root=False):
        """Recursively write BVH HIERARCHY section."""
        prefix = "  " * indent
        if is_root:
            lines.append(f"{prefix}ROOT {name}")
        else:
            lines.append(f"{prefix}JOINT {name}")

        lines.append(f"{prefix}{{")

        # Compute offset from parent
        pos = _get_joint_position(name, rest_pts)
        parent_name = self._parent_map.get(name)
        if parent_name is not None:
            parent_pos = _get_joint_position(parent_name, rest_pts)
            offset = (pos - parent_pos) * 100  # Convert to cm scale
        else:
            offset = np.array([0.0, 0.0, 0.0])

        lines.append(f"{prefix}  OFFSET {offset[0]:.4f} {-offset[1]:.4f} {offset[2]:.4f}")

        if is_root:
            lines.append(f"{prefix}  CHANNELS 6 Xposition Yposition Zposition Zrotation Xrotation Yrotation")
        else:
            lines.append(f"{prefix}  CHANNELS 3 Zrotation Xrotation Yrotation")

        if len(children) == 0:
            # End site
            lines.append(f"{prefix}  End Site")
            lines.append(f"{prefix}  {{")
            lines.append(f"{prefix}    OFFSET 0.0000 0.0000 0.0000")
            lines.append(f"{prefix}  }}")
        else:
            for child_name, child_children in children:
                self._write_hierarchy(lines, child_name, child_children,
                                      rest_pts, indent + 1)

        lines.append(f"{prefix}}}")

    # ------------------------------------------------------------------
    # Private: frame data computation
    # ------------------------------------------------------------------
    def _compute_frame_data(self, landmarks_3d: np.ndarray) -> List[float]:
        """
        Compute BVH channel values for one frame.

        Returns list of floats: root position (3) + rotations for each joint (3 each).
        """
        data: List[float] = []

        for i, name in enumerate(self._joint_order):
            pos = _get_joint_position(name, landmarks_3d)

            if i == 0:
                # Root: position (scaled to cm) + rotation
                data.extend([pos[0] * 100, -pos[1] * 100, pos[2] * 100])

            # Compute rotation relative to rest pose
            parent_name = self._parent_map.get(name)
            if parent_name is not None:
                # Current direction: parent -> this joint
                parent_pos = _get_joint_position(parent_name, landmarks_3d)
                curr_dir = pos - parent_pos

                # Rest direction
                rest_pos = _get_joint_position(name, self._rest_pose)
                rest_parent = _get_joint_position(parent_name, self._rest_pose)
                rest_dir = rest_pos - rest_parent

                if np.linalg.norm(curr_dir) > 1e-8 and np.linalg.norm(rest_dir) > 1e-8:
                    R = _rotation_matrix_from_vectors(rest_dir, curr_dir)
                    zr, xr, yr = _rotation_to_euler_zxy(R)
                else:
                    zr, xr, yr = 0.0, 0.0, 0.0
            else:
                # Root rotation: compute from hip orientation
                left_hip = landmarks_3d[23]
                right_hip = landmarks_3d[24]
                hip_vec = right_hip - left_hip
                rest_hip = self._rest_pose[24] - self._rest_pose[23]
                if np.linalg.norm(hip_vec) > 1e-8 and np.linalg.norm(rest_hip) > 1e-8:
                    R = _rotation_matrix_from_vectors(rest_hip, hip_vec)
                    zr, xr, yr = _rotation_to_euler_zxy(R)
                else:
                    zr, xr, yr = 0.0, 0.0, 0.0

            data.extend([zr, xr, yr])

        return data
