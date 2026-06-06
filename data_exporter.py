"""
data_exporter.py - Export pose landmarks and joint angles to JSON or CSV.

The BVH exporter is useful for animation tools. This exporter is aimed at
reports, debugging, and experiments where raw landmarks and kinematics are
easier to inspect in a spreadsheet or script.
"""

import csv
import json
from pathlib import Path
from typing import Dict, List, Optional, Tuple


Point3D = Tuple[float, float, float]


class MotionDataExporter:
    """Collect and save frame-by-frame pose data."""

    def __init__(self):
        self._frames: List[Dict] = []

    @property
    def frame_count(self) -> int:
        return len(self._frames)

    def add_frame(
        self,
        frame_index: int,
        timestamp_ms: float,
        detection: Optional[Dict],
        angles: Optional[Dict[str, float]],
        kinematics: Optional[Dict] = None,
    ) -> None:
        """Append one frame of data."""
        if detection is None:
            self._frames.append({
                "frame": frame_index,
                "timestamp_ms": timestamp_ms,
                "detected": False,
                "landmarks": [],
                "visibility": [],
                "angles": {},
                "kinematics": kinematics or {},
            })
            return

        self._frames.append({
            "frame": frame_index,
            "timestamp_ms": timestamp_ms,
            "detected": True,
            "landmarks": [
                {"x": float(x), "y": float(y), "z": float(z)}
                for x, y, z in detection["landmarks_3d"]
            ],
            "visibility": [float(v) for v in detection["visibility"]],
            "angles": {name: float(value) for name, value in (angles or {}).items()},
            "kinematics": kinematics or {},
        })

    def save_json(self, filepath: str) -> None:
        """Save collected frames to JSON."""
        path = Path(filepath)
        path.parent.mkdir(parents=True, exist_ok=True)
        payload = {
            "format": "human-motion-capture-json-v1",
            "frame_count": len(self._frames),
            "frames": self._frames,
        }
        with path.open("w", encoding="utf-8") as f:
            json.dump(payload, f, indent=2)
        print(f"[DATA] Saved {len(self._frames)} frames to {path}")

    def save_csv(self, filepath: str) -> None:
        """Save landmarks and angles to a wide CSV file."""
        path = Path(filepath)
        path.parent.mkdir(parents=True, exist_ok=True)

        angle_names = sorted({
            angle_name
            for frame in self._frames
            for angle_name in frame.get("angles", {}).keys()
        })

        headers = ["frame", "timestamp_ms", "detected"]
        for idx in range(33):
            headers.extend([f"lm{idx}_x", f"lm{idx}_y", f"lm{idx}_z", f"lm{idx}_visibility"])
        headers.extend([f"angle_{name}" for name in angle_names])

        with path.open("w", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=headers)
            writer.writeheader()

            for frame in self._frames:
                row = {
                    "frame": frame["frame"],
                    "timestamp_ms": f"{frame['timestamp_ms']:.3f}",
                    "detected": int(frame["detected"]),
                }
                landmarks = frame.get("landmarks", [])
                visibility = frame.get("visibility", [])
                for idx in range(33):
                    if idx < len(landmarks):
                        lm = landmarks[idx]
                        row[f"lm{idx}_x"] = f"{lm['x']:.8f}"
                        row[f"lm{idx}_y"] = f"{lm['y']:.8f}"
                        row[f"lm{idx}_z"] = f"{lm['z']:.8f}"
                        row[f"lm{idx}_visibility"] = f"{visibility[idx]:.6f}"
                    else:
                        row[f"lm{idx}_x"] = ""
                        row[f"lm{idx}_y"] = ""
                        row[f"lm{idx}_z"] = ""
                        row[f"lm{idx}_visibility"] = ""

                for name in angle_names:
                    value = frame.get("angles", {}).get(name)
                    row[f"angle_{name}"] = "" if value is None else f"{value:.4f}"
                writer.writerow(row)

        print(f"[DATA] Saved {len(self._frames)} frames to {path}")
