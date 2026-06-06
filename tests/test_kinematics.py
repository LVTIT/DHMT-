import math
import unittest

import numpy as np

from kinematics import (
    angle_between,
    compute_all_angles,
    compute_bone_lengths,
    solve_two_bone_ik,
)
from motion_filter import LandmarkSmoother


class KinematicsTest(unittest.TestCase):
    def test_angle_between_right_angle(self):
        deg = angle_between(
            np.array([1.0, 0.0, 0.0]),
            np.array([0.0, 0.0, 0.0]),
            np.array([0.0, 1.0, 0.0]),
        )
        self.assertAlmostEqual(deg, 90.0, places=5)

    def test_compute_all_angles_has_elbow(self):
        landmarks = [(0.0, 0.0, 0.0)] * 33
        landmarks[11] = (0.0, 1.0, 0.0)
        landmarks[13] = (0.0, 0.0, 0.0)
        landmarks[15] = (1.0, 0.0, 0.0)
        visibility = [1.0] * 33

        angles = compute_all_angles(landmarks, visibility)

        self.assertIn("LEFT_ELBOW", angles)
        self.assertAlmostEqual(angles["LEFT_ELBOW"], 90.0, places=5)

    def test_bone_lengths_returns_segments(self):
        landmarks = [(float(i), 0.0, 0.0) for i in range(33)]

        lengths = compute_bone_lengths(landmarks)

        self.assertGreater(len(lengths), 0)
        self.assertTrue(all(value >= 0.0 for value in lengths.values()))

    def test_two_bone_ik_preserves_lengths(self):
        root = (0.0, 0.0, 0.0)
        mid = (1.0, 0.0, 0.0)
        end = (2.0, 0.0, 0.0)
        target = (1.0, 1.0, 0.0)

        new_root, new_mid, new_end = solve_two_bone_ik(root, mid, end, target)

        len_a = math.dist(new_root, new_mid)
        len_b = math.dist(new_mid, new_end)
        self.assertAlmostEqual(len_a, 1.0, places=5)
        self.assertAlmostEqual(len_b, 1.0, places=5)


class MotionFilterTest(unittest.TestCase):
    def test_landmark_smoother_ema(self):
        smoother = LandmarkSmoother(alpha=0.5)
        base = {
            "landmarks_3d": [(0.0, 0.0, 0.0)],
            "landmarks_2d": [(0, 0)],
            "visibility": [1.0],
        }
        moved = dict(base)
        moved["landmarks_3d"] = [(2.0, 0.0, 0.0)]

        smoother.apply(base)
        result = smoother.apply(moved)

        self.assertEqual(result["landmarks_3d"][0], (1.0, 0.0, 0.0))


if __name__ == "__main__":
    unittest.main()
