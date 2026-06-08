import math
import unittest

import numpy as np

from kinematics import (
    angle_between,
    compute_all_angles,
    compute_bone_lengths,
    fabrik_ik,
    solve_two_bone_ik,
)
from gesture_classifier import GestureClassifier
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

    def test_fabrik_reaches_target_and_preserves_lengths(self):
        chain = [(0.0, 0.0, 0.0), (1.0, 0.0, 0.0), (2.0, 0.0, 0.0)]
        target = (1.0, 1.0, 0.0)

        solved = fabrik_ik(chain, target, tolerance=1e-5, max_iterations=30)

        self.assertLess(math.dist(solved[-1], target), 1e-3)
        self.assertAlmostEqual(math.dist(solved[0], solved[1]), 1.0, places=5)
        self.assertAlmostEqual(math.dist(solved[1], solved[2]), 1.0, places=5)


class GestureClassifierTest(unittest.TestCase):
    def _standing_detection(self):
        landmarks_2d = [(0.0, 0.0)] * 33
        landmarks_3d = [(0.0, 0.0, 0.0)] * 33
        visibility = [1.0] * 33

        landmarks_2d[0] = (320, 60)
        landmarks_2d[11] = (270, 140)
        landmarks_2d[12] = (370, 140)
        landmarks_2d[13] = (245, 205)
        landmarks_2d[14] = (395, 205)
        landmarks_2d[15] = (235, 260)
        landmarks_2d[16] = (405, 260)
        landmarks_2d[23] = (285, 285)
        landmarks_2d[24] = (355, 285)
        landmarks_2d[25] = (285, 405)
        landmarks_2d[26] = (355, 405)
        landmarks_2d[27] = (285, 520)
        landmarks_2d[28] = (355, 520)

        landmarks_3d[0] = (0.0, -1.55, 0.0)
        landmarks_3d[11] = (-0.25, -1.0, 0.0)
        landmarks_3d[12] = (0.25, -1.0, 0.0)
        landmarks_3d[13] = (-0.45, -0.55, 0.0)
        landmarks_3d[14] = (0.45, -0.55, 0.0)
        landmarks_3d[15] = (-0.48, -0.15, 0.0)
        landmarks_3d[16] = (0.48, -0.15, 0.0)
        landmarks_3d[23] = (-0.18, 0.0, 0.0)
        landmarks_3d[24] = (0.18, 0.0, 0.0)
        landmarks_3d[25] = (-0.18, 0.75, 0.0)
        landmarks_3d[26] = (0.18, 0.75, 0.0)
        landmarks_3d[27] = (-0.18, 1.45, 0.0)
        landmarks_3d[28] = (0.18, 1.45, 0.0)

        return {
            "landmarks_2d": landmarks_2d,
            "landmarks_3d": landmarks_3d,
            "visibility": visibility,
        }

    def test_classifier_detects_left_hand_up(self):
        detection = self._standing_detection()
        detection["landmarks_2d"][15] = (235, 78)
        result = GestureClassifier().classify(detection)
        self.assertEqual(result.label, "LEFT_HAND_UP")
        self.assertGreater(result.confidence, 0.5)

    def test_classifier_detects_standing(self):
        result = GestureClassifier().classify(self._standing_detection())
        self.assertEqual(result.label, "STANDING")


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
