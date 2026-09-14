import unittest
import numpy as np
from build_target_reference import screen

class ScreeningTests(unittest.TestCase):
    def test_excludes_opposite_direction_and_invalid_sample(self):
        accepted,centroid,audit=screen(np.array([[1,0],[1,.1],[1,-.1],[-1,0],[np.nan,0]]),.45,2)
        self.assertEqual(audit['accepted_rows'],[0,1,2])
        self.assertEqual(audit['excluded_rows'],[3,4])
        self.assertTrue(np.isfinite(accepted).all())
        self.assertAlmostEqual(float(np.linalg.norm(centroid)),1,places=6)
    def test_rejects_split_identity_without_majority(self):
        with self.assertRaises(ValueError):screen(np.array([[1,0],[1,0],[-1,0],[-1,0]]),.45,2)
    def test_rejects_wrong_model_dimensions(self):
        with self.assertRaises(ValueError):screen(np.ones((4,3)),.45,192)
if __name__=='__main__':unittest.main()
