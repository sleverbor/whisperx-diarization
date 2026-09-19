import unittest

from overlap_review_policy import apply_policy


class OverlapReviewPolicyTest(unittest.TestCase):
    def evaluate(self, sf, dp, baseline_overlap=False):
        base = {"baseline_index": 7, "start": 1.0, "end": 2.0,
                "text": "words", "baseline_speaker": "Target_Speaker",
                "baseline_target_non_target_overlap": baseline_overlap}
        return apply_policy(
            {"segments": [{**base, "overlap_fraction": sf}]},
            {"segments": [{**base, "overlap_fraction": dp}]},
        )["segments"][0]

    def test_strong_sortformer_triggers_separation_review(self):
        row = self.evaluate(.30, 0)
        self.assertEqual(row["tier"], "separation_review")
        self.assertFalse(row["speaker_identity_changed"])

    def test_weak_signals_only_add_uncertainty(self):
        self.assertEqual(self.evaluate(.10, 0)["tier"], "uncertainty_evidence")
        row = self.evaluate(0, .20)
        self.assertEqual(row["tier"], "uncertainty_evidence")
        self.assertEqual(row["reasons"], ["strong_diaper_rescue"])

    def test_preserves_existing_baseline_overlap_review(self):
        row = self.evaluate(0, 0, baseline_overlap=True)
        self.assertEqual(row["tier"], "separation_review")
        self.assertEqual(row["reasons"], ["baseline_overlap_evidence"])

    def test_below_threshold_signals_do_nothing(self):
        self.assertEqual(self.evaluate(.029, .199)["tier"], "none")


if __name__ == "__main__":
    unittest.main()
