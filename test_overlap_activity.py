import unittest

from evaluate_overlap_activity import evaluate


class OverlapActivityTest(unittest.TestCase):
    def test_reports_overlap_and_control_agreement(self):
        overlap = [{"source": "overlapping_speakers", "details": {"target_and_non_target": True}}]
        baseline = {"segments": [
            {"start": 0.0, "end": 1.0, "text": "both", "evidence": overlap},
            {"start": 1.0, "end": 2.0, "text": "one", "evidence": []},
        ]}
        rttm = [
            {"start": 0.0, "end": 2.0, "speaker": "a"},
            {"start": 0.2, "end": 0.8, "speaker": "b"},
        ]
        result = evaluate(baseline, rttm)
        self.assertEqual(result["summary"]["corroborated_overlap_segments"], 1)
        self.assertEqual(result["summary"]["overlap_on_control_segments"], 0)


if __name__ == "__main__":
    unittest.main()
