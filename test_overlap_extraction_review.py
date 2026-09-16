import unittest

from review_overlap_extraction import classify_extraction, select_overlap_segments


class OverlapExtractionReviewTests(unittest.TestCase):
    def test_selects_only_target_non_target_overlap(self):
        baseline = {"segments": [
            {"text": "yes", "evidence": [{"source": "overlapping_speakers", "details": {"target_and_non_target": True}}]},
            {"text": "no", "evidence": [{"source": "overlapping_speakers", "details": {"target_and_non_target": False}}]},
            {"text": "plain", "evidence": []},
        ]}
        selected = select_overlap_segments(baseline)
        self.assertEqual([row[0] for row in selected], [0])

    def test_low_energy_is_suppressed_even_if_asr_hallucinates(self):
        self.assertEqual(
            classify_extraction(0.01, 0.40, 0.075, "My name is Jack."),
            "likely_suppressed_residual",
        )

    def test_stronger_retained_target_is_candidate(self):
        self.assertEqual(
            classify_extraction(0.18, 0.43, 0.108, "I'm not answering questions."),
            "candidate_target_speech",
        )

    def test_borderline_result_abstains(self):
        self.assertEqual(
            classify_extraction(0.25, 0.31, 0.15, "Maybe."),
            "unresolved",
        )


if __name__ == "__main__":
    unittest.main()
