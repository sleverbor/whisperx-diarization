import unittest

import numpy as np

from review_overlap_extraction import (
    classify_extraction, corroborated_novel_words, select_overlap_segments,
    stereo_metrics,
)


class OverlapExtractionReviewTests(unittest.TestCase):
    def test_selects_only_target_non_target_overlap(self):
        baseline = {"segments": [
            {"text": "yes", "evidence": [{"source": "overlapping_speakers", "details": {"target_and_non_target": True}}]},
            {"text": "no", "evidence": [{"source": "overlapping_speakers", "details": {"target_and_non_target": False}}]},
            {"text": "plain", "evidence": []},
        ]}
        selected = select_overlap_segments(baseline)
        self.assertEqual([row[0] for row in selected], [0])

    def test_policy_adds_strong_rows_without_dropping_baseline_rows(self):
        baseline = {"segments": [
            {"start": 0.0, "end": 1.0, "evidence": [{
                "source": "overlapping_speakers",
                "details": {"target_and_non_target": True},
            }]},
            {"start": 1.0, "end": 2.0, "evidence": []},
            {"start": 2.0, "end": 3.0, "evidence": []},
        ]}
        policy = {"segments": [
            {"baseline_index": 0, "start": 0.0, "end": 1.0,
             "tier": "separation_review"},
            {"baseline_index": 1, "start": 1.0, "end": 2.0,
             "tier": "separation_review", "reasons": ["strong_sortformer_overlap"]},
            {"baseline_index": 2, "start": 2.0, "end": 3.0,
             "tier": "uncertainty_evidence"},
        ]}
        selected = select_overlap_segments(baseline, policy)
        self.assertEqual([row[0] for row in selected], [0, 1])
        self.assertEqual(selected[0][2]["source"], "overlapping_speakers")
        self.assertEqual(selected[1][2]["source"], "supplemental_overlap_policy")

    def test_policy_must_match_baseline_timing(self):
        baseline = {"segments": [{"start": 0.0, "end": 1.0, "evidence": []}]}
        policy = {"segments": [{"baseline_index": 0, "start": 9.0, "end": 10.0,
                                "tier": "separation_review"}]}
        with self.assertRaises(ValueError):
            select_overlap_segments(baseline, policy)

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

    def test_duplicated_mono_is_not_analyzed_as_stereo(self):
        mono = np.linspace(-1, 1, 1600, dtype=np.float32)
        result = stereo_metrics(np.column_stack([mono, mono]))
        self.assertTrue(result["available"])
        self.assertFalse(result["distinct"])

    def test_meaningfully_different_channels_are_detected(self):
        time = np.arange(1600, dtype=np.float32) / 16000
        left = np.sin(2 * np.pi * 220 * time)
        right = np.sin(2 * np.pi * 370 * time)
        result = stereo_metrics(np.column_stack([left, right]))
        self.assertTrue(result["distinct"])
        self.assertLess(result["correlation"], .98)

    def test_novel_word_requires_two_channel_views(self):
        transcripts = {
            "left": "No, not on that property. Well.",
            "right": "No, not on that property.",
            "middle": "No, not on that property.",
            "difference": "No, not on that property. Well. Yes.",
        }
        result = corroborated_novel_words(
            transcripts, "No, not on that property."
        )
        self.assertEqual(result, [{
            "word": "well", "support": 2,
            "views": ["difference", "left"],
        }])

    def test_repeated_substitution_is_not_recovered_speech(self):
        transcripts = {
            "left": "I am going to see you.",
            "right": "I am going to see you.",
            "middle": "I am going to see you.",
        }
        self.assertEqual(
            corroborated_novel_words(
                transcripts, "I am going to sue you."
            ),
            [],
        )


if __name__ == "__main__":
    unittest.main()
