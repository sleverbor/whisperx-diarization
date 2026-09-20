import copy
import unittest

from mossformer2_review_policy import apply_review_policy, prepare_labels
from run_mossformer2_separation_experiment import token_f1


class MossFormer2ReviewPolicyTest(unittest.TestCase):
    def test_runner_has_self_contained_token_metric(self):
        self.assertEqual(token_f1("there's no", "there's no"), 1.0)
        self.assertEqual(token_f1("", "words"), 0.0)

    def report(self, first, second):
        return {"results": [{
            "exchange_id": "example", "baseline_index": 3,
            "start": 1.0, "end": 2.0,
            "baseline_text": "candidate words",
            "streams": [
                {"stream": 1, "target_similarity": first,
                 "audio": "one.wav", "transcription": {"text": "candidate"}},
                {"stream": 2, "target_similarity": second,
                 "audio": "two.wav", "transcription": {"text": "other"}},
            ],
        }]}

    def test_strong_target_stream_is_review_only(self):
        row = apply_review_policy(self.report(.335, .009))["segments"][0]
        self.assertEqual(row["disposition"], "review_candidate_target_stream")
        self.assertEqual(row["selected_stream"], 1)
        self.assertTrue(row["mixed_speaker_risk"])
        self.assertFalse(row["automatic_text_insertion"])
        self.assertFalse(row["speaker_identity_changed"])
        self.assertEqual(row["candidate_transcription_status"],
                         "unverified_review_hint")
        self.assertFalse(row["automatic_candidate_transcription_use"])
        self.assertEqual(row["transcript_review_class"],
                         "baseline_corroboration")

    def test_novel_words_are_flagged_even_with_strong_voice_identity(self):
        report = self.report(.50, .01)
        report["results"][0]["baseline_text"] = "known baseline"
        report["results"][0]["streams"][0]["transcription"]["text"] = (
            "entirely different phrase"
        )
        row = apply_review_policy(report)["segments"][0]
        self.assertEqual(row["transcript_review_class"],
                         "novel_or_conflicting_words")
        self.assertFalse(row["automatic_candidate_transcription_use"])

    def test_low_absolute_and_margin_are_rejected(self):
        row = apply_review_policy(self.report(.043, .024))["segments"][0]
        self.assertEqual(row["disposition"], "rejected_low_target_evidence")
        self.assertIsNone(row["selected_stream"])
        self.assertFalse(row["review_required"])

    def test_high_absolute_but_ambiguous_margin_is_rejected(self):
        row = apply_review_policy(self.report(.30, .25))["segments"][0]
        self.assertEqual(row["disposition"], "rejected_low_target_evidence")

    def test_prepare_preserves_baseline(self):
        baseline = {"segments": [{
            "start": 1.0, "end": 2.0, "text": "original",
            "evidence": [{"source": "overlapping_speakers",
                          "details": {"target_and_non_target": True}}],
        }]}
        before = copy.deepcopy(baseline)
        prepared = prepare_labels(baseline)
        self.assertEqual(baseline, before)
        self.assertEqual(prepared["labels"][0]["baseline_text"], "original")


if __name__ == "__main__":
    unittest.main()
