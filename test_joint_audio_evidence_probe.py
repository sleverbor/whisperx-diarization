import unittest

from joint_audio_evidence_probe import (common_contiguous_spans,
    cross_family_wording, joint_decision)


class JointAudioEvidenceTests(unittest.TestCase):
    def test_shared_suffix_survives_changed_prefix(self):
        spans = common_contiguous_spans(
            "Not doing it right here because this is public property",
            "Now I'm doing it right here because this is public property")
        self.assertEqual(spans[0], "doing it right here because this is public property")

    def test_same_family_agreement_is_not_corroboration(self):
        rows = [{"name": "original", "family": "original", "text": "wrong words"},
                {"name": "filtered", "family": "original", "text": "wrong words"}]
        self.assertEqual(cross_family_wording(rows), [])

    def test_cross_family_span_and_target_voice_yields_candidate(self):
        rows = [{"name": "original", "family": "original", "text": "I am not panhandling",
                 "speaker_resolution": "Target_Speaker"},
                {"name": "separated", "family": "separator", "text": "not panhandling",
                 "speaker_resolution": "Target_Speaker"}]
        spans = cross_family_wording(rows)
        self.assertEqual(spans[0]["text"], "not panhandling")
        self.assertEqual(joint_decision(rows, spans), "target_wording_candidate")

    def test_target_identity_does_not_validate_unstable_words(self):
        rows = [{"name": "original", "family": "original", "text": "paying him",
                 "speaker_resolution": "Target_Speaker"}]
        self.assertEqual(joint_decision(rows, []), "target_identity_only")

    def test_competing_identified_speaker_blocks_target_wording_candidate(self):
        rows = [{"name": "a", "family": "a", "text": "same phrase",
                 "speaker_resolution": "Target_Speaker"},
                {"name": "b", "family": "b", "text": "same phrase",
                 "speaker_resolution": "Opening_officer"}]
        self.assertEqual(joint_decision(rows, cross_family_wording(rows)), "target_identity_only")

    def test_mixed_sentence_boundary_overrides_whole_crop_voice(self):
        rows = [{"name": "whole", "family": "original", "text": "we can test that",
                 "speaker_resolution": "Target_Speaker"}]
        self.assertEqual(joint_decision(rows, [], "mixed"), "mixed_boundary")


if __name__ == "__main__": unittest.main()
