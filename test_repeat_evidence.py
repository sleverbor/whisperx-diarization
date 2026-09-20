import unittest

from chainofrules import Baseline, Evidence, TimelineSegment
from repeat_evidence import (find_repeat_groups, find_text_repeat_candidates, build_repeat_proposals,
                             repeat_target_corroboration,
                             resolve_repeat_target_corroboration)


def segment(start, text, speaker="Uncertain", confidence=0.0):
    value = TimelineSegment(start, start + 1.0, text, Baseline("S0", "S0"))
    value.final_speaker = speaker
    value.final_confidence = confidence
    return value


class RepeatEvidenceTests(unittest.TestCase):
    def test_short_repeated_dialogue_is_review_candidate(self):
        timeline = [
            segment(0, "Unrelated opening words are spoken here."),
            segment(10, "I cannot force him to leave the property."),
            segment(12, "I can talk to him and ask him to move down."),
            segment(40, "Different intervening conversation happens now."),
            segment(70, "I can't force him to leave the property."),
            segment(72, "I can talk to him and ask him to move down."),
        ]
        candidates = find_text_repeat_candidates(timeline)
        match = max(candidates, key=lambda x: x["text_similarity"])
        self.assertAlmostEqual(match["left_start"], 10.0)
        self.assertAlmostEqual(match["right_start"], 70.0)
        self.assertTrue(match["review_required"])
        self.assertFalse(match["automatic_text_replacement"])
        self.assertFalse(match["automatic_speaker_change"])

    def test_short_repeat_requires_meaningful_window(self):
        timeline = [segment(0, "All right."), segment(20, "All right.")]
        self.assertEqual(find_text_repeat_candidates(timeline), [])

    def test_presentation_group_marks_sequence_support_without_auto_action(self):
        timeline = [
            segment(0, "I cannot force him to leave the property."),
            segment(2, "I can talk to him and ask him to move down."),
            segment(30, "I can't force him to leave the property."),
            segment(32, "I can talk to him and ask him to move down."),
        ]
        groups = [{"id": "repeat_01", "left_start": 0, "left_end": 3,
                   "right_start": 30, "right_end": 33}]
        match = find_text_repeat_candidates(
            timeline, presentation_groups=groups
        )[0]
        self.assertEqual(match["evidence_tier"], "sequence_supported")
        self.assertEqual(match["presentation_group_support"], ["repeat_01"])
        self.assertFalse(match["automatic_text_replacement"])
        self.assertFalse(match["automatic_speaker_change"])

    def test_detects_ordered_repeated_presentation(self):
        first = [
            segment(0, "They can ask you to get off their property."),
            segment(10, "Are you going to leave the property?"),
            segment(20, "We will arrest you on the property if you do not leave."),
            segment(30, "Where does the property begin?"),
        ]
        second = [
            segment(100, "They can ask you to get off their property."),
            segment(110, "Are you going to leave the property?"),
            segment(120, "We are going to arrest you on the property if you do not leave."),
            segment(130, "Where does the property begin here?"),
        ]
        groups = find_repeat_groups(first + second)
        self.assertEqual(len(groups), 1)
        self.assertEqual(len(groups[0]["anchors"]), 4)
        self.assertAlmostEqual(groups[0]["offset_seconds"], 100.0)

    def test_bracketed_corruption_receives_donor_candidate(self):
        first = [
            segment(0, "They can ask you to get off their property."),
            segment(10, "I'm a garbled person."),
            segment(20, "We will arrest you on the property if you do not leave."),
            segment(30, "Where does the property begin?"),
            segment(40, "My name is Matthew Cox."),
        ]
        second = [
            segment(100, "They can ask you to get off their property."),
            segment(110, "I am engaged constitutionally.", "Target_Speaker", 0.9),
            segment(120, "We are going to arrest you on the property if you do not leave."),
            segment(130, "Where does the property begin here?"),
            segment(140, "My name is Matthew Cox."),
        ]
        timeline = first + second
        groups = find_repeat_groups(timeline)
        proposals = build_repeat_proposals(timeline, groups)
        self.assertEqual(proposals[1][0]["donor_text"], "I am engaged constitutionally.")
        self.assertEqual(proposals[1][0]["donor_final_speaker"], "Target_Speaker")
        self.assertEqual(first[1].text, "I'm a garbled person.")

    def test_isolated_repeated_phrase_is_not_a_presentation(self):
        timeline = [
            segment(0, "Have a nice day."),
            segment(60, "Have a nice day."),
            segment(120, "Something unrelated happened here."),
        ]
        self.assertEqual(find_repeat_groups(timeline), [])

    def test_exact_repeat_can_corroborate_strong_target_donor(self):
        recipient = segment(10, "I mean, I understand.", "SPEAKER_03", 0.77)
        recipient.evidence.append(Evidence(
            "local_voice", -1.0, 0.65, {"similarity": 0.095}
        ))
        proposal = {
            "group_id": "repeat_01", "donor_start": 110, "donor_end": 111,
            "donor_text": "I mean, I understand.",
            "donor_final_speaker": "Target_Speaker",
            "donor_final_confidence": 0.90,
            "alignment_confidence": 0.75,
            "timing_error_seconds": 0.1,
        }
        corroboration = repeat_target_corroboration(recipient, [proposal])
        self.assertIsNotNone(corroboration)
        self.assertTrue(resolve_repeat_target_corroboration(recipient, corroboration))
        self.assertEqual(recipient.final_speaker, "Target_Speaker")
        self.assertEqual(recipient.final_confidence, 0.55)
        self.assertEqual(recipient.text, "I mean, I understand.")

    def test_repeat_corroboration_rejects_weak_or_partial_evidence(self):
        base = {
            "group_id": "repeat_01", "donor_start": 110, "donor_end": 111,
            "donor_text": "I mean, I understand.",
            "donor_final_speaker": "Target_Speaker",
            "donor_final_confidence": 0.90,
            "alignment_confidence": 0.75,
            "timing_error_seconds": 0.1,
        }
        cases = [
            ({**base, "donor_final_confidence": 0.74}, 0.10,
             "I mean, I understand."),
            ({**base, "alignment_confidence": 0.71}, 0.10,
             "I mean, I understand."),
            (base, 0.049, "I mean, I understand."),
            (base, 0.10, "I understand a different request entirely."),
            ({**base, "donor_final_speaker": "SPEAKER_02"}, 0.10,
             "I mean, I understand."),
        ]
        for proposal, local_similarity, text in cases:
            with self.subTest(proposal=proposal, local_similarity=local_similarity,
                              text=text):
                recipient = segment(10, text, "SPEAKER_03", 0.8)
                recipient.evidence.append(Evidence(
                    "local_voice", -1.0, 0.8,
                    {"similarity": local_similarity}
                ))
                self.assertIsNone(
                    repeat_target_corroboration(recipient, [proposal])
                )

    def test_overlap_is_never_reassigned_by_repeat(self):
        recipient = segment(10, "I mean, I understand.",
                            "Overlapping_Speakers", 0.0)
        recipient.evidence.append(Evidence(
            "local_voice", 1.0, 1.0, {"similarity": 0.8}
        ))
        proposal = {
            "donor_text": recipient.text,
            "donor_final_speaker": "Target_Speaker",
            "donor_final_confidence": 1.0,
            "alignment_confidence": 1.0,
        }
        self.assertIsNone(repeat_target_corroboration(recipient, [proposal]))


if __name__ == "__main__":
    unittest.main()
