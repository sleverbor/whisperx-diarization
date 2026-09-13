"""Behavior checks for conversational attribution, independent of model downloads."""
import unittest
from chainofrules import (Baseline, Evidence, TimelineSegment, short_voice_crop,
                          add_question_response_evidence, resolve_segment)


class ShortAnswerTests(unittest.TestCase):
    def question(self, text="Do you have any weapons?", strength=0.9, speaker="SPEAKER_00"):
        segment = TimelineSegment(10.0, 12.83, text, Baseline(speaker, speaker))
        segment.final_speaker, segment.final_confidence = speaker, strength
        return segment

    def reply(self, start=12.89, end=12.99, text="No.", raw="SPEAKER_00"):
        segment = TimelineSegment(start, end, text, Baseline(raw, raw))
        segment.evidence.append(Evidence("local_voice", 0.0, 0.0))
        return segment

    def infer(self, reply, question, tracks=("SPEAKER_00", "SPEAKER_01"), mapping=1.0):
        add_question_response_evidence(reply, question, set(tracks), "SPEAKER_01", mapping)
        resolve_segment(reply, "SPEAKER_01", mapping)
        return reply

    def test_brief_answer_has_weak_alternative_identity(self):
        reply = self.infer(self.reply(), self.question())
        self.assertEqual(reply.final_speaker, "Target_Speaker")
        self.assertLessEqual(reply.final_confidence, 0.25)
        self.assertTrue(any("not voice-verified" in reason for reason in reply.reasons))

    def test_target_question_does_not_force_target_answer(self):
        reply = self.infer(self.reply(raw="SPEAKER_01"), self.question(speaker="Target_Speaker"))
        self.assertEqual(reply.final_speaker, "SPEAKER_00")

    def test_three_speakers_leave_answer_unresolved(self):
        reply = self.infer(self.reply(), self.question(), ("SPEAKER_00", "SPEAKER_01", "SPEAKER_02"))
        self.assertEqual(reply.final_speaker, "Uncertain")

    def test_no_question_or_weak_question_leaves_answer_unresolved(self):
        for question in (None, self.question("You are on private property."),
                         self.question("Why are you here?"), self.question(strength=0.35)):
            with self.subTest(question=question):
                self.assertEqual(self.infer(self.reply(), question).final_speaker, "Uncertain")

    def test_gap_overlap_and_long_answer_are_not_inferred(self):
        for reply in (self.reply(13.5, 13.6), self.reply(12.8, 12.9),
                      self.reply(12.89, 14.5), self.reply(text="No one should be doing that.")):
            with self.subTest(reply=reply):
                result = self.infer(reply, self.question())
                self.assertFalse(any(item.source == "question_response" for item in result.evidence))
                if reply.end - reply.start < 0.4:
                    self.assertEqual(result.final_speaker, "Uncertain")

    def test_weak_target_mapping_does_not_infer(self):
        self.assertEqual(self.infer(self.reply(), self.question(), mapping=0.4).final_speaker, "Uncertain")

    def test_available_voice_is_not_overridden_by_conversation(self):
        reply = self.reply()
        reply.evidence = [Evidence("local_voice", -1.0, 0.5)]
        self.assertEqual(self.infer(reply, self.question()).final_speaker, "SPEAKER_00")

    def test_padded_short_audio_does_not_claim_high_confidence(self):
        reply = self.reply(raw="SPEAKER_01")
        reply.evidence = [Evidence("local_voice", -0.64, 0.28)]
        resolve_segment(reply, "SPEAKER_01", 1.0)
        self.assertLessEqual(reply.final_confidence, 0.30)

    def test_independent_voice_corrects_short_baseline_conflict(self):
        reply = self.reply(start=0.25, end=0.92, raw="SPEAKER_01", text="Question")
        reply.evidence = [Evidence("local_voice", -0.57, 0.55,
            {"best_track": "SPEAKER_00", "track_margin": 0.15,
             "track_similarities": {"SPEAKER_00": 0.31, "SPEAKER_01": 0.16}})]
        resolve_segment(reply, "SPEAKER_01", 1.0)
        self.assertEqual(reply.final_speaker, "SPEAKER_00")

    def test_poor_profile_match_cannot_correct_identity(self):
        reply = self.reply(start=0.25, end=0.92, raw="SPEAKER_01", text="Question")
        reply.evidence = [Evidence("local_voice", -0.8, 0.55,
            {"best_track": "SPEAKER_00", "track_margin": 0.15,
             "track_similarities": {"SPEAKER_00": 0.20, "SPEAKER_01": 0.05}})]
        resolve_segment(reply, "SPEAKER_01", 1.0)
        self.assertEqual(reply.final_speaker, "Uncertain")

    def test_crop_respects_both_neighbors(self):
        following = TimelineSegment(13.01, 15.6, "Next", Baseline("SPEAKER_00", "SPEAKER_00"))
        start, end = short_voice_crop(self.reply(), self.question(), following, 30.0)
        self.assertAlmostEqual(start, 12.83)
        self.assertAlmostEqual(end, 13.01)
        self.assertLess(end-start, 0.4)

    def test_overlapping_timing_does_not_trim_reply(self):
        preceding = self.question()
        preceding.end = 12.92
        reply = self.reply()
        self.assertEqual(short_voice_crop(reply, preceding, None, 30.0), (reply.start, reply.end))


if __name__ == "__main__":
    unittest.main()
