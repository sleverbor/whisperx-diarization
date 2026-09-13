"""Behavior checks for conversational attribution, independent of model downloads."""
import unittest
import numpy as np
from chainofrules import (Baseline, Evidence, TimelineSegment, short_voice_crop,
                          add_question_response_evidence, add_echo_question_evidence, resolve_segment, collect_visual_evidence)


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

    def test_face_presence_alone_does_not_override_voice(self):
        reply = self.reply(start=28.0, end=29.0)
        reply.evidence = [Evidence("local_voice", -0.8, 0.7,
            {"track_margin": 0.03, "track_similarities": {"SPEAKER_00": 0.23, "SPEAKER_01": 0.20}}),
            Evidence("target_face_visible", 1.0, 1.0)]
        resolve_segment(reply, "SPEAKER_01", 1.0)
        self.assertEqual(reply.final_speaker, "SPEAKER_00")

    def test_mouth_hint_is_tentative_only_with_ambiguous_profiles(self):
        for margin, expected in ((0.03, "Target_Speaker"), (0.20, "SPEAKER_00")):
            reply = self.reply(start=28.0, end=29.0)
            reply.evidence = [Evidence("local_voice", -0.8, 0.7,
                {"track_margin": margin, "track_similarities": {"SPEAKER_00": 0.23, "SPEAKER_01": 0.20}}),
                Evidence("target_mouth_motion", 1.0, 0.2)]
            resolve_segment(reply, "SPEAKER_01", 1.0)
            self.assertEqual(reply.final_speaker, expected)
            if margin < 0.05:
                self.assertLessEqual(reply.final_confidence, 0.25)

    def test_face_identity_continues_through_head_turn_but_not_bbox_jump(self):
        class Capture:
            index = 0
            def get(self, prop): return 20
            def set(self, prop, value): self.index = int(value)
            def read(self): return True, np.zeros((200, 200, 3), dtype=np.uint8)
        class Face:
            pass
        for jump in (False, True):
            capture = Capture()
            class Analyzer:
                def get(self, frame):
                    face = Face()
                    face.embedding = np.array([0.7, np.sqrt(1-0.7**2)]) if capture.index < 2 else np.array([0.3, np.sqrt(1-0.3**2)])
                    face.bbox = np.array([10,10,80,100]) if not jump or capture.index < 2 else np.array([120,120,190,200])
                    points = np.zeros((68,3))
                    points[64,0] = 10
                    points[66,1] = 0.4 if capture.index % 2 else 1.2
                    face.landmark_3d_68 = points
                    return [face]
            reply = self.reply(start=0.5, end=1.4)
            collect_visual_evidence(reply, capture, 8.0, Analyzer(), np.array([1.0, 0.0]))
            motion = next(item for item in reply.evidence if item.source == "target_mouth_motion")
            self.assertEqual(motion.confidence > 0, not jump)

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


    def test_echo_requires_weak_supporting_voice_and_visible_target(self):
        previous = self.question("You are being arrested for criminal loitering.")
        reply = self.reply(start=13.3, end=13.88, text="Criminal loitering?")
        reply.evidence = [Evidence("local_voice", -.8, .48,
            {"best_track": "SPEAKER_01", "track_margin": .07,
             "track_similarities": {"SPEAKER_00": .05, "SPEAKER_01": .12}}),
            Evidence("target_face_visible", 0, 0, {"target_visible_hint": True})]
        add_echo_question_evidence(reply, previous, {"SPEAKER_00", "SPEAKER_01"}, "SPEAKER_01", 1)
        resolve_segment(reply, "SPEAKER_01", 1)
        self.assertEqual(reply.final_speaker, "Target_Speaker")
        self.assertEqual(reply.final_confidence, .20)
        reply.evidence = [e for e in reply.evidence if e.source != "target_face_visible"]
        resolve_segment(reply, "SPEAKER_01", 1)
        self.assertNotEqual(reply.final_speaker, "Target_Speaker")

    def test_echo_does_not_override_clear_voice_or_choose_among_three_people(self):
        previous = self.question("You are being arrested for criminal loitering.")
        reply = self.reply(start=13.3, end=13.88, text="Criminal loitering?")
        reply.evidence = [Evidence("local_voice", -1, .6,
            {"best_track": "SPEAKER_00", "track_margin": .4,
             "track_similarities": {"SPEAKER_00": .5, "SPEAKER_01": .1}}),
            Evidence("target_face_visible", 0, 0, {"target_visible_hint": True})]
        add_echo_question_evidence(reply, previous, {"SPEAKER_00", "SPEAKER_01"}, "SPEAKER_01", 1)
        resolve_segment(reply, "SPEAKER_01", 1)
        self.assertEqual(reply.final_speaker, "SPEAKER_00")
        reply.evidence = []
        add_echo_question_evidence(reply, previous, {"SPEAKER_00", "SPEAKER_01", "SPEAKER_02"}, "SPEAKER_01", 1)
        self.assertEqual(reply.evidence, [])

    def test_padded_but_weak_voice_allows_tentative_answer(self):
        reply = self.reply()
        reply.evidence = [Evidence("local_voice", -1, .28,
            {"best_track": "SPEAKER_00", "track_similarities": {"SPEAKER_00": .23, "SPEAKER_01": .08}})]
        self.infer(reply, self.question())
        self.assertEqual(reply.final_speaker, "Target_Speaker")
        self.assertEqual(reply.final_confidence, .20)


if __name__ == "__main__":
    unittest.main()
