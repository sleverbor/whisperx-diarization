import unittest

from chainofrules import Baseline, Evidence, TimelineSegment, add_overlap_evidence, resolve_segment


class OverlapResolutionTests(unittest.TestCase):
    def segment(self):
        value = TimelineSegment(7.21, 9.01, "My name is Jeff.", Baseline("SPEAKER_01", "SPEAKER_01"))
        value.evidence.append(Evidence("local_voice", -1.0, 1.0, {}))
        return value

    def test_target_non_target_overlap_stays_unassigned(self):
        segment = self.segment()
        add_overlap_evidence(segment, [
            {"start": 7.18, "end": 8.01, "speaker": "SPEAKER_00"},
            {"start": 7.44, "end": 9.14, "speaker": "SPEAKER_01"},
        ], "SPEAKER_00")
        resolve_segment(segment, "SPEAKER_00", 1.0)
        self.assertEqual(segment.final_speaker, "Overlapping_Speakers")
        self.assertEqual(segment.final_confidence, 0.0)

    def test_adjacent_tracks_are_not_overlap(self):
        segment = self.segment()
        add_overlap_evidence(segment, [
            {"start": 7.2, "end": 8.0, "speaker": "SPEAKER_00"},
            {"start": 8.0, "end": 9.1, "speaker": "SPEAKER_01"},
        ], "SPEAKER_00")
        self.assertFalse(any(e.source == "overlapping_speakers" for e in segment.evidence))

    def test_tiny_boundary_overlap_is_ignored(self):
        segment = self.segment()
        add_overlap_evidence(segment, [
            {"start": 7.2, "end": 8.05, "speaker": "SPEAKER_00"},
            {"start": 8.0, "end": 9.1, "speaker": "SPEAKER_01"},
        ], "SPEAKER_00")
        self.assertFalse(any(e.source == "overlapping_speakers" for e in segment.evidence))

    def test_localized_overlap_preserves_strong_dominant_target(self):
        segment = TimelineSegment(
            10.0, 13.0, "What are you doing?",
            Baseline("SPEAKER_00", "Target_Speaker"),
        )
        segment.evidence.append(Evidence("local_voice", 0.70, 0.80, {
            "similarity": 0.72,
            "track_similarities": {"SPEAKER_00": 0.71, "SPEAKER_01": 0.04},
            "best_track": "SPEAKER_00",
            "track_margin": 0.67,
        }))
        add_overlap_evidence(segment, [
            {"start": 10.0, "end": 13.0, "speaker": "SPEAKER_00"},
            {"start": 11.1, "end": 11.7, "speaker": "SPEAKER_01"},
        ], "SPEAKER_00")
        resolve_segment(segment, "SPEAKER_00", 1.0)
        self.assertEqual(segment.final_speaker, "Target_Speaker")
        self.assertGreaterEqual(segment.final_confidence, 0.35)
        overlap = next(e for e in segment.evidence if e.source == "overlapping_speakers")
        self.assertAlmostEqual(overlap.details["overlap_fraction"], 0.2)
        self.assertIn("remains unresolved", segment.reasons[-1])

    def test_broad_overlap_remains_unassigned_despite_target_voice(self):
        segment = TimelineSegment(
            10.0, 13.0, "Two people talking.",
            Baseline("SPEAKER_00", "Target_Speaker"),
        )
        segment.evidence.append(Evidence("local_voice", 0.70, 0.80, {
            "similarity": 0.72,
            "track_similarities": {"SPEAKER_00": 0.71, "SPEAKER_01": 0.04},
            "best_track": "SPEAKER_00",
            "track_margin": 0.67,
        }))
        add_overlap_evidence(segment, [
            {"start": 10.0, "end": 13.0, "speaker": "SPEAKER_00"},
            {"start": 11.0, "end": 12.5, "speaker": "SPEAKER_01"},
        ], "SPEAKER_00")
        resolve_segment(segment, "SPEAKER_00", 1.0)
        self.assertEqual(segment.final_speaker, "Overlapping_Speakers")
        self.assertEqual(segment.final_confidence, 0.0)


if __name__ == "__main__":
    unittest.main()
