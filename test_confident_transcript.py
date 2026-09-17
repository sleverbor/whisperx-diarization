import json
import tempfile
import unittest
from pathlib import Path

from export_confident_transcript import classify, export_transcript, timestamp


def row(speaker, confidence, text="Words", start=1.0, end=2.0):
    return {"start": start, "end": end, "final_speaker": speaker,
            "final_confidence": confidence, "text": text}


class ConfidentTranscriptTests(unittest.TestCase):
    def test_uses_separate_target_and_other_thresholds(self):
        self.assertTrue(classify(row("Target_Speaker", .35))[0])
        self.assertFalse(classify(row("Target_Speaker", .349))[0])
        self.assertTrue(classify(row("SPEAKER_03", .65))[0])
        self.assertFalse(classify(row("SPEAKER_03", .649))[0])

    def test_uncertain_overlap_and_empty_text_are_reviewed(self):
        cases = [
            (row("Uncertain", 1), "uncertain_identity"),
            (row("Overlapping_Speakers", 1), "overlapping_speakers"),
            (row("Target_Speaker", 1, " "), "empty_text"),
        ]
        for value, reason in cases:
            with self.subTest(reason=reason):
                self.assertEqual(classify(value), (False, reason))

    def test_target_like_secondary_track_is_quarantined(self):
        self.assertEqual(
            classify(row("SPEAKER_03", 1.0), ambiguous_target_tracks={"SPEAKER_03"}),
            (False, "ambiguous_target_like_track"),
        )

    def test_export_preserves_every_row_in_one_output(self):
        payload = {"target_candidate": "SPEAKER_04",
                   "cluster_voice_means": {"SPEAKER_04": .42, "SPEAKER_02": .08},
                   "segments": [
            row("Target_Speaker", .5, "Target line"),
            row("SPEAKER_02", .8, "Other line", 2, 3),
            row("Uncertain", .1, "Review me", 3, 4),
            row("Overlapping_Speakers", 0, "Two people", 4, 5),
        ]}
        with tempfile.TemporaryDirectory() as directory:
            summary = export_transcript(payload, directory)
            confident = Path(directory, "confident_transcript.txt").read_text()
            review = json.loads(Path(directory, "review_segments.json").read_text())
        self.assertEqual(summary["included_segments"], 2)
        self.assertEqual(summary["review_segments"], 2)
        self.assertIn("Target line", confident)
        self.assertIn("Other line", confident)
        self.assertEqual([item["baseline_index"] for item in review], [2, 3])
        self.assertFalse(summary["baseline_modified"])

    def test_export_detects_globally_ambiguous_target_track(self):
        payload = {
            "target_candidate": "SPEAKER_04",
            "cluster_voice_means": {
                "SPEAKER_04": .415, "SPEAKER_03": .261, "SPEAKER_02": .056,
            },
            "segments": [
                row("SPEAKER_03", 1.0, "Could be target"),
                row("SPEAKER_02", 1.0, "Clearly other", 2, 3),
            ],
        }
        with tempfile.TemporaryDirectory() as directory:
            summary = export_transcript(payload, directory)
            review = json.loads(Path(directory, "review_segments.json").read_text())
        self.assertEqual(summary["ambiguous_target_tracks"], ["SPEAKER_03"])
        self.assertEqual(summary["included_segments"], 1)
        self.assertEqual(review[0]["disposition"], "ambiguous_target_like_track")

    def test_timestamp_supports_long_videos(self):
        self.assertEqual(timestamp(3661.25), "01:01:01.250")


if __name__ == "__main__":
    unittest.main()
