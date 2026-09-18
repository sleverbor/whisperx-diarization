import unittest

from evaluate_diaper_overlap import evaluate


def evidence(overlap):
    return ([{"source": "overlapping_speakers", "details": {
        "target_and_non_target": True}}] if overlap else [])


class DiaPerOverlapTest(unittest.TestCase):
    def test_maps_speakers_and_counts_overlap_without_mutating_baseline(self):
        baseline = {
            "target_candidate": "SPEAKER_04",
            "segments": [
                {"start": 0.0, "end": 1.0, "text": "target",
                 "baseline": {"raw_speaker_track": "SPEAKER_04"},
                 "final_speaker": "Target_Speaker", "evidence": evidence(False)},
                {"start": 1.0, "end": 2.0, "text": "other",
                 "baseline": {"raw_speaker_track": "SPEAKER_03"},
                 "final_speaker": "SPEAKER_03", "evidence": evidence(False)},
                {"start": 2.0, "end": 3.0, "text": "both",
                 "baseline": {"raw_speaker_track": "SPEAKER_04"},
                 "final_speaker": "Overlapping_Speakers", "evidence": evidence(True)},
            ],
        }
        original = repr(baseline)
        rttm = [
            {"start": 0.0, "end": 1.0, "speaker": "dia0"},
            {"start": 1.0, "end": 2.0, "speaker": "dia1"},
            {"start": 2.0, "end": 3.0, "speaker": "dia0"},
            {"start": 2.2, "end": 2.8, "speaker": "dia1"},
        ]
        result = evaluate(baseline, rttm)
        self.assertEqual(repr(baseline), original)
        self.assertEqual(result["summary"]["target_diaper_speaker"], "dia0")
        self.assertEqual(result["summary"]["diaper_corroborated_overlap_segments"], 1)
        self.assertEqual(result["summary"]["diaper_target_plus_other_segments"], 1)
        self.assertEqual(result["summary"]["diaper_overlap_on_control_segments"], 0)


if __name__ == "__main__":
    unittest.main()
