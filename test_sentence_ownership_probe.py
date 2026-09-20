import unittest

from sentence_ownership_probe import (conclude_sentence, crop_variants,
    resolve_scores, stable_resolution, word_runs)


class SentenceOwnershipProbeTests(unittest.TestCase):
    def test_word_runs_and_long_gap_preserve_boundary(self):
        words = [
            {"word": "We", "start": 1.0, "end": 1.1, "speaker": "other"},
            {"word": "can", "start": 1.12, "end": 1.3, "speaker": "other"},
            {"word": "test", "start": 3.3, "end": 3.5, "speaker": "target"},
            {"word": "that", "start": 3.52, "end": 3.7, "speaker": "target"},
        ]
        runs, gaps = word_runs(words)
        self.assertEqual([run["text"] for run in runs], ["We can", "test that"])
        self.assertAlmostEqual(max(gap["seconds"] for gap in gaps), 2.0)

    def test_text_coherence_cannot_bridge_acoustic_boundary(self):
        runs = [
            {"track": "other", "resolution": "unresolved"},
            {"track": "target", "resolution": "target"},
        ]
        gaps = [{"seconds": 1.9}]
        self.assertEqual(conclude_sentence(runs, gaps, "target", "target", .8, ("target",)), "mixed")

    def test_all_runs_and_full_span_target(self):
        runs = [{"track": "wrong_cluster", "resolution": "target"}]
        self.assertEqual(conclude_sentence(runs, [{"seconds": .05}], "target", "target"), "target")

    def test_competing_voice_makes_mixed(self):
        runs = [{"track": "target", "resolution": "target"},
                {"track": "other", "resolution": "non_target"}]
        self.assertEqual(conclude_sentence(runs, [], "target", "unresolved"), "mixed")

    def test_voice_gate_and_variant_instability(self):
        self.assertEqual(resolve_scores({"Target_Speaker": .31, "Other": .1}), "target")
        self.assertEqual(resolve_scores({"Target_Speaker": .26, "Other": .22}), "unresolved")
        variants = [{"scores": {"Target_Speaker": .31, "Other": .1}},
                    {"scores": {"Target_Speaker": .20, "Other": .21}}]
        self.assertEqual(stable_resolution(variants), "unresolved")

    def test_crop_variants_never_cross_media_bounds(self):
        self.assertEqual(crop_variants(.02, .5, 10)[0], (.02, .5))
        self.assertTrue(all(0 <= left < right <= 10 for left, right in crop_variants(.02, .5, 10)))


if __name__ == "__main__":
    unittest.main()
