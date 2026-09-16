import unittest

from chainofrules import voice_mapping_confidence


class SingleSpeakerMappingTests(unittest.TestCase):
    def test_strong_well_sampled_single_track_can_map(self):
        self.assertAlmostEqual(
            voice_mapping_confidence(["S0"], {"S0": 0.304}, {"S0": 13}),
            (0.304 - 0.18) / 0.15,
        )

    def test_weak_or_under_sampled_single_track_stays_uncertain(self):
        self.assertEqual(voice_mapping_confidence(["S0"], {"S0": 0.18}, {"S0": 13}), 0.0)
        self.assertEqual(voice_mapping_confidence(["S0"], {"S0": 0.40}, {"S0": 2}), 0.0)

    def test_multi_track_behavior_still_uses_separation(self):
        self.assertAlmostEqual(
            voice_mapping_confidence(
                ["target", "other"],
                {"target": 0.319, "other": 0.055},
                {"target": 3, "other": 2},
            ),
            1.0,
        )


if __name__ == "__main__":
    unittest.main()
