import unittest
from run_mossformer2_corroborated_asr import (
    classify_cross_audio, longest_common_words, surrounding_prompt, token_f1,
)


class CorroboratedAsrTest(unittest.TestCase):
    def test_overlap_is_order_aware_for_review(self):
        self.assertEqual(longest_common_words("I do not consent", "I don't consent"), ["i", "consent"])
        self.assertGreater(token_f1("give me your ID", "give your ID"), .7)

    def test_strong_requires_two_ordered_words(self):
        row = classify_cross_audio("I do not consent", "I do not consent")
        self.assertEqual(row["level"], "strong_cross_audio_corroboration")
        row = classify_cross_audio("yes", "yes")
        self.assertNotEqual(row["level"], "strong_cross_audio_corroboration")

    def test_prompt_excludes_current_segment(self):
        segments = [{"text": "before"}, {"text": "secret current"}, {"text": "after"}]
        prompt = surrounding_prompt(segments, 1)
        self.assertIn("before", prompt)
        self.assertIn("after", prompt)
        self.assertNotIn("secret current", prompt)


if __name__ == "__main__":
    unittest.main()
