import unittest

from build_repeat_generalization_review import page


class RepeatGeneralizationReviewTests(unittest.TestCase):
    def test_page_has_clear_labels_and_stable_notes(self):
        item = {"review_id": "run-01", "run_id": "run", "left_start": 1.0,
                "left_end": 2.0, "right_start": 10.0, "right_end": 11.0,
                "left_text": "one", "right_text": "two",
                "left_video": "left.mp4", "right_video": "right.mp4",
                "text_similarity": .75}
        html = page([item])
        self.assertIn("similar_words_different_event", html)
        self.assertIn("same_dialogue_partial", html)
        self.assertIn("repeat-generalization-labels.json", html)
        self.assertIn("addEventListener('input'", html)


if __name__ == "__main__":
    unittest.main()
