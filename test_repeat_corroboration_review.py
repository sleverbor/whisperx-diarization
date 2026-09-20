import unittest
from build_repeat_corroboration_review import differences, page


class CorroborationReviewTests(unittest.TestCase):
    def test_differences_preserve_alternatives(self):
        rows = differences("check the property", "shake the property")
        self.assertEqual(rows, [{"operation": "replace", "first": "check", "second": "shake"}])

    def test_prediction_is_hidden_from_page(self):
        item = {"review_id":"x", "left_video":"a.mp4", "right_video":"b.mp4",
                "left_text":"a", "right_text":"b", "differences":[],
                "blinded_quality_prediction":"second"}
        html = page([item])
        self.assertNotIn("blinded_quality_prediction", html)
        self.assertIn("repeat-corroboration-labels.json", html)


if __name__ == "__main__": unittest.main()
