import unittest
from build_caption_assisted_mossformer2_review import html, merge_caption_evidence


class CaptionAssistedReviewTest(unittest.TestCase):
    def test_caption_is_provenance_limited(self):
        items = [{"review_id": "one", "start": 1.0, "end": 2.0}]
        report = {"caption_type": "youtube_automatic", "results": [{
            "review_id": "one", "caption_text": "candidate words",
            "relation": "similar_support",
        }]}
        merged = merge_caption_evidence(items, report)
        evidence = merged[0]["caption_evidence"]
        self.assertEqual(evidence["text"], "candidate words")
        self.assertFalse(evidence["speaker_identity_available"])
        self.assertFalse(evidence["automatic_text_insertion"])

    def test_page_explains_caption_limit(self):
        page = html([])
        self.assertIn("does not identify the speaker", page)
        self.assertIn("never changes the transcript automatically", page)


if __name__ == "__main__":
    unittest.main()
