import unittest
from build_caption_gap_review import (
    duplicates_nearby_transcript, neighboring_segments, uncovered_groups,
)


class CaptionGapReviewTest(unittest.TestCase):
    def test_only_uncovered_words_are_grouped(self):
        words = [{"time": 1.0, "text": "covered"}, {"time": 3.0, "text": "new"}, {"time": 3.4, "text": "speech"}]
        segments = [{"start": .5, "end": 1.5, "text": "covered"}]
        groups = uncovered_groups(words, segments, coverage_padding=0, minimum_words=2)
        self.assertEqual([[row["text"] for row in group] for group in groups], [["new", "speech"]])

    def test_neighbors_do_not_overlap_candidate(self):
        segments = [{"start": 1, "end": 2, "text": "before"}, {"start": 5, "end": 6, "text": "after"}]
        before, after = neighboring_segments(segments, 3, 4)
        self.assertEqual(before["text"], "before")
        self.assertEqual(after["text"], "after")

    def test_caption_timing_duplicate_is_suppressed(self):
        before = {"text": "All right, thanks."}
        self.assertTrue(duplicates_nearby_transcript(
            ">> All right, thanks.", before, None
        ))
        self.assertFalse(duplicates_nearby_transcript(
            "I don't know how much clearer it gets", before, None
        ))


if __name__ == "__main__":
    unittest.main()
