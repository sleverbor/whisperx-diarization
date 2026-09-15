import unittest
from review_transcript_regions import select_review_regions


class RegionTests(unittest.TestCase):
    def test_good_long_segment_is_not_selected(self):
        rows = [{"start": 1, "end": 4, "final_speaker": "Target_Speaker",
                 "final_confidence": .9}]
        self.assertEqual(select_review_regions(rows, 5), [])

    def test_uncertain_short_and_gap_are_selected_without_changing_rows(self):
        rows = [{"start": 5, "end": 5.5, "final_speaker": "Uncertain",
                 "final_confidence": .1},
                {"start": 20, "end": 24, "final_speaker": "SPEAKER_04",
                 "final_confidence": .8}]
        snapshot = [dict(row) for row in rows]
        result = select_review_regions(rows, 30, context=1, minimum_gap=5)
        self.assertEqual(rows, snapshot)
        self.assertTrue(any("uncertain_speaker" in row["reasons"] for row in result))
        self.assertTrue(any("transcript_gap" in row["reasons"] for row in result))

    def test_windows_are_bounded_and_external_controls_are_data(self):
        rows = [{"start": 1, "end": 99, "final_speaker": "Uncertain",
                 "final_confidence": 0}]
        result = select_review_regions(rows, 100, context=0, minimum_gap=200,
                                       maximum_window=30, overlap=4,
                                       extra_regions=[(40, 50)])
        self.assertTrue(all(0 < row["end"]-row["start"] <= 30 for row in result))
        self.assertTrue(any("external_review_control" in row["reasons"] for row in result))

    def test_invalid_region_is_rejected(self):
        with self.assertRaises(ValueError):
            select_review_regions([], 10, extra_regions=[(9, 11)])


if __name__ == "__main__":
    unittest.main()
