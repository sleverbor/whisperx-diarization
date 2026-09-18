import unittest

from build_overlap_label_set import categorize


class LabelSetSelectionTest(unittest.TestCase):
    def test_selects_each_disagreement_category(self):
        base = {"baseline_index": 0, "start": 0.0, "end": 1.0, "text": "x",
                "baseline_speaker": "x"}
        pairs = [
            (False, .5, .5, "both_on_control"),
            (True, .5, 0, "sortformer_only_on_baseline_overlap"),
            (False, .5, 0, "sortformer_only_on_control"),
            (False, 0, .5, "diaper_only_on_control"),
            (True, 0, 0, "neither_on_baseline_overlap"),
        ]
        sortformer, diaper = [], []
        for index, (expected, sf, dp, _) in enumerate(pairs):
            row = {**base, "baseline_index": index, "start": float(index),
                   "baseline_target_non_target_overlap": expected, "overlap_fraction": sf}
            sortformer.append(row)
            diaper.append({**row, "overlap_fraction": dp})
        selected = categorize(sortformer, diaper)
        self.assertEqual({row["selection_category"] for row in selected},
                         {item[3] for item in pairs})


if __name__ == "__main__":
    unittest.main()
