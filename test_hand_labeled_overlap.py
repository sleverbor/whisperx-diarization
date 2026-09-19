import json
import tempfile
import unittest
from pathlib import Path
from evaluate_hand_labeled_overlap import evaluate

class HandLabeledOverlapTest(unittest.TestCase):
    def test_excludes_unclear_and_scores_detectors(self):
        rows = [
            {"overlap_label":"true_overlap","sortformer_overlap_fraction":.5,"diaper_overlap_fraction":0},
            {"overlap_label":"rapid_turn_boundary","sortformer_overlap_fraction":0,"diaper_overlap_fraction":.4},
            {"overlap_label":"unclear_or_noisy","sortformer_overlap_fraction":1,"diaper_overlap_fraction":1},
        ]
        with tempfile.TemporaryDirectory() as d:
            p=Path(d)/"labels.json"; p.write_text(json.dumps({"labels":rows}))
            result=evaluate(p)
        self.assertEqual(result["usable"],2)
        self.assertEqual(result["sortformer_any"]["tp"],1)
        self.assertEqual(result["diaper_any"]["fp"],1)

if __name__ == "__main__": unittest.main()
