import unittest
from build_separation_review_set import choose_exchanges, build_html

class SeparationReviewSetTest(unittest.TestCase):
    def test_selects_spread_of_confirmed_target_overlaps(self):
        rows=[{"start":i,"overlap_label":"true_overlap","target_in_overlap":"yes","notes":"known"} for i in range(9)]
        chosen=choose_exchanges(rows,5)
        self.assertEqual([x["start"] for x in chosen],[0,2,4,6,8])
    def test_rejects_unusable_rows_and_builds_focus_safe_ui(self):
        rows=[{"start":0,"overlap_label":"unclear_or_noisy","target_in_overlap":"yes","notes":"x"},{"start":1,"overlap_label":"true_overlap","target_in_overlap":"yes","notes":""}]
        self.assertEqual(choose_exchanges(rows,5),[])
        html=build_html([])
        self.assertIn("separation-labels.json",html)
        self.assertNotIn("oninput=render",html)

if __name__=="__main__":unittest.main()
