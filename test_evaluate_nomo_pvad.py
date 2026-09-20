import unittest

from evaluate_nomo_pvad import (active_runs, chunk_times, confusion,
    interval_probabilities, summarize)


class NomoPvadEvaluationTests(unittest.TestCase):
    def test_interval_selects_overlapping_chunks(self):
        times = chunk_times(10, 4)
        self.assertEqual(interval_probabilities([.1,.2,.3,.4], times, 10.17, 10.47), [.2,.3])

    def test_summary_threshold_fractions(self):
        row = summarize([.4,.6,.8])
        self.assertAlmostEqual(row["mean"], .6, places=6)
        self.assertAlmostEqual(row["fraction_at_or_above"]["0.7"], 1/3)

    def test_confusion_ignores_mixed_cases(self):
        rows = [{"label":"target","summary":{"mean":.8}},
                {"label":"non_target","summary":{"mean":.2}},
                {"label":"mixed","summary":{"mean":.9}}]
        result = confusion(rows, .7)
        self.assertEqual(result["true_positive"], 1)
        self.assertEqual(result["true_negative"], 1)
        self.assertEqual(result["false_positive"], 0)

    def test_active_runs_require_consecutive_chunks(self):
        times = chunk_times(5, 5)
        runs = active_runs([.8,.2,.6,.7,.8], times, .5, 2)
        self.assertEqual(len(runs), 1)
        self.assertAlmostEqual(runs[0]["start"], 5.32)
        self.assertEqual(runs[0]["chunks"], 3)


if __name__ == "__main__": unittest.main()
