import unittest

import torch

from ctc_candidate_rescore import (
    ctc_candidate_loss, encode_candidate, greedy_decode, normalize_ctc_text,
    rank_scores)


class CtcCandidateRescoreTests(unittest.TestCase):
    def test_normalization(self):
        self.assertEqual(normalize_ctc_text("I'm not panhandling."), "I'M NOT PANHANDLING")

    def test_encoding_uses_word_separator(self):
        normalized, ids = encode_candidate("A B", ("-", "|", "A", "B"))
        self.assertEqual(normalized, "A B")
        self.assertEqual(ids, [2, 1, 3])

    def test_ctc_prefers_matching_candidate(self):
        # Labels: blank, A, B. Three frames strongly encode A, blank, B.
        logits = torch.tensor([[0., 8., 0.], [8., 0., 0.], [0., 0., 8.]])
        log_probs = torch.log_softmax(logits, dim=-1)
        _, matching = ctc_candidate_loss(log_probs, [1, 2])
        _, wrong = ctc_candidate_loss(log_probs, [2, 1])
        self.assertLess(matching, wrong)

    def test_rank_scores_lower_is_better(self):
        rows = [{"normalized_ctc_loss": 2., "raw_ctc_loss": 4.},
                {"normalized_ctc_loss": 1., "raw_ctc_loss": 3.}]
        self.assertEqual(rank_scores(rows)[0]["rank"], 1)
        self.assertEqual(rank_scores(rows)[0]["normalized_ctc_loss"], 1.)

    def test_greedy_decode_collapses_repeats_and_blank(self):
        emissions = torch.tensor([[0., 5., 0.], [0., 5., 0.], [5., 0., 0.],
                                  [0., 0., 5.]])
        self.assertEqual(greedy_decode(emissions, ("-", "A", "B")), "AB")


if __name__ == "__main__": unittest.main()
