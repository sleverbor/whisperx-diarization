import unittest

from run_sortformer_overlap import parse_segment, unwrap


class SortformerOutputTest(unittest.TestCase):
    def test_parses_documented_string_output(self):
        self.assertEqual(parse_segment("1.25 2.50 speaker_0"), (1.25, 2.5, "speaker_0"))

    def test_parses_sequence_and_unwraps_single_file_batch(self):
        self.assertEqual(parse_segment((1, 2, "speaker_1")), (1.0, 2.0, "speaker_1"))
        self.assertEqual(unwrap([["0 1 speaker_0"]]), ["0 1 speaker_0"])


if __name__ == "__main__":
    unittest.main()
