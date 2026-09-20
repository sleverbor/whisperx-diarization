import unittest
from evaluate_youtube_captions_on_priority import (
    caption_for_interval, timed_caption_words, token_f1,
)


class CaptionPriorityTest(unittest.TestCase):
    def test_word_offsets_and_append_events(self):
        data = {"events": [
            {"tStartMs": 1000, "segs": [{"utf8": "hello"}, {"utf8": " world", "tOffsetMs": 500}]},
            {"tStartMs": 1500, "aAppend": 1, "segs": [{"utf8": "\n"}]},
        ]}
        rows = timed_caption_words(data)
        self.assertEqual(rows, [{"time": 1.0, "text": "hello"}, {"time": 1.5, "text": " world"}])
        self.assertEqual(caption_for_interval(rows, 1.4, 1.6, padding=0), "world")

    def test_token_overlap(self):
        self.assertGreater(token_f1("I do not consent", "I don't consent"), .5)


if __name__ == "__main__":
    unittest.main()
