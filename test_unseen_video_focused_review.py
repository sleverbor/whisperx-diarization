import unittest
from pathlib import Path

from build_unseen_video_focused_review import assemble_items, html


RESULTS = Path("/home/think/Documents/Codex/2026-09-13/referenced-chatgpt-conversation-this-is-an/diarization-results-19")


class FocusedReviewTests(unittest.TestCase):
    def test_review_set(self):
        items = assemble_items(RESULTS)
        self.assertEqual(len(items), 15)
        counts = {kind: sum(x["type"] == kind for x in items) for kind in
                  ("target_candidate", "caption_gap", "mossformer2", "repeat_pair")}
        self.assertEqual(counts, {"target_candidate": 5, "caption_gap": 4,
                                  "mossformer2": 3, "repeat_pair": 3})
        target = next(x for x in items if abs(x.get("start", 0) - 425.95) < .1)
        self.assertIn("mossformer2", target)
        self.assertFalse(any(x.get("baseline_index") == 200 for x in items))
        self.assertTrue(all(x.get("baseline_text") for x in items if x["type"] == "mossformer2"))

    def test_page_instructions_and_download(self):
        page = html(assemble_items(RESULTS))
        self.assertIn("You do not need to watch the full video", page)
        self.assertIn("unseen-video-focused-labels.json", page)
        self.assertIn("Baseline transcript:", page)
        self.assertIn("addEventListener('input'", page)


if __name__ == "__main__":
    unittest.main()
