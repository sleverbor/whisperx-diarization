import unittest
from export_reviewed_conservative_transcript import export


class ReviewedTranscriptTests(unittest.TestCase):
    def test_human_non_target_review_promotes_and_mixed_stays_withheld(self):
        evidence = {"segments": [
            {"start": 1, "end": 2, "text": "confirmed", "final_speaker": "S1", "final_confidence": .4},
            {"start": 3, "end": 4, "text": "mixed", "final_speaker": "S2", "final_confidence": .4},
        ]}
        review = [{"baseline_index": 0, "disposition": "below_confidence_threshold"},
                  {"baseline_index": 1, "disposition": "below_confidence_threshold"}]
        labels = [
            {"type":"target_candidate", "review_id":"a", "start":1, "end":2,
             "Speaker identity":"non_target", "Machine wording":"correct"},
            {"type":"target_candidate", "review_id":"b", "start":3, "end":4,
             "Speaker identity":"mixed_or_overlapping", "Machine wording":"correct"},
        ]
        published, appendix = export(evidence, review, labels)
        self.assertEqual([x["text"] for x in published], ["confirmed"])
        self.assertEqual([x["text"] for x in appendix], ["mixed"])


if __name__ == "__main__": unittest.main()
