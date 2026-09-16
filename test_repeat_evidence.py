import unittest

from chainofrules import Baseline, TimelineSegment
from repeat_evidence import find_repeat_groups, build_repeat_proposals


def segment(start, text, speaker="Uncertain", confidence=0.0):
    value = TimelineSegment(start, start + 1.0, text, Baseline("S0", "S0"))
    value.final_speaker = speaker
    value.final_confidence = confidence
    return value


class RepeatEvidenceTests(unittest.TestCase):
    def test_detects_ordered_repeated_presentation(self):
        first = [
            segment(0, "They can ask you to get off their property."),
            segment(10, "Are you going to leave the property?"),
            segment(20, "We will arrest you on the property if you do not leave."),
            segment(30, "Where does the property begin?"),
        ]
        second = [
            segment(100, "They can ask you to get off their property."),
            segment(110, "Are you going to leave the property?"),
            segment(120, "We are going to arrest you on the property if you do not leave."),
            segment(130, "Where does the property begin here?"),
        ]
        groups = find_repeat_groups(first + second)
        self.assertEqual(len(groups), 1)
        self.assertEqual(len(groups[0]["anchors"]), 4)
        self.assertAlmostEqual(groups[0]["offset_seconds"], 100.0)

    def test_bracketed_corruption_receives_donor_candidate(self):
        first = [
            segment(0, "They can ask you to get off their property."),
            segment(10, "I'm a garbled person."),
            segment(20, "We will arrest you on the property if you do not leave."),
            segment(30, "Where does the property begin?"),
            segment(40, "My name is Matthew Cox."),
        ]
        second = [
            segment(100, "They can ask you to get off their property."),
            segment(110, "I am engaged constitutionally.", "Target_Speaker", 0.9),
            segment(120, "We are going to arrest you on the property if you do not leave."),
            segment(130, "Where does the property begin here?"),
            segment(140, "My name is Matthew Cox."),
        ]
        timeline = first + second
        groups = find_repeat_groups(timeline)
        proposals = build_repeat_proposals(timeline, groups)
        self.assertEqual(proposals[1][0]["donor_text"], "I am engaged constitutionally.")
        self.assertEqual(proposals[1][0]["donor_final_speaker"], "Target_Speaker")
        self.assertEqual(first[1].text, "I'm a garbled person.")

    def test_isolated_repeated_phrase_is_not_a_presentation(self):
        timeline = [
            segment(0, "Have a nice day."),
            segment(60, "Have a nice day."),
            segment(120, "Something unrelated happened here."),
        ]
        self.assertEqual(find_repeat_groups(timeline), [])


if __name__ == "__main__":
    unittest.main()
