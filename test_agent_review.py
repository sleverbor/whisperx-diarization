import json
from pathlib import Path
import tempfile
import unittest
from types import SimpleNamespace

from agent_review import add_anchor, add_annotation, add_comment, init_session, load, save, select_item


class AgentReviewTest(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory(); self.root=Path(self.temp.name); self.session=self.root/'session'
        source=self.root/'source.json'; source.write_text(json.dumps({"segments":[{"baseline_index":4,"start":10,"end":11,"text":"hello"},{"baseline_index":5,"start":20,"end":21,"text":"next"}]}))
        self.assertEqual(init_session(self.session,source,"video"),2)
        self.assertIn("do not need to repeat", load(self.session/'queue.json')[0]["review_prompt"].casefold())
    def tearDown(self): self.temp.cleanup()
    def test_comment_anchor_annotation_and_navigation(self):
        save(self.session/'player-state.json',{"media_time":10.4,"playing":True})
        observation=add_comment(self.session,"target says hello", "voice")
        self.assertEqual(observation["input_modality"], "voice")
        grounded=add_anchor(self.session,observation["observation_id"],"here")
        self.assertEqual(grounded["status"],"grounded")
        args=SimpleNamespace(observation_id=1,speaker="target",text="hello",overlap="yes",basis="voice,context",speaker_confidence=.9,word_confidence=.8)
        row=add_annotation(self.session,args); self.assertEqual(row["attribution_basis"],["voice","context"])
        select_item(self.session,1); self.assertEqual(load(self.session/'session.json')["current_index"],1)
    def test_start_and_end_are_both_required(self):
        save(self.session/'player-state.json',{"media_time":10.1}); obs=add_comment(self.session,"phrase")
        self.assertEqual(add_anchor(self.session,obs["observation_id"],"start")["status"],"needs_grounding")
        save(self.session/'player-state.json',{"media_time":10.8})
        self.assertEqual(add_anchor(self.session,obs["observation_id"],"end")["status"],"grounded")

if __name__ == "__main__": unittest.main()
