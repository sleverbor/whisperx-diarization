import unittest
from recover_transcript_gaps import uncovered_intervals,recovery_windows,collect_candidates,preserve_baseline
class GapTests(unittest.TestCase):
 def test_union_handles_overlaps_and_edges(self):
  self.assertEqual(uncovered_intervals([{'start':2,'end':5},{'start':4,'end':8},{'start':12,'end':14}],17),[(0.,2.),(8.,12.),(14.,17)])
 def test_windows_cover_gap_without_tiny_tail(self):
  windows=recovery_windows((10,30.00001),40,size=8,overlap=4)
  self.assertTrue(all(0<r-l<=8.000001 for l,r in windows));self.assertEqual(windows[0][0],9.5);self.assertEqual(windows[-1][1],30.50001)
  self.assertTrue(all(b[0]<=a[1] for a,b in zip(windows,windows[1:])))
 def test_context_is_bounded_and_invalid_context_rejected(self):
  windows=recovery_windows((1,39),40,size=20,overlap=10,context=2)
  self.assertEqual(windows[0][0],0);self.assertEqual(windows[-1][1],40)
  for context in (-1,float('nan'),float('inf')):
   with self.assertRaises(ValueError):recovery_windows((10,30),40,context=context)
 def test_repetition_needs_distinct_windows_and_stays_in_gap(self):
  words=[{'start':4,'end':4.5,'word':'Height?','probability':.9,'window_index':0},{'start':4.1,'end':4.6,'word':'height','probability':.8,'window_index':1},{'start':1,'end':2,'word':'existing','probability':1,'window_index':1}]
  candidates=collect_candidates(words,(3,6));self.assertEqual(len(candidates),1);self.assertTrue(candidates[0]['repeated_in_overlapping_windows']);self.assertTrue(candidates[0]['review_required'])
 def test_candidate_does_not_splice_disagreeing_windows(self):
  words=[{'start':4,'end':4.2,'word':'one','probability':.9,'window_index':0},{'start':4.3,'end':4.5,'word':'answer','probability':.6,'window_index':0},{'start':4,'end':4.2,'word':'another','probability':.6,'window_index':1},{'start':4.3,'end':4.5,'word':'answer','probability':.9,'window_index':1}]
  candidates=collect_candidates(words,(3,6));self.assertEqual(len(candidates),1);self.assertEqual(len({w['window_index'] for w in candidates[0]['words']}),1)
 def test_zero_duration_word_kept_with_phrase_not_invented_timing(self):
  words=[{'start':4,'end':4.5,'word':' How','probability':.9,'window_index':0},{'start':4.5,'end':4.5,'word':' tall?','probability':.9,'window_index':0}]
  result=collect_candidates(words,(3,6));self.assertEqual(result[0]['text'],'How tall?');self.assertEqual(result[0]['words'][1]['start'],result[0]['words'][1]['end'])
  self.assertEqual(collect_candidates(words[1:],(3,6)),[])
 def test_keeps_baseline_nested_fields_and_identity_unchanged(self):
  original={'segments':[{'start':0,'end':1,'text':'works','final_speaker':'Target_Speaker','words':[{'word':'works'}],'reasons':['original']}]}
  result=preserve_baseline(original,[{'text':'new','speaker':'Uncertain'}]);self.assertEqual(result['segments'],original['segments']);result['segments'][0]['words'][0]['word']='mutated';self.assertEqual(original['segments'][0]['words'][0]['word'],'works')
if __name__=='__main__':unittest.main()
