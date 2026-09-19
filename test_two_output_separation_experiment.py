import unittest
import numpy as np
from types import SimpleNamespace
from run_two_output_separation_experiment import aligned_words,crop_stream,shifted_window

class TwoOutputSeparationTest(unittest.TestCase):
 def test_crop_stream_maps_context_to_original_interval(self):
  wave=np.arange(100,dtype=np.float32)
  self.assertEqual(crop_stream(wave,10,5,7,9).tolist(),list(range(20,40)))
 def test_shifted_window_preserves_interval(self):
  self.assertEqual(shifted_window(10,12,30,3,-1),(6,14))
  self.assertEqual(shifted_window(10,12,30,3,1),(8,16))
  with self.assertRaises(ValueError):shifted_window(10,12,30,3,4)
 def test_aligned_words_use_word_midpoints(self):
  segments=[SimpleNamespace(words=[SimpleNamespace(start=0,end=.4,word=' before'),SimpleNamespace(start=.8,end=1.2,word=' keep'),SimpleNamespace(start=1.8,end=2.2,word=' after')])]
  self.assertEqual(aligned_words(segments,.7,1.3),'keep')
if __name__=='__main__':unittest.main()
