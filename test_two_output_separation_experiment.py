import unittest
import numpy as np
from run_two_output_separation_experiment import crop_stream

class TwoOutputSeparationTest(unittest.TestCase):
 def test_crop_stream_maps_context_to_original_interval(self):
  wave=np.arange(100,dtype=np.float32)
  self.assertEqual(crop_stream(wave,10,5,7,9).tolist(),list(range(20,40)))
if __name__=='__main__':unittest.main()
