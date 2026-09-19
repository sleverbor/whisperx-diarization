import unittest
import numpy as np
from run_contextual_wesep_experiment import crop_context_output,token_f1

class ContextualWesepTest(unittest.TestCase):
    def test_crop_returns_original_interval(self):
        wave=np.arange(200,dtype=np.float32)
        self.assertEqual(crop_context_output(wave,10,5,7,9).tolist(),list(range(20,40)))
    def test_token_f1_distinguishes_target_and_other(self):
        self.assertEqual(token_f1("I am hard of hearing","I am hard of hearing"),1)
        self.assertEqual(token_f1("what protects you","I am hard of hearing"),0)

if __name__=="__main__":unittest.main()
