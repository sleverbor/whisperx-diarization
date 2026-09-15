import unittest
from types import SimpleNamespace
from bounded_silero_vad import bounded_chunks

class ChunkTests(unittest.TestCase):
    def chunks(self, pairs, size=20):
        return bounded_chunks([SimpleNamespace(start=a, end=b) for a,b in pairs], size)
    def test_empty(self):
        self.assertEqual(self.chunks([]), [])
    def test_no_bridge_across_silence(self):
        self.assertEqual([(x['start'],x['end']) for x in self.chunks([(0,1.5),(1.6,13.6),(13.9,30)],30)], [(0,1.5),(1.6,13.6),(13.9,30)])
    def test_long_single_interval_is_capped_without_duplicates(self):
        r=self.chunks([(2,47)],20)
        self.assertEqual([(x['start'],x['end']) for x in r], [(2,22),(22,42),(42,47)])
    def test_overlapping_context_not_decoded_twice(self):
        r=self.chunks([(5,9),(1,6),(15,18)])
        self.assertEqual([(x['start'],x['end']) for x in r],[(1,9),(15,18)])
    def test_invalid(self):
        for pairs,size in [([(1,0)],20),([(-1,1)],20),([],0),([],float('nan'))]:
            with self.assertRaises(ValueError): self.chunks(pairs,size)

if __name__=='__main__': unittest.main()
