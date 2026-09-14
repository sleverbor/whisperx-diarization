import unittest
import copy
import numpy as np
from collect_duplicate_evidence import best_waveform_matches,collect,SAMPLE_RATE

class DuplicateTests(unittest.TestCase):
    def setUp(self):
        self.rng=np.random.default_rng(17);self.a=self.rng.normal(size=SAMPLE_RATE)
    def test_gain_and_endpoint(self):
        for query,start in [(self.a*2,0),(np.r_[np.zeros(137),self.a*.5],137)]:
            m=best_waveform_matches(self.a,query)
            self.assertEqual(len(m),1);self.assertEqual(m[0]['sample_start'],start)
    def test_noise_not_same_recording(self):
        self.assertEqual(best_waveform_matches(self.a,self.rng.normal(size=3*SAMPLE_RATE)),[])
    def test_short_and_silent_rejected(self):
        self.assertEqual(best_waveform_matches(self.a[:400],self.a),[])
        self.assertEqual(best_waveform_matches(np.zeros(SAMPLE_RATE),self.a),[])
    def test_repeated_match_ambiguity_retained(self):
        self.assertEqual(len(best_waveform_matches(self.a,np.r_[self.a,np.zeros(1000),self.a])),2)
    def test_preservation_scoping_and_no_independent_enrollment(self):
        source={'segments':[{'start':0,'end':1,'text':'one','final_speaker':'SPEAKER_01','final_confidence':.7}]}
        baseline={'segments':[{'start':10,'end':11,'text':'two','raw_speaker_track':'SPEAKER_08','final_speaker':'SPEAKER_08','final_confidence':.3,'evidence':[{'original':True}]}]}
        snapshot=copy.deepcopy(baseline)
        out,m=collect(baseline,source,self.a,self.a,query_offset=10,source_name='opening')
        self.assertEqual(baseline,snapshot);self.assertTrue(m)
        clean=copy.deepcopy(out);clean['segments'][0].pop('supplemental_evidence')
        self.assertEqual(clean,snapshot)
        self.assertEqual(m[0]['speaker_hypothesis'],'opening:SPEAKER_01')
        self.assertFalse(m[0]['counts_as_independent_voice_sample'])
    def test_target_identity_kept_while_raw_track_ids_are_scoped(self):
        s={'segments':[{'start':0,'end':1,'final_speaker':'Target_Speaker'}]}
        _,m=collect({'segments':[]},s,self.a,self.a)
        self.assertEqual(m[0]['speaker_hypothesis'],'Target_Speaker')

if __name__=='__main__':unittest.main()
