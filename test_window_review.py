import unittest
from review_audio_window import baseline_evidence,resolve_voice,decoder_sentence_bounds,resolve_timing_evidence,decoder_quality_flags
class WindowTests(unittest.TestCase):
    def test_confident_beep_words_do_not_prove_speech(self):
        s=[{'start':0,'end':4.2,'no_speech_prob':.808,'avg_logprob':-.584,'words':[{'probability':.94}]}]
        flags,_=decoder_quality_flags(s,1,2)
        self.assertTrue(flags)
    def test_unrelated_decoder_warning_not_applied(self):
        flags,_=decoder_quality_flags([{'start':0,'end':1,'no_speech_prob':.9}],3,4)
        self.assertEqual(flags,[])
    def test_consistent_crops_with_one_qualified_match(self):
        self.assertEqual(resolve_timing_evidence([{'Target':.34,'Other':.18},{'Target':.27,'Other':.21}]),'Target')
    def test_conflicting_crops_do_not_choose_the_stronger_match(self):
        self.assertEqual(resolve_timing_evidence([{'Target':.7,'Other':.1},{'Target':.1,'Other':.3}]),'Uncertain')
    def test_two_weak_crops_do_not_accumulate_confidence(self):
        self.assertEqual(resolve_timing_evidence([{'Target':.2,'Other':.1},{'Target':.21,'Other':.1}]),'Uncertain')
    def test_decoder_words_follow_repeated_sentences_in_order(self):
        d=[{'text':'I agree. I agree.','words':[{'word':'I','start':0,'end':.1},{'word':'agree.','start':.1,'end':1},{'word':'I','start':2,'end':2.1},{'word':'agree.','start':2.1,'end':3}]}]
        a=[{'text':'I agree.'},{'text':'I agree.'}]
        self.assertEqual(decoder_sentence_bounds(d,a),[(0,1),(2,3)])
    def test_missing_word_links_do_not_invent_timings(self):
        self.assertEqual(decoder_sentence_bounds([{'text':'Hi.'}],[{'text':'Hi.'}]),[None])
    def test_raw_ids_preserved_and_offsets_applied(self):
        a=[{'start':0,'end':2,'raw_speaker_track':'SPEAKER_03','final_speaker':'Target_Speaker'},{'start':2,'end':4,'raw_speaker_track':'SPEAKER_07','final_speaker':'SPEAKER_07'}]
        result=baseline_evidence(a,101,103,offset=100)
        self.assertEqual(result['raw_track_overlap_seconds'],{'SPEAKER_03':1,'SPEAKER_07':1})
        self.assertEqual(a[0]['raw_speaker_track'],'SPEAKER_03')
    def test_current_pipeline_nested_baseline_schema(self):
        s=[{'start':0,'end':2,'baseline':{'raw_speaker_track':'SPEAKER_08','speaker':'SPEAKER_08'},'final_speaker':'SPEAKER_08'}]
        self.assertEqual(baseline_evidence(s,0,1)['raw_track_overlap_seconds'],{'SPEAKER_08':1})
    def test_no_observation_is_not_negative_identity_evidence(self):
        self.assertEqual(baseline_evidence([],0,2)['segments'],[])
        self.assertEqual(resolve_voice({}),'Uncertain')
    def test_short_response_cannot_borrow_questioners_identity(self):
        self.assertEqual(resolve_voice({'Target_Speaker':.22,'Officer':.21}),'Uncertain')
    def test_close_profiles_abstain_even_if_similarity_is_high(self):
        self.assertEqual(resolve_voice({'Target_Speaker':.6,'Other':.57}),'Uncertain')
    def test_strong_separated_profile_produces_review_hypothesis(self):
        self.assertEqual(resolve_voice({'Target_Speaker':.5,'Other':.1}),'Target_Speaker')
if __name__=='__main__':unittest.main()
