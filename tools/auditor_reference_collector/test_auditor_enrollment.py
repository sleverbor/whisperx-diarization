import unittest
from pathlib import Path
from tempfile import TemporaryDirectory
import numpy as np
from auditor_enrollment import voice_reference,approved_short,decide,export_reference,compare_voice,rank_candidates,youtube_url
from auditor_collector_engine import speech_chunks
from types import SimpleNamespace as S


def record(duration=3,decision=None,index=0):
    return {'id':str(index),'duration':duration,'voice_decision':decision,'face_decision':None,
            'voice_vector':[1.]+[0.]*191,'face_vector':[1.]+[0.]*511,'source_url':'https://www.youtube.com/watch?v=abcdefghijk',
            'source_start':index*4,'source_end':index*4+duration,'text':'No.','audio':'clips/sample.wav','clip':'clips/sample.mp4'}


class EnrollmentTests(unittest.TestCase):
    def test_prediction_never_enrolls(self):
        r=record();r['voice_similarity']=.99
        self.assertEqual(voice_reference([r]),[])
        self.assertEqual(approved_short([r]),[])
    def test_separate_face_and_voice_approval(self):
        r=decide(record(),'no','yes','normal')
        self.assertEqual(voice_reference([r]),[])
        self.assertEqual(r['face_decision'],'yes')
    def test_short_voice_is_not_main_reference(self):
        r=decide(record(.8),'yes','not_visible','quiet','No')
        self.assertEqual(voice_reference([r]),[])
        self.assertEqual(len(approved_short([r])),1)
    def test_rejected_and_unsure_do_not_rank_as_pending(self):
        rows=[record(decision=x,index=i) for i,x in enumerate(['yes','no','unsure',None])]
        self.assertEqual([r['id'] for r in rank_candidates(rows,rows)],['3'])
    def test_export_uses_only_approved_long_voice_and_faces(self):
        rows=[decide(record(index=i),'yes','yes','normal') for i in range(3)]
        rows+=[decide(record(.8,index=4),'yes','not_visible','quiet','No'),record(index=5)]
        with TemporaryDirectory() as d:
            out=Path(d)/'reference';audit=export_reference({'seeds':[],'candidates':rows},out)
            self.assertEqual(np.load(out/'voice_embeddings.npy').shape,(3,192))
            self.assertEqual(audit['voice_sources'],['0','1','2'])
            self.assertEqual(audit['short_library_ids'],['4'])
            with self.assertRaises(FileExistsError):export_reference({'seeds':[],'candidates':rows},out)
    def test_comparison_returns_scores_without_identity_or_enrollment(self):
        rows=[decide(record(.8),'yes','not_visible','normal','No')]
        result=compare_voice(rows[0]['voice_vector'],rows,'NO!')
        self.assertIsNone(result['identity_decision'])
        self.assertEqual(len(result['short_matches']),1)
        self.assertEqual(compare_voice(rows[0]['voice_vector'],rows,'yes')['short_matches'],[])
    def test_only_youtube_https_urls(self):
        for url in ['http://youtube.com','https://youtube.com.evil.test','file:///tmp/input','https://me:secret@youtube.com']:
            with self.assertRaises(ValueError):youtube_url(url)
    def test_long_asr_segments_are_bounded(self):
        words=[S(start=i,end=i+1,word=' word') for i in range(20)]
        chunks=speech_chunks([S(start=0,end=20,text='words',words=words)])
        self.assertEqual(len(chunks),3)
        self.assertTrue(all(x['end']-x['start']<=8 for x in chunks))

if __name__=='__main__':unittest.main()
