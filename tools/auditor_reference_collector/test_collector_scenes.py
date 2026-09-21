import unittest
from auditor_collector_engine import channel_videos_url,video_entries,matched_scenes
class SceneTests(unittest.TestCase):
    def test_channel_landing_page_becomes_video_listing(self):
        self.assertEqual(channel_videos_url('https://www.youtube.com/@HONORYOUROATH'),'https://www.youtube.com/@HONORYOUROATH/videos')
    def test_nested_tabs_are_flattened_and_duplicate_ids_removed(self):
        data={'id':'channel','entries':[{'id':'tab','entries':[{'id':'abcdefghijk'},{'id':'abcdefghijk'},{'id':'12345678901'}]}]}
        self.assertEqual([x['id'] for x in video_entries(data,3)],['abcdefghijk','12345678901'])
    def test_other_face_or_missing_face_breaks_scene(self):
        observations=[{'time':i*2,'matched':ok,'similarity':.6} for i,ok in enumerate([True,True,True,False,True,True,True])]
        scenes=matched_scenes(observations,14)
        self.assertEqual(len(scenes),2)
        self.assertLessEqual(scenes[0]['end'],5)
        self.assertGreaterEqual(scenes[1]['start'],7)
    def test_isolated_matching_frame_is_not_a_scene(self):
        self.assertEqual(matched_scenes([{'time':2,'matched':True,'similarity':.6}],20),[])
if __name__=='__main__':unittest.main()
