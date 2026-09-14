import json
from pathlib import Path
from collections import defaultdict
from difflib import SequenceMatcher
root=Path('work/kaggle_latest');out=Path('outputs/timing_diagnostic')
clip=json.loads((root/'results/height_weight_evidence.json').read_text());full=json.loads((root/'results/full_video_evidence.json').read_text())
cache=next(p for p in (root/'checkpoints/stage-cache').iterdir() if p.name.startswith('bd352'));tracks=json.loads((cache/'diarization.json').read_text())
rows=[]
for s in clip['segments']:
 overlaps=defaultdict(float)
 for d in tracks:overlaps[d['speaker']]+=max(0,min(s['end'],d['end'])-max(s['start'],d['start']))
 candidates=[r for r in full['segments'] if 625<=r['start']<695]
 match=max(candidates,key=lambda r:SequenceMatcher(None,s['text'].lower(),r['text'].lower()).ratio())
 ratio=SequenceMatcher(None,s['text'].lower(),match['text'].lower()).ratio()
 rows.append({'clip_start':s['start'],'source_start':625+s['start'],'text':s['text'],'baseline_track':s['baseline']['raw_speaker_track'],'track_overlap_seconds':dict(overlaps),'closest_full_video_text':match['text'],'full_video_start':match['start'],'text_similarity':ratio,'start_difference_seconds':match['start']-(625+s['start'])})
(out/'saved_timing_audit.json').write_text(json.dumps({'clip_voice_means':clip['cluster_voice_means'],'diarization':tracks,'segments':rows},indent=2)+'\n')
for r in rows:print(round(r['source_start'],2),r['text'],r['track_overlap_seconds'],'matching full-video delta',round(r['start_difference_seconds'],2))
