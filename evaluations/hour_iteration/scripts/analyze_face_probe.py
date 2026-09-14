import json
from pathlib import Path
import numpy as np
p=Path('outputs/hour_iteration/face_probe');rows=json.loads((p/'face_observations.json').read_text())
# Listening roles used only to inspect evidence coverage, never to label faces.
intervals=[('key_reflection',1087.11,1092.99),('target_agreement',1093.15,1095.53),('target_long_agreement',1098.21,1099.38),('key_stuck_key',1103.15,1109.43),('original_cuff_offer',1112.305,1115.85),('target_space',1121.94,1123.96),('original_reply',1124.16,1126.24),('key_new_key',1130.92,1139.84)]
result=[]
for name,a,b in intervals:
 items=[r for r in rows if a<=r['time']<=b];confident=[r for r in items if r['det_score']>=.7];times={r['time'] for r in confident};stats={}
 for track in {r['face_track'] for r in items}:
  v=[r['mouth_aperture'] for r in items if r['face_track']==track and r['mouth_aperture'] is not None]
  if v:stats[track]={'observations':len(v),'median_aperture':float(np.median(v)),'aperture_range':float(np.ptp(v))}
 result.append({'review_interval':name,'start':a,'end':b,'frames_with_confident_face':len(times),'estimated_sampled_frames':round((b-a)*4),'mouth_geometry':stats,'speaker_identity_inferred':False})
(p/'coverage_diagnostic.json').write_text(json.dumps({'intervals':result,'all_face_identity_clusters':len({r['face_track'] for r in rows}),'max_target_face_similarity':max(r['target_face_similarity'] for r in rows),'limitations':['Embedding-only clustering splits the visible key officer across poses','Original officer occludes the key officer later','Face presence and estimated aperture do not verify speech','4 Hz sampling and uncorrected head pose are insufficient for lip reading'],'promoted_to_resolver':False},indent=2)+'\n')
for r in result:print(r['review_interval'],r['frames_with_confident_face'],r['estimated_sampled_frames'],r['mouth_geometry'])
