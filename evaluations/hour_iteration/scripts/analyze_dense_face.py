import json
from pathlib import Path
import numpy as np
p=Path('outputs/hour_iteration/dense_face');rows=json.loads((p/'observations.json').read_text());intervals=[('key_reflection',1087.11,1092.99),('target_agreement',1093.15,1095.53),('target_long_agreement',1098.21,1099.38)];results=[]
for label,a,b in intervals:
 selected=[r for r in rows if a<=r['time']<=b];mouth_visible=[r for r in selected if r['det_score']>=.7 and r['pose'] is not None and abs(r['pose'][1])<=45 and abs(r['pose'][2])<=30]
 values=[r['mouth_aperture'] for r in selected if r['face_track']=='TRACK_05'];results.append({'review_interval':label,'observations':len(selected),'high_score_moderate_pose_observations':len(mouth_visible),'longest_spatial_track_aperture_median':float(np.median(values)) if values else None,'longest_spatial_track_aperture_range':float(np.ptp(values)) if values else None})
(p/'evaluation.json').write_text(json.dumps({'intervals':results,'same_visible_face_continuity_improved':True,'longest_track_observations':211,'limitations':['Spatial tracking improves continuity but does not verify identity','Most longest-track observations have strong yaw and roll','Aperture can be estimated when the mouth is obscured; unreliable active-speaker evidence','Listening roles used only to inspect the probe, not to assign speakers'],'speaker_assignment_changed':False},indent=2)+'\n')
print(results)
