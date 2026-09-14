import json
from pathlib import Path
p=Path('outputs/hour_iteration/filter_profiles');rows=json.loads((p/'scores.json').read_text())
labels={'height_weight':{25:'Target_Speaker',26:'Opening_officer_reference',27:'Target_Speaker',28:'Opening_officer_reference',29:'Target_Speaker',30:'Target_Speaker'},'handcuffs':{3:'Other',4:'Target_Speaker',5:'Target_Speaker',6:'Target_Speaker',7:'Other',8:'Other',10:'Target_Speaker',11:'Target_Speaker',12:'Opening_officer_reference',13:'Opening_officer_reference',14:'Other',15:'Other',16:'Other'}}
reviewed={3:'Scene_key_officer',4:'Target_Speaker',5:'Target_Speaker',6:'Target_Speaker',7:'Scene_key_officer',8:'Scene_key_officer',10:'Target_Speaker',11:'Target_Speaker',12:'Scene_original_officer',13:'Scene_original_officer'}
def resolve(scores):
 ranked=sorted(scores,key=scores.get,reverse=True);a,b=ranked[:2];return a if scores[a]>=.25 and scores[a]-scores[b]>=.08 else 'Uncertain'
summaries=[]
for filter in sorted({r['filter'] for r in rows}):
 for mode in ['independent_opening_reference','reviewed_scene_reference_diagnostic']:
  evaluated=[]
  for r in rows:
   if r['filter']!=filter:continue
   if mode=='independent_opening_reference':
    expected=labels.get(r['scene'],{}).get(r['index']);profiles=['Target_Speaker','Opening_officer_reference']
   else:
    if r['scene']!='handcuffs' or r['overlaps_reviewed_seed']:continue
    expected=reviewed.get(r['index']);profiles=['Target_Speaker','Scene_original_officer','Scene_key_officer']
   if expected is None:continue
   scores={name:r['scores'][name] for name in profiles};prediction=resolve(scores)
   evaluated.append({'scene':r['scene'],'index':r['index'],'expected':expected,'hypothesis':prediction,'scores':scores})
  known=[r for r in evaluated if r['expected']!='Other'];other=[r for r in evaluated if r['expected']=='Other']
  summaries.append({'filter':filter,'mode':mode,'known_evaluated':len(known),'known_correct':sum(r['hypothesis']==r['expected'] for r in known),'known_wrong':sum(r['hypothesis'] not in ('Uncertain',r['expected']) for r in known),'other_evaluated':len(other),'false_known_on_other':sum(r['hypothesis']!='Uncertain' for r in other),'rows':evaluated})
(p/'evaluation.json').write_text(json.dumps({'limitations':['Small manually reviewed set; no calibrated accuracy claim','Reviewed scene seeds are not automatic speaker identification','No filter thresholds tuned to this clip'],'labels_evaluation_only':labels,'summaries':summaries},indent=2)+'\n')
for r in summaries: print(r['filter'],r['mode'],r['known_correct'],r['known_wrong'],r['false_known_on_other'])
