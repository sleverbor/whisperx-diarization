import json
from pathlib import Path
p=Path('outputs/hour_iteration/jitter_voice');rows=json.loads((p/'observations.json').read_text())
expected={'height_weight':{25:'Target_Speaker',26:'Opening_officer_reference',27:'Target_Speaker',28:'Opening_officer_reference',29:'Target_Speaker',30:'Target_Speaker'},'handcuffs':{3:'Other',4:'Target_Speaker',5:'Target_Speaker',6:'Target_Speaker',7:'Other',8:'Other',10:'Target_Speaker',11:'Target_Speaker',12:'Opening_officer_reference',13:'Opening_officer_reference',14:'Other',15:'Other',16:'Other'}}
reviewed={3:'Scene_key_officer',4:'Target_Speaker',5:'Target_Speaker',6:'Target_Speaker',7:'Scene_key_officer',8:'Scene_key_officer',10:'Target_Speaker',11:'Target_Speaker',12:'Scene_original_officer',13:'Scene_original_officer'}
summaries=[]
for mode in ['independent_reference','reviewed_reference_diagnostic']:
 for radius in (0,.1,.2):
  evaluated=[]
  for r in rows:
   label=expected.get(r['scene'],{}).get(r['index']) if mode=='independent_reference' else reviewed.get(r['index']) if r['scene']=='handcuffs' else None
   if label is None:continue
   if radius==0:
    if mode=='independent_reference':prediction=r['center_hypothesis']
    else:prediction=next((v['reviewed_reference_hypothesis'] for v in r['observations'] if v['shift_seconds']==0),'Uncertain')
   else:prediction=r['stability_hypotheses'][str(radius)]['hypothesis' if mode=='independent_reference' else 'reviewed_reference_hypothesis']
   evaluated.append({'scene':r['scene'],'index':r['index'],'expected':label,'hypothesis':prediction})
  known=[r for r in evaluated if r['expected']!='Other'];other=[r for r in evaluated if r['expected']=='Other']
  summaries.append({'mode':mode,'jitter_radius_seconds':radius,'known_evaluated':len(known),'known_correct':sum(r['expected']==r['hypothesis'] for r in known),'known_wrong':sum(r['hypothesis'] not in ('Uncertain',r['expected']) for r in known),'other_evaluated':len(other),'false_known_on_other':sum(r['hypothesis']!='Uncertain' for r in other),'rows':evaluated})
(p/'evaluation.json').write_text(json.dumps({'evaluation_only_listening_roles':expected,'reviewed_reference_seed_roles_not_automatic':True,'summaries':summaries},indent=2)+'\n')
for r in summaries:print(r['mode'],r['jitter_radius_seconds'],r['known_correct'],r['known_wrong'],r['false_known_on_other'])
for r in rows:
 if r['scene']=='height_weight' and r['index'] in [25,27,29,30]:print(r['index'],r['center_hypothesis'],r['stability_hypotheses'])
