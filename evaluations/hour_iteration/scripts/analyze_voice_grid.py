import json
from pathlib import Path
from collections import defaultdict
import numpy as np
p=Path('outputs/hour_iteration');rows=json.loads((p/'voice_grid.json').read_text());vectors=np.load('work/hour_iteration/voice_grid_vectors.npz');labels={1:'Target_Speaker',2:'Target_Speaker',4:'Other',5:'Other',6:'Other',7:'Opening_officer_reference',9:'Opening_officer_reference',11:'Other',12:'Other',13:'Other',14:'Other'}
(p/'evaluation_labels.json').write_text(json.dumps({'basis':'User listening notes; aligned utterance boundaries are approximate; labels are evaluation only, never model prompts','utterance_labels':labels,'excluded_indices':[0,3,8,10]},indent=2)+'\n')
variants=defaultdict(list)
for r in rows:
 if r['kind']=='utterance':variants[(r['signal'],r['mode'])].append(r)
report=[]
for (signal,mode),items in variants.items():
 result=[]
 for r in items:
  index=r['segment_index']
  if index not in labels:continue
  scores=r['scores'];ranked=sorted(scores,key=scores.get,reverse=True);prediction=ranked[0] if scores[ranked[0]]>=.25 and scores[ranked[0]]-scores[ranked[1]]>=.08 else 'Uncertain'
  result.append({'segment_index':index,'expected':labels[index],'hypothesis':prediction,'scores':scores})
 known=[r for r in result if r['expected']!='Other'];other=[r for r in result if r['expected']=='Other'];summary={'signal':signal,'mode':mode,'correct_known':sum(r['expected']==r['hypothesis'] for r in known),'known_evaluated':len(known),'wrong_known':sum(r['hypothesis']!='Uncertain' and r['expected']!=r['hypothesis'] for r in known),'false_known_on_other':sum(r['hypothesis']!='Uncertain' for r in other),'other_evaluated':len(other),'rows':result};report.append(summary)
 print({k:v for k,v in summary.items() if k!='rows'})
(p/'voice_grid_evaluation.json').write_text(json.dumps(report,indent=2)+'\n')
# A cluster of same-identity examples must separate from other identity examples.
comparisons=[]
for signal in ['raw','light','full']:
 for mode in ['complete','voiced_only']:
  key_vectors={i:vectors[f'utterance_{i}_{signal}_{mode}'] for i in [4,6,11,12,13,14] if f'utterance_{i}_{signal}_{mode}' in vectors}
  officer={i:vectors[f'utterance_{i}_{signal}_{mode}'] for i in [7,9] if f'utterance_{i}_{signal}_{mode}' in vectors}
  within=[float(a@b) for i,a in key_vectors.items() for j,b in key_vectors.items() if i<j];cross=[float(a@b) for a in key_vectors.values() for b in officer.values()]
  comparisons.append({'signal':signal,'mode':mode,'within_key_voice_mean':float(np.mean(within)),'cross_officer_mean':float(np.mean(cross)),'within_key_voice_range':[min(within),max(within)],'cross_officer_range':[min(cross),max(cross)]})
 print('Consistency',comparisons[-1])
(p/'profile_separation_evaluation.json').write_text(json.dumps(comparisons,indent=2)+'\n')
