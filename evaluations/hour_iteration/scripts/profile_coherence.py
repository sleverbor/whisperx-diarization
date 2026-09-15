import json,numpy as np
from pathlib import Path
p=Path('outputs/hour_iteration');data=np.load(p/'complete_voice/vectors.npz');prior=np.load('/home/think/projects/whisperx_diarization/references/youtube-v1/voice_embeddings.npy')
def unit(v):
 v=np.asarray(v).reshape(-1);return v/np.linalg.norm(v)
prior=np.array([unit(v) for v in prior]);loo=[float(v@unit(np.mean(np.delete(prior,i,axis=0),axis=0))) for i,v in enumerate(prior)];independent=unit(np.load('outputs/speaker_separation/officer_reference.npy'))
# The independently reviewed seed-crop comparison is already in the filter
# diagnostic; these similarities inspect domain shift, not identity truth.
rows=json.load(open(p/'filter_profiles/scores.json'));raw=[r for r in rows if r['filter']=='raw' and r['scene']=='handcuffs'];selected=[r for r in raw if r['index'] in [3,9,12,14,15,16]]
result={'target_reference_samples':len(prior),'target_leave_one_out_cosines':loo,'target_reference_loo_min':min(loo),'target_reference_loo_median':float(np.median(loo)),'reviewed_scene_rows':selected,'notes':['Reference consistency is not transcription or speaker accuracy','Two officers share high scene-profile similarities despite the independent opening profile being weaker','Do not use transcript text or hand-reviewed roles to train automatic identity rules']}
(p/'profile_coherence.json').write_text(json.dumps(result,indent=2)+'\n');print('Target reference LOO',min(loo),np.median(loo))
