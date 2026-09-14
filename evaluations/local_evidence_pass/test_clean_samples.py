"""Nonoverlapping clean sample screening; separate candidate profile only."""
import json
from pathlib import Path
import numpy as np
import torch
import soundfile as sf
from speechbrain.inference.speaker import SpeakerRecognition

def unit(x):
 x=np.asarray(x,dtype=np.float32).reshape(-1);return x/np.linalg.norm(x)
p=Path('outputs/local_evidence_pass');audio,sr=sf.read('outputs/caption_test/cuffs_original.wav',dtype='float32');tracks=json.loads(Path('outputs/handcuff_separation/scene_diarization.json').read_text());activity=json.loads(Path('outputs/speech_detection/silero_intervals.json').read_text())['0.5']
model=SpeakerRecognition.from_hparams(source='speechbrain/spkrec-ecapa-voxceleb',savedir='/home/think/projects/whisperx_diarization/pretrained_models/spkrec-ecapa-voxceleb',run_opts={'device':'cpu'});torch.set_num_threads(4)
profiles={}
for name in sorted({r['speaker'] for r in tracks}):
 samples=[];vectors=[]
 for t in tracks:
  if t['speaker']!=name:continue
  left=t['start']
  while t['end']-left>=1.2:
   right=min(left+2.4,t['end']);duration=right-left
   speech=sum(max(0,min(right,r['end'])-max(left,r['start'])) for r in activity)/duration
   overlap=sum(max(0,min(right,r['end'])-max(left,r['start'])) for r in tracks if r['speaker']!=name)/duration
   if speech>=.8 and overlap<=.1:
    with torch.no_grad():v=unit(model.encode_batch(torch.from_numpy(audio[int((left-1078)*sr):int((right-1078)*sr)]).unsqueeze(0)).flatten().numpy())
    samples.append({'start':left,'end':right,'speech_fraction':speech,'other_track_overlap_fraction':overlap});vectors.append(v)
   left=right
 kept=[];loo=[]
 if len(vectors)>=3:
  matrix=np.array(vectors);similarity=matrix@matrix.T;medoid=int(np.argmax(np.median(similarity,axis=1)));kept=[i for i in range(len(vectors)) if similarity[medoid,i]>=.45]
  if len(kept)>=3 and len(kept)>len(vectors)/2:
   loo=[float(vectors[i]@unit(np.mean([vectors[j] for j in kept if j!=i],axis=0))) for i in kept]
 accepted=len(kept)>=3 and len(kept)>len(vectors)/2 and bool(loo) and min(loo)>=.4
 profiles[name]={'samples':samples,'retained_indices':kept,'leave_one_out_similarities':loo,'coherent_candidate':accepted,'does_not_establish_person_identity':True}
 if accepted:np.save(p/('candidate_'+name+'.npy'),unit(np.mean([vectors[i] for i in kept],axis=0)))
(p/'clean_sample_review.json').write_text(json.dumps(profiles,indent=2)+'\n');print(json.dumps(profiles,indent=2),flush=True)
