"""Compare scene-local anonymous tracks and split saved aligned words by overlap."""
import json
from pathlib import Path
from collections import defaultdict
import numpy as np
import soundfile as sf
import torch
from speechbrain.inference.speaker import SpeakerRecognition

def unit(x):
 x=np.asarray(x,dtype=np.float32).reshape(-1);return x/np.linalg.norm(x)
p=Path('outputs/handcuff_separation');tracks=json.loads((p/'scene_diarization.json').read_text());aligned=json.loads((p/'local_alignment.json').read_text());audio,sr=sf.read('outputs/caption_test/cuffs_original.wav',dtype='float32');assert sr==16000
prior=np.load('/home/think/projects/whisperx_diarization/references/youtube-v1/voice_embeddings.npy');target=unit(np.mean([unit(v) for v in prior],axis=0));officer=unit(np.load('outputs/speaker_separation/officer_reference.npy'))
model=SpeakerRecognition.from_hparams(source='speechbrain/spkrec-ecapa-voxceleb',savedir='/home/think/projects/whisperx_diarization/pretrained_models/spkrec-ecapa-voxceleb',run_opts={'device':'cpu'});torch.set_num_threads(4)
samples=defaultdict(list)
for s in tracks:
 duration=s['end']-s['start']
 if duration<1.2:continue
 overlap=sum(max(0,min(s['end'],r['end'])-max(s['start'],r['start'])) for r in tracks if r['speaker']!=s['speaker'])
 if overlap/duration>.2:continue
 crop=audio[max(0,int((s['start']-1078)*sr)):min(len(audio),int((s['end']-1078)*sr))]
 with torch.no_grad():v=unit(model.encode_batch(torch.from_numpy(crop).unsqueeze(0)).flatten().numpy())
 samples[s['speaker']].append(v)
profiles={};mapping={}
for track in sorted({s['speaker'] for s in tracks}):
 vectors=samples[track];scores={}
 if vectors:
  centroid=unit(np.mean(vectors,axis=0));scores={'Target_Speaker':float(centroid@target),'Opening_officer_reference':float(centroid@officer)}
 label=track
 if scores:
  ranked=sorted(scores,key=scores.get,reverse=True)
  if scores[ranked[0]]>=.25 and scores[ranked[0]]-scores[ranked[1]]>=.08:label=ranked[0]
 profiles[track]={'clean_samples':len(vectors),'voice_similarities':scores,'tentative_label':label};mapping[track]=label
lines=[];runs=[]
for segment in aligned:
 words=segment.get('words',[])
 for w in words:
  if 'start' not in w or 'end' not in w:continue
  start=1078+w['start'];end=1078+w['end'];scores=defaultdict(float)
  for t in tracks:scores[t['speaker']]+=max(0,min(end,t['end'])-max(start,t['start']))
  ranked=sorted(scores,key=scores.get,reverse=True);speaker='Uncertain'
  if ranked and scores[ranked[0]]>0:
   # Nearly tied overlaps can be simultaneous speech or unstable segmentation.
   second=scores[ranked[1]] if len(ranked)>1 else 0
   if scores[ranked[0]]-second>=.03:speaker=ranked[0]
  if runs and runs[-1]['raw_track']==speaker and start-runs[-1]['end']<.7:
   runs[-1]['end']=max(end,runs[-1]['end']);runs[-1]['words'].append(w['word'])
  else:runs.append({'start':start,'end':end,'raw_track':speaker,'speaker_hypothesis':mapping.get(speaker,speaker),'words':[w['word']]})
for r in runs:r['text']=' '.join(r.pop('words'));r['review_required']=True
(p/'scene_profiles.json').write_text(json.dumps(profiles,indent=2)+'\n');(p/'scene_word_runs.json').write_text(json.dumps(runs,indent=2)+'\n')
(p/'scene_transcript.txt').write_text('\n'.join(f"[{r['start']:.2f}–{r['end']:.2f}] {r['speaker_hypothesis']} ({r['raw_track']}): {r['text']}" for r in runs)+'\n')
print('Profiles',profiles,flush=True)
for r in runs:print(r['start'],r['speaker_hypothesis'],r['text'],flush=True)
