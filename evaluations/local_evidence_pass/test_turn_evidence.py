"""Sliding voice evidence and a reviewed anonymous profile; immutable baseline."""
import json
from pathlib import Path
from collections import defaultdict
import numpy as np
import torch
import soundfile as sf
from speechbrain.inference.speaker import SpeakerRecognition

def unit(x):
 x=np.asarray(x,dtype=np.float32).reshape(-1);return x/np.linalg.norm(x)
p=Path('outputs/local_evidence_pass');audio,sr=sf.read('outputs/caption_test/cuffs_original.wav',dtype='float32');assert sr==16000
activity=json.loads(Path('outputs/speech_detection/silero_intervals.json').read_text())['0.5'];tracks=json.loads(Path('outputs/handcuff_separation/scene_diarization.json').read_text());aligned=json.loads(Path('outputs/handcuff_separation/local_alignment.json').read_text());frozen=json.dumps(aligned,sort_keys=True)
prior=np.load('/home/think/projects/whisperx_diarization/references/youtube-v1/voice_embeddings.npy');target=unit(np.mean([unit(v) for v in prior],axis=0));officer=unit(np.load('outputs/speaker_separation/officer_reference.npy'))
model=SpeakerRecognition.from_hparams(source='speechbrain/spkrec-ecapa-voxceleb',savedir='/home/think/projects/whisperx_diarization/pretrained_models/spkrec-ecapa-voxceleb',run_opts={'device':'cpu'});torch.set_num_threads(4)
cache={}
def embed(start,end):
 key=(round(start,6),round(end,6))
 if key not in cache:
  crop=audio[max(0,int((start-1078)*sr)):min(len(audio),int((end-1078)*sr))]
  with torch.no_grad():cache[key]=unit(model.encode_batch(torch.from_numpy(crop).unsqueeze(0)).flatten().numpy())
 return cache[key]
def speech_fraction(start,end):return min(1.,sum(max(0,min(end,r['end'])-max(start,r['start'])) for r in activity)/(end-start))
samples=defaultdict(list)
for s in tracks:
 duration=s['end']-s['start'];overlap=sum(max(0,min(s['end'],r['end'])-max(s['start'],r['start'])) for r in tracks if r['speaker']!=s['speaker'])
 if duration>=1.2 and overlap/duration<=.2 and speech_fraction(s['start'],s['end'])>=.7:samples[s['speaker']].append(dict(s,vector=embed(s['start'],s['end'])))
profile_reviews={};accepted={}
for track,items in samples.items():
 vectors=[s['vector'] for s in items];centroid=unit(np.mean(vectors,axis=0));loo=[float(v@unit(np.mean([x for j,x in enumerate(vectors) if j!=i],axis=0))) for i,v in enumerate(vectors)] if len(items)>=3 else []
 known={'Target_Speaker':float(centroid@target),'Opening_officer_reference':float(centroid@officer)}
 accept=len(items)>=3 and min(loo)>=.4 and max(known.values())<.25
 profile_reviews[track]={'samples':[{k:v for k,v in s.items() if k!='vector'} for s in items],'leave_one_out_similarities':loo,'known_reference_similarities':known,'accepted_for_experiment':accept}
 if accept:accepted[track]=items
(p/'anonymous_profile_review.json').write_text(json.dumps(profile_reviews,indent=2)+'\n')
# Sliding windows are local speaker evidence, not immutable word identities.
observations=[]
for start in np.arange(1090,1148.8,.4):
 start=float(start);end=start+1.2
 if speech_fraction(start,end)<.5:continue
 v=embed(start,end);scores={'Target_Speaker':float(v@target),'Opening_officer_reference':float(v@officer)};supports={}
 for track,items in accepted.items():
  independent=[s['vector'] for s in items if s['end']<=start or s['start']>=end]
  supports[track]=len(independent)
  if len(independent)>=2:scores['Scene_'+track]=float(v@unit(np.mean(independent,axis=0)))
 ranked=sorted(scores,key=scores.get,reverse=True);margin=scores[ranked[0]]-scores[ranked[1]]
 label=ranked[0] if scores[ranked[0]]>=.25 and margin>=.08 else 'Uncertain'
 observations.append({'start':start,'end':end,'speech_fraction':speech_fraction(start,end),'voice_similarities':scores,'hypothesis':label,'margin':margin,'independent_profile_counts':supports})
(p/'sliding_voice_evidence.json').write_text(json.dumps(observations,indent=2)+'\n')
words=[]
for segment in aligned:
 for w in segment.get('words',[]):
  if 'start' not in w or 'end' not in w:continue
  start=1078+w['start'];end=1078+w['end'];center=(start+end)/2
  candidates=[r for r in observations if r['start']<=center<=r['end']]
  votes=defaultdict(int)
  for r in candidates:votes[r['hypothesis']]+=1
  ranked=sorted(votes,key=votes.get,reverse=True);label='Uncertain'
  if ranked and ranked[0]!='Uncertain' and votes[ranked[0]]>=2 and votes[ranked[0]]/len(candidates)>=2/3:label=ranked[0]
  words.append({'start':start,'end':end,'word':w['word'],'speaker_hypothesis':label,'voice_window_votes':dict(votes),'speech_fraction':speech_fraction(max(1090,start-.1),min(1150,max(end,start+.01)+.1)),'review_required':True})
runs=[]
for w in words:
 if runs and runs[-1]['speaker_hypothesis']==w['speaker_hypothesis'] and w['start']-runs[-1]['end']<.7:
  runs[-1]['end']=max(runs[-1]['end'],w['end']);runs[-1]['words'].append(w)
 else:runs.append({'start':w['start'],'end':w['end'],'speaker_hypothesis':w['speaker_hypothesis'],'words':[w]})
for r in runs:r['text']=' '.join(w['word'] for w in r['words']);r['review_required']=True
(p/'word_evidence.json').write_text(json.dumps(words,indent=2)+'\n');(p/'review_runs.json').write_text(json.dumps(runs,indent=2)+'\n')
(p/'review_transcript.txt').write_text('\n'.join(f"[{r['start']:.2f}–{r['end']:.2f}] {r['speaker_hypothesis']}: {r['text']}" for r in runs)+'\n')
assert json.dumps(aligned,sort_keys=True)==frozen
print('Profiles',profile_reviews,flush=True)
for r in runs:print(r['start'],r['speaker_hypothesis'],r['text'],flush=True)
