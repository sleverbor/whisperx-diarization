"""Frozen recovered-text voice comparison; no captions/listening-note prompts."""
import json,re
from pathlib import Path
import numpy as np
import torch
import soundfile as sf
import whisperx
from speechbrain.inference.speaker import SpeakerRecognition

def unit(x):
 x=np.asarray(x,dtype=np.float32).reshape(-1);return x/np.linalg.norm(x)
p=Path('outputs/handcuff_separation');source=Path('outputs/caption_test/cuffs_audio_only/transcript_with_candidates.json');baseline=json.loads(source.read_text());saved=source.read_bytes()
audio,sr=sf.read('outputs/caption_test/cuffs_original.wav',dtype='float32');assert sr==16000
# Retain the saved candidate hypotheses, rather than supplying the manual reference.
# Align each candidate in its contextual crop and split sentences using WhisperX.
items=[]
for s in baseline['segments']:
 if 1089<=1078+s['start']<1148:items.append(dict(s,kind='baseline'))
for s in baseline['gap_recovery_candidates']:
 if 1090<=1078+s['start']<1141:items.append(dict(s,kind='recovered'))
items.sort(key=lambda r:r['start'])
model,metadata=whisperx.load_align_model(language_code='en',device='cpu');aligned=[]
for s in items:
 left=max(0,s['start']-.5);right=min(len(audio)/sr,s['end']+.5)
 r=whisperx.align([{'start':s['start']-left,'end':s['end']-left,'text':s['text']}],model,metadata,audio[int(left*sr):int(right*sr)],'cpu',return_char_alignments=False)
 for segment in r['segments']:
  segment['start']+=left;segment['end']+=left
  for w in segment.get('words',[]):
   for k in ('start','end'):
    if k in w:w[k]+=left
  aligned.append(dict(segment,kind=s['kind']))
(p/'local_alignment.json').write_text(json.dumps(aligned,indent=2)+'\n');del model
prior=np.load('/home/think/projects/whisperx_diarization/references/youtube-v1/voice_embeddings.npy');target=unit(np.mean([unit(v) for v in prior],axis=0));officer=unit(np.load('outputs/speaker_separation/officer_reference.npy'))
embedding=SpeakerRecognition.from_hparams(source='speechbrain/spkrec-ecapa-voxceleb',savedir='/home/think/projects/whisperx_diarization/pretrained_models/spkrec-ecapa-voxceleb',run_opts={'device':'cpu'});torch.set_num_threads(4)
rows=[];vectors={}
for index,s in enumerate(aligned):
 row={'start':1078+s['start'],'end':1078+s['end'],'text':s['text'],'kind':s['kind'],'speaker_hypothesis':'Uncertain','voice_similarities':{},'review_required':True}
 if s['end']-s['start']>=.4:
  crop=audio[int(s['start']*sr):int(s['end']*sr)]
  with torch.no_grad():v=unit(embedding.encode_batch(torch.from_numpy(crop).unsqueeze(0)).flatten().numpy())
  vectors[index]=v;scores={'Target_Speaker':float(v@target),'Opening_officer_reference':float(v@officer)};row['voice_similarities']=scores
  ranked=sorted(scores,key=scores.get,reverse=True)
  if scores[ranked[0]]>=.25 and scores[ranked[0]]-scores[ranked[1]]>=.08:row['speaker_hypothesis']=ranked[0]
 rows.append(row)
# Look for direct pairwise agreement among unmatched crops. Connected components
# are exploratory and do not assert that every member is one person.
unknown=[i for i in vectors if rows[i]['speaker_hypothesis']=='Uncertain' and rows[i]['end']-rows[i]['start']>=1.2]
pairs=[]
for pos,i in enumerate(unknown):
 for j in unknown[pos+1:]:
  similarity=float(vectors[i]@vectors[j]);pairs.append({'left_index':i,'right_index':j,'similarity':similarity})
(p/'unknown_voice_pairs.json').write_text(json.dumps(pairs,indent=2)+'\n')
(p/'voice_hypotheses.json').write_text(json.dumps(rows,indent=2)+'\n')
(p/'transcript.txt').write_text('\n'.join(f"[{r['start']:.2f}–{r['end']:.2f}] {r['speaker_hypothesis']}: {r['text']}" for r in rows)+'\n')
assert source.read_bytes()==saved
for i,r in enumerate(rows):print(i,r,flush=True)
print('Unknown pairs',sorted(pairs,key=lambda r:r['similarity'],reverse=True)[:8],flush=True)
