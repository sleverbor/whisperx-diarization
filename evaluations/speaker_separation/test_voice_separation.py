"""Frozen text/timing voice-reference comparison. No lexical identity rules."""
import json
from pathlib import Path
import numpy as np
import torch
import soundfile as sf
from speechbrain.inference.speaker import SpeakerRecognition

def unit(v):
 v=np.asarray(v,dtype=np.float32).reshape(-1);return v/np.linalg.norm(v)
root=Path('outputs/speaker_separation');cache=next(p for p in Path('work/kaggle_latest/checkpoints/stage-cache').iterdir() if p.name.startswith('aa70'))
opening=json.loads(Path('work/kaggle_latest/results/opening_evidence.json').read_text())['segments'];officer=[];sources=[]
for s in opening:
 if s['final_speaker']=='SPEAKER_01' and s['final_confidence']>=.65 and s['end']-s['start']>=1.2:
  file=cache/f"voice_{int(s['start']*16000)}_{int(s['end']*16000)}.json"
  if file.exists():officer.append(unit(json.loads(file.read_text())));sources.append({'start':s['start'],'end':s['end'],'track':s['final_speaker'],'checkpoint':file.name})
assert len(officer)>=3
priors=np.load('/home/think/projects/whisperx_diarization/references/youtube-v1/voice_embeddings.npy');target=unit(np.mean([unit(v) for v in priors],axis=0));other=unit(np.mean(officer,axis=0))
profiles={'Target_Speaker':target,'Opening_officer_reference':other}
np.save(root/'officer_reference.npy',other)
(root/'reference_provenance.json').write_text(json.dumps({'officer_sources':sources,'officer_target_similarity':float(other@target),'selection':'Non-target opening track, saved strength >=0.65, duration >=1.2 seconds; no phrase selection','thresholds':{'minimum_crop_seconds':.4,'minimum_similarity':.25,'minimum_margin':.08},'scores_are_calibrated':False},indent=2)+'\n')
audio,sr=sf.read('outputs/timing_diagnostic/last_25s.wav',dtype='float32');assert sr==16000
aligned=json.loads(Path('outputs/context_alignment/complete_local_alignment.json').read_text());snapshot=json.dumps(aligned,sort_keys=True)
model=SpeakerRecognition.from_hparams(source='speechbrain/spkrec-ecapa-voxceleb',savedir='/home/think/projects/whisperx_diarization/pretrained_models/spkrec-ecapa-voxceleb',run_opts={'device':'cpu'})
torch.set_num_threads(4);rows=[]
for s in aligned['segments']:
 duration=s['end']-s['start'];scores={};label='Uncertain';reason='Too short for reliable embedding'
 if duration>=.4:
  crop=audio[int(s['start']*sr):int(s['end']*sr)]
  with torch.no_grad():v=unit(model.encode_batch(torch.from_numpy(crop).unsqueeze(0)).flatten().numpy())
  scores={name:float(v@p) for name,p in profiles.items()};ranked=sorted(scores,key=scores.get,reverse=True);margin=scores[ranked[0]]-scores[ranked[1]]
  if scores[ranked[0]]>=.25 and margin>=.08:label=ranked[0];reason='Tentative voice-reference match'
  else:reason='Weak or conflicting reference matches; may be another voice'
 rows.append({'start':665+s['start'],'end':665+s['end'],'text':s['text'],'speaker_hypothesis':label,'voice_similarities':scores,'reason':reason,'review_required':True})
 print(rows[-1],flush=True)
assert json.dumps(aligned,sort_keys=True)==snapshot
(root/'speaker_hypotheses.json').write_text(json.dumps(rows,indent=2)+'\n')
(root/'transcript.txt').write_text('\n'.join(f"[{r['start']:.2f}–{r['end']:.2f}] {r['speaker_hypothesis']}: {r['text']}" for r in rows)+'\n')
