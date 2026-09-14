"""Crop-timing sensitivity diagnostic; fixed shifts, no identity/text tuning."""
import json,subprocess,time
from pathlib import Path
import numpy as np,torch
from speechbrain.inference.speaker import SpeakerRecognition
p=Path('outputs/hour_iteration/jitter_voice');p.mkdir(parents=True,exist_ok=True);torch.set_num_threads(3);start=time.monotonic()
root=Path('/home/think/projects/whisperx_diarization');model=SpeakerRecognition.from_hparams(source='speechbrain/spkrec-ecapa-voxceleb',savedir=str(root/'pretrained_models/spkrec-ecapa-voxceleb'),run_opts={'device':'cpu'})
def unit(v):
 v=np.asarray(v,dtype=np.float32).reshape(-1);return v/max(np.linalg.norm(v),1e-12)
def encode(a):
 with torch.no_grad():return unit(model.encode_batch(torch.from_numpy(a.copy()).unsqueeze(0)).numpy())
def resolve(scores):
 a,b=sorted(scores,key=scores.get,reverse=True)[:2];return a if scores[a]>=.25 and scores[a]-scores[b]>=.08 else 'Uncertain'
target=unit(np.mean([unit(v) for v in np.load(root/'references/youtube-v1/voice_embeddings.npy')],axis=0));officer=unit(np.load('outputs/speaker_separation/officer_reference.npy'));rows=[]
for scene,offset,duration in [('height_weight',625,65),('handcuffs',1078,90)]:
 audio=np.frombuffer(subprocess.check_output(['ffmpeg','-nostdin','-hide_banner','-loglevel','error','-ss',str(offset),'-i',str(root/'video.mp4'),'-t',str(duration),'-vn','-ar','16000','-ac','1','-f','f32le','pipe:1']),dtype='<f4').copy();profiles={'Target_Speaker':target,'Opening_officer_reference':officer};reviewed_profiles={'Target_Speaker':target}
 if scene=='handcuffs':
  seeds={'Scene_original_officer':[(1112.30596875,1115.84971875)],'Scene_key_officer':[(1133.72034375,1135.91409375),(1136.97721875,1139.94721875)]}
  for name,times in seeds.items():reviewed_profiles[name]=unit(np.mean([encode(audio[round((a-offset)*16000):round((b-offset)*16000)]) for a,b in times],axis=0))
 segments=json.loads(Path('outputs/hour_iteration/complete_voice',scene+'_hypotheses.json').read_text())
 for s in segments:
  if s['end']-s['start']<.4:continue
  observations=[]
  for shift in (-.2,-.1,0,.1,.2):
   a=max(0,s['start']-offset+shift);b=min(duration,s['end']-offset+shift)
   if b-a<.4:continue
   v=encode(audio[round(a*16000):round(b*16000)]);scores={k:float(v@r) for k,r in profiles.items()};review_scores={k:float(v@r) for k,r in reviewed_profiles.items()} if len(reviewed_profiles)>1 else {}
   observations.append({'shift_seconds':shift,'scores':scores,'hypothesis':resolve(scores),'reviewed_reference_scores':review_scores,'reviewed_reference_hypothesis':resolve(review_scores) if review_scores else None})
  center=next((r for r in observations if r['shift_seconds']==0),None)
  stable={}
  for radius in (.1,.2):
   selected=[r for r in observations if abs(r['shift_seconds'])<=radius];labels={r['hypothesis'] for r in selected};review_labels={r['reviewed_reference_hypothesis'] for r in selected}
   stable[str(radius)]={'hypothesis':next(iter(labels)) if len(labels)==1 and len(selected)>=3 else 'Uncertain','reviewed_reference_hypothesis':next(iter(review_labels)) if len(review_labels)==1 and len(selected)>=3 else 'Uncertain'}
  rows.append({'scene':scene,'index':s['index'],'start':s['start'],'end':s['end'],'text':s['text'],'center_hypothesis':center['hypothesis'] if center else None,'stability_hypotheses':stable,'observations':observations})
 (p/'observations.json').write_text(json.dumps(rows,indent=2)+'\n');print('Finished jitter scene',scene,'elapsed',round(time.monotonic()-start),flush=True)
print('Jitter complete',flush=True)
