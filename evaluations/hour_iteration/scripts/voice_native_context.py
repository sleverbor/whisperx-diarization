import json,subprocess
from pathlib import Path
import numpy as np,torch
from speechbrain.inference.speaker import SpeakerRecognition
p=Path('outputs/hour_iteration/contextual_scenes');torch.set_num_threads(2)
model=SpeakerRecognition.from_hparams(source='speechbrain/spkrec-ecapa-voxceleb',savedir='/home/think/projects/whisperx_diarization/pretrained_models/spkrec-ecapa-voxceleb',run_opts={'device':'cpu'})
def unit(v):
 v=np.asarray(v,dtype=np.float32).reshape(-1);return v/max(np.linalg.norm(v),1e-12)
target=unit(np.mean([unit(v) for v in np.load('/home/think/projects/whisperx_diarization/references/youtube-v1/voice_embeddings.npy')],axis=0));officer=unit(np.load('outputs/speaker_separation/officer_reference.npy'));windows=json.loads((p/'height_weight_windows.json').read_text());rows=[]
for w in windows:
 raw=subprocess.check_output(['ffmpeg','-nostdin','-hide_banner','-loglevel','error','-ss',str(w['start']),'-i','/home/think/projects/whisperx_diarization/video.mp4','-t',str(w['end']-w['start']),'-vn','-ar','16000','-ac','1','-f','f32le','pipe:1']);audio=np.frombuffer(raw,dtype='<f4').copy()
 for s in w['segments']:
  scores={};hypothesis='Uncertain'
  if s['end']-s['start']>=.4:
   with torch.no_grad():v=unit(model.encode_batch(torch.from_numpy(audio[round((s['start']-w['start'])*16000):round((s['end']-w['start'])*16000)]).unsqueeze(0)).numpy())
   scores={'Target_Speaker':float(v@target),'Opening_officer_reference':float(v@officer)};ranked=sorted(scores,key=scores.get,reverse=True)
   if scores[ranked[0]]>=.25 and scores[ranked[0]]-scores[ranked[1]]>=.08:hypothesis=ranked[0]
  rows.append({'window_start':w['start'],'start':s['start'],'end':s['end'],'text':s['text'],'scores':scores,'speaker_hypothesis':hypothesis,'review_required':True})
(p/'native_word_boundary_voice.json').write_text(json.dumps(rows,indent=2)+'\n')
for r in rows:print(round(r['start'],2),r['speaker_hypothesis'],r['scores'],r['text'],flush=True)
