import json,subprocess,sys
from pathlib import Path
import numpy as np,torch
sys.path.insert(0,'outputs/hour_iteration/code')
from review_audio_window import decoder_sentence_bounds,resolve_voice
from speechbrain.inference.speaker import SpeakerRecognition
p=Path('outputs/hour_iteration/timing_sources');p.mkdir(parents=True,exist_ok=True);torch.set_num_threads(2)
root=Path('/home/think/projects/whisperx_diarization');model=SpeakerRecognition.from_hparams(source='speechbrain/spkrec-ecapa-voxceleb',savedir=str(root/'pretrained_models/spkrec-ecapa-voxceleb'),run_opts={'device':'cpu'})
def unit(v):
 v=np.asarray(v,dtype=np.float32).reshape(-1);return v/max(np.linalg.norm(v),1e-12)
profiles={'Target_Speaker':unit(np.mean([unit(v) for v in np.load(root/'references/youtube-v1/voice_embeddings.npy')],axis=0)),'Opening_officer':unit(np.load('outputs/speaker_separation/officer_reference.npy'))}
rows=[]
for start in (665,640):
 path=Path(f'outputs/hour_iteration/runner_test_{start}')
 if not (path/'alignment.json').exists():continue
 asr=json.loads((path/'asr.json').read_text());align=json.loads((path/'alignment.json').read_text());bounds=decoder_sentence_bounds(asr['segments'],align['segments']);audio=np.frombuffer(subprocess.check_output(['ffmpeg','-nostdin','-hide_banner','-loglevel','error','-ss',str(start),'-i',str(root/'video.mp4'),'-t','25','-vn','-ar','16000','-ac','1','-f','f32le','pipe:1']),dtype='<f4').copy()
 for index,(s,b) in enumerate(zip(align['segments'],bounds)):
  results={}
  for name,crop in [('alignment',(s['start'],s['end'])),('decoder',b)]:
   if crop is None:continue
   a,z=crop;scores={}
   if z-a>=.4:
    with torch.no_grad():v=unit(model.encode_batch(torch.from_numpy(audio[round(a*16000):round(z*16000)]).unsqueeze(0)).numpy())
    scores={k:float(v@r) for k,r in profiles.items()}
   results[name]={'start':start+a,'end':start+z,'scores':scores,'hypothesis':resolve_voice(scores)}
  rows.append({'window_start':start,'index':index,'text':s['text'],'timing_sources':results})
(p/'comparison.json').write_text(json.dumps(rows,indent=2)+'\n')
for r in rows:
 if r['window_start']==665 and 'found guilty' in r['text'] or r['window_start']==665 and 'I do have' in r['text'] or r['window_start']==665 and 'I have' in r['text'] or r['window_start']==665 and 'fake information' in r['text']:print(r,flush=True)
