import json,time
from pathlib import Path
import numpy as np
import torch
import soundfile as sf
from speechbrain.inference.speaker import SpeakerRecognition

def unit(v):
 v=np.asarray(v,dtype=np.float32).reshape(-1);return v/max(np.linalg.norm(v),1e-12)
base=Path('work/hour_iteration');out=Path('outputs/hour_iteration');original,sr=sf.read('outputs/caption_test/cuffs_original.wav',dtype='float32');assert sr==16000
signals={'raw':original}
for name in ('light','full'):signals[name]=sf.read(base/'denoised_cuffs'/f'{name}.wav',dtype='float32')[0]
segments=json.loads(Path('outputs/handcuff_separation/local_alignment.json').read_text());activity=json.loads(Path('outputs/speech_detection/silero_intervals.json').read_text())['0.5'];priors=np.load('/home/think/projects/whisperx_diarization/references/youtube-v1/voice_embeddings.npy');target=unit(np.mean([unit(v) for v in priors],axis=0));officer=unit(np.load('outputs/speaker_separation/officer_reference.npy'))
model=SpeakerRecognition.from_hparams(source='speechbrain/spkrec-ecapa-voxceleb',savedir='/home/think/projects/whisperx_diarization/pretrained_models/spkrec-ecapa-voxceleb',run_opts={'device':'cpu'});torch.set_num_threads(4)
arrays={};observations=[]

def crop(signal,left,right,voiced=False):
 if not voiced:return signal[max(0,int(left*sr)):min(len(signal),int(right*sr))]
 chunks=[]
 for r in activity:
  a=max(left,r['start']-1078);b=min(right,r['end']-1078)
  if b>a:chunks.append(signal[int(a*sr):int(b*sr)])
 return np.concatenate(chunks) if chunks else np.array([],dtype=np.float32)

def encode(x):
 with torch.no_grad():return unit(model.encode_batch(torch.from_numpy(x).unsqueeze(0)).flatten().numpy())
start=time.monotonic()
for index,segment in enumerate(segments):
 for signal_name,signal in signals.items():
  for mode in ('complete','voiced_only'):
   x=crop(signal,segment['start'],segment['end'],mode=='voiced_only')
   if len(x)<6400:continue
   v=encode(x);key=f'utterance_{index}_{signal_name}_{mode}';arrays[key]=v
   scores={'Target_Speaker':float(v@target),'Opening_officer_reference':float(v@officer)};observations.append({'key':key,'kind':'utterance','segment_index':index,'start':1078+segment['start'],'end':1078+segment['end'],'text':segment['text'],'signal':signal_name,'mode':mode,'scores':scores,'samples':len(x)})
 print('Utterance',index+1,'/',len(segments),'elapsed',round(time.monotonic()-start),flush=True)
# Local duration sensitivity, evaluated separately from true person identity.
for signal_name,signal in signals.items():
 for duration in (.8,1.2,1.8,2.4,3.6):
  for center in np.arange(1090.5,1149.6,.5):
   left=center-duration/2-1078;right=center+duration/2-1078
   x=crop(signal,left,right,False)
   if len(x)<6400:continue
   # Speech activity only gates computation; it does not identify a person.
   speech=crop(signal,left,right,True)
   if len(speech)/len(x)<.5:continue
   v=encode(x);key=f'window_{signal_name}_{duration}_{center:.1f}';arrays[key]=v
   observations.append({'key':key,'kind':'window','center':float(center),'duration':duration,'signal':signal_name,'mode':'complete','start':1078+left,'end':1078+right,'scores':{'Target_Speaker':float(v@target),'Opening_officer_reference':float(v@officer)}})
  print('Windows',signal_name,duration,'elapsed',round(time.monotonic()-start),flush=True)
np.savez_compressed(base/'voice_grid_vectors.npz',**arrays)
(out/'voice_grid.json').write_text(json.dumps(observations,indent=2)+'\n');print('Complete',len(observations),'elapsed',time.monotonic()-start,flush=True)
