"""Rebuild matching ECAPA references for each filter; reviewed seeds diagnostic only."""
import json,subprocess,time
from pathlib import Path
import numpy as np,torch
from scipy.signal import butter,sosfiltfilt
from speechbrain.inference.speaker import SpeakerRecognition
p=Path('outputs/hour_iteration/filter_profiles');p.mkdir(parents=True,exist_ok=True);torch.set_num_threads(3);started=time.monotonic()
root=Path('/home/think/projects/whisperx_diarization');refs=root/'references';model=SpeakerRecognition.from_hparams(source='speechbrain/spkrec-ecapa-voxceleb',savedir=str(root/'pretrained_models/spkrec-ecapa-voxceleb'),run_opts={'device':'cpu'})
def unit(v):
 v=np.asarray(v,dtype=np.float32).reshape(-1);return v/max(np.linalg.norm(v),1e-12)
def encode(a):
 with torch.no_grad():return unit(model.encode_batch(torch.from_numpy(np.asarray(a,dtype=np.float32).copy()).unsqueeze(0)).numpy())
def decode(path,start=None,duration=None):
 args=['ffmpeg','-nostdin','-hide_banner','-loglevel','error']
 if start is not None:args+=['-ss',str(start)]
 args+=['-i',str(path)]
 if duration is not None:args+=['-t',str(duration)]
 args+=['-vn','-ar','16000','-ac','1','-f','f32le','pipe:1'];return np.frombuffer(subprocess.check_output(args),dtype='<f4').copy()
manifest=json.loads((refs/'source-clips/sources.json').read_text())['sources'];sources={s['url']:decode(refs/'source-clips'/s['path']) for s in manifest};metadata=json.loads((refs/'youtube-v1/reference.json').read_text());opening=decode(root/'short.mp4');cuffs=decode(root/'video.mp4',1078,90);height=decode(root/'video.mp4',625,65)
seed_times={'Scene_original_officer':[(1112.30596875,1115.84971875)],'Scene_key_officer':[(1133.72034375,1135.91409375),(1136.97721875,1139.94721875)]}
officer_times=json.loads(Path('outputs/speaker_separation/reference_provenance.json').read_text())['officer_sources'];cuff_rows=json.loads(Path('outputs/hour_iteration/complete_voice/handcuffs_hypotheses.json').read_text());height_rows=json.loads(Path('outputs/hour_iteration/complete_voice/height_weight_hypotheses.json').read_text());rows=[]
filters={'raw':None,'highpass100':butter(4,100,btype='highpass',fs=16000,output='sos'),'band150_4000':butter(4,[150,4000],btype='bandpass',fs=16000,output='sos'),'band300_3000':butter(4,[300,3000],btype='bandpass',fs=16000,output='sos'),'lowpass2500':butter(4,2500,btype='lowpass',fs=16000,output='sos')}
for name,filter in filters.items():
 def signal(a):return a if filter is None else sosfiltfilt(filter,a).astype(np.float32)
 target_items=[]
 # Filter whole recording first, not each crop, to avoid edge transients.
 filtered_sources={k:signal(a) for k,a in sources.items()}
 for i in metadata['voice']['accepted_rows']:
  s=metadata['voice']['sample_provenance'][i];a=filtered_sources[s['source']];target_items.append(encode(a[int(s['start']*16000):int(s['end']*16000)]))
 o=signal(opening);c=signal(cuffs);h=signal(height)
 profiles={'Target_Speaker':unit(np.mean(target_items,axis=0)),'Opening_officer_reference':unit(np.mean([encode(o[int(s['start']*16000):int(s['end']*16000)]) for s in officer_times],axis=0))}
 for label,windows in seed_times.items():profiles[label]=unit(np.mean([encode(c[int((a-1078)*16000):int((b-1078)*16000)]) for a,b in windows],axis=0))
 for scene,segments,audio,offset in [('handcuffs',cuff_rows,c,1078),('height_weight',height_rows,h,625)]:
  for s in segments:
   if s['end']-s['start']<.4:continue
   v=encode(audio[int((s['start']-offset)*16000):int((s['end']-offset)*16000)]);scores={k:float(v@r) for k,r in profiles.items()};rows.append({'filter':name,'scene':scene,'index':s['index'],'start':s['start'],'end':s['end'],'text':s['text'],'scores':scores,'overlaps_reviewed_seed':scene=='handcuffs' and any(s['start']<b and s['end']>a for times in seed_times.values() for a,b in times)})
 (p/'scores.json').write_text(json.dumps(rows,indent=2)+'\n');print('Finished filter',name,'elapsed',round(time.monotonic()-started),flush=True)
(p/'provenance.json').write_text(json.dumps({'target_samples':len(target_items),'matching_references_rebuilt':True,'reviewed_seed_roles_not_automatic':True,'reviewed_seed_times':seed_times,'filters':list(filters),'models':'ECAPA','elapsed_seconds':time.monotonic()-started},indent=2)+'\n')
