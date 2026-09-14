"""Contextual audio-only scene hypothesis comparison; baseline not modified."""
import json,time,subprocess
from pathlib import Path
import numpy as np
from faster_whisper import WhisperModel
p=Path('outputs/hour_iteration/contextual_scenes');p.mkdir(parents=True,exist_ok=True)
raw=subprocess.check_output(['ffmpeg','-nostdin','-hide_banner','-loglevel','error','-i','/home/think/projects/whisperx_diarization/video.mp4','-vn','-ar','16000','-ac','1','-f','f32le','pipe:1']);audio=np.frombuffer(raw,dtype='<f4').copy();sr=16000
model=WhisperModel('large-v2',device='cpu',compute_type='int8',cpu_threads=4);start=time.monotonic()
scenes=[('opening',0,30),('height_weight',620,695),('handcuffs',1078,1168),('replay',925,970)]
for name,left,right in scenes:
 existing=p/(name+'_windows.json')
 windows=json.loads(existing.read_text()) if existing.exists() else []
 starts=list(np.arange(left,max(left,right-30)+.01,20))
 final=max(left,right-30)
 if not starts or final-starts[-1]>5:starts.append(final)
 elif right-starts[-1]>30:starts[-1]=final
 for index,a in enumerate(starts):
  if any(abs(w['start']-a)<.001 for w in windows):continue
  b=min(a+30,right);segments,_=model.transcribe(audio[int(a*sr):int(b*sr)],language='en',beam_size=5,vad_filter=False,condition_on_previous_text=False,word_timestamps=True)
  rows=[]
  for s in segments:
   rows.append({'start':float(a+s.start),'end':float(a+s.end),'text':s.text,'avg_logprob':float(s.avg_logprob),'no_speech_prob':float(s.no_speech_prob),'compression_ratio':float(s.compression_ratio),'quality_passed':bool(s.avg_logprob>=-1 and s.no_speech_prob<=.6 and s.compression_ratio<=2.4),'words':[{'start':float(a+w.start),'end':float(a+w.end),'word':w.word,'probability':float(w.probability)} for w in s.words or []]})
  windows.append({'window_index':index,'start':float(a),'end':float(b),'segments':rows})
  (p/(name+'_windows.json')).write_text(json.dumps(windows,indent=2)+'\n')
  print(name,index+1,'window',a,b,'elapsed',round(time.monotonic()-start),flush=True)
 # Record one-window ownership, retaining alternate hypotheses separately.
 observations=[]
 for w in windows:
  for s in w['segments']:
   if not s['quality_passed']:continue
   for word in s['words']:
    center=(word['start']+word['end'])/2
    owners=[x for x in windows if x['start']<=center<=x['end']]
    owner=min(owners,key=lambda x:abs(center-(x['start']+x['end'])/2))
    if owner['window_index']==w['window_index']:observations.append(dict(word,window_index=w['window_index']))
 windows.sort(key=lambda w:w['start'])
 (p/(name+'_word_hypotheses.json')).write_text(json.dumps(sorted(observations,key=lambda x:x['start']),indent=2)+'\n')
 print('Finished scene',name,flush=True)
print('All scenes complete, elapsed',time.monotonic()-start,flush=True)
