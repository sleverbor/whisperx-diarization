"""Controlled chunk/decode sensitivity; no baseline changes."""
import json,subprocess,time
from pathlib import Path
import numpy as np
import torch
import whisperx
from dataclasses import replace
from silero_activity_adapter import SileroActivity
from whisperx.diarize import Segment
p=Path('outputs/hour_iteration/chunk_sweep');p.mkdir(parents=True,exist_ok=True);torch.set_num_threads(4)
raw=subprocess.check_output(['ffmpeg','-nostdin','-hide_banner','-loglevel','error','-ss','0','-i','/home/think/projects/whisperx_diarization/video.mp4','-t','30','-vn','-ar','16000','-ac','1','-f','f32le','pipe:1']);audio=np.frombuffer(raw,dtype='<f4').copy()
class BoundedActivity(SileroActivity):
 @staticmethod
 def merge_chunks(segments,chunk_size,onset=.5,offset=None):
  if not segments:return []
  merged=[]
  for s in segments:
   if merged and s.start<=merged[-1][1]:merged[-1][1]=max(merged[-1][1],s.end)
   else:merged.append([s.start,s.end])
  chunks=[]
  for a,b in merged:
   left=a
   while left<b:
    right=min(left+chunk_size,b);chunks.append({'start':left,'end':right,'segments':[(left,right)]})
    if right>=b:break
    left=right-min(2,chunk_size/4)
  return chunks
model=whisperx.load_model('large-v2','cpu',compute_type='int8',language='en',vad_model=BoundedActivity())
start=time.monotonic()
for chunk in (10,15,20,30):
 result=model.transcribe(audio,batch_size=4,chunk_size=chunk,language='en')
 (p/f'opening_{chunk}s.json').write_text(json.dumps(result,indent=2)+'\n');print('Chunk',chunk,'elapsed',round(time.monotonic()-start),'text',' '.join(s['text'] for s in result['segments']),flush=True)
# Diagnostic only: batched WhisperX documents non-timestamp mode. Test tokens
# without adopting this unsupported configuration in the production runner.
model.options=replace(model.options,without_timestamps=False)
result=model.transcribe(audio,batch_size=4,chunk_size=30,language='en')
(p/'opening_timestamp_tokens_diagnostic.json').write_text(json.dumps(result,indent=2)+'\n');print('Timestamp-token diagnostic elapsed',round(time.monotonic()-start),'text',result,flush=True)
