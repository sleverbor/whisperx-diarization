import gc,json,subprocess,time
from pathlib import Path
import numpy as np
import torch
import whisperx
from silero_activity_adapter import SileroActivity
p=Path('outputs/hour_iteration/silero_asr');p.mkdir(parents=True,exist_ok=True);torch.set_num_threads(4)
model=whisperx.load_model('large-v2','cpu',compute_type='int8',language='en',vad_model=SileroActivity())
start=time.monotonic()
for name,left,duration in [('opening',0,30),('height_weight',625,65),('handcuffs',1078,90)]:
 raw=subprocess.check_output(['ffmpeg','-nostdin','-hide_banner','-loglevel','error','-ss',str(left),'-i','/home/think/projects/whisperx_diarization/video.mp4','-t',str(duration),'-vn','-ar','16000','-ac','1','-f','f32le','pipe:1']);audio=np.frombuffer(raw,dtype='<f4').copy()
 result=model.transcribe(audio,batch_size=4,chunk_size=30,language='en')
 result['source_offset_seconds']=left;result['settings']={'vad':'faster-whisper Silero activity adapter','threshold':.5,'context_seconds':.5,'chunk_size':30,'device':'cpu','compute_type':'int8','batch_size':4}
 (p/(name+'_transcription.json')).write_text(json.dumps(result,indent=2)+'\n')
 print('Finished',name,'elapsed',round(time.monotonic()-start),'segments',len(result['segments']),flush=True)
print('All Silero ASR tests complete',flush=True)
