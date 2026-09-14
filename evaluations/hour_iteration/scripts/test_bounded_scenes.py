import sys,json,subprocess,time
from pathlib import Path
sys.path.insert(0,'outputs/hour_iteration/code')
from bounded_silero_vad import make_vad
import numpy as np,torch,whisperx
p=Path('outputs/hour_iteration/bounded_asr');p.mkdir(parents=True,exist_ok=True);torch.set_num_threads(4)
model=whisperx.load_model('large-v2','cpu',compute_type='int8',language='en',vad_model=make_vad())
start=time.monotonic()
for name,offset,duration in [('opening',0,30),('height_weight',625,65),('handcuffs',1078,90)]:
 raw=subprocess.check_output(['ffmpeg','-nostdin','-hide_banner','-loglevel','error','-ss',str(offset),'-i','/home/think/projects/whisperx_diarization/video.mp4','-t',str(duration),'-vn','-ar','16000','-ac','1','-f','f32le','pipe:1']);audio=np.frombuffer(raw,dtype='<f4').copy()
 result=model.transcribe(audio,batch_size=4,chunk_size=20,language='en');result['source_offset_seconds']=offset;result['settings']={'vad':'bounded_silero','threshold':.5,'context_seconds':.5,'chunk_seconds':20,'overlap_seconds':0}
 (p/(name+'_transcription.json')).write_text(json.dumps(result,indent=2)+'\n');print(name,'elapsed',round(time.monotonic()-start),result,flush=True)
