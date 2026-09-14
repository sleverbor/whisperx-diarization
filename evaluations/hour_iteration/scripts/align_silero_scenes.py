import json,subprocess
from pathlib import Path
import numpy as np
import torch
import whisperx
p=Path('outputs/hour_iteration/silero_asr');torch.set_num_threads(2)
model,metadata=whisperx.load_align_model(language_code='en',device='cpu')
for name,duration in [('opening',30),('height_weight',65),('handcuffs',90)]:
 source=json.loads((p/(name+'_transcription.json')).read_text());offset=source['source_offset_seconds']
 raw=subprocess.check_output(['ffmpeg','-nostdin','-hide_banner','-loglevel','error','-ss',str(offset),'-i','/home/think/projects/whisperx_diarization/video.mp4','-t',str(duration),'-vn','-ar','16000','-ac','1','-f','f32le','pipe:1']);audio=np.frombuffer(raw,dtype='<f4').copy()
 result=whisperx.align(source['segments'],model,metadata,audio,'cpu',return_char_alignments=False);result['source_offset_seconds']=offset
 (p/(name+'_alignment.json')).write_text(json.dumps(result,indent=2)+'\n')
 print('Aligned',name,len(result['segments']),'segments',flush=True)
