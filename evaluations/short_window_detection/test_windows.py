"""Compare detected speech coverage; anonymous IDs remain local to each window."""
import os,json
from pathlib import Path
import torch
import soundfile as sf
from dotenv import load_dotenv
from whisperx.diarize import DiarizationPipeline
load_dotenv('/home/think/projects/whisperx_diarization/.env');torch.set_num_threads(4)
p=Path('outputs/short_window_detection');audio,sr=sf.read('outputs/caption_test/cuffs_original.wav',dtype='float32');assert sr==16000
model=DiarizationPipeline(token=os.environ.get('HF_TOKEN') or os.environ.get('HUGGINGFACE_TOKEN'),device='cpu')
results=[]
whole_only=os.environ.get('WHOLE_CONTROL_ONLY')=='1'
starts=[1090] if whole_only else range(1090,1141,10)
for index,start in enumerate(starts):
 end=start+(60 if whole_only else 20)
 # Unconstrained count: not all three people necessarily speak in each window.
 r=model(audio[int((start-1078)*sr):int((end-1078)*sr)])
 rows=r[['start','end','speaker']].to_dict(orient='records')
 for row in rows:row['start']+=start;row['end']+=start
 results.append({'window_index':index,'start':start,'end':end,'tracks':rows})
 (p/('whole_unconstrained.json' if whole_only else 'window_diarization.json')).write_text(json.dumps(results,indent=2)+'\n')
 print('Completed window',start,end,len(rows),'intervals',flush=True)
