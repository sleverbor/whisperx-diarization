"""Scene-local diarization test; three speakers is user-provided scene metadata."""
import json,os
from pathlib import Path
import soundfile as sf
from dotenv import load_dotenv
from whisperx.diarize import DiarizationPipeline
load_dotenv('/home/think/projects/whisperx_diarization/.env')
p=Path('outputs/handcuff_separation');audio,sr=sf.read('outputs/caption_test/cuffs_original.wav',dtype='float32');assert sr==16000
offset=12.;clip=audio[int(offset*sr):int(72*sr)]
model=DiarizationPipeline(token=os.environ.get('HF_TOKEN') or os.environ.get('HUGGINGFACE_TOKEN'),device='cpu')
result=model(clip,num_speakers=3)
rows=result[['start','end','speaker']].to_dict(orient='records')
for r in rows:r['start']+=1090;r['end']+=1090
(p/'scene_diarization.json').write_text(json.dumps(rows,indent=2)+'\n')
print('Scene diarization complete:',len(rows),'intervals',flush=True)
