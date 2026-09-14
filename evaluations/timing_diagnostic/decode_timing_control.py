import json
from pathlib import Path
import numpy as np
import soundfile as sf
from faster_whisper import WhisperModel
p=Path('outputs/timing_diagnostic');audio,sr=sf.read(p/'last_25s.wav',dtype='float32');assert sr==16000
model=WhisperModel('large-v2',device='cpu',compute_type='int8',cpu_threads=4)
segments,_=model.transcribe(audio,language='en',beam_size=5,vad_filter=False,condition_on_previous_text=False,word_timestamps=True)
rows=[]
for s in segments:
 rows.append({'start':665+s.start,'end':665+s.end,'text':s.text,'avg_logprob':float(s.avg_logprob),'words':[{'start':665+w.start,'end':665+w.end,'word':w.word,'probability':float(w.probability)} for w in s.words or []]})
 print(round(665+s.start,2),s.text,flush=True)
(p/'audio_timing_control.json').write_text(json.dumps(rows,indent=2)+'\n')
