import os,json,time
from pathlib import Path
os.environ['MPLBACKEND']='Agg'
import whisperx
from whisperx.vads.vad import Vad
class FullInterval(Vad):
 @staticmethod
 def preprocess_audio(audio):return audio
 def __call__(self,inputs):return [{'start':0.,'end':len(inputs['waveform'])/inputs['sample_rate']}]
 @staticmethod
 def merge_chunks(segments,chunk_size,onset,offset):return segments
base=Path('/home/think/Documents/Codex/2026-09-13/referenced-chatgpt-conversation-this-is-an/outputs/height_exchange')
model=whisperx.load_model('large-v2','cpu',compute_type='int8',language='en',vad_model=FullInterval(.5))
result=model.transcribe(whisperx.load_audio(str(base/'raw.wav')),batch_size=16)
result['source_offset_seconds']=640
(base/'whisperx_full_interval.json').write_text(json.dumps(result,indent=2)+'\n')
(base/'whisperx_full_interval.txt').write_text('\n'.join(s['text'].strip() for s in result['segments'])+'\n')
print('Original WhisperX decoder with full interval coverage completed.')
