import os,time,json
from pathlib import Path
os.environ['MPLBACKEND']='Agg'
from faster_whisper import WhisperModel
base=Path('/home/think/Documents/Codex/2026-09-13/referenced-chatgpt-conversation-this-is-an/outputs/height_exchange')
start=time.monotonic()
model=WhisperModel('large-v2',device='cpu',compute_type='int8',cpu_threads=4)
for name in ('raw','cleaned'):
 segments,info=model.transcribe(str(base/(name+'.wav')),language='en',vad_filter=False,beam_size=5,condition_on_previous_text=False,word_timestamps=True)
 rows=[{'start':s.start,'end':s.end,'text':s.text,'avg_logprob':s.avg_logprob,'no_speech_prob':s.no_speech_prob,'words':[{'start':w.start,'end':w.end,'word':w.word,'probability':w.probability} for w in s.words or []]} for s in segments]
 (base/(name+'_transcription.json')).write_text(json.dumps({'model':'large-v2','vad_filter':False,'source_offset_seconds':640,'segments':rows},indent=2)+'\n')
 (base/(name+'_transcript.txt')).write_text('\n'.join(f"[{s['start']+640:.2f}-{s['end']+640:.2f}] {s['text'].strip()}" for s in rows)+'\n')
 print(name,'completed',round((time.monotonic()-start)/60,1),'minutes',flush=True)
print('Both direct-ASR comparisons complete',flush=True)
