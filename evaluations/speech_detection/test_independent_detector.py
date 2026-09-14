"""Independent Silero speech-activity comparison; no transcription changes."""
import json
from pathlib import Path
import numpy as np
import soundfile as sf
from faster_whisper.vad import get_speech_timestamps,VadOptions
p=Path('outputs/speech_detection');audio,sr=sf.read('outputs/caption_test/cuffs_original.wav',dtype='float32');assert sr==16000
clip=audio[12*sr:72*sr]
results={}
for threshold in (.5,.35,.2):
 intervals=get_speech_timestamps(clip,vad_options=VadOptions(threshold=threshold,min_speech_duration_ms=100,min_silence_duration_ms=200,speech_pad_ms=0),sampling_rate=sr)
 results[str(threshold)]=[{'start':1090+r['start']/sr,'end':1090+r['end']/sr} for r in intervals]
 silence=get_speech_timestamps(np.zeros(5*sr,dtype=np.float32),vad_options=VadOptions(threshold=threshold,speech_pad_ms=0),sampling_rate=sr)
 assert not silence,'Detector labels digital silence as speech'
(p/'silero_intervals.json').write_text(json.dumps(results,indent=2)+'\n')
old=json.loads(Path('outputs/short_window_detection/whole_unconstrained.json').read_text())[0]['tracks'];speech=json.loads(Path('outputs/handcuff_separation/voice_hypotheses.json').read_text());rows=[]
def fraction(intervals,s):
 spans=sorted((r['start'],r['end']) for r in intervals);merged=[]
 for a,b in spans:
  if merged and a<=merged[-1][1]:merged[-1][1]=max(merged[-1][1],b)
  else:merged.append([a,b])
 return sum(max(0,min(s['end'],b)-max(s['start'],a)) for a,b in merged)/(s['end']-s['start'])
for s in speech:
 rows.append({'start':s['start'],'end':s['end'],'text':s['text'],'pyannote_coverage':fraction(old,s),'silero_coverage':{t:fraction(intervals,s) for t,intervals in results.items()}})
# This is a listening candidate, not an established key-only/no-speech ground truth.
noise_region={'start':1100.,'end':1102.}
sf.write(p/'noise_candidate_listen.wav',audio[int((1100-1078)*sr):int((1102-1078)*sr)]*.6,sr,subtype='PCM_16')
report={'region':[1090,1150],'silero_options':{'minimum_speech_ms':100,'minimum_silence_ms':200,'padding_ms':0},'rows':rows,'digital_silence_control_passed':True,'noise_candidate':{'source_region':noise_region,'verified_speech_free':False,'coverage':{t:fraction(intervals,noise_region) for t,intervals in results.items()}},'coverage_is_not_accuracy':True,'baseline_unchanged':True}
(p/'comparison.json').write_text(json.dumps(report,indent=2)+'\n')
for r in rows:print(r['text'],round(r['pyannote_coverage'],2),{t:round(v,2) for t,v in r['silero_coverage'].items()})
print('Noise candidate',report['noise_candidate'])
