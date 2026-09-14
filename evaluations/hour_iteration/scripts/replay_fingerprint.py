"""Waveform duplicate evidence; does not add independent enrollment samples."""
import json,subprocess
from pathlib import Path
import numpy as np
from scipy.signal import correlate,butter,sosfiltfilt,find_peaks
p=Path('outputs/hour_iteration/replay_fingerprint');p.mkdir(parents=True,exist_ok=True);sr=16000

def audio(start,duration):
 raw=subprocess.check_output(['ffmpeg','-nostdin','-hide_banner','-loglevel','error','-ss',str(start),'-i','/home/think/projects/whisperx_diarization/video.mp4','-t',str(duration),'-vn','-ar',str(sr),'-ac','1','-f','f32le','pipe:1']);return np.frombuffer(raw,dtype='<f4').copy()
filter=butter(4,[150,4500],btype='bandpass',fs=sr,output='sos');opening=sosfiltfilt(filter,audio(0,30));replay=sosfiltfilt(filter,audio(925,45));control=sosfiltfilt(filter,audio(600,45));segments=json.loads(Path('work/kaggle_latest/results/opening_evidence.json').read_text())['segments'];full=json.loads(Path('work/kaggle_latest/results/full_video_evidence.json').read_text())['segments']

def matches(template,query):
 denominator=np.cumsum(np.r_[0.,query*query]);denominator=denominator[len(template):]-denominator[:-len(template)];dots=correlate(query,template,mode='valid',method='fft');scores=dots/np.sqrt(np.maximum(denominator*np.sum(template*template),1e-16));peaks,_=find_peaks(scores,distance=int(sr*.5));best=sorted(peaks,key=lambda i:scores[i],reverse=True)[:3];return [(int(i),float(scores[i])) for i in best]
rows=[]
for index,s in enumerate(segments):
 if s['end']-s['start']<.6:continue
 template=opening[int(s['start']*sr):int(s['end']*sr)];found=matches(template,replay);negative=matches(template,control)
 if not found:continue
 start=925+found[0][0]/sr;end=start+len(template)/sr;overlap=[r for r in full if r['start']<end and r['end']>start]
 rows.append({'opening_segment_index':index,'text':s['text'],'opening_start':s['start'],'opening_end':s['end'],'source_speaker_hypothesis':s['final_speaker'],'source_strength':s['final_confidence'],'replay_start':start,'replay_end':end,'waveform_similarity':found[0][1],'negative_control_max_similarity':negative[0][1] if negative else None,'duplicate_match_accepted':found[0][1]>=.85,'replay_overlapping_labels':[{'start':r['start'],'text':r['text'],'speaker':r['final_speaker']} for r in overlap],'counts_as_independent_voice_sample':False})
(p/'matches.json').write_text(json.dumps(rows,indent=2)+'\n')
for r in rows:print(round(r['opening_start'],2),round(r['replay_start'],2),round(r['waveform_similarity'],3),round(r['negative_control_max_similarity'],3),r['duplicate_match_accepted'],r['text'],flush=True)
