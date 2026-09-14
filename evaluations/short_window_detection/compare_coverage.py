import json
from pathlib import Path
p=Path('outputs/short_window_detection')

def union(intervals):
 out=[]
 for a,b in sorted(intervals):
  a=max(1090,a);b=min(1150,b)
  if b<=a:continue
  if out and a<=out[-1][1]:out[-1][1]=max(out[-1][1],b)
  else:out.append([a,b])
 return out

def coverage(intervals,start,end):return sum(max(0,min(end,b)-max(start,a)) for a,b in intervals)
old=json.loads(Path('outputs/handcuff_separation/scene_diarization.json').read_text());windows=json.loads((p/'window_diarization.json').read_text());whole=json.loads((p/'whole_unconstrained.json').read_text())[0]['tracks']
sets={'prior_60s_fixed_three':union([(r['start'],r['end']) for r in old]),'control_60s_unconstrained':union([(r['start'],r['end']) for r in whole]),'union_20s_windows':union([(r['start'],r['end']) for w in windows for r in w['tracks']])}
speech=json.loads(Path('outputs/handcuff_separation/voice_hypotheses.json').read_text());rows=[]
for s in speech:
 if s['end']>1150:continue
 rows.append({'start':s['start'],'end':s['end'],'text':s['text'],'coverage_fraction':{name:coverage(intervals,s['start'],s['end'])/(s['end']-s['start']) for name,intervals in sets.items()}})
report={'region':[1090,1150],'speech_seconds':{name:sum(b-a for a,b in intervals) for name,intervals in sets.items()},'window_count':len(windows),'window_seconds':20,'overlap_seconds':10,'window_speaker_count':'unconstrained','window_ids_not_stitched':True,'coverage_is_not_accuracy':True,'speech_spans':'saved recovered hypotheses; not manually timed ground truth','rows':rows}
(p/'coverage_report.json').write_text(json.dumps(report,indent=2)+'\n')
lines=['# Short-window speech-detection test','','Compared six 20-second community-1 windows with 10-second overlap against a 60-second unconstrained pass and the earlier fixed-three-speaker pass. Same raw audio/model, CPU, unchanged text/reference. Window IDs are not stitched. Comparison region is source 1090–1150 seconds; the last window extends beyond this and is clipped for coverage calculations.','','Speech coverage means overlap with saved recovered/aligned utterance spans, not measured transcription or diarization accuracy. Noise and mixed-speaker spans can count as coverage.','','| Recovered hypothesis | 60s unconstrained | 20s-window union |','|---|---:|---:|']
for r in rows:lines.append(f"| {r['text']} | {r['coverage_fraction']['control_60s_unconstrained']:.0%} | {r['coverage_fraction']['union_20s_windows']:.0%} |")
lines+=['','## Findings','','Shorter windows recover more portions of some unknown/key-officer speech, but do not fix the important missed target agreement or cuff-tightening exchange. The window at source 1120–1140 detects no intervals despite audible speech. Similarity-based speaker identification cannot use an interval that diarization fails to detect.','','The known longer target agreement has a direct target voice match from the previous experiment but zero diarization coverage here. Keep that independent evidence; a missing diarization interval should not be interpreted as silence.','','Do not adopt window shortening alone as a solution. Next bounded test: inspect or compare speech activity detection itself, with the recovered/audio-supported spans retained for review even when diarization misses them. A detector should also be checked on key-only noise to avoid replacing missed speech with false detections. No identity thresholds or production defaults were changed.']
(p/'RESULTS.md').write_text('\n'.join(lines)+'\n')
print(report['speech_seconds'])
for r in rows:
 if 'syllable' in r['text'] or 'not trying' in r['text']:print(r)
