import json,sys
from pathlib import Path
from recover_transcript_gaps import collect_candidates
for name in sys.argv[1:]:
 out=Path(name);path=out/'transcript_with_candidates.json';result=json.loads(path.read_text());segments=result['segments'];candidates=[]
 for index,gap in enumerate(result['gap_recovery_settings']['gaps']):
  observations=[]
  for window in json.loads((out/'window_decodes.json').read_text()):
   if window['gap_index']!=index:continue
   for row in window['segments']:
    if row['quality_filter_passed']:observations.extend(row['words'])
  for candidate in collect_candidates(observations,gap):candidate['gap_index']=index;candidates.append(candidate)
 result['gap_recovery_candidates']=candidates;result['gap_recovery_settings']['phrase_selection']='Single-window phrases ranked by distinct-window word support; no mixing hypotheses'
 assert result['segments']==segments
 path.write_text(json.dumps(result,indent=2)+'\n');offset=result.get('source_offset_seconds',0);lines=[]
 for s in segments:lines.append((s['start'],f"[{s['start']+offset:.2f}-{s['end']+offset:.2f}] {s.get('final_speaker','Unknown')}: {s['text']}"))
 for s in candidates:
  support=f"overlap support {s['supported_word_count']}/{len(s['words'])} words" if s['supported_word_count'] else 'single decode'
  lines.append((s['start'],f"[{s['start']+offset:.2f}-{s['end']+offset:.2f}] REVIEW ({support}; speaker unknown): {s['text']}"))
 (out/'review_transcript.txt').write_text('\n'.join(text for _,text in sorted(lines))+'\n');print(name,len(candidates),'candidates')
