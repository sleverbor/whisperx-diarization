"""Attach independent speech activity without changing transcription/identity."""
import argparse,copy,json
from pathlib import Path

def fraction(intervals,start,end):
 if end<=start:return None
 clipped=sorted((max(start,r['start']),min(end,r['end'])) for r in intervals if r['end']>start and r['start']<end);merged=[]
 for a,b in clipped:
  if b<=a:continue
  if merged and a<=merged[-1][1]:merged[-1][1]=max(merged[-1][1],b)
  else:merged.append([a,b])
 return sum(b-a for a,b in merged)/(end-start)

if __name__=='__main__':
 p=argparse.ArgumentParser(description=__doc__);p.add_argument('baseline',type=Path);p.add_argument('--activity',required=True,type=Path,help='JSON list of start/end intervals in source seconds');p.add_argument('--source-offset',type=float,default=0);p.add_argument('--region-start',type=float,required=True);p.add_argument('--region-end',type=float,required=True);p.add_argument('--output',required=True,type=Path);a=p.parse_args()
 if a.output.exists():p.error('Use a new output file')
 if a.region_end<=a.region_start:p.error('Invalid analyzed region')
 original=json.loads(a.baseline.read_text());output=copy.deepcopy(original);activity=json.loads(a.activity.read_text());evidence=[]
 for kind,key in [('baseline','segments'),('review_candidate','gap_recovery_candidates')]:
  for index,s in enumerate(original.get(key,[])):
   start=a.source_offset+s['start'];end=a.source_offset+s['end'];evidence.append({'kind':kind,'index':index,'start':start,'end':end,'speech_coverage_fraction':fraction(activity,start,end) if start>=a.region_start and end<=a.region_end else None,'fully_observed':start>=a.region_start and end<=a.region_end,'source':'independent_silero','identifies_speaker':False})
 output['independent_speech_activity_evidence']=evidence
 assert output['segments']==original['segments']
 assert output.get('gap_recovery_candidates')==original.get('gap_recovery_candidates')
 a.output.parent.mkdir(parents=True,exist_ok=True);a.output.write_text(json.dumps(output,indent=2)+'\n')
