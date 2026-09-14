import ast,json
from pathlib import Path
from dataclasses import dataclass,field,asdict
import numpy as np,re
source=Path('/home/think/projects/whisperx_diarization/chainofrules.py').read_text();tree=ast.parse(source);tree.body=[n for n in tree.body if isinstance(n,(ast.FunctionDef,ast.ClassDef)) and n.name!='main'];exec(compile(tree,'chainofrules.py','exec'))
p=Path('outputs/reference_comparison/opening_original.json');j=json.loads(p.read_text());checks={s['start']:s['fresh_original_visual'] for s in json.loads(Path('outputs/reference_comparison/opening_visual_verification.json').read_text())};previous=None;tracks={s['baseline']['raw_speaker_track'] for s in j['segments']}
result=[]
for s in j['segments']:
 segment=TimelineSegment(s['start'],s['end'],s['text'],Baseline(**s['baseline']),s.get('words',[]))
 segment.evidence=[Evidence(**e) for e in s['evidence'] if e['source'] not in ['question_response','brief_exchange','echo_question'] and (s['start'] not in checks or e['source'] not in ['target_face_visible','visual_context','target_mouth_motion'])]
 if s['start'] in checks:segment.evidence.extend(Evidence(**e) for e in checks[s['start']])
 add_question_response_evidence(segment,previous,tracks,j['target_candidate'],j['mapping_strength']);add_brief_exchange_evidence(segment,previous,tracks,j['target_candidate'],j['mapping_strength']);add_echo_question_evidence(segment,previous,tracks,j['target_candidate'],j['mapping_strength']);resolve_segment(segment,j['target_candidate'],j['mapping_strength'])
 if s['start'] in checks:print(segment.text,s['final_speaker'],'-> same-video original',segment.final_speaker)
 result.append(asdict(segment));previous=segment
j['segments']=result;j['verification_note']='Fresh original visual evidence on identical local video for the two lines whose labels differed in the first comparison; all context/resolution replayed.'
Path('outputs/reference_comparison/opening_original_verified.json').write_text(json.dumps(j,indent=2)+'\n')
