"""Align a complete local audio-only hypothesis; do not alter baseline results."""
import json
from pathlib import Path
from difflib import SequenceMatcher
import numpy as np
import soundfile as sf
import whisperx
root=Path('outputs/context_alignment');control=json.loads(Path('outputs/timing_diagnostic/audio_timing_control.json').read_text());audio,sr=sf.read('outputs/timing_diagnostic/last_25s.wav',dtype='float32');assert sr==16000
# One complete local text block mirrors the old broad-block alignment, but includes
# intervening speech. No user reference or YouTube captions are supplied.
hypothesis=[{'start':0.,'end':len(audio)/sr,'text':' '.join(r['text'].strip() for r in control)}]
model,metadata=whisperx.load_align_model(language_code='en',device='cpu')
result=whisperx.align(hypothesis,model,metadata,audio,'cpu',return_char_alignments=False)
(root/'complete_local_alignment.json').write_text(json.dumps(result,indent=2)+'\n')
old=json.loads(Path('outputs/timing_diagnostic/confirmed_timing_shifts.json').read_text());rows=[]
for s in old:
 match=max(result['segments'],key=lambda r:SequenceMatcher(None,s['text'].lower(),r['text'].lower()).ratio())
 rows.append({'baseline_text':s['text'],'saved_source_start':s['saved_start_seconds'],'audio_control_start':s['control_start_seconds'],'complete_alignment_source_start':665+match['start'],'complete_alignment_text':match['text'],'absolute_error_vs_audio_control':abs(665+match['start']-s['control_start_seconds'])})
(root/'comparison.json').write_text(json.dumps(rows,indent=2)+'\n')
for row in rows:print(row,flush=True)
