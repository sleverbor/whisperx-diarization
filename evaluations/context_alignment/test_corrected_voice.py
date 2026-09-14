import json
from pathlib import Path
import numpy as np
import torch
import soundfile as sf
from speechbrain.inference.speaker import SpeakerRecognition
p=Path('outputs/context_alignment');audio,sr=sf.read('outputs/timing_diagnostic/last_25s.wav',dtype='float32');assert sr==16000
prior=np.load('/home/think/projects/whisperx_diarization/references/youtube-v1/voice_embeddings.npy');prior=prior/np.linalg.norm(prior,axis=1,keepdims=True);target=prior.mean(axis=0);target/=np.linalg.norm(target)
model=SpeakerRecognition.from_hparams(source='speechbrain/spkrec-ecapa-voxceleb',savedir='/home/think/projects/whisperx_diarization/pretrained_models/spkrec-ecapa-voxceleb',run_opts={'device':'cpu'})
comparison=json.loads((p/'comparison.json').read_text());aligned=json.loads((p/'complete_local_alignment.json').read_text())['segments'];old=json.loads(Path('work/kaggle_latest/results/height_weight_evidence.json').read_text())['segments'];rows=[]
for c in comparison:
 s=next(r for r in aligned if r['text']==c['complete_alignment_text']);crop=audio[int(s['start']*sr):int(s['end']*sr)]
 with torch.no_grad():v=model.encode_batch(torch.from_numpy(crop).unsqueeze(0)).flatten().numpy()
 v/=np.linalg.norm(v);baseline=next(r for r in old if r['text'].strip()==c['baseline_text'].strip());e=next(e for e in baseline['evidence'] if e['source']=='local_voice')
 rows.append({'text':s['text'],'start':665+s['start'],'end':665+s['end'],'old_target_similarity':e['details']['similarity'],'corrected_target_similarity':float(v@target)})
 print(rows[-1],flush=True)
(p/'voice_comparison.json').write_text(json.dumps(rows,indent=2)+'\n')
