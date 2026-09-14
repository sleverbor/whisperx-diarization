"""Compare the cached diarization embedding model; matching references rebuilt."""
import json,subprocess,time
from pathlib import Path
import numpy as np
import torch
import soundfile as sf
from pyannote.audio.pipelines.speaker_verification import PretrainedSpeakerEmbedding

def unit(v):
 v=np.asarray(v,dtype=np.float32).reshape(-1)
 if not np.isfinite(v).all() or np.linalg.norm(v)==0:raise ValueError('Invalid embedding')
 return v/np.linalg.norm(v)
p=Path('outputs/hour_iteration/native_embedding');p.mkdir(parents=True,exist_ok=True);torch.set_num_threads(4)
checkpoint=next(Path('/home/think/.cache/huggingface/hub/models--pyannote--speaker-diarization-community-1/snapshots').glob('*/embedding/pytorch_model.bin'))
model=PretrainedSpeakerEmbedding(str(checkpoint),device=torch.device('cpu'))

def encode(audio):return unit(model(torch.from_numpy(audio).reshape(1,1,-1))[0])

def decode(path):
 raw=subprocess.check_output(['ffmpeg','-nostdin','-hide_banner','-loglevel','error','-i',str(path),'-vn','-ar','16000','-ac','1','-f','f32le','pipe:1']);return np.frombuffer(raw,dtype='<f4').copy()
references=Path('/home/think/projects/whisperx_diarization/references');manifest=json.loads((references/'source-clips/sources.json').read_text())['sources'];source_audio={r['url']:decode(references/'source-clips'/r['path']) for r in manifest};metadata=json.loads((references/'youtube-v1/reference.json').read_text());vectors=[]
for i in metadata['voice']['accepted_rows']:
 r=metadata['voice']['sample_provenance'][i];vectors.append(encode(source_audio[r['source']][int(r['start']*16000):int(r['end']*16000)]))
 print('Target reference',len(vectors),flush=True)
target=unit(np.mean(vectors,axis=0));np.save(p/'target_reference.npy',target)
# Use the same independent opening officer crop selection as the ECAPA test.
provenance=json.loads(Path('outputs/speaker_separation/reference_provenance.json').read_text())['officer_sources'];opening=decode('/home/think/projects/whisperx_diarization/short.mp4');officer_vectors=[]
for r in provenance:officer_vectors.append(encode(opening[int(r['start']*16000):int(r['end']*16000)]))
officer=unit(np.mean(officer_vectors,axis=0));np.save(p/'officer_reference.npy',officer)
audio,sr=sf.read('outputs/caption_test/cuffs_original.wav',dtype='float32');assert sr==16000;segments=json.loads(Path('outputs/handcuff_separation/local_alignment.json').read_text());signals={'raw':audio,'light':sf.read('work/hour_iteration/denoised_cuffs/light.wav',dtype='float32')[0]};rows=[];arrays={}
for name,signal in signals.items():
 for i,s in enumerate(segments):
  if s['end']-s['start']<.4:continue
  try:v=encode(signal[int(s['start']*sr):int(s['end']*sr)])
  except (ValueError,RuntimeError):continue
  scores={'Target_Speaker':float(v@target),'Opening_officer_reference':float(v@officer)}
  rows.append({'segment_index':i,'signal':name,'text':s['text'],'start':1078+s['start'],'end':1078+s['end'],'scores':scores});arrays[f'{name}_{i}']=v
 print('Finished native model',name,flush=True)
np.savez_compressed(p/'utterance_vectors.npz',**arrays);(p/'scores.json').write_text(json.dumps(rows,indent=2)+'\n')
(p/'reference_provenance.json').write_text(json.dumps({'model_checkpoint':checkpoint.name,'target_reference_samples':len(vectors),'officer_reference_samples':provenance,'dimensions':len(target),'not_ecapa_compatible':True},indent=2)+'\n')
print('Native embedding comparison complete',flush=True)
