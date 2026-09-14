"""Create lightly mixed and fully denoised speech candidates without clipping."""
import argparse,hashlib,json,time
from pathlib import Path
import numpy as np
import soundfile as sf
import torch
from denoiser import pretrained
p=argparse.ArgumentParser(description=__doc__);p.add_argument('audio',type=Path);p.add_argument('--output-dir',type=Path,required=True);p.add_argument('--model-cache',type=Path,required=True);p.add_argument('--wet',type=float,default=.5);args=p.parse_args()
if not 0<=args.wet<=1:p.error('Wet mix must be between zero and one')
if args.output_dir.exists():p.error('Choose a new output directory')
audio,rate=sf.read(args.audio,dtype='float32')
if rate!=16000 or audio.ndim!=1:raise ValueError('Input must be 16 kHz mono')
if not np.isfinite(audio).all():raise ValueError('Non-finite input')
torch.set_num_threads(4);torch.hub.set_dir(str(args.model_cache));device='cuda' if torch.cuda.is_available() else 'cpu'
start=time.monotonic();model=pretrained.dns64().eval().to(device)
with torch.inference_mode():enhanced=model(torch.from_numpy(audio).reshape(1,1,-1).to(device)).reshape(-1).cpu().numpy()
if enhanced.shape!=audio.shape or not np.isfinite(enhanced).all():raise ValueError('Denoiser changed duration or returned invalid audio')
light=(1-args.wet)*audio+args.wet*enhanced
args.output_dir.mkdir(parents=True)
for name,waveform in [('original',audio),('light',light),('full',enhanced),('removed',audio-enhanced)]:sf.write(args.output_dir/(name+'.wav'),waveform,rate,subtype='FLOAT')
weights=list((args.model_cache/'checkpoints').glob('dns64*'))
summary={'model':'Meta Denoiser DNS64','device':device,'wet_fraction':args.wet,'sample_rate':rate,'samples':len(audio),'input_sha256':hashlib.sha256(args.audio.read_bytes()).hexdigest(),'weights_sha256':hashlib.sha256(weights[0].read_bytes()).hexdigest() if weights else None,'elapsed_seconds':time.monotonic()-start,'save_format':'unclipped FLOAT; no loudness normalization'}
(args.output_dir/'denoising_summary.json').write_text(json.dumps(summary,indent=2)+'\n');print('Denoising complete',flush=True)
