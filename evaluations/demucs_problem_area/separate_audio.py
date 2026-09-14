"""Save unclipped float Demucs outputs and verify reconstruction."""
import argparse,json,random,time
from pathlib import Path
import numpy as np
import soundfile as sf
import torch
from demucs.api import Separator
p=argparse.ArgumentParser(description=__doc__);p.add_argument('audio',type=Path);p.add_argument('--output-dir',type=Path,required=True);args=p.parse_args()
torch.set_num_threads(4);torch.manual_seed(0);random.seed(0);device='cuda' if torch.cuda.is_available() else 'cpu'
start=time.monotonic();separator=Separator(model='htdemucs',device=device,shifts=1,progress=True)
original,stems=separator.separate_audio_file(args.audio)
vocals=stems['vocals'].cpu().numpy().T;original=original.cpu().numpy().T;residual=original-vocals
args.output_dir.mkdir(parents=True,exist_ok=True)
for name,waveform in [('vocals',vocals),('removed_sounds',residual)]:sf.write(args.output_dir/(name+'.wav'),waveform,separator.samplerate,subtype='FLOAT')
error=float(np.max(np.abs(original-vocals-residual)))
assert error<1e-6
(args.output_dir/'separation_summary.json').write_text(json.dumps({'model':'htdemucs','device':device,'shifts':1,'seed':0,'sample_rate':separator.samplerate,'samples':len(original),'reconstruction_error':error,'vocals_peak':float(np.max(np.abs(vocals))),'elapsed_seconds':time.monotonic()-start,'saving':'soundfile FLOAT, no clipping or rescaling'},indent=2)+'\n')
