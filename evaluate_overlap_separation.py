"""Separate short overlapping-speech crops and compare each stream to a target reference."""
import argparse,json
from pathlib import Path
import numpy as np
import torch,torchaudio
from speechbrain.inference.separation import SepformerSeparation
from speechbrain.inference.speaker import SpeakerRecognition
from faster_whisper import WhisperModel


def unit(v):
 v=np.asarray(v,dtype=np.float32).reshape(-1);return v/max(float(np.linalg.norm(v)),1e-9)

def main():
 p=argparse.ArgumentParser();p.add_argument('--crop',type=Path,action='append',required=True);p.add_argument('--voice-priors',type=Path,required=True);p.add_argument('--output-dir',type=Path,required=True);a=p.parse_args();a.output_dir.mkdir(parents=True,exist_ok=True)
 target=unit(np.mean(np.stack([unit(x) for x in np.load(a.voice_priors)]),axis=0))
 sepdir=Path('pretrained_models/sepformer-whamr');ecapadir=Path('pretrained_models/spkrec-ecapa-voxceleb')
 separator=SepformerSeparation.from_hparams(source=str(sepdir),savedir=str(sepdir),run_opts={'device':'cpu'})
 speaker=SpeakerRecognition.from_hparams(source=str(ecapadir),savedir=str(ecapadir),run_opts={'device':'cpu'})
 whisper=WhisperModel('small',device='cpu',compute_type='int8')
 report=[]
 for crop in a.crop:
  separated=separator.separate_file(path=str(crop.resolve()))[0].detach().cpu().T
  for index,wave8 in enumerate(separated):
   wave16=torchaudio.functional.resample(wave8,8000,16000)
   emb=unit(speaker.encode_batch(wave16.unsqueeze(0)).flatten().detach().cpu().numpy())
   peak=wave16.abs().max().clamp_min(1e-6);save_wave=(wave16/peak*.9).unsqueeze(0)
   path=a.output_dir/f'{crop.stem}-stream-{index+1}.wav';torchaudio.save(str(path),save_wave,16000)
   segments,_=whisper.transcribe(wave16.numpy(),vad_filter=False,condition_on_previous_text=False)
   text=' '.join(s.text.strip() for s in segments if s.text.strip())
   report.append({'crop':str(crop),'stream':index+1,'target_similarity':float(np.dot(target,emb)),'transcript':text,'audio':str(path)})
 (a.output_dir/'report.json').write_text(json.dumps(report,indent=2)+'\n')
 for row in report:print(f"{Path(row['crop']).name} stream {row['stream']}: similarity={row['target_similarity']:.3f}: {row['transcript']}")
if __name__=='__main__':main()
