#!/usr/bin/env python3
"""Blind two-output separation followed by target-speaker scoring."""
import argparse,json,re,subprocess
from pathlib import Path
import numpy as np

from review_overlap_extraction import transcribe,unit
from run_contextual_wesep_experiment import token_f1

def crop_stream(wave,sample_rate,window_start,start,end):
 left=round((start-window_start)*sample_rate);right=round((end-window_start)*sample_rate)
 return wave[max(0,left):max(0,right)]

def main():
 p=argparse.ArgumentParser();p.add_argument('--video',type=Path,required=True);p.add_argument('--labels',type=Path,required=True);p.add_argument('--voice-priors',type=Path,required=True);p.add_argument('--output-dir',type=Path,required=True);p.add_argument('--context',type=float,default=3.0);p.add_argument('--device',default='cpu');p.add_argument('--whisper-model',default='large-v2');a=p.parse_args();a.output_dir.mkdir(parents=True,exist_ok=True)
 import soundfile as sf
 import torch,torchaudio
 from speechbrain.inference.separation import SepformerSeparation
 from speechbrain.inference.speaker import SpeakerRecognition
 from faster_whisper import WhisperModel
 labels=json.loads(a.labels.read_text())['labels'];source=a.output_dir/'source-16khz.wav'
 subprocess.run(['ffmpeg','-nostdin','-hide_banner','-loglevel','error','-y','-i',str(a.video),'-vn','-ac','1','-ar','16000',str(source)],check=True)
 full,sr=sf.read(source,dtype='float32');assert sr==16000
 sepdir=Path('pretrained_models/sepformer-whamr');spkdir=Path('pretrained_models/spkrec-ecapa-voxceleb')
 separator=SepformerSeparation.from_hparams(source=str(sepdir),savedir=str(sepdir),run_opts={'device':a.device});speaker=SpeakerRecognition.from_hparams(source=str(spkdir),savedir=str(spkdir),run_opts={'device':a.device})
 whisper=WhisperModel(a.whisper_model,device='cuda' if a.device.startswith('cuda') else 'cpu',compute_type='float16' if a.device.startswith('cuda') else 'int8')
 priors=np.load(a.voice_priors);target=unit(np.mean(np.stack([unit(x) for x in priors]),axis=0));audio=a.output_dir/'audio';audio.mkdir(exist_ok=True);rows=[]
 def similarity(w):
  emb=unit(speaker.encode_batch(torch.from_numpy(np.asarray(w,dtype=np.float32)).unsqueeze(0)).flatten().detach().cpu().numpy());return float(np.dot(target,emb))
 for label in labels:
  start,end=float(label['start']),float(label['end']);ws=max(0,start-a.context);we=min(len(full)/sr,end+a.context);window=torch.from_numpy(full[round(ws*sr):round(we*sr)]).float()
  wave8=torchaudio.functional.resample(window,16000,8000).unsqueeze(0);separated=separator.separate_batch(wave8)[0].detach().cpu()
  exchange=[]
  for index in range(separated.shape[-1]):
   stream8=separated[:,index].numpy();cropped8=crop_stream(stream8,8000,ws,start,end);cropped16=torchaudio.functional.resample(torch.from_numpy(cropped8),8000,16000).numpy();expected=round((end-start)*16000)
   if len(cropped16)<expected:cropped16=np.pad(cropped16,(0,expected-len(cropped16)))
   cropped16=cropped16[:expected];path=audio/f"{label['exchange_id']}-stream-{index+1}.wav";sf.write(path,cropped16,16000)
   asr=transcribe(whisper,cropped16);row={'stream':index+1,'audio':str(path.relative_to(a.output_dir)),'target_similarity':similarity(cropped16),'transcription':asr,'target_word_f1':token_f1(label.get('target_words',''),asr['text']),'other_word_f1':token_f1(label.get('other_words',''),asr['text'])};exchange.append(row)
  ranked=sorted(exchange,key=lambda r:r['target_similarity'],reverse=True);margin=ranked[0]['target_similarity']-ranked[1]['target_similarity']
  rows.append({'exchange_id':label['exchange_id'],'baseline_index':label['baseline_index'],'start':start,'end':end,'context_seconds':a.context,'target_words':label.get('target_words',''),'other_words':label.get('other_words',''),'streams':exchange,'similarity_selected_stream':ranked[0]['stream'],'similarity_margin':margin})
  print(label['exchange_id'],f"selected={ranked[0]['stream']} margin={margin:.3f}",'; '.join(f"s{x['stream']} sim={x['target_similarity']:.3f} target_f1={x['target_word_f1']:.2f} other_f1={x['other_word_f1']:.2f}: {x['transcription']['text']}" for x in exchange),flush=True)
 report={'schema_version':1,'method':'blind_sepformer_whamr_then_ecapa','labels_used_for_separation':False,'context_seconds':a.context,'results':rows};(a.output_dir/'report.json').write_text(json.dumps(report,indent=2)+'\n')
if __name__=='__main__':main()
