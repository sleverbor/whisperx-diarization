"""Compare original, Demucs vocals and residual with identical ASR settings."""
import argparse,json,subprocess,time
from pathlib import Path
import numpy as np
from faster_whisper import WhisperModel
import torch
import soundfile as sf
from importlib.metadata import version

def main():
 p=argparse.ArgumentParser(description=__doc__)
 p.add_argument('--original',type=Path,required=True);p.add_argument('--vocals',type=Path,required=True);p.add_argument('--residual',type=Path,required=True);p.add_argument('--output-dir',type=Path,required=True);p.add_argument('--source-offset',type=float,default=0);p.add_argument('--language',default='en');args=p.parse_args()
 args.output_dir.mkdir(parents=True,exist_ok=True)
 audio={}
 for name,path in [('original',args.original),('vocals',args.vocals),('residual',args.residual)]:
  raw=subprocess.check_output(['ffmpeg','-nostdin','-hide_banner','-loglevel','error','-i',str(path),'-ar','16000','-ac','1','-f','f32le','pipe:1'])
  audio[name]=np.frombuffer(raw,dtype='<f4').copy()
 lengths={k:len(v) for k,v in audio.items()}
 if len(set(lengths.values()))!=1:raise ValueError('Track lengths differ: '+str(lengths))
 residual_error=float(np.max(np.abs(audio['original']-audio['vocals']-audio['residual'])))
 if residual_error>1e-4:raise ValueError('Vocals plus residual do not reconstruct original: '+str(residual_error))
 for name,waveform in audio.items():sf.write(args.output_dir/(name+'_asr.wav'),waveform,16000,subtype='FLOAT')
 device='cuda' if torch.cuda.is_available() else 'cpu';model=WhisperModel('large-v2',device=device,compute_type='float16' if device=='cuda' else 'int8',cpu_threads=4)
 summary={'model':'large-v2','device':device,'language':args.language,'vad_filter':False,'beam_size':5,'condition_on_previous_text':False,'word_timestamps':True,'source_offset_seconds':args.source_offset,'samples_16000':lengths,'reconstruction_max_error':residual_error,'runs':{},'versions':{name:version(name) for name in ['demucs','faster-whisper','torch','soundfile']}}
 for name,waveform in audio.items():
  start=time.monotonic();segments,info=model.transcribe(waveform,language=args.language,vad_filter=False,beam_size=5,condition_on_previous_text=False,word_timestamps=True)
  rows=[{'start':s.start,'end':s.end,'text':s.text,'avg_logprob':s.avg_logprob,'no_speech_prob':s.no_speech_prob,'compression_ratio':s.compression_ratio,'words':[{'start':w.start,'end':w.end,'word':w.word,'probability':w.probability} for w in s.words or []]} for s in segments]
  (args.output_dir/(name+'_transcription.json')).write_text(json.dumps({'source_offset_seconds':args.source_offset,'segments':rows},indent=2)+'\n')
  (args.output_dir/(name+'_transcript.txt')).write_text('\n'.join(f"[{s['start']+args.source_offset:.2f}-{s['end']+args.source_offset:.2f}] {s['text'].strip()}" for s in rows)+'\n')
  summary['runs'][name]={'elapsed_seconds':time.monotonic()-start,'segments':len(rows)};(args.output_dir/'summary.json').write_text(json.dumps(summary,indent=2)+'\n');print('Completed',name,flush=True)
if __name__=='__main__':main()
