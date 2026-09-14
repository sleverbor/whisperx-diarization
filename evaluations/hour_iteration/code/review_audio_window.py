"""Optional local audio-window review using the project's existing models.

Writes independent ASR/alignment/voice hypotheses, never edits a baseline.
No expected transcript text, named-video rules, clothing rules, or HF token.
"""
import argparse
import gc
import hashlib
import json
import math
from pathlib import Path
import subprocess
import time


def baseline_evidence(segments, start, end, offset=0):
    tracks = {}
    sources = []
    for index, s in enumerate(segments):
        overlap=max(0,min(end,float(s['end'])+offset)-max(start,float(s['start'])+offset))
        if not overlap: continue
        track=s.get('raw_speaker_track',s.get('baseline',{}).get('raw_speaker_track',s.get('base_track',s.get('speaker'))))
        sources.append({'segment_index':index,'raw_speaker_track':track,
            'baseline_speaker':s.get('final_speaker',s.get('speaker')),
            'overlap_seconds':overlap})
        if track is not None: tracks[track]=tracks.get(track,0)+overlap
    return {'source':'baseline_overlap','raw_track_overlap_seconds':tracks,
            'segments':sources,'identity_verified':False}


def resolve_voice(scores, min_similarity=.25, min_margin=.08):
    """Conservative review hypothesis; thresholds are not calibrated confidence."""
    if not scores: return 'Uncertain'
    ranked=sorted(scores,key=scores.get,reverse=True)
    # A margin requires at least one competing profile. With only a target
    # reference, use similarity alone but keep the explicit review requirement.
    runner=scores[ranked[1]] if len(ranked)>1 else None
    if scores[ranked[0]] < min_similarity: return 'Uncertain'
    if runner is not None and scores[ranked[0]]-runner < min_margin: return 'Uncertain'
    return ranked[0]


def sha(path):
    h=hashlib.sha256()
    with path.open('rb') as f:
        for chunk in iter(lambda:f.read(1024*1024),b''): h.update(chunk)
    return h.hexdigest()


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--video',type=Path,required=True)
    p.add_argument('--start',type=float,required=True)
    p.add_argument('--duration',type=float,required=True)
    p.add_argument('--target-reference',type=Path,required=True)
    p.add_argument('--other-reference',action='append',default=[],metavar='NAME=PATH')
    p.add_argument('--baseline',type=Path)
    p.add_argument('--baseline-offset',type=float,default=0,
                   help='Add this offset to baseline times; default assumes absolute video times')
    p.add_argument('--output-dir',type=Path,required=True)
    p.add_argument('--model',default='large-v2')
    p.add_argument('--language',default='en')
    p.add_argument('--device',choices=['auto','cpu','cuda'],default='auto')
    p.add_argument('--threads',type=int,default=4)
    p.add_argument('--chunk-seconds',type=float,default=20)
    p.add_argument('--context-seconds',type=float,default=.5)
    p.add_argument('--min-similarity',type=float,default=.25)
    p.add_argument('--min-margin',type=float,default=.08)
    p.add_argument('--speechbrain-cache',type=Path)
    a=p.parse_args()
    if not math.isfinite(a.start) or a.start<0 or not math.isfinite(a.duration) or a.duration<=0:
        p.error('Provide a nonnegative start and positive duration')
    if not math.isfinite(a.baseline_offset) or a.threads<1:
        p.error('Invalid offset or thread count')
    if not math.isfinite(a.chunk_seconds) or not 1<=a.chunk_seconds<=30:
        p.error('Chunk length must be between 1 and 30 seconds')
    if not -1<=a.min_similarity<=1 or not 0<=a.min_margin<=2:
        p.error('Invalid voice gates')
    references={'Target_Speaker':a.target_reference}
    for item in a.other_reference:
        if '=' not in item:p.error('Other reference must be NAME=PATH')
        name,path=item.split('=',1)
        if not name or name in references or name in ('Unknown','Uncertain'):p.error('Reference name must be unique')
        references[name]=Path(path)
    paths=[a.video,*references.values()]+([a.baseline] if a.baseline else [])
    for path in paths:
        if not path.is_file():p.error(f'File not found: {path}')
    planned=[a.output_dir/x for x in ('asr.json','alignment.json','review_hypotheses.json','review_transcript.txt')]
    if any(path.resolve() in [x.resolve() for x in paths] for path in planned):
        p.error('Outputs must be separate from input files')
    if any(path.exists() for path in planned):
        p.error('Choose an empty output directory to preserve earlier experiments')
    import numpy as np
    import torch
    import whisperx
    from bounded_silero_vad import make_vad
    torch.set_num_threads(a.threads)
    device=('cuda' if torch.cuda.is_available() else 'cpu') if a.device=='auto' else a.device
    if device=='cuda' and not torch.cuda.is_available():p.error('CUDA is unavailable')
    a.output_dir.mkdir(parents=True,exist_ok=True)
    started=time.monotonic()
    raw=subprocess.check_output(['ffmpeg','-nostdin','-hide_banner','-loglevel','error',
        '-ss',str(a.start),'-i',str(a.video),'-t',str(a.duration),'-vn',
        '-ar','16000','-ac','1','-f','f32le','pipe:1'])
    audio=np.frombuffer(raw,dtype='<f4').copy()
    if not len(audio) or not np.isfinite(audio).all():p.error('No valid audio decoded')
    asr=whisperx.load_model(a.model,device,compute_type='float16' if device=='cuda' else 'int8',
        language=a.language,vad_model=make_vad(context=a.context_seconds))
    result=asr.transcribe(audio,batch_size=4,chunk_size=a.chunk_seconds,language=a.language)
    (a.output_dir/'asr.json').write_text(json.dumps(result,indent=2)+'\n')
    del asr;gc.collect()
    if device=='cuda':torch.cuda.empty_cache()
    if result['segments']:
        aligner,metadata=whisperx.load_align_model(language_code=a.language,device=device)
        aligned=whisperx.align(result['segments'],aligner,metadata,audio,device,return_char_alignments=False)
        del aligner;gc.collect()
        if device=='cuda':torch.cuda.empty_cache()
    else:aligned={'segments':[],'word_segments':[]}
    (a.output_dir/'alignment.json').write_text(json.dumps(aligned,indent=2)+'\n')
    from speechbrain.inference.speaker import SpeakerRecognition
    options={'source':'speechbrain/spkrec-ecapa-voxceleb','run_opts':{'device':device}}
    if a.speechbrain_cache:options['savedir']=str(a.speechbrain_cache)
    encoder=SpeakerRecognition.from_hparams(**options)
    def unit(v):
        v=np.asarray(v,dtype=np.float32).reshape(-1)
        if not np.isfinite(v).all() or np.linalg.norm(v)<1e-8:raise ValueError('Invalid embedding')
        return v/np.linalg.norm(v)
    profiles={}
    for name,path in references.items():
        values=np.load(path,allow_pickle=False)
        if values.ndim==1:values=values[None,:]
        if values.ndim!=2 or not len(values):raise ValueError('Reference must contain one or more embeddings')
        profiles[name]=unit(np.mean([unit(v) for v in values],axis=0))
    baseline=json.loads(a.baseline.read_text()) if a.baseline else {'segments':[]}
    rows=[]
    for index,s in enumerate(aligned['segments']):
        left,right=float(s['start']),float(s['end']);scores={};reasons=[]
        if right-left>=.4:
            crop=torch.from_numpy(audio[round(left*16000):round(right*16000)]).unsqueeze(0).to(device)
            with torch.no_grad():v=unit(encoder.encode_batch(crop).detach().cpu().numpy())
            for name,profile in profiles.items():
                if v.shape!=profile.shape:raise ValueError('Reference embedding model/dimension mismatch')
                scores[name]=float(v@profile)
        else:reasons.append('Voice crop shorter than 0.4 seconds')
        speaker=resolve_voice(scores,a.min_similarity,a.min_margin)
        near_edge=left<=.25 or right>=len(audio)/16000-.25
        if near_edge:reasons.append('Near audio-window boundary; wording or timing may be incomplete')
        if speaker=='Uncertain':reasons.append('Insufficient voice similarity or separation between profiles')
        if len(profiles)==1:reasons.append('No competing voice reference; target-only match needs review')
        absolute_start,absolute_end=a.start+left,a.start+right
        rows.append({'index':index,'start':absolute_start,'end':absolute_end,'text':s['text'],
            'speaker_hypothesis':speaker,'review_required':True,'near_window_boundary':near_edge,'reasons':reasons,
            'evidence':[baseline_evidence(baseline['segments'],absolute_start,absolute_end,a.baseline_offset),
                {'source':'local_voice','similarities':scores,'min_similarity':a.min_similarity,
                 'min_margin':a.min_margin,'identity_probability_calibrated':False}],
            'words':[{**w,**({'start':a.start+w['start']} if 'start' in w else {}),
                      **({'end':a.start+w['end']} if 'end' in w else {})} for w in s.get('words',[])]})
    document={'baseline_modified':False,'experimental':True,'segments':rows,
        'provenance':{'video':str(a.video.resolve()),'video_sha256':sha(a.video),
            'window_start':a.start,'decoded_duration':len(audio)/16000,
            'models':{'asr':a.model,'voice':'speechbrain/spkrec-ecapa-voxceleb'},
            'device':device,'vad':'bounded_silero','chunk_seconds':a.chunk_seconds,
            'context_seconds':a.context_seconds,'references':{k:{'path':str(v.resolve()),'sha256':sha(v)} for k,v in references.items()},
            'baseline':str(a.baseline.resolve()) if a.baseline else None,
            'baseline_sha256':sha(a.baseline) if a.baseline else None,'elapsed_seconds':time.monotonic()-started}}
    (a.output_dir/'review_hypotheses.json').write_text(json.dumps(document,indent=2)+'\n')
    (a.output_dir/'review_transcript.txt').write_text('\n'.join(f"[{r['start']:.2f}–{r['end']:.2f}] {r['speaker_hypothesis']}: {r['text']}" for r in rows)+'\n')
    print(f'Wrote {len(rows)} review hypotheses to {a.output_dir}; baseline preserved.')

if __name__=='__main__':main()
