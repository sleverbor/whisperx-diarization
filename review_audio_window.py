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


def decoder_sentence_bounds(decoded_segments, aligned_segments):
    """Link generated sentence text to its own decoder words, in sequence.

    This matches two representations of the same ASR hypothesis, not supplied
    expected dialogue. Missing links yield None rather than invented timings.
    """
    text_parts=[];word_spans=[];base=0
    for segment in decoded_segments:
        body=' '.join(segment['text'].split());cursor=0
        for word in segment.get('words',[]):
            token=' '.join(word.get('word','').split())
            location=body.find(token,cursor) if token else -1
            if location<0:continue
            word_spans.append((base+location,base+location+len(token),float(word['start']),float(word['end'])))
            cursor=location+len(token)
        text_parts.append(body);base+=len(body)+1
    text=' '.join(text_parts);cursor=0;bounds=[]
    for segment in aligned_segments:
        sentence=' '.join(segment['text'].split());left=text.find(sentence,cursor) if sentence else -1
        if left<0:
            bounds.append(None);continue
        right=left+len(sentence);cursor=right
        words=[w for w in word_spans if w[0]<right and w[1]>left]
        if not words or max(w[3] for w in words)<=min(w[2] for w in words):
            bounds.append(None)
        else:bounds.append((min(w[2] for w in words),max(w[3] for w in words)))
    return bounds


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


def resolve_timing_evidence(variants, min_similarity=.25, min_margin=.08):
    """Use one evidence family: agree on the leading voice, with qualified support.

    Alternate crops are not independent votes and do not raise confidence.
    Conflicting leading identities abstain, even if one crop matches strongly.
    """
    usable=[scores for scores in variants if scores]
    if not usable:return 'Uncertain'
    leaders={max(scores,key=scores.get) for scores in usable}
    if len(leaders)!=1:return 'Uncertain'
    leader=next(iter(leaders))
    return leader if any(resolve_voice(scores,min_similarity,min_margin)==leader for scores in usable) else 'Uncertain'


def decoder_quality_flags(segments, start, end):
    """Keep decoder warnings as evidence; word probability is not accuracy."""
    observed=[s for s in segments if s['start']<end and s['end']>start]
    flags=[]
    if any(s.get('no_speech_prob',0)>.6 for s in observed):
        flags.append('Decoder marks this passage as possible non-speech')
    if any(s.get('avg_logprob',0)<-1 for s in observed):
        flags.append('Low decoder support for wording')
    if any(s.get('compression_ratio',0)>2.4 for s in observed):
        flags.append('Decoder wording may be repetitive')
    return flags, [{'start':s['start'],'end':s['end'],**{k:s[k] for k in ('avg_logprob','no_speech_prob','compression_ratio') if k in s}} for s in observed]


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
    p.add_argument('--asr-engine',choices=['native','bounded-whisperx'],default='native')
    p.add_argument('--timing-source',choices=['consensus','decoder','alignment'],default='consensus')
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
    if a.asr_engine=='native':
        from faster_whisper import WhisperModel
        asr=WhisperModel(a.model,device=device,compute_type='float16' if device=='cuda' else 'int8',cpu_threads=a.threads)
        decoded,_=asr.transcribe(audio,language=a.language,beam_size=5,
            vad_filter=False,condition_on_previous_text=False,word_timestamps=True)
        result={'language':a.language,'segments':[{'start':float(s.start),'end':float(s.end),'text':s.text,
            'avg_logprob':float(s.avg_logprob),'no_speech_prob':float(s.no_speech_prob),
            'compression_ratio':float(s.compression_ratio),
            'words':[{'start':float(w.start),'end':float(w.end),'word':w.word,'probability':float(w.probability)} for w in s.words or []]} for s in decoded]}
    else:
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
    decoder_bounds=decoder_sentence_bounds(result['segments'],aligned['segments'])
    for index,s in enumerate(aligned['segments']):
        alignment_left,alignment_right=float(s['start']),float(s['end']);scores={};reasons=[]
        bounds=decoder_bounds[index]
        use_decoder=a.timing_source in ('decoder','consensus') and bounds is not None
        left,right=bounds if use_decoder else (alignment_left,alignment_right)
        if a.timing_source in ('decoder','consensus') and bounds is None:reasons.append('Decoder word boundaries unavailable; alignment fallback needs review')
        if right-left>=.4:
            crop=torch.from_numpy(audio[round(left*16000):round(right*16000)]).unsqueeze(0).to(device)
            with torch.no_grad():v=unit(encoder.encode_batch(crop).detach().cpu().numpy())
            for name,profile in profiles.items():
                if v.shape!=profile.shape:raise ValueError('Reference embedding model/dimension mismatch')
                scores[name]=float(v@profile)
        else:reasons.append('Voice crop shorter than 0.4 seconds')
        variants=[{'source':'decoder' if use_decoder else 'alignment','start':a.start+left,'end':a.start+right,'scores':scores}]
        if a.timing_source=='consensus' and use_decoder and alignment_right-alignment_left>=.4 and (abs(left-alignment_left)>1/16000 or abs(right-alignment_right)>1/16000):
            crop=torch.from_numpy(audio[round(alignment_left*16000):round(alignment_right*16000)]).unsqueeze(0).to(device)
            with torch.no_grad():alternate=unit(encoder.encode_batch(crop).detach().cpu().numpy())
            variants.append({'source':'alignment','start':a.start+alignment_left,'end':a.start+alignment_right,'scores':{name:float(alternate@profile) for name,profile in profiles.items()}})
        speaker=resolve_timing_evidence([v['scores'] for v in variants],a.min_similarity,a.min_margin) if a.timing_source=='consensus' else resolve_voice(scores,a.min_similarity,a.min_margin)
        quality_flags,quality_signals=decoder_quality_flags(result['segments'],*(bounds if bounds is not None else (left,right)))
        reasons.extend(quality_flags)
        near_edge=left<=.25 or right>=len(audio)/16000-.25
        if near_edge:reasons.append('Near audio-window boundary; wording or timing may be incomplete')
        if speaker=='Uncertain':reasons.append('Insufficient voice similarity or separation between profiles')
        if len(profiles)==1:reasons.append('No competing voice reference; target-only match needs review')
        absolute_start,absolute_end=a.start+left,a.start+right
        rows.append({'index':index,'start':absolute_start,'end':absolute_end,'text':s['text'],
            'speaker_hypothesis':speaker,'transcription_status':'decoder_warning' if quality_flags else 'review_hypothesis','review_required':True,'near_window_boundary':near_edge,'reasons':reasons,
            'evidence':[{'source':'decoder_support','flags':quality_flags,'signals':quality_signals,'word_probability_is_accuracy':False},{'source':'timing_comparison','selected_source':'decoder' if use_decoder else 'alignment',
                'alignment_start':a.start+alignment_left,'alignment_end':a.start+alignment_right,
                'decoder_start':a.start+bounds[0] if bounds else None,'decoder_end':a.start+bounds[1] if bounds else None},baseline_evidence(baseline['segments'],absolute_start,absolute_end,a.baseline_offset),
                {'source':'local_voice','similarities':scores,'timing_variants':variants,'variants_are_independent_votes':False,'min_similarity':a.min_similarity,
                 'min_margin':a.min_margin,'identity_probability_calibrated':False}],
            'alignment_words':[{**w,**({'start':a.start+w['start']} if 'start' in w else {}),
                      **({'end':a.start+w['end']} if 'end' in w else {})} for w in s.get('words',[])]})
    document={'baseline_modified':False,'experimental':True,'segments':rows,
        'provenance':{'video':str(a.video.resolve()),'video_sha256':sha(a.video),
            'window_start':a.start,'decoded_duration':len(audio)/16000,
            'models':{'asr':a.model,'voice':'speechbrain/spkrec-ecapa-voxceleb'},
            'device':device,'requested_timing_source':a.timing_source,'asr_engine':a.asr_engine,'vad':'disabled' if a.asr_engine=='native' else 'bounded_silero','chunk_seconds':a.chunk_seconds if a.asr_engine=='bounded-whisperx' else None,
            'context_seconds':a.context_seconds if a.asr_engine=='bounded-whisperx' else None,'references':{k:{'path':str(v.resolve()),'sha256':sha(v)} for k,v in references.items()},
            'baseline':str(a.baseline.resolve()) if a.baseline else None,
            'baseline_sha256':sha(a.baseline) if a.baseline else None,'elapsed_seconds':time.monotonic()-started}}
    (a.output_dir/'review_hypotheses.json').write_text(json.dumps(document,indent=2)+'\n')
    (a.output_dir/'review_transcript.txt').write_text('\n'.join(f"[{r['start']:.2f}–{r['end']:.2f}] {r['speaker_hypothesis']}{' [decoder warning]' if r['transcription_status']=='decoder_warning' else ''}: {r['text']}" for r in rows)+'\n')
    print(f'Wrote {len(rows)} review hypotheses to {a.output_dir}; baseline preserved.')

if __name__=='__main__':main()
