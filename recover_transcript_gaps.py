"""Recover review candidates only inside uncovered transcript intervals.

Existing segments are copied unchanged. Candidates are separate, have no speaker
identity, and require review. Detection of a transcript gap does not prove speech.
"""
import argparse
import copy
import hashlib
import json
from pathlib import Path
import re
import subprocess
import numpy as np


def uncovered_intervals(segments, duration, minimum_gap=2.0):
    cursor=0.0; gaps=[]
    for segment in sorted(segments,key=lambda s:s['start']):
        start=max(0.0,min(duration,float(segment['start'])))
        end=max(start,min(duration,float(segment['end'])))
        if start-cursor>=minimum_gap: gaps.append((cursor,start))
        cursor=max(cursor,end)
    if duration-cursor>=minimum_gap:gaps.append((cursor,duration))
    return gaps


def recovery_windows(gap,duration,size=8.0,overlap=4.0,context=.5):
    if size<=0 or not 0<=overlap<size:raise ValueError('Invalid window size/overlap')
    left=max(0.,gap[0]-context); right=min(duration,gap[1]+context)
    if right-left<=size:return [(left,right)]
    starts=[];start=left
    while start+size<right:
        starts.append(start); start+=size-overlap
    final=max(left,right-size)
    if not starts or abs(final-starts[-1])>1e-6:starts.append(final)
    return [(s,min(s+size,right)) for s in starts]


def word_key(text):return re.sub(r'[^\w]+','',text.casefold())


def collect_candidates(observations,gap):
    # Retain all eligible words for review; repeated words need distinct windows.
    clusters=[]
    for word in sorted(observations,key=lambda w:(w['start'],w['window_index'])):
        if word['start']<gap[0] or word['end']>gap[1] or word['end']<=word['start']:continue
        key=word_key(word['word'])
        if not key:continue
        matches=[c for c in clusters if c['key']==key and abs(c['anchor']-(word['start']+word['end'])/2)<=.6 and word['window_index'] not in {x['window_index'] for x in c['observations']}]
        if matches:
            closest=min(matches,key=lambda c:abs(c['anchor']-(word['start']+word['end'])/2));closest['observations'].append(word)
        else:clusters.append({'key':key,'anchor':(word['start']+word['end'])/2,'observations':[word]})
    # Keep words from one decoder window together; never splice hypotheses.
    support={}
    for cluster in clusters:
        indices=sorted({w['window_index'] for w in cluster['observations']})
        for w in cluster['observations']:
            support[(w['window_index'],w['start'],w['end'],w['word'])]=indices
    hypotheses=[]
    for index in sorted({w['window_index'] for w in observations}):
        runs=[]
        for w in sorted((w for w in observations if w['window_index']==index and w['start']>=gap[0] and w['end']<=gap[1] and w['end']>w['start'] and word_key(w['word'])),key=lambda w:w['start']):
            word=dict(w,supporting_windows=support.get((index,w['start'],w['end'],w['word']),[index]))
            if runs and word['start']-runs[-1]['end']<=.65:
                runs[-1]['words'].append(word);runs[-1]['end']=max(runs[-1]['end'],word['end'])
            else:runs.append({'start':word['start'],'end':word['end'],'words':[word],'window_index':index})
        for run in runs:
            repeated=sum(len(w['supporting_windows'])>=2 for w in run['words'])
            run.update(text=''.join(w['word'] for w in run['words']).strip(),supported_word_count=repeated,supported_word_fraction=repeated/len(run['words']),repeated_in_overlapping_windows=repeated/len(run['words'])>=.5,review_required=True,speaker='Uncertain')
            hypotheses.append(run)
    selected=[]
    for run in sorted(hypotheses,key=lambda r:(r['supported_word_count'],sum(w['probability'] for w in r['words'])/len(r['words']),r['end']-r['start']),reverse=True):
        if any(min(run['end'],chosen['end'])-max(run['start'],chosen['start'])>.15 for chosen in selected):continue
        selected.append(run)
    return sorted(selected,key=lambda r:r['start'])


def preserve_baseline(baseline,candidates):
    result=copy.deepcopy(baseline)
    result['gap_recovery_candidates']=copy.deepcopy(candidates)
    result['gap_recovery_note']='Existing segments unchanged; candidates require review and have no attributed speaker. Repeated decoding is not ground truth.'
    return result


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('video',type=Path)
    parser.add_argument('--baseline',required=True,type=Path)
    parser.add_argument('--output-dir',required=True,type=Path)
    parser.add_argument('--minimum-gap',type=float,default=2.)
    parser.add_argument('--window-seconds',type=float,default=8.)
    parser.add_argument('--overlap-seconds',type=float,default=4.)
    parser.add_argument('--language',default='en',help='Language of the working transcript')
    args=parser.parse_args()
    if args.output_dir.exists():parser.error('Choose a new output directory; existing results are never overwritten')
    if args.minimum_gap<=0:parser.error('Minimum gap must be positive')
    if args.window_seconds<=0 or not 0<=args.overlap_seconds<args.window_seconds:parser.error('Invalid overlap/window duration')
    baseline=json.loads(args.baseline.read_text())
    raw=subprocess.check_output(['ffmpeg','-nostdin','-hide_banner','-loglevel','error','-i',str(args.video),'-vn','-ar','16000','-ac','1','-f','f32le','pipe:1'])
    audio=np.frombuffer(raw,dtype='<f4').copy();duration=len(audio)/16000
    gaps=uncovered_intervals(baseline['segments'],duration,args.minimum_gap)
    args.output_dir.mkdir(parents=True)
    candidates=[];decodes=[];model=None;device=None
    if gaps:
        import torch
        from faster_whisper import WhisperModel
        device='cuda' if torch.cuda.is_available() else 'cpu'
        model=WhisperModel('large-v2',device=device,compute_type='float16' if device=='cuda' else 'int8',cpu_threads=4)
    for gap_index,gap in enumerate(gaps):
        observations=[]
        for window_index,(left,right) in enumerate(recovery_windows(gap,duration,args.window_seconds,args.overlap_seconds)):
            segments,info=model.transcribe(audio[int(left*16000):int(right*16000)],language=args.language,vad_filter=False,beam_size=5,condition_on_previous_text=False,word_timestamps=True)
            rows=[]
            for segment in segments:
                eligible=segment.avg_logprob>=-1.0 and segment.no_speech_prob<=.6 and segment.compression_ratio<=2.4
                row={'start':left+segment.start,'end':left+segment.end,'text':segment.text,'avg_logprob':segment.avg_logprob,'no_speech_prob':segment.no_speech_prob,'compression_ratio':segment.compression_ratio,'quality_filter_passed':eligible,'words':[]}
                for word in segment.words or []:
                    item={'start':left+word.start,'end':left+word.end,'word':word.word,'probability':word.probability,'window_index':window_index}
                    row['words'].append(item)
                    item['low_confidence']=word.probability<.4
                    if eligible:observations.append(item)
                rows.append(row)
            decodes.append({'gap_index':gap_index,'window_index':window_index,'window_start':left,'window_end':right,'segments':rows})
            (args.output_dir/'window_decodes.json').write_text(json.dumps(decodes,indent=2)+'\n')
            print('Decoded gap',gap_index+1,'window',window_index+1,f'{left:.2f}-{right:.2f}',flush=True)
        for candidate in collect_candidates(observations,gap):
            candidate['gap_index']=gap_index;candidates.append(candidate)
    output=preserve_baseline(baseline,candidates)
    assert output['segments']==baseline['segments'],'Existing transcript changed'
    offset=float(baseline.get('source_offset_seconds',0))
    output['gap_recovery_settings']={'model':'large-v2','device':device,'vad_filter':False,'condition_on_previous_text':False,'window_seconds':args.window_seconds,'overlap_seconds':args.overlap_seconds,'minimum_gap':args.minimum_gap,'gaps':gaps,'baseline_sha256':hashlib.sha256(args.baseline.read_bytes()).hexdigest(),'video_sha256':hashlib.sha256(args.video.read_bytes()).hexdigest()}
    (args.output_dir/'transcript_with_candidates.json').write_text(json.dumps(output,indent=2)+'\n')
    lines=[]
    for s in baseline['segments']:lines.append((s['start'],f"[{s['start']+offset:.2f}-{s['end']+offset:.2f}] {s.get('final_speaker',s.get('speaker','Unknown'))}: {s['text']}"))
    for s in candidates:
        support=f"overlap support {s['supported_word_count']}/{len(s['words'])} words" if s['supported_word_count'] else 'single decode'
        lines.append((s['start'],f"[{s['start']+offset:.2f}-{s['end']+offset:.2f}] REVIEW ({support}; speaker unknown): {s['text']}"))
    (args.output_dir/'review_transcript.txt').write_text('\n'.join(text for _,text in sorted(lines))+'\n')
    print('Completed; preserved',len(baseline['segments']),'existing segments;',len(candidates),'review candidates',flush=True)

if __name__=='__main__':main()
