"""Collect exact-recording evidence without changing transcript or speaker fields.

Useful for edited videos containing replays. Same words or similar voices are
not enough: this requires high normalized waveform correlation. Raw source
track IDs are scoped to their own transcript; never merged across runs.
"""
import argparse
import copy
import json
import math
from pathlib import Path
import subprocess
import numpy as np
from scipy.signal import butter, correlate, find_peaks, sosfiltfilt

SAMPLE_RATE = 16000


def read_audio(path, start, duration):
    raw = subprocess.check_output(['ffmpeg', '-nostdin', '-hide_banner',
        '-loglevel', 'error', '-ss', str(start), '-i', str(path), '-t',
        str(duration), '-vn', '-ar', str(SAMPLE_RATE), '-ac', '1',
        '-f', 'f32le', 'pipe:1'])
    audio = np.frombuffer(raw, dtype='<f4').copy()
    if not len(audio) or not np.isfinite(audio).all():
        raise ValueError('Empty or non-finite decoded audio')
    return audio


def best_waveform_matches(template, query, threshold=.95, min_seconds=.6):
    """All well-separated strong matches; preserve ambiguity of repeated clips."""
    if not .85 <= threshold <= 1:
        raise ValueError('Correlation threshold must be between .85 and 1')
    template, query = np.asarray(template, dtype=float), np.asarray(query, dtype=float)
    if template.ndim != 1 or query.ndim != 1:
        raise ValueError('Expected mono arrays')
    if not np.isfinite(template).all() or not np.isfinite(query).all():
        raise ValueError('Non-finite waveform')
    if len(template) < SAMPLE_RATE*min_seconds or len(template) > len(query):
        return []
    energy = np.dot(template, template)
    if energy < 1e-10:
        return []
    cumulative = np.cumsum(np.r_[0., query*query])
    window_energy = np.maximum(cumulative[len(template):]-cumulative[:-len(template)], 0)
    dots = correlate(query, template, mode='valid', method='fft')
    scores = np.clip(dots / np.sqrt(np.maximum(window_energy*energy, 1e-20)), -1, 1)
    # Padding allows a genuine match at either endpoint to be found.
    peaks, _ = find_peaks(np.r_[-np.inf, scores, -np.inf],
                         height=threshold, distance=max(1, len(template)//2))
    return [{'sample_start':int(i-1),'waveform_similarity':float(scores[i-1])}
            for i in peaks]


def collect(document, source_document, source_audio, query_audio,
            source_offset=0., query_offset=0., source_name='source', threshold=.95):
    """Add separate evidence; confidence values are inherited, not calibrated."""
    result = copy.deepcopy(document)
    matches = []
    band = butter(4,[150,4500],btype='bandpass',fs=SAMPLE_RATE,output='sos')
    source_audio = sosfiltfilt(band,source_audio)
    query_audio = sosfiltfilt(band,query_audio)
    for index, s in enumerate(source_document['segments']):
        left, right = float(s['start'])-source_offset, float(s['end'])-source_offset
        if left < 0 or right > len(source_audio)/SAMPLE_RATE or right <= left:
            continue
        template = source_audio[round(left*SAMPLE_RATE):round(right*SAMPLE_RATE)]
        found = best_waveform_matches(template, query_audio, threshold=threshold)
        label = s.get('final_speaker',s.get('speaker','Uncertain'))
        if label not in ('Target_Speaker','Uncertain','Unknown',None):
            label = source_name+':'+str(label)
        for m in found:
            a = query_offset+m['sample_start']/SAMPLE_RATE
            b = a+len(template)/SAMPLE_RATE
            evidence = {'source':'duplicate_recording','source_document':source_name,
                'source_segment_index':index,'source_start':s['start'],
                'source_end':s['end'],'query_start':a,'query_end':b,
                'speaker_hypothesis':label,
                'source_identity_strength':s.get('final_confidence'),
                'waveform_similarity':m['waveform_similarity'],
                'counts_as_independent_voice_sample':False,
                'multiple_recording_matches':len(found)>1,
                'identity_probability_calibrated':False}
            matches.append(evidence)
            for row in result['segments']:
                overlap = max(0,min(float(row['end']),b)-max(float(row['start']),a))
                if overlap <= 0:
                    continue
                duration = float(row['end'])-float(row['start'])
                item = dict(evidence,segment_overlap_fraction=overlap/duration if duration>0 else 0)
                row.setdefault('supplemental_evidence',[]).append(item)
    return result, matches


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--video',type=Path,required=True)
    p.add_argument('--source-transcript',type=Path,required=True)
    p.add_argument('--transcript',type=Path,required=True)
    p.add_argument('--source-start',type=float,required=True)
    p.add_argument('--source-duration',type=float,required=True)
    p.add_argument('--query-start',type=float,required=True)
    p.add_argument('--query-duration',type=float,required=True)
    p.add_argument('--source-name',default='source')
    p.add_argument('--source-times-local',action='store_true')
    p.add_argument('--threshold',type=float,default=.95)
    p.add_argument('--output',type=Path,required=True)
    a=p.parse_args()
    for v in (a.source_start,a.query_start,a.source_duration,a.query_duration):
        if not math.isfinite(v) or v<0: p.error('Times must be nonnegative and finite')
    if not a.source_duration or not a.query_duration: p.error('Duration must be positive')
    if a.output.resolve() in (a.transcript.resolve(),a.source_transcript.resolve()):
        p.error('Output must be a separate file to preserve input transcripts')
    result,matches=collect(json.loads(a.transcript.read_text()),
        json.loads(a.source_transcript.read_text()),
        read_audio(a.video,a.source_start,a.source_duration),
        read_audio(a.video,a.query_start,a.query_duration),
        source_offset=0 if a.source_times_local else a.source_start,
        query_offset=a.query_start,source_name=a.source_name,threshold=a.threshold)
    a.output.parent.mkdir(parents=True,exist_ok=True)
    a.output.write_text(json.dumps(result,indent=2)+'\n')
    a.output.with_suffix('.matches.json').write_text(json.dumps(matches,indent=2)+'\n')
    print(f'Collected {len(matches)} duplicate-recording matches; baseline fields preserved.')

if __name__=='__main__': main()
