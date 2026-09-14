"""Create a reversible conservative transcript view from saved evidence.
Alignment scores are screening signals, not probabilities of correct wording.
"""
import argparse
import copy
import hashlib
import json
import math
from pathlib import Path
from statistics import mean


def screen(segment, text_threshold=.4, speaker_threshold=.6):
    scores=[float(w['score']) for w in segment.get('words',[]) if isinstance(w.get('score'),(float,int)) and math.isfinite(w['score'])]
    alignment=mean(scores) if scores else None
    weak_fraction=sum(s<.1 for s in scores)/len(scores) if scores else None
    reasons=[]
    if not segment.get('text','').strip():reasons.append('empty text')
    if alignment is None:reasons.append('no word-alignment evidence')
    elif alignment<text_threshold:reasons.append('mean alignment below threshold')
    if weak_fraction is not None and weak_fraction>.5:reasons.append('majority of scored words have very weak alignment')
    speaker=segment.get('final_speaker','Uncertain')
    attribution=float(segment.get('final_confidence',0))
    speaker_known=speaker not in ('Uncertain','Unknown',None) and math.isfinite(attribution) and attribution>=speaker_threshold
    return {'include_text':not reasons,'display_speaker':(speaker or 'Uncertain'),'speaker_uncertain':not speaker_known,'mean_alignment':alignment,'very_weak_word_fraction':weak_fraction,'speaker_strength':attribution,'omission_reasons':reasons}


def timestamp(seconds):
    seconds=max(0,round(seconds));return f'{seconds//60:02d}:{seconds%60:02d}'


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('source',type=Path);p.add_argument('--output-dir',type=Path,required=True)
    p.add_argument('--text-threshold',type=float,default=.4);p.add_argument('--speaker-threshold',type=float,default=.6)
    args=p.parse_args()
    if not all(math.isfinite(t) and 0<=t<=1 for t in (args.text_threshold,args.speaker_threshold)):p.error('Thresholds must be finite and between zero and one')
    if args.output_dir.exists():p.error('Use a new output directory')
    baseline=json.loads(args.source.read_text());snapshot=copy.deepcopy(baseline)
    lines=['Conservative transcript — provisional screening thresholds',
           'Doubtful text is omitted. Existing speaker labels are preserved; a speaker uncertain note flags weak attribution.',
           'Alignment scores and speaker strengths are not calibrated accuracy probabilities. Gaps may be silence or missed speech.',
           'Speaker track IDs are model labels; separate IDs may belong to the same person.','']
    decisions=[];omitted=[];kept=[];cursor=0.;uncertain=0
    for index,s in enumerate(sorted(baseline['segments'],key=lambda r:r['start'])):
        if s['start']-cursor>=5:lines.append(f'[{timestamp(cursor)}–{timestamp(s["start"])}] [No transcript available; not reviewed]')
        decision=screen(s,args.text_threshold,args.speaker_threshold);decisions.append({'segment_index_in_time_order':index,'start':s['start'],'end':s['end'],**decision})
        span=f'[{timestamp(s["start"])}–{timestamp(s["end"])}]'
        if decision['include_text']:
            kept.append(copy.deepcopy(s));uncertain+=decision['speaker_uncertain']
            note=' (speaker uncertain)' if decision['speaker_uncertain'] and decision['display_speaker'] not in ('Uncertain','Unknown') else ''
            lines.append(f'{span} {decision["display_speaker"]}{note}: {s["text"].strip()}')
        else:
            omitted.append({'original_segment':copy.deepcopy(s),'screening':decision})
            lines.append(f'{span} [Doubtful material omitted]')
        cursor=max(cursor,s['end'])
    assert baseline==snapshot
    args.output_dir.mkdir(parents=True)
    (args.output_dir/'transcript.txt').write_text('\n'.join(lines)+'\n')
    (args.output_dir/'omitted_material.json').write_text(json.dumps(omitted,indent=2)+'\n')
    (args.output_dir/'omitted_material.txt').write_text('\n'.join(f'[{timestamp(r["original_segment"]["start"])}–{timestamp(r["original_segment"]["end"])}] {r["original_segment"]["text"]}\n  Reasons: {", ".join(r["screening"]["omission_reasons"])}; mean alignment={r["screening"]["mean_alignment"]}' for r in omitted)+'\n')
    report={'source_sha256':hashlib.sha256(args.source.read_bytes()).hexdigest(),'text_threshold':args.text_threshold,'speaker_threshold':args.speaker_threshold,'very_weak_word_threshold':.1,'maximum_very_weak_fraction':.5,'confidence_is_calibrated':False,'original_segments':len(baseline['segments']),'retained_segments':len(kept),'omitted_segments':len(omitted),'retained_with_uncertain_speaker':uncertain,'decisions':decisions,'pending_recovery_candidates':baseline.get('gap_recovery_candidates',[])}
    (args.output_dir/'screening_report.json').write_text(json.dumps(report,indent=2)+'\n')
    print({k:v for k,v in report.items() if k not in ('decisions','pending_recovery_candidates')})

if __name__=='__main__':main()
