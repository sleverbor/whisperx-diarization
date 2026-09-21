"""Auditor-only enrollment data; predictions never become enrollment approvals."""
from pathlib import Path
from datetime import datetime, timezone
import hashlib
import json
import re
from urllib.parse import urlparse
import numpy as np


def youtube_url(value):
    parsed = urlparse(value)
    host = (parsed.hostname or '').lower()
    if parsed.scheme != 'https' or host not in ('youtube.com','www.youtube.com','m.youtube.com','youtu.be') or parsed.username or parsed.password:
        raise ValueError('Use an HTTPS YouTube channel, playlist, or video URL')
    return value


def unit(vector):
    x = np.asarray(vector, dtype=np.float32).reshape(-1)
    norm = np.linalg.norm(x)
    if not np.isfinite(x).all() or norm <= 0:
        raise ValueError('Invalid embedding')
    return x / norm


def word_key(text):
    return ' '.join(re.findall(r"[a-z0-9']+", text.lower()))


def voice_reference(records):
    # Short speech is retained in a separate library; it cannot contaminate the
    # longer-speech identity reference even when manually approved.
    return [r for r in records if r.get('voice_decision') == 'yes' and r['duration'] >= 2 and r.get('voice_vector')]


def approved_short(records):
    return [r for r in records if r.get('voice_decision') == 'yes' and .3 <= r['duration'] <= 1.6 and r.get('voice_vector')]


def cosine_scores(vector, rows):
    q = unit(vector)
    return [float(q @ unit(x)) for x in rows]


def rank_candidates(candidates, confirmed, goal='balanced'):
    voices = voice_reference(confirmed)
    vectors = [r['voice_vector'] for r in voices]
    ordered = []
    for row in candidates:
        if row.get('voice_decision') in ('yes','no','unsure'):
            continue
        x = dict(row)
        scores = cosine_scores(row['voice_vector'],vectors) if vectors and row.get('voice_vector') else []
        x['voice_similarity'] = float(np.median(sorted(scores, reverse=True)[:3])) if scores else None
        score = .45 * max(row.get('face_similarity') or 0,0) + .55 * max(x['voice_similarity'] or 0,0)
        if goal == 'short' and .3 <= row['duration'] <= 1.6: score += .5
        if goal == 'long' and row['duration'] >= 2: score += .3
        if goal in ('front','side') and row.get('view_hint') == goal: score += .3
        # Diversity is a scheduling preference, never an identity vote.
        approved_here = sum(r.get('voice_decision')=='yes' and r.get('source_url')==row.get('source_url') for r in confirmed)
        score -= min(.3, approved_here * .04)
        x['ranking_score'] = score
        ordered.append(x)
    return sorted(ordered,key=lambda x:x['ranking_score'],reverse=True)


def decide(row, voice, face, style, corrected_text=None):
    if voice not in ('yes','no','unsure') or face not in ('yes','no','unsure','not_visible'):
        raise ValueError('Confirm face and voice separately')
    if style not in ('normal','raised','quiet','other'):
        raise ValueError('Invalid speaking style')
    if voice == 'yes' and not row.get('voice_vector'):
        raise ValueError('No usable voice embedding')
    if face == 'yes' and not row.get('face_vector'):
        raise ValueError('No unambiguous face sample; use a seed image instead')
    result = dict(row)
    result.update(voice_decision=voice,face_decision=face,style=style,
                  reviewed_at=datetime.now(timezone.utc).isoformat())
    if corrected_text is not None:
        result['confirmed_text'] = corrected_text.strip()[:1000]
    return result


def export_reference(session, output):
    from build_target_reference import screen
    records = session['candidates']
    voice = voice_reference(records)
    faces = session['seeds'] + [r for r in records if r.get('face_decision')=='yes' and r.get('face_vector')]
    if len(voice) < 3 or len(faces) < 3:
        raise ValueError('Need at least 3 approved voice clips of 2+ seconds and 3 confirmed face samples')
    raw = {'voice': np.stack([r['voice_vector'] for r in voice]),
           'face': np.stack([r['face_vector'] for r in faces])}
    selected = {}
    audits = {}
    for kind, dimensions in [('voice',192),('face',512)]:
        accepted, centroid, audit = screen(raw[kind],.45,dimensions)
        selected[kind] = (accepted,centroid)
        audits[kind] = audit
    output = Path(output)
    output.mkdir(parents=True,exist_ok=False)
    for kind,(accepted,centroid) in selected.items():
        np.save(output/f'{kind}_raw_embeddings.npy',raw[kind])
        np.save(output/f'{kind}_embeddings.npy',accepted)
        np.save(output/f'{kind}_embedding.npy',centroid)
    short = approved_short(records)
    audit = {'method':'manual identity approval followed by consistency screening',
             'screening':audits,'voice_sources':[r['id'] for r in voice],
             'face_sources':[r['id'] for r in faces],
             'short_library_ids':[r['id'] for r in short],
             'note':'No inference is an approval. Short samples excluded from the main voice centroid. Hold evaluation videos out of enrollment.'}
    (output/'reference.json').write_text(json.dumps(audit,indent=2)+'\n')
    # The session remains the full provenance and media library; export a manifest
    # without duplicating rejected/uncertain embeddings or their identities.
    library = [{k:r.get(k) for k in ('id','source_url','source_start','source_end','confirmed_text','style','clip','audio','voice_vector')} for r in short]
    (output/'short_speech_library.json').write_text(json.dumps(library,indent=2)+'\n')
    return audit


def compare_voice(vector, records, phrase=''):
    short = approved_short(records)
    if phrase:
        short = [r for r in short if word_key(r.get('confirmed_text','')) == word_key(phrase)]
    matches = []
    for r in short:
        matches.append({'id':r['id'],'similarity':cosine_scores(vector,[r['voice_vector']])[0],
                        'text':r.get('confirmed_text',r.get('text','')),'audio':r['audio'],
                        'source_url':r['source_url']})
    long = voice_reference(records)
    long_scores = cosine_scores(vector,[r['voice_vector'] for r in long]) if long else []
    return {'short_matches': sorted(matches,key=lambda x:x['similarity'],reverse=True)[:5],
            'long_reference_median_similarity':float(np.median(long_scores)) if long_scores else None,
            'identity_decision':None,
            'note':'Similarity is not identity probability. No automatic enrollment or speaker assignment.'}
