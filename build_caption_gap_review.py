#!/usr/bin/env python3
"""Find caption-timed speech outside baseline coverage and build a video review."""

import argparse
import json
from pathlib import Path
import subprocess

from evaluate_youtube_captions_on_priority import timed_caption_words, token_f1


def uncovered_groups(timed_words, segments, coverage_padding=.35, group_gap=1.25,
                     minimum_words=2):
    uncovered = [row for row in timed_words if not any(
        float(segment["start"]) - coverage_padding <= row["time"]
        <= float(segment["end"]) + coverage_padding
        for segment in segments
    )]
    groups = []
    for row in uncovered:
        if not groups or row["time"] - groups[-1][-1]["time"] > group_gap:
            groups.append([row])
        else:
            groups[-1].append(row)
    return [group for group in groups if len(group) >= minimum_words]


def caption_excerpt(timed_words, start, end):
    return " ".join(
        row["text"].strip() for row in timed_words
        if start <= row["time"] <= end
    ).strip()


def neighboring_segments(segments, start, end):
    before = [row for row in segments if float(row["end"]) < start]
    after = [row for row in segments if float(row["start"]) > end]
    return (before[-1] if before else None, after[0] if after else None)


def duplicates_nearby_transcript(caption_text, before, after, threshold=.8):
    """Suppress caption timing drift when the same words are already present."""
    neighbors = [row.get("text", "") for row in (before, after) if row]
    return any(token_f1(caption_text, text) >= threshold for text in neighbors)


def review_html(items):
    data = json.dumps(items, ensure_ascii=False).replace("</", "<\\/")
    return f'''<!doctype html><html><head><meta charset="utf-8"><title>Caption missing-speech review</title><style>
body{{font:16px system-ui;max-width:1000px;margin:24px auto;padding:0 16px;background:#f4f6f8;color:#17202a}}.intro,.card{{background:white;padding:16px;border-radius:10px;margin:16px 0}}video{{width:100%}}.caption{{background:#fff7d6;border-left:5px solid #d39e00;padding:12px}}.context{{color:#4f5963}}label{{display:block;margin:7px 0}}textarea{{width:100%;min-height:60px}}.done{{border-left:7px solid #1f9d55}}.sticky{{position:sticky;bottom:0;background:#e8edf3;padding:10px}}button{{padding:10px;font-size:16px}}
</style></head><body><h1>Possible missing speech</h1><div class="intro">These four intervals contain timed YouTube caption words outside the existing transcript coverage. Watch each clip and decide whether speech is actually missing. Captions are wording hints only and do not identify the speaker.</div><div id="root"></div><div class="sticky"><span id="progress"></span> <button onclick="downloadLabels()">Download caption-gap-labels.json</button></div><script>
const items={data},key='caption-gap-review-v1',saved=JSON.parse(localStorage.getItem(key)||'{{}}'),outcomes=['missing_speech','caption_duplicate_or_timing_error','noise_or_not_speech','already_covered','unclear'],speakers=['target','non_target','multiple_or_overlapping','unknown'];const esc=s=>String(s??'').replace(/[&<>"']/g,c=>({{'&':'&amp;','<':'&lt;','>':'&gt;','"':'&quot;',"'":'&#39;'}}[c]));
function render(){{items.forEach((x,i)=>{{let v=saved[x.review_id]||{{}},c=document.createElement('div');c.className='card '+(v.outcome?'done':'');c.innerHTML=`<b>${{i+1}}/${{items.length}} · source ${{x.candidate_start.toFixed(2)}}–${{x.candidate_end.toFixed(2)}} sec</b><div class="caption"><b>Automatic caption near the gap:</b> ${{esc(x.caption_text)}}</div><p class="context"><b>Transcript before:</b> ${{esc(x.before_text)||'<i>none</i>'}}<br><b>Transcript after:</b> ${{esc(x.after_text)||'<i>none</i>'}}</p><video controls preload="metadata" src="${{esc(x.video)}}"></video><fieldset><legend>What is this gap?</legend>${{outcomes.map(y=>`<label><input type="radio" name="outcome-${{x.review_id}}" value="${{y}}" ${{v.outcome===y?'checked':''}}> ${{y.replaceAll('_',' ')}}</label>`).join('')}}</fieldset><fieldset><legend>If speech is missing, who speaks?</legend>${{speakers.map(y=>`<label><input type="radio" name="speaker-${{x.review_id}}" value="${{y}}" ${{v.speaker===y?'checked':''}}> ${{y.replaceAll('_',' ')}}</label>`).join('')}}</fieldset><textarea placeholder="Correct words or optional notes">${{esc(v.notes||'')}}</textarea>`;c.querySelectorAll('input,textarea').forEach(e=>e.oninput=e.onchange=()=>save(c,x.review_id));root.appendChild(c)}});updateProgress()}}
function save(c,id){{let outcome=c.querySelector('input[name^="outcome-"]:checked'),speaker=c.querySelector('input[name^="speaker-"]:checked');saved[id]={{outcome:outcome?.value||'',speaker:speaker?.value||'',notes:c.querySelector('textarea').value}};localStorage.setItem(key,JSON.stringify(saved));c.classList.toggle('done',Boolean(outcome));updateProgress()}}function updateProgress(){{document.getElementById('progress').textContent=`Reviewed ${{items.filter(x=>saved[x.review_id]?.outcome).length}} of ${{items.length}}`}}function downloadLabels(){{let out={{schema_version:1,labels:items.map(x=>({{...x,...(saved[x.review_id]||{{}})}}))}},a=document.createElement('a');a.href=URL.createObjectURL(new Blob([JSON.stringify(out,null,2)],{{type:'application/json'}}));a.download='caption-gap-labels.json';a.click()}}render();</script></body></html>'''


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--captions", type=Path, required=True)
    parser.add_argument("--baseline", type=Path, required=True)
    parser.add_argument("--video", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()
    timed = timed_caption_words(json.loads(args.captions.read_text()))
    segments = json.loads(args.baseline.read_text())["segments"]
    groups = uncovered_groups(timed, segments)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    media = args.output_dir / "media"; media.mkdir(exist_ok=True)
    items = []
    suppressed_duplicates = 0
    for index, group in enumerate(groups, 1):
        candidate_start = max(0.0, group[0]["time"] - .6)
        candidate_end = group[-1]["time"] + .8
        clip_start, clip_end = max(0, candidate_start - 2), candidate_end + 2
        before, after = neighboring_segments(segments, candidate_start, candidate_end)
        caption_text = caption_excerpt(
            timed, candidate_start - .4, candidate_end + .4
        )
        if duplicates_nearby_transcript(caption_text, before, after):
            suppressed_duplicates += 1
            continue
        filename = f"gap-{index:02d}.mp4"
        subprocess.run([
            "ffmpeg", "-nostdin", "-hide_banner", "-loglevel", "error", "-y",
            "-ss", str(clip_start), "-to", str(clip_end), "-i", str(args.video),
            "-map", "0:v:0", "-map", "0:a:0", "-c:v", "libx264", "-preset", "veryfast",
            "-crf", "24", "-c:a", "aac", "-movflags", "+faststart", str(media / filename),
        ], check=True)
        items.append({
            "review_id": f"gap-{index:02d}",
            "candidate_start": candidate_start, "candidate_end": candidate_end,
            "clip_start": clip_start, "clip_end": clip_end,
            "video": f"media/{filename}",
            "caption_text": caption_text,
            "before_text": before.get("text", "") if before else "",
            "after_text": after.get("text", "") if after else "",
            "caption_does_not_identify_speaker": True,
            "automatic_text_insertion": False,
        })
    (args.output_dir / "manifest.json").write_text(json.dumps(items, indent=2) + "\n")
    (args.output_dir / "index.html").write_text(review_html(items))
    print(
        f"Built {len(items)} caption-gap reviews; suppressed "
        f"{suppressed_duplicates} nearby duplicates: {args.output_dir / 'index.html'}"
    )


if __name__ == "__main__":
    main()
