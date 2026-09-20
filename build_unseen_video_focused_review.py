#!/usr/bin/env python3
"""Build the focused human review for the uxOLBG1OcI0 unseen-video run."""

from __future__ import annotations

import argparse
import json
import shutil
import subprocess
from pathlib import Path


TARGET_TIMES = [193.97, 323.39, 419.71, 425.95, 463.77]
MOSS_INDICES = [0, 10, 220]
REPEAT_PAIRS = [
    ("repeat-a", 212.94, 216.94, 239.46, 245.62,
     "I can't force him to leave … I can talk to him …"),
    ("repeat-b", 226.84, 235.23, 255.79, 263.89,
     "Property-line explanation"),
    ("repeat-c", 354.05, 360.27, 366.33, 373.56,
     "Do his thing … what he does for a living …"),
]


def read_json(path: Path):
    return json.loads(path.read_text())


def nearest(rows, start, tolerance=0.15):
    row = min(rows, key=lambda x: abs(float(x["start"]) - start))
    if abs(float(row["start"]) - start) > tolerance:
        raise ValueError(f"No row near {start}; nearest is {row['start']}")
    return row


def assemble_items(results_dir: Path) -> list[dict]:
    targeted = read_json(results_dir / "targeted-review/review_hypotheses.json")["segments"]
    moss = read_json(results_dir / "mossformer2-review/review-evidence.json")["segments"]
    gaps = read_json(results_dir / "caption-gap-review/manifest.json")

    items = []
    for number, start in enumerate(TARGET_TIMES, 1):
        row = nearest(targeted, start)
        item = {
            "review_id": f"target-{number:02d}", "type": "target_candidate",
            "start": row["start"], "end": row["end"], "text": row["text"].strip(),
            "machine_hypothesis": row.get("review_speaker_hypothesis", "Target_Speaker"),
        }
        if start == 425.95:
            m = next(x for x in moss if x["baseline_index"] == 200)
            item["mossformer2"] = {
                "selected_stream": m["selected_stream"],
                "selected_audio_source": m["selected_audio"],
                "candidate_transcription": m["candidate_transcription"],
                "target_similarity": m["target_similarity"],
                "target_margin": m["target_margin"],
            }
        items.append(item)

    for gap in gaps:
        items.append({**gap, "type": "caption_gap"})

    for baseline_index in MOSS_INDICES:
        row = next(x for x in moss if x["baseline_index"] == baseline_index)
        items.append({
            "review_id": f"moss-{baseline_index:04d}", "type": "mossformer2",
            "start": row["start"], "end": row["end"],
            "baseline_index": baseline_index,
            "candidate_transcription": row["candidate_transcription"],
            "selected_stream": row["selected_stream"],
            "selected_audio_source": row["selected_audio"],
            "target_similarity": row["target_similarity"], "target_margin": row["target_margin"],
        })

    for review_id, a0, a1, b0, b1, text in REPEAT_PAIRS:
        items.append({"review_id": review_id, "type": "repeat_pair",
                      "first_start": a0, "first_end": a1,
                      "second_start": b0, "second_end": b1, "text": text})
    return items


def extract(video: Path, output: Path, start: float, end: float, context=2.0):
    if output.is_file() and output.stat().st_size > 0:
        return
    clip_start = max(0, start - context)
    duration = end - start + 2 * context
    subprocess.run([
        "ffmpeg", "-hide_banner", "-loglevel", "error", "-y", "-ss", str(clip_start),
        "-i", str(video), "-t", str(duration), "-map", "0:v:0", "-map", "0:a:0?",
        "-c:v", "libx264", "-preset", "veryfast", "-crf", "25", "-c:a", "aac", "-b:a", "128k",
        "-movflags", "+faststart", str(output),
    ], check=True)


def materialize_media(items, results_dir: Path, video: Path, output_dir: Path):
    media = output_dir / "media"
    media.mkdir(parents=True, exist_ok=True)
    for item in items:
        kind, rid = item["type"], item["review_id"]
        if kind == "repeat_pair":
            for label in ("first", "second"):
                name = f"{rid}-{label}.mp4"
                extract(video, media / name, item[f"{label}_start"], item[f"{label}_end"], 1.5)
                item[f"{label}_video"] = f"media/{name}"
            continue
        name = f"{rid}.mp4"
        start = item["candidate_start"] if "candidate_start" in item else item["start"]
        end = item["candidate_end"] if "candidate_end" in item else item["end"]
        extract(video, media / name, start, end)
        item["video"] = f"media/{name}"
        source = item.get("selected_audio_source")
        if source:
            audio_name = f"{rid}-selected.wav"
            shutil.copy2(results_dir / "mossformer2-review/inference" / source, media / audio_name)
            item["selected_audio"] = f"media/{audio_name}"
        embedded = item.get("mossformer2")
        if embedded:
            audio_name = f"{rid}-mossformer2.wav"
            shutil.copy2(results_dir / "mossformer2-review/inference" /
                         embedded["selected_audio_source"], media / audio_name)
            embedded["selected_audio"] = f"media/{audio_name}"


def html(items):
    data = json.dumps(items).replace("</", "<\\/")
    return f'''<!doctype html><html><head><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1"><title>Unseen-video focused review</title><style>
body{{font:16px system-ui,sans-serif;max-width:1040px;margin:auto;padding:24px;background:#f4f5f7;color:#18202a}}h1{{margin-bottom:6px}}.intro,.card{{background:white;border-radius:12px;padding:18px;margin:14px 0;box-shadow:0 1px 4px #0002}}.card.done{{border-left:7px solid #25864b}}video{{width:100%;max-height:430px;background:#111}}audio{{width:100%}}fieldset{{border:0;padding:6px 0;margin:8px 0}}legend{{font-weight:700;margin-bottom:5px}}label{{display:inline-block;margin:4px 14px 4px 0}}textarea{{width:100%;min-height:64px;box-sizing:border-box}}.machine{{background:#eef2f8;padding:10px;border-radius:8px}}.pair{{display:grid;grid-template-columns:1fr 1fr;gap:12px}}.sticky{{position:sticky;bottom:0;background:#18202a;color:white;padding:13px;border-radius:10px;display:flex;justify-content:space-between;align-items:center}}button{{font:inherit;padding:9px 14px}}small{{color:#556}}@media(max-width:700px){{.pair{{grid-template-columns:1fr}}}}</style></head><body>
<h1>Focused review: unseen video</h1><div class="intro"><b>Review 15 focused questions.</b> Play only the clips on each card. You do not need to watch the full video or transcribe every line. Judge what you can hear and see; choose <i>unclear</i> when the clip does not support a decision. Your work is saved in this browser as you go.</div><div id="root"></div><div class="sticky"><span id="progress"></span><button onclick="downloadLabels()">Download unseen-video-focused-labels.json</button></div>
<script>const items={data},key='unseen-video-focused-review-v1',saved=JSON.parse(localStorage.getItem(key)||'{{}}');
const esc=s=>String(s??'').replace(/[&<>"']/g,c=>({{'&':'&amp;','<':'&lt;','>':'&gt;','"':'&quot;',"'":'&#39;'}}[c]));
const radios=(id,name,values)=>`<fieldset><legend>${{name}}</legend>${{values.map(v=>`<label><input type="radio" name="${{id}}-${{name}}" value="${{v}}"> ${{v.replaceAll('_',' ')}}</label>`).join('')}}</fieldset>`;
function body(x){{let m=`<h2>${{esc(x.review_id)}}</h2>`;
if(x.type==='target_candidate'){{m+=`<p>Could this later line be the target speaker? Check identity, wording, and whether the video offers usable visual evidence.</p><video controls preload="metadata" src="${{x.video}}"></video><div class="machine"><b>Machine text:</b> ${{esc(x.text)}} <small>(${{x.start.toFixed(2)}}–${{x.end.toFixed(2)}})</small></div>`+radios(x.review_id,'Speaker identity',['target','non_target','mixed_or_overlapping','unclear'])+radios(x.review_id,'Machine wording',['correct','partly_correct','wrong','unclear'])+radios(x.review_id,'Target visible',['yes','no','unclear'])+radios(x.review_id,'Face usable',['yes','no','unclear']);if(x.mossformer2)m+=`<div class="machine"><b>Separated target candidate:</b> ${{esc(x.mossformer2.candidate_transcription)}}</div><audio controls src="${{x.mossformer2.selected_audio}}"></audio>`;}}
if(x.type==='caption_gap'){{m+=`<p>Does the caption reveal speech missing from the baseline transcript?</p><video controls preload="metadata" src="${{x.video}}"></video><div class="machine"><b>Caption:</b> ${{esc(x.caption_text)}}<br><small>Before: ${{esc(x.before_text)}} — After: ${{esc(x.after_text)}}</small></div>`+radios(x.review_id,'Gap outcome',['missing_speech','caption_duplicate_or_timing_error','noise_or_not_speech','already_covered','unclear'])+radios(x.review_id,'Speaker',['target','non_target','multiple_or_overlapping','unknown']);}}
if(x.type==='mossformer2'){{m+=`<p>Compare the original video with the separated audio. Does the separated stream recover useful target speech?</p><video controls preload="metadata" src="${{x.video}}"></video><div class="machine"><b>Separated text hint:</b> ${{esc(x.candidate_transcription)}}</div><audio controls src="${{x.selected_audio}}"></audio>`+radios(x.review_id,'Separated result',['adds_missing_target_speech','cleaner_same_target_line','mixed_but_useful','wrong_or_garbled','not_target','unclear']);}}
if(x.type==='repeat_pair'){{m+=`<p>Are these two clips presentations of the same event? If so, can the clearer version corroborate the other?</p><div class="machine">${{esc(x.text)}}</div><div class="pair"><div><b>First</b><video controls preload="metadata" src="${{x.first_video}}"></video></div><div><b>Second</b><video controls preload="metadata" src="${{x.second_video}}"></video></div></div>`+radios(x.review_id,'Same event',['same_event','not_same_event','unclear'])+radios(x.review_id,'Clearer version',['first','second','same','unclear'])+radios(x.review_id,'Speech wording',['same','different','unclear']);}}
return m+`<label><b>Optional notes</b><textarea></textarea></label>`}}
function required(x){{return x.type==='target_candidate'?['Speaker identity','Machine wording','Target visible','Face usable']:x.type==='caption_gap'?['Gap outcome','Speaker']:x.type==='mossformer2'?['Separated result']:['Same event','Clearer version','Speech wording']}}
function render(){{const root=document.getElementById('root');items.forEach(x=>{{const c=document.createElement('section');c.className='card';c.dataset.id=x.review_id;c.innerHTML=body(x);root.appendChild(c);const old=saved[x.review_id]||{{}};c.querySelectorAll('input').forEach(i=>{{if(old[i.name.split(x.review_id+'-')[1]]===i.value)i.checked=true;i.addEventListener('change',()=>save(c,x))}});const t=c.querySelector('textarea');t.value=old.notes||'';t.addEventListener('input',()=>save(c,x));mark(c,x)}});progress()}}
function save(c,x){{let o={{notes:c.querySelector('textarea').value}};c.querySelectorAll('input:checked').forEach(i=>o[i.name.split(x.review_id+'-')[1]]=i.value);saved[x.review_id]=o;localStorage.setItem(key,JSON.stringify(saved));mark(c,x);progress()}}
function mark(c,x){{c.classList.toggle('done',required(x).every(k=>saved[x.review_id]?.[k]))}}function progress(){{document.getElementById('progress').textContent=`Completed ${{items.filter(x=>required(x).every(k=>saved[x.review_id]?.[k])).length}} of ${{items.length}}`}}
function downloadLabels(){{const out={{schema_version:1,review_name:'unseen-video-focused-review',labels:items.map(x=>({{...x,...(saved[x.review_id]||{{}})}}))}};const a=document.createElement('a');a.href=URL.createObjectURL(new Blob([JSON.stringify(out,null,2)],{{type:'application/json'}}));a.download='unseen-video-focused-labels.json';a.click()}}render();</script></body></html>'''


def build(results_dir: Path, video: Path, output_dir: Path):
    output_dir.mkdir(parents=True, exist_ok=True)
    items = assemble_items(results_dir)
    materialize_media(items, results_dir, video, output_dir)
    (output_dir / "manifest.json").write_text(json.dumps(items, indent=2))
    (output_dir / "index.html").write_text(html(items))
    return items


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--results-dir", type=Path, required=True)
    p.add_argument("--video", type=Path, required=True)
    p.add_argument("--output-dir", type=Path, required=True)
    a = p.parse_args()
    print(json.dumps({"cards": len(build(a.results_dir, a.video, a.output_dir)),
                      "output": str(a.output_dir)}, indent=2))


if __name__ == "__main__":
    main()
