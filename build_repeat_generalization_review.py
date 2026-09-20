#!/usr/bin/env python3
"""Build paired-video review cards for text-repeat generalization candidates."""

import argparse
import json
import subprocess
from pathlib import Path


def extract(source: Path, destination: Path, start: float, end: float):
    if destination.is_file() and destination.stat().st_size:
        return
    clip_start = max(0.0, start - 1.5)
    duration = end - start + 3.0
    subprocess.run([
        "ffmpeg", "-hide_banner", "-loglevel", "error", "-y",
        "-ss", str(clip_start), "-i", str(source), "-t", str(duration),
        "-map", "0:v:0", "-map", "0:a:0?", "-c", "copy",
        "-avoid_negative_ts", "make_zero", str(destination),
    ], check=True)


def prepare_items(data, output_dir: Path):
    media = output_dir / "media"
    media.mkdir(parents=True, exist_ok=True)
    items = []
    for run in data["runs"]:
        video = Path(run["video"])
        if not video.is_file():
            raise FileNotFoundError(video)
        for number, candidate in enumerate(run["candidates"], 1):
            review_id = f"{run['run_id']}-{number:02d}"
            item = {**candidate, "review_id": review_id,
                    "run_id": run["run_id"]}
            for side in ("left", "right"):
                name = f"{review_id}-{side}.mp4"
                extract(video, media / name, candidate[f"{side}_start"],
                        candidate[f"{side}_end"])
                item[f"{side}_video"] = f"media/{name}"
            items.append(item)
    return items


def page(items):
    encoded = json.dumps(items).replace("</", "<\\/")
    return f'''<!doctype html><html><head><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1"><title>Repeated-scene generalization review</title><style>
body{{font:16px system-ui,sans-serif;max-width:1200px;margin:auto;padding:24px;background:#f3f5f7;color:#17202a}}.intro,.card{{background:white;padding:18px;margin:14px 0;border-radius:12px;box-shadow:0 1px 4px #0002}}.done{{border-left:7px solid #25864b}}.pair{{display:grid;grid-template-columns:1fr 1fr;gap:14px}}video{{width:100%;max-height:380px;background:#111}}.text{{background:#eef2f8;padding:10px;border-radius:8px;margin-top:6px}}fieldset{{border:0;padding:5px 0}}legend{{font-weight:700}}label{{display:inline-block;margin:4px 14px 4px 0}}textarea{{width:100%;min-height:58px;box-sizing:border-box}}.sticky{{position:sticky;bottom:0;background:#17202a;color:white;padding:13px;border-radius:10px;display:flex;justify-content:space-between;align-items:center}}button{{font:inherit;padding:9px 14px}}small{{color:#556}}@media(max-width:760px){{.pair{{grid-template-columns:1fr}}}}</style></head><body>
<h1>Repeated-scene generalization review</h1><div class="intro">Review each pair without assuming the matching words come from the same event. Choose whether this is the same recorded event, similar wording from different events, a false text match, or unclear. If it is the same event, note whether the shared dialogue is complete or only partial. Answers save automatically.</div><div id="root"></div><div class="sticky"><span id="progress"></span><button onclick="downloadLabels()">Download repeat-generalization-labels.json</button></div>
<script>const items={encoded},key='repeat-generalization-review-v1',saved=JSON.parse(localStorage.getItem(key)||'{{}}');const esc=s=>String(s??'').replace(/[&<>"']/g,c=>({{'&':'&amp;','<':'&lt;','>':'&gt;','"':'&quot;',"'":'&#39;'}}[c]));
const choices=(id,title,values)=>`<fieldset><legend>${{title}}</legend>${{values.map(v=>`<label><input type="radio" name="${{id}}-${{title}}" value="${{v}}"> ${{v.replaceAll('_',' ')}}</label>`).join('')}}</fieldset>`;
function body(x){{return `<h2>${{esc(x.review_id)}} <small>similarity ${{x.text_similarity.toFixed(3)}}</small></h2><div class="pair"><div><b>First · ${{x.left_start.toFixed(2)}}–${{x.left_end.toFixed(2)}}</b><video controls preload="metadata" src="${{x.left_video}}"></video><div class="text">${{esc(x.left_text)}}</div></div><div><b>Second · ${{x.right_start.toFixed(2)}}–${{x.right_end.toFixed(2)}}</b><video controls preload="metadata" src="${{x.right_video}}"></video><div class="text">${{esc(x.right_text)}}</div></div></div>`+choices(x.review_id,'Relationship',['same_event','similar_words_different_event','false_match','unclear'])+choices(x.review_id,'Dialogue alignment',['same_dialogue_complete','same_dialogue_partial','misleading_text_match','unclear'])+choices(x.review_id,'Clearer recording',['first','second','same','unclear'])+`<label><b>Optional notes</b><textarea></textarea></label>`}}
const required=['Relationship','Dialogue alignment','Clearer recording'];function done(x){{return required.every(k=>saved[x.review_id]?.[k])}}function save(c,x){{let o={{notes:c.querySelector('textarea').value}};c.querySelectorAll('input:checked').forEach(i=>o[i.name.split(x.review_id+'-')[1]]=i.value);saved[x.review_id]=o;localStorage.setItem(key,JSON.stringify(saved));c.classList.toggle('done',done(x));progress()}}function render(){{const root=document.getElementById('root');items.forEach(x=>{{let c=document.createElement('section');c.className='card';c.innerHTML=body(x);root.appendChild(c);let old=saved[x.review_id]||{{}};c.querySelectorAll('input').forEach(i=>{{if(old[i.name.split(x.review_id+'-')[1]]===i.value)i.checked=true;i.addEventListener('change',()=>save(c,x))}});let t=c.querySelector('textarea');t.value=old.notes||'';t.addEventListener('input',()=>save(c,x));c.classList.toggle('done',done(x))}});progress()}}function progress(){{document.getElementById('progress').textContent=`Completed ${{items.filter(done).length}} of ${{items.length}}`}}function downloadLabels(){{let out={{schema_version:1,review_name:'repeat-generalization-review',labels:items.map(x=>({{...x,...(saved[x.review_id]||{{}})}}))}},a=document.createElement('a');a.href=URL.createObjectURL(new Blob([JSON.stringify(out,null,2)],{{type:'application/json'}}));a.download='repeat-generalization-labels.json';a.click()}}render();</script></body></html>'''


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--candidates", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    items = prepare_items(json.loads(args.candidates.read_text()), args.output_dir)
    (args.output_dir / "manifest.json").write_text(json.dumps(items, indent=2) + "\n")
    (args.output_dir / "index.html").write_text(page(items))
    print(json.dumps({"cards": len(items), "output": str(args.output_dir)}, indent=2))


if __name__ == "__main__":
    main()
