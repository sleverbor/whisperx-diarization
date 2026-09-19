#!/usr/bin/env python3
"""Build a small offline benchmark UI for speaker-separation experiments."""

import argparse
import json
from pathlib import Path
import shutil
import subprocess


def choose_exchanges(labels, count=5):
    rows = [
        row for row in labels
        if row.get("overlap_label") == "true_overlap"
        and row.get("target_in_overlap") == "yes"
        and row.get("notes", "").strip()
    ]
    rows.sort(key=lambda row: float(row["start"]))
    if len(rows) <= count:
        return rows
    indices = {round(i * (len(rows) - 1) / (count - 1)) for i in range(count)}
    return [rows[i] for i in sorted(indices)]


def build_html(items):
    data = json.dumps(items).replace("</", "<\\/")
    return f'''<!doctype html><html lang="en"><head><meta charset="utf-8"><meta name="viewport" content="width=device-width">
<title>Separation experiment review</title><style>
body{{font:16px system-ui;max-width:1050px;margin:24px auto;padding:0 16px;background:#f6f7f9;color:#17202a}}h1{{margin-bottom:4px}}
.intro,.card{{background:#fff;padding:16px;border-radius:10px}}.card{{margin:16px 0;box-shadow:0 1px 4px #0002}}video,audio{{width:100%;margin:8px 0}}
.meta{{color:#52606d;font-size:14px}}label{{display:inline-block;margin:5px 16px 5px 0}}textarea{{width:100%;min-height:58px;margin:5px 0}}
select,input[type=range]{{margin:5px}}button{{font-size:16px;padding:10px 15px;margin:6px}}.done{{border-left:7px solid #1f9d55}}.sticky{{position:sticky;bottom:0;background:#eef2f7;padding:10px;border-radius:10px}}
</style></head><body><h1>Five-exchange separation benchmark</h1>
<div class="intro"><p>Review the original video, then the current speaker-conditioned extraction. Describe the target and other speaker separately. Judge the extracted stream by what it actually contains; do not reward it merely for sounding cleaner.</p><p>These labels remain fixed ground truth for contextual WeSep and later two-output comparisons.</p></div>
<div id="items"></div><div class="sticky"><span id="progress"></span><button onclick="exportLabels()">Download separation-labels.json</button><button onclick="clearLabels()">Clear saved labels</button></div>
<script>const items={data};const key='separation-experiment-labels-v1';const saved=JSON.parse(localStorage.getItem(key)||'{{}}');
function esc(s){{return String(s??'').replace(/[&<>"']/g,c=>({{'&':'&amp;','<':'&lt;','>':'&gt;','"':'&quot;',"'":'&#39;'}}[c]))}}
function render(){{const root=document.getElementById('items');root.innerHTML='';items.forEach((x,i)=>{{let v=saved[x.exchange_id]||{{}};let c=document.createElement('div');c.className='card '+(v.extraction_identity?'done':'');c.innerHTML=`<div class="meta">${{i+1}}/${{items.length}} · source ${{x.start.toFixed(2)}}–${{x.end.toFixed(2)}} sec · baseline ${{x.baseline_index}}</div><p><b>Earlier hand label:</b> ${{esc(x.prior_notes)}}</p><video controls preload="metadata" src="${{esc(x.video)}}"></video><p><b>Current speaker-conditioned extraction</b></p>${{x.current_extraction?`<audio controls preload="none" src="${{esc(x.current_extraction)}}"></audio>`:'<p>Not available.</p>'}}
<label>Extracted stream contains <select data-f="extraction_identity"><option value="">Choose…</option>${{['target_only','mostly_target','mixed_speakers','mostly_other','other_only','no_intelligible_speech','unclear'].map(y=>`<option ${{v.extraction_identity===y?'selected':''}} value="${{y}}">${{y.replaceAll('_',' ')}}</option>`).join('')}}</select></label>
<p>Target words</p><textarea data-f="target_words">${{esc(v.target_words||'')}}</textarea><p>Other speaker words</p><textarea data-f="other_words">${{esc(v.other_words||'')}}</textarea>
<fieldset><legend>Basis for target attribution</legend>${{['voice','visible_face_or_lips','conversation_context'].map(y=>`<label><input data-b="${{y}}" type="checkbox" ${{(v.basis||[]).includes(y)?'checked':''}}> ${{y.replaceAll('_',' ')}}</label>`).join('')}}</fieldset>
<label>Speaker confidence <input data-f="speaker_confidence" type="range" min="0" max="100" value="${{v.speaker_confidence??50}}"> <span>${{v.speaker_confidence??50}}</span>%</label><p>Notes</p><textarea data-f="notes">${{esc(v.notes||'')}}</textarea>`;
c.querySelectorAll('textarea,select,input').forEach(el=>{{el.oninput=()=>save(c,x.exchange_id);el.onchange=()=>save(c,x.exchange_id)}});root.appendChild(c)}});progress()}}
function save(c,id){{let v={{}};c.querySelectorAll('[data-f]').forEach(e=>v[e.dataset.f]=e.type==='range'?Number(e.value):e.value);v.basis=[...c.querySelectorAll('[data-b]:checked')].map(e=>e.dataset.b);saved[id]=v;localStorage.setItem(key,JSON.stringify(saved));c.classList.toggle('done',Boolean(v.extraction_identity));let r=c.querySelector('input[type=range]');if(r)r.nextElementSibling.textContent=r.value;progress()}}
function progress(){{document.getElementById('progress').textContent=`Reviewed ${{items.filter(x=>saved[x.exchange_id]?.extraction_identity).length}} of ${{items.length}}`}}
function exportLabels(){{let out={{schema_version:1,purpose:'fixed_ground_truth_for_separation_experiments',labels:items.map(x=>({{...x,...(saved[x.exchange_id]||{{}})}}))}};let a=document.createElement('a');a.href=URL.createObjectURL(new Blob([JSON.stringify(out,null,2)],{{type:'application/json'}}));a.download='separation-labels.json';a.click();URL.revokeObjectURL(a.href)}}
function clearLabels(){{if(confirm('Clear all saved labels?')){{localStorage.removeItem(key);location.reload()}}}}render();</script></body></html>'''


def build(args):
    payload = json.loads(args.labels.read_text())
    selected = choose_exchanges(payload["labels"], args.count)
    if len(selected) < args.count:
        raise ValueError(f"Only {len(selected)} suitable labeled exchanges found")
    report = json.loads(args.report.read_text())
    reports = {int(row["baseline_index"]): row for row in report["segments"]}
    if args.output_dir.exists():
        shutil.rmtree(args.output_dir)
    media = args.output_dir / "media"; media.mkdir(parents=True)
    items = []
    for number, row in enumerate(selected, 1):
        start=max(0.0,float(row["start"])-args.context); end=float(row["end"])+args.context
        stem=f"{number:02d}-{int(row['baseline_index']):04d}"
        video=media/f"{stem}-source.mp4"
        subprocess.run(["ffmpeg","-nostdin","-hide_banner","-loglevel","error","-y","-ss",str(start),"-i",str(args.video),"-t",str(end-start),"-c:v","libx264","-preset","veryfast","-crf","25","-c:a","aac","-movflags","+faststart",str(video)],check=True)
        extracted=None; detail=reports.get(int(row["baseline_index"]),{})
        rel=detail.get("extracted",{}).get("audio")
        if rel:
            source=args.extraction_dir/Path(rel).name
            if not source.is_file():
                source=args.extraction_dir/f"current-{int(row['baseline_index']):04d}.wav"
            if source.is_file():
                target=media/f"{stem}-current.wav";shutil.copy2(source,target);extracted=f"media/{target.name}"
        items.append({"exchange_id":stem,"baseline_index":int(row["baseline_index"]),"start":float(row["start"]),"end":float(row["end"]),"clip_start":start,"clip_end":end,"video":f"media/{video.name}","current_extraction":extracted,"baseline_text":row.get("text",""),"prior_notes":row.get("notes","")})
    (args.output_dir/"manifest.json").write_text(json.dumps(items,indent=2)+"\n")
    (args.output_dir/"index.html").write_text(build_html(items))
    (args.output_dir/"README.txt").write_text("Open index.html, review all five exchanges, and download separation-labels.json. Labels save in the browser.\n")
    return items


def main():
    p=argparse.ArgumentParser();p.add_argument("--labels",type=Path,required=True);p.add_argument("--video",type=Path,required=True);p.add_argument("--report",type=Path,required=True);p.add_argument("--extraction-dir",type=Path,required=True);p.add_argument("--output-dir",type=Path,required=True);p.add_argument("--count",type=int,default=5);p.add_argument("--context",type=float,default=4.0)
    args=p.parse_args();items=build(args);print(f"Built {len(items)} exchanges in {args.output_dir}")

if __name__=="__main__":main()
