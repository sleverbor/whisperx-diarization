#!/usr/bin/env python3
"""Build a blinded wording-corroboration review from confirmed replay pairs."""

import argparse
from difflib import SequenceMatcher
import json
import os
from pathlib import Path
import re
import shutil


def words(text):
    return re.findall(r"[a-z']+", text.lower())


def differences(left, right):
    a, b = words(left), words(right)
    rows = []
    for operation, i1, i2, j1, j2 in SequenceMatcher(None, a, b).get_opcodes():
        if operation != "equal":
            rows.append({"operation": operation, "first": " ".join(a[i1:i2]),
                         "second": " ".join(b[j1:j2])})
    return rows


def quality(evidence_rows, start, end):
    scores = []
    for segment in evidence_rows:
        if float(segment["end"]) < start or float(segment["start"]) > end:
            continue
        for word in segment.get("words", []):
            if word.get("score") is not None and float(word.get("end", start)) >= start \
                    and float(word.get("start", end)) <= end:
                scores.append(float(word["score"]))
    return {
        "scored_words": len(scores),
        "mean_alignment_score": sum(scores) / len(scores) if scores else None,
        "low_score_fraction": (sum(score < .30 for score in scores) / len(scores)
                               if scores else None),
    }


def prepare_items(labels, evidence_rows, prior_dir: Path, output_dir: Path):
    media = output_dir / "media"
    media.mkdir(parents=True, exist_ok=True)
    items = []
    for source in labels:
        if source.get("Relationship") != "same_event":
            continue
        diffs = differences(source["left_text"], source["right_text"])
        if not diffs:
            continue
        item = {key: source[key] for key in (
            "review_id", "run_id", "left_start", "left_end", "right_start",
            "right_end", "left_text", "right_text", "text_similarity"
        )}
        item["differences"] = diffs
        item["first_quality"] = quality(evidence_rows, source["left_start"], source["left_end"])
        item["second_quality"] = quality(evidence_rows, source["right_start"], source["right_end"])
        first_score = item["first_quality"]["mean_alignment_score"]
        second_score = item["second_quality"]["mean_alignment_score"]
        item["blinded_quality_prediction"] = (
            "unavailable" if first_score is None or second_score is None else
            "first" if first_score > second_score + .05 else
            "second" if second_score > first_score + .05 else "same"
        )
        for side in ("left", "right"):
            source_path = prior_dir / source[f"{side}_video"]
            name = f"{source['review_id']}-{side}.mp4"
            destination = media / name
            if not destination.exists():
                try:
                    os.link(source_path, destination)
                except OSError:
                    shutil.copy2(source_path, destination)
            item[f"{side}_video"] = f"media/{name}"
        items.append(item)
    return items


def page(items):
    review_items = [
        {key: value for key, value in item.items()
         if key != "blinded_quality_prediction"}
        for item in items
    ]
    encoded = json.dumps(review_items).replace("</", "<\\/")
    return f'''<!doctype html><html><head><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1"><title>Replay wording corroboration</title><style>
body{{font:16px system-ui,sans-serif;max-width:1200px;margin:auto;padding:24px;background:#f3f5f7;color:#17202a}}.intro,.card{{background:white;padding:18px;margin:14px 0;border-radius:12px;box-shadow:0 1px 4px #0002}}.done{{border-left:7px solid #25864b}}.pair{{display:grid;grid-template-columns:1fr 1fr;gap:14px}}video{{width:100%;max-height:380px;background:#111}}.text,.diff{{background:#eef2f8;padding:10px;border-radius:8px;margin-top:6px}}.diff{{background:#fff4d6}}fieldset{{border:0;padding:5px 0}}legend{{font-weight:700}}label{{display:inline-block;margin:4px 14px 4px 0}}textarea{{width:100%;min-height:58px;box-sizing:border-box}}.sticky{{position:sticky;bottom:0;background:#17202a;color:white;padding:13px;border-radius:10px;display:flex;justify-content:space-between;align-items:center}}button{{font:inherit;padding:9px 14px}}small{{color:#556}}@media(max-width:760px){{.pair{{grid-template-columns:1fr}}}}</style></head><body>
<h1>Replay wording corroboration</h1><div class="intro">These pairs were already confirmed as the same recorded event. Judge which transcript wording matches the audio. The automatic quality prediction is hidden so it cannot bias the review. Select <i>combined</i> when each recording preserves different correct words. No transcript is changed by this page.</div><div id="root"></div><div class="sticky"><span id="progress"></span><button onclick="downloadLabels()">Download repeat-corroboration-labels.json</button></div>
<script>const items={encoded},key='repeat-corroboration-review-v1',saved=JSON.parse(localStorage.getItem(key)||'{{}}');const esc=s=>String(s??'').replace(/[&<>"']/g,c=>({{'&':'&amp;','<':'&lt;','>':'&gt;','"':'&quot;',"'":'&#39;'}}[c]));const choices=(id,title,values)=>`<fieldset><legend>${{title}}</legend>${{values.map(v=>`<label><input type="radio" name="${{id}}-${{title}}" value="${{v}}"> ${{v.replaceAll('_',' ')}}</label>`).join('')}}</fieldset>`;
function body(x){{let diffs=x.differences.map(d=>`<div><b>${{d.operation}}</b> · first: “${{esc(d.first||'∅')}}” · second: “${{esc(d.second||'∅')}}”</div>`).join('');return `<h2>${{esc(x.review_id)}}</h2><div class="pair"><div><b>First</b><video controls preload="metadata" src="${{x.left_video}}"></video><div class="text">${{esc(x.left_text)}}</div></div><div><b>Second</b><video controls preload="metadata" src="${{x.right_video}}"></video><div class="text">${{esc(x.right_text)}}</div></div></div><div class="diff"><b>Transcript differences</b>${{diffs}}</div>`+choices(x.review_id,'More accurate wording',['first','second','combined','equivalent','neither','unclear'])+choices(x.review_id,'Missing speech recovery',['first_recovers','second_recovers','each_recovers_different_words','neither','unclear'])+choices(x.review_id,'Safe as corroboration',['yes','no','unclear'])+`<label><b>Corrected wording or optional notes</b><textarea></textarea></label>`}}
const required=['More accurate wording','Missing speech recovery','Safe as corroboration'];function done(x){{return required.every(k=>saved[x.review_id]?.[k])}}function save(c,x){{let o={{notes:c.querySelector('textarea').value}};c.querySelectorAll('input:checked').forEach(i=>o[i.name.split(x.review_id+'-')[1]]=i.value);saved[x.review_id]=o;localStorage.setItem(key,JSON.stringify(saved));c.classList.toggle('done',done(x));progress()}}function render(){{let root=document.getElementById('root');items.forEach(x=>{{let c=document.createElement('section');c.className='card';c.innerHTML=body(x);root.appendChild(c);let old=saved[x.review_id]||{{}};c.querySelectorAll('input').forEach(i=>{{if(old[i.name.split(x.review_id+'-')[1]]===i.value)i.checked=true;i.addEventListener('change',()=>save(c,x))}});let t=c.querySelector('textarea');t.value=old.notes||'';t.addEventListener('input',()=>save(c,x));c.classList.toggle('done',done(x))}});progress()}}function progress(){{document.getElementById('progress').textContent=`Completed ${{items.filter(done).length}} of ${{items.length}}`}}function downloadLabels(){{let out={{schema_version:1,review_name:'repeat-corroboration-review',labels:items.map(x=>({{...x,...(saved[x.review_id]||{{}})}}))}},a=document.createElement('a');a.href=URL.createObjectURL(new Blob([JSON.stringify(out,null,2)],{{type:'application/json'}}));a.download='repeat-corroboration-labels.json';a.click()}}render();</script></body></html>'''


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--labels", type=Path, required=True)
    p.add_argument("--evidence", type=Path, required=True)
    p.add_argument("--prior-review-dir", type=Path, required=True)
    p.add_argument("--output-dir", type=Path, required=True)
    a = p.parse_args(); a.output_dir.mkdir(parents=True, exist_ok=True)
    labels = json.loads(a.labels.read_text())["labels"]
    evidence = json.loads(a.evidence.read_text())["segments"]
    items = prepare_items(labels, evidence, a.prior_review_dir, a.output_dir)
    (a.output_dir / "manifest.json").write_text(json.dumps(items, indent=2) + "\n")
    (a.output_dir / "index.html").write_text(page(items))
    print(json.dumps({"cards": len(items), "output": str(a.output_dir)}, indent=2))


if __name__ == "__main__": main()
