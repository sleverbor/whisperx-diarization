#!/usr/bin/env python3
"""Add aligned YouTube wording to a MossFormer2 review without changing its media."""

import argparse
import json
from pathlib import Path


def merge_caption_evidence(items, caption_report):
    captions = {row["review_id"]: row for row in caption_report["results"]}
    merged = []
    for item in items:
        row = dict(item)
        caption = captions.get(item["review_id"])
        row["caption_evidence"] = None if caption is None else {
            "text": caption.get("caption_text", ""),
            "relation": caption.get("relation", ""),
            "caption_type": caption_report.get("caption_type", "youtube_automatic"),
            "speaker_identity_available": False,
            "automatic_text_insertion": False,
        }
        merged.append(row)
    return merged


def html(items):
    encoded = json.dumps(items, ensure_ascii=False).replace("</", "<\\/")
    return f'''<!doctype html><html><head><meta charset="utf-8"><title>MossFormer2 caption-assisted review</title><style>
body{{font:16px system-ui;max-width:1050px;margin:24px auto;padding:0 16px;background:#f4f6f8;color:#17202a}}.card,.intro{{background:white;padding:16px;border-radius:10px;margin:16px 0}}video,audio{{width:100%;margin:6px 0}}.meta{{color:#59636e}}label{{display:inline-block;margin:5px 14px 5px 0}}textarea{{width:100%;min-height:50px}}.done{{border-left:7px solid #1f9d55}}.sticky{{position:sticky;bottom:0;background:#e8edf3;padding:10px}}button{{padding:10px;font-size:16px}}.caption{{background:#fff7d6;border-left:5px solid #d39e00;padding:12px;margin:12px 0}}.caption small{{display:block;color:#665b32;margin-top:6px}}
</style></head><body><h1>MossFormer2 review with caption evidence</h1><div class="intro">Listen before judging. The yellow box is the automatic YouTube caption aligned by time. It can help with wording, but it does not identify the speaker and never changes the transcript automatically.</div><div id="root"></div><div class="sticky"><span id="progress"></span> <button onclick="downloadLabels()">Download mossformer2-priority-labels.json</button></div><script>
const items={encoded},key='mossformer2-priority-review-v1',saved=JSON.parse(localStorage.getItem(key)||'{{}}');const outcomes=['adds_missing_target_speech','cleaner_same_target_line','mixed_but_useful','wrong_or_garbled','not_target','unclear'];const esc=s=>String(s??'').replace(/[&<>"']/g,c=>({{'&':'&amp;','<':'&lt;','>':'&gt;','"':'&quot;',"'":'&#39;'}}[c]));
function render(){{items.forEach((x,i)=>{{let v=saved[x.review_id]||{{}},c=document.createElement('div'),caption=x.caption_evidence;c.className='card '+(v.outcome?'done':'');c.innerHTML=`<div class="meta">${{i+1}}/${{items.length}} · ${{x.start.toFixed(2)}}–${{x.end.toFixed(2)}} sec · baseline ${{x.baseline_index}}</div><p><b>Baseline:</b> ${{esc(x.baseline_text)}}</p><p><b>MossFormer2 selected text:</b> ${{esc(x.candidate_text)}}</p>${{caption?`<div class="caption"><b>YouTube automatic caption:</b> ${{esc(caption.text)||'<i>No caption words in this interval</i>'}}<small>Wording and timing evidence only · no speaker identity · relation: ${{esc(caption.relation).replaceAll('_',' ')}}</small></div>`:''}}<video controls preload="metadata" src="${{esc(x.video)}}"></video>${{x.streams.map(s=>`<h3>Stream ${{s.stream}}${{s.stream===x.selected_stream?' (selected)':''}}</h3><audio controls preload="none" src="${{esc(s.audio)}}"></audio><details><summary>Machine evidence</summary>Similarity ${{s.similarity.toFixed(3)}} · ASR: ${{esc(s.transcript)}}</details>`).join('')}}<fieldset><legend>What did the selected output provide?</legend>${{outcomes.map(y=>`<label><input type="radio" name="${{x.review_id}}" value="${{y}}" ${{v.outcome===y?'checked':''}}> ${{y.replaceAll('_',' ')}}</label>`).join('')}}</fieldset><textarea placeholder="Correct target words or optional notes">${{esc(v.notes||'')}}</textarea>`;c.querySelectorAll('input,textarea').forEach(e=>e.oninput=e.onchange=()=>save(c,x.review_id));root.appendChild(c)}});updateProgress()}}
function save(c,id){{let checked=c.querySelector('input:checked');saved[id]={{outcome:checked?.value||'',notes:c.querySelector('textarea').value}};localStorage.setItem(key,JSON.stringify(saved));c.classList.toggle('done',Boolean(checked));updateProgress()}}function updateProgress(){{document.getElementById('progress').textContent=`Reviewed ${{items.filter(x=>saved[x.review_id]?.outcome).length}} of ${{items.length}}`}}function downloadLabels(){{let out={{schema_version:1,labels:items.map(x=>({{...x,...(saved[x.review_id]||{{}})}}))}},a=document.createElement('a');a.href=URL.createObjectURL(new Blob([JSON.stringify(out,null,2)],{{type:'application/json'}}));a.download='mossformer2-priority-labels.json';a.click()}}render();</script></body></html>'''


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--caption-report", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    source = json.loads(args.manifest.read_text())
    items = source if isinstance(source, list) else source.get("items")
    if items is None:
        raise ValueError("Manifest must be a list or contain an items list")
    merged = merge_caption_evidence(items, json.loads(args.caption_report.read_text()))
    args.output.write_text(html(merged))
    args.manifest.write_text(json.dumps({"schema_version": 2, "items": merged}, indent=2) + "\n")
    print(f"Added caption evidence to {len(merged)} review items: {args.output}")


if __name__ == "__main__":
    main()
