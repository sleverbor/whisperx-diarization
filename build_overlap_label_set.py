"""Build a balanced, offline listening set for overlap ground-truth labels."""

import argparse
import csv
import json
from pathlib import Path
import shutil
import subprocess


CATEGORY_LIMITS = {
    "both_on_control": 7,
    "sortformer_only_on_baseline_overlap": 7,
    "sortformer_only_on_control": 8,
    "diaper_only_on_control": 6,
    "neither_on_baseline_overlap": 7,
}


def spread(rows, count):
    rows = sorted(rows, key=lambda row: row["start"])
    if len(rows) <= count:
        return rows
    if count == 1:
        return [rows[len(rows) // 2]]
    indices = {round(index * (len(rows) - 1) / (count - 1)) for index in range(count)}
    return [rows[index] for index in sorted(indices)]


def categorize(sortformer_rows, diaper_rows, threshold=0.10):
    categories = {name: [] for name in CATEGORY_LIMITS}
    for sortformer, diaper in zip(sortformer_rows, diaper_rows):
        if sortformer["baseline_index"] != diaper["baseline_index"]:
            raise ValueError("Comparison rows are not aligned")
        expected = bool(sortformer["baseline_target_non_target_overlap"])
        sort_positive = sortformer["overlap_fraction"] >= threshold
        diaper_positive = diaper["overlap_fraction"] >= threshold
        row = dict(sortformer)
        row.update(
            sortformer_overlap_fraction=sortformer["overlap_fraction"],
            diaper_overlap_fraction=diaper["overlap_fraction"],
        )
        if not expected and sort_positive and diaper_positive:
            categories["both_on_control"].append(row)
        elif expected and sort_positive and not diaper_positive:
            categories["sortformer_only_on_baseline_overlap"].append(row)
        elif not expected and sort_positive and not diaper_positive:
            categories["sortformer_only_on_control"].append(row)
        elif not expected and not sort_positive and diaper_positive:
            categories["diaper_only_on_control"].append(row)
        elif expected and not sort_positive and not diaper_positive:
            categories["neither_on_baseline_overlap"].append(row)
    selected = []
    for category, limit in CATEGORY_LIMITS.items():
        for row in spread(categories[category], limit):
            selected.append({"selection_category": category, **row})
    return sorted(selected, key=lambda row: (row["selection_category"], row["start"]))


def build_html(items):
    data = json.dumps(items).replace("</", "<\\/")
    return f'''<!doctype html>
<html lang="en"><head><meta charset="utf-8"><meta name="viewport" content="width=device-width">
<title>Overlap hand-label set</title>
<style>
body{{font:16px system-ui,sans-serif;max-width:1050px;margin:24px auto;padding:0 16px;background:#f6f7f9;color:#17202a}}
h1{{margin-bottom:4px}} .intro{{background:white;padding:16px;border-radius:10px}}
.card{{background:white;margin:16px 0;padding:16px;border-radius:10px;box-shadow:0 1px 4px #0002}}
.meta{{color:#52606d;font-size:14px}} .text{{font-size:18px;margin:10px 0}} audio{{width:100%}}
fieldset{{border:0;padding:8px 0}} label{{margin-right:18px;display:inline-block;padding:4px}}
textarea{{width:100%;min-height:45px}} button{{font-size:16px;padding:10px 15px;margin:6px}}
.done{{border-left:7px solid #1f9d55}} .sticky{{position:sticky;bottom:0;background:#eef2f7;padding:10px;border-radius:10px}}
</style></head><body>
<h1>Overlap hand-label set</h1>
<div class="intro"><p>Listen for <strong>simultaneous intelligible speech</strong>. A quick handoff where one person stops before another starts is a rapid turn boundary, not overlap. Use unclear/noisy when the distinction cannot be heard reliably.</p>
<p>Speaker identity is a separate label: indicate whether the known target is one of the concurrent speakers only when overlap is present.</p></div>
<div id="items"></div>
<div class="sticky"><span id="progress"></span><button onclick="exportLabels()">Download labels.json</button><button onclick="clearLabels()">Clear saved labels</button></div>
<script>
const items={data}; const saved=JSON.parse(localStorage.getItem('overlap-labels-v1')||'{{}}');
const choices=['true_overlap','rapid_turn_boundary','single_speaker','unclear_or_noisy'];
function esc(s){{return String(s).replace(/[&<>"']/g,c=>({{'&':'&amp;','<':'&lt;','>':'&gt;','"':'&quot;',"'":'&#39;'}}[c]))}}
function render(){{
 const root=document.getElementById('items'); root.innerHTML='';
 items.forEach((item,i)=>{{const id=String(item.review_id), value=saved[id]||{{}}; const card=document.createElement('div'); card.className='card '+(value.overlap_label?'done':'');
 card.innerHTML=`<div class="meta">${{i+1}}/${{items.length}} · ${{esc(item.selection_category)}} · ${{item.start.toFixed(2)}}–${{item.end.toFixed(2)}}s · baseline ${{item.baseline_target_non_target_overlap?'overlap':'control'}} · Sortformer ${{item.sortformer_overlap_fraction.toFixed(2)}} · DiaPer ${{item.diaper_overlap_fraction.toFixed(2)}}</div><div class="text">${{esc(item.text)}}</div><audio controls preload="none" src="${{esc(item.audio)}}"></audio>
 <fieldset><legend>What is audible?</legend>${{choices.map(c=>`<label><input type="radio" name="o-${{id}}" value="${{c}}" ${{value.overlap_label===c?'checked':''}}> ${{c.replaceAll('_',' ')}}</label>`).join('')}}</fieldset>
 <fieldset><legend>If true overlap, is the known target one of the speakers?</legend>${{['yes','no','unclear','not_applicable'].map(c=>`<label><input type="radio" name="t-${{id}}" value="${{c}}" ${{value.target_in_overlap===c?'checked':''}}> ${{c.replaceAll('_',' ')}}</label>`).join('')}}</fieldset>
 <textarea placeholder="Optional notes">${{esc(value.notes||'')}}</textarea>`;
 card.querySelectorAll('input').forEach(input=>input.onchange=()=>saveCard(card,id)); card.querySelector('textarea').oninput=()=>saveCard(card,id); root.appendChild(card);
 }}); updateProgress();
}}
function saveCard(card,id){{const o=card.querySelector(`input[name="o-${{id}}"]:checked`);const t=card.querySelector(`input[name="t-${{id}}"]:checked`);saved[id]={{overlap_label:o?.value||'',target_in_overlap:t?.value||'',notes:card.querySelector('textarea').value}};localStorage.setItem('overlap-labels-v1',JSON.stringify(saved));render()}}
function updateProgress(){{const n=items.filter(x=>saved[x.review_id]?.overlap_label).length;document.getElementById('progress').textContent=`Labeled ${{n}} of ${{items.length}}`}}
function exportLabels(){{const result={{schema_version:1,labels:items.map(item=>({{...item,...(saved[item.review_id]||{{}})}}))}};const a=document.createElement('a');a.href=URL.createObjectURL(new Blob([JSON.stringify(result,null,2)],{{type:'application/json'}}));a.download='overlap-labels.json';a.click();URL.revokeObjectURL(a.href)}}
function clearLabels(){{if(confirm('Clear all saved labels?')){{localStorage.removeItem('overlap-labels-v1');location.reload()}}}}
render();
</script></body></html>'''


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--audio", type=Path, required=True)
    parser.add_argument("--sortformer", type=Path, required=True)
    parser.add_argument("--diaper", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--context-seconds", type=float, default=1.25)
    args = parser.parse_args()
    sortformer = json.loads(args.sortformer.read_text())["segments"]
    diaper = json.loads(args.diaper.read_text())["segments"]
    selected = categorize(sortformer, diaper)
    if args.output_dir.exists():
        shutil.rmtree(args.output_dir)
    clips = args.output_dir / "audio"
    clips.mkdir(parents=True)
    items = []
    for review_id, row in enumerate(selected, 1):
        clip_start = max(0.0, float(row["start"]) - args.context_seconds)
        clip_end = float(row["end"]) + args.context_seconds
        filename = f"{review_id:03d}-{row['baseline_index']:04d}-{row['selection_category']}.wav"
        subprocess.run([
            "ffmpeg", "-nostdin", "-hide_banner", "-loglevel", "error", "-y",
            "-ss", str(clip_start), "-i", str(args.audio), "-t", str(clip_end-clip_start),
            "-ac", "1", "-ar", "16000", str(clips/filename),
        ], check=True)
        items.append({
            "review_id": review_id, "audio": f"audio/{filename}",
            "clip_start": clip_start, "clip_end": clip_end,
            **{key: row[key] for key in (
                "selection_category", "baseline_index", "start", "end", "text",
                "baseline_speaker", "baseline_target_non_target_overlap",
                "sortformer_overlap_fraction", "diaper_overlap_fraction",
            )},
        })
    (args.output_dir/"manifest.json").write_text(json.dumps(items, indent=2)+"\n")
    with (args.output_dir/"manifest.csv").open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=items[0].keys())
        writer.writeheader(); writer.writerows(items)
    (args.output_dir/"index.html").write_text(build_html(items))
    (args.output_dir/"README.txt").write_text(
        "Open index.html in a browser, label every clip, then click Download labels.json.\n"
        "Return overlap-labels.json for analysis. Labels save automatically in this browser.\n"
    )
    print(f"Built {len(items)} review clips in {args.output_dir}")


if __name__ == "__main__":
    main()
