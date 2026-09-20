#!/usr/bin/env python3
"""Export a conservative transcript with traceable human-review decisions."""

import argparse
import json
from pathlib import Path
import re


def overlaps(row, start, end):
    return float(row["end"]) >= start and float(row["start"]) <= end


def timestamp(seconds):
    milliseconds = round(seconds * 1000)
    hours, remainder = divmod(milliseconds, 3_600_000)
    minutes, remainder = divmod(remainder, 60_000)
    secs, millis = divmod(remainder, 1000)
    return f"{hours:02d}:{minutes:02d}:{secs:02d}.{millis:03d}"


def export(evidence, review_rows, labels):
    segments = evidence["segments"]
    withheld = {int(row["baseline_index"]): row for row in review_rows}
    decisions = {index: [] for index in range(len(segments))}
    promoted = set()
    insertions = []

    for label in labels:
        kind = label.get("type")
        if kind == "target_candidate":
            affected = [i for i, row in enumerate(segments)
                        if overlaps(row, float(label["start"]), float(label["end"]))]
            identity = label.get("Speaker identity")
            wording = label.get("Machine wording")
            for index in affected:
                decisions[index].append({
                    "source": label["review_id"], "identity": identity,
                    "wording": wording, "notes": label.get("notes", "")
                })
                if identity == "non_target" and wording == "correct":
                    promoted.add(index)
                elif identity in ("mixed_or_overlapping", "unclear"):
                    withheld.setdefault(index, {
                        "disposition": "human_review_mixed_or_unclear"
                    })
        elif kind == "caption_gap":
            outcome = label.get("Gap outcome")
            note = label.get("notes", "")
            corrected = re.search(r'transcribed should be\s+["“](.+?)["”]', note,
                                  flags=re.IGNORECASE)
            if corrected:
                text = corrected.group(1).strip()
                matching = [i for i, row in enumerate(segments)
                            if overlaps(row, float(label["candidate_start"]),
                                        float(label["candidate_end"]))
                            and text.lower().strip("?.!") in row["text"].lower().strip("?.!")]
                for index in matching:
                    promoted.add(index)
                    decisions[index].append({"source": label["review_id"],
                                             "reviewed_wording": text,
                                             "speaker": label.get("Speaker")})
            if outcome == "missing_speech":
                insertions.append({
                    "start": float(label["candidate_start"]),
                    "end": float(label["candidate_end"]),
                    "speaker": ("Non_Target_Speaker" if label.get("Speaker") == "non_target"
                                else "Overlapping_Speakers" if label.get("Speaker") == "multiple_or_overlapping"
                                else "Unknown_Speaker"),
                    "text": label["caption_text"],
                    "status": "human_confirmed_missing_speech_from_caption_review",
                    "source": label["review_id"],
                    "note": "Caption wording was reviewed as audible missing speech; identity is limited to the reviewed class."
                })

    published, appendix = [], []
    repeats = evidence.get("text_repeat_candidates", [])
    for index, segment in enumerate(segments):
        row = {
            "baseline_index": index, "start": segment["start"], "end": segment["end"],
            "speaker": segment["final_speaker"], "confidence": segment["final_confidence"],
            "text": segment["text"], "status": "baseline_confident",
            "review_decisions": decisions[index],
        }
        repeat_sources = [candidate["left_start"] for candidate in repeats
                          if overlaps(segment, candidate["right_start"], candidate["right_end"])]
        if repeat_sources:
            row["repeated_presentation_of"] = repeat_sources
        if index in promoted:
            row["status"] = "human_reviewed_promotion"
            published.append(row)
        elif index not in withheld:
            published.append(row)
        else:
            row["status"] = "withheld"
            row["withheld_reason"] = withheld[index].get("disposition", "review_required")
            appendix.append(row)

    published.extend(insertions)
    published.sort(key=lambda row: (row["start"], row["end"], row.get("baseline_index", 10**9)))
    return published, appendix


def markdown(title, rows, appendix=False):
    lines = [f"# {title}", ""]
    if appendix:
        lines += ["These intervals are preserved for audit but omitted from the conservative transcript.", ""]
    for row in rows:
        note = ""
        if row.get("repeated_presentation_of"):
            sources = ", ".join(timestamp(x) for x in row["repeated_presentation_of"])
            note += f" [repeated presentation of event near {sources}]"
        if row.get("status") == "human_reviewed_promotion":
            note += " [human reviewed]"
        if row.get("withheld_reason"):
            note += f" [{row['withheld_reason']}]"
        lines.append(
            f"[{timestamp(row['start'])}–{timestamp(row['end'])}] "
            f"**{row['speaker']}**{note}: {row['text']}"
        )
    return "\n".join(lines) + "\n"


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--evidence", type=Path, required=True)
    p.add_argument("--review-segments", type=Path, required=True)
    p.add_argument("--labels", type=Path, required=True)
    p.add_argument("--output-dir", type=Path, required=True)
    a = p.parse_args(); a.output_dir.mkdir(parents=True, exist_ok=True)
    evidence = json.loads(a.evidence.read_text())
    review_rows = json.loads(a.review_segments.read_text())
    labels = json.loads(a.labels.read_text())["labels"]
    published, appendix = export(evidence, review_rows, labels)
    report = {
        "schema_version": 1, "baseline_modified": False,
        "published_segments": len(published), "withheld_segments": len(appendix),
        "human_reviewed_published": sum(x["status"].startswith("human_") for x in published),
        "caption_insertions": sum("baseline_index" not in x for x in published),
        "segments": published, "appendix": appendix,
    }
    (a.output_dir / "reviewed_transcript.json").write_text(json.dumps(report, indent=2) + "\n")
    (a.output_dir / "reviewed_transcript.md").write_text(
        markdown("Conservative reviewed transcript", published))
    (a.output_dir / "review_appendix.md").write_text(
        markdown("Withheld and uncertain intervals", appendix, appendix=True))
    print(json.dumps({key: report[key] for key in (
        "published_segments", "withheld_segments", "human_reviewed_published",
        "caption_insertions")}, indent=2))


if __name__ == "__main__": main()
