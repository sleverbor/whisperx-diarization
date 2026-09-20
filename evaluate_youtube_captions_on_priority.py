#!/usr/bin/env python3
"""Align YouTube word timings to reviewed overlap clips without changing transcript text."""

import argparse
from collections import Counter
import hashlib
import json
from pathlib import Path
import re


def words(text):
    return re.findall(r"[a-z0-9']+", str(text).casefold())


def token_f1(left, right):
    a, b = words(left), words(right)
    if not a or not b:
        return 0.0
    common = sum((Counter(a) & Counter(b)).values())
    if not common:
        return 0.0
    precision, recall = common / len(b), common / len(a)
    return 2 * precision * recall / (precision + recall)


def timed_caption_words(caption_data):
    rows = []
    for event in caption_data.get("events", []):
        if event.get("aAppend") or not event.get("segs"):
            continue
        start = float(event.get("tStartMs", 0)) / 1000
        for segment in event["segs"]:
            text = segment.get("utf8", "")
            if not text.strip():
                continue
            offset = float(segment.get("tOffsetMs", 0)) / 1000
            rows.append({"time": start + offset, "text": text})
    return rows


def caption_for_interval(timed_words, start, end, padding=.35):
    return " ".join(
        row["text"].strip() for row in timed_words
        if start - padding <= row["time"] <= end + padding
    ).strip()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--captions", type=Path, required=True)
    parser.add_argument("--labels", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--video-id", required=True)
    args = parser.parse_args()

    raw = args.captions.read_bytes()
    timed = timed_caption_words(json.loads(raw))
    labels = json.loads(args.labels.read_text())["labels"]
    results = []
    for label in labels:
        caption = caption_for_interval(timed, float(label["start"]), float(label["end"]))
        baseline_f1 = token_f1(caption, label["baseline_text"])
        candidate_f1 = token_f1(caption, label["candidate_text"])
        if not caption:
            relation = "no_caption_words"
        elif max(baseline_f1, candidate_f1) < .35:
            relation = "conflicts_with_both"
        elif baseline_f1 >= candidate_f1 + .15:
            relation = "supports_baseline_more"
        elif candidate_f1 >= baseline_f1 + .15:
            relation = "supports_separated_candidate_more"
        else:
            relation = "similar_support"
        results.append({
            "review_id": label["review_id"],
            "baseline_index": label["baseline_index"],
            "start": label["start"], "end": label["end"],
            "human_outcome": label["outcome"],
            "baseline_text": label["baseline_text"],
            "separated_candidate_text": label["candidate_text"],
            "caption_text": caption,
            "caption_baseline_f1": baseline_f1,
            "caption_candidate_f1": candidate_f1,
            "relation": relation,
            "speaker_identity_from_captions": False,
            "automatic_text_insertion": False,
        })
    counts = Counter(row["relation"] for row in results)
    output = {
        "schema_version": 1,
        "video_id": args.video_id,
        "caption_type": "youtube_automatic",
        "caption_sha256": hashlib.sha256(raw).hexdigest(),
        "policy": {
            "captions_are_independent_wording_and_timing_evidence": True,
            "captions_do_not_identify_speakers": True,
            "automatic_text_insertion": False,
        },
        "summary": {"segments": len(results), "relations": dict(counts)},
        "results": results,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(output, indent=2) + "\n")
    print(json.dumps(output["summary"], indent=2))


if __name__ == "__main__":
    main()
