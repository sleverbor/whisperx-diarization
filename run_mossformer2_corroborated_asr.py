#!/usr/bin/env python3
"""Compare mixture and MossFormer2 ASR without trusting either transcript alone."""

import argparse
from collections import Counter
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


def longest_common_words(left, right):
    """Return an ordered common subsequence for human review, not new text."""
    a, b = words(left), words(right)
    table = [[0] * (len(b) + 1) for _ in range(len(a) + 1)]
    for i, x in enumerate(a, 1):
        for j, y in enumerate(b, 1):
            table[i][j] = table[i - 1][j - 1] + 1 if x == y else max(
                table[i - 1][j], table[i][j - 1]
            )
    out, i, j = [], len(a), len(b)
    while i and j:
        if a[i - 1] == b[j - 1]:
            out.append(a[i - 1]); i -= 1; j -= 1
        elif table[i - 1][j] >= table[i][j - 1]:
            i -= 1
        else:
            j -= 1
    return list(reversed(out))


def classify_cross_audio(mixture_text, target_text):
    overlap = token_f1(mixture_text, target_text)
    common = longest_common_words(mixture_text, target_text)
    if overlap >= 0.65 and len(common) >= 2:
        level = "strong_cross_audio_corroboration"
    elif overlap >= 0.40 and common:
        level = "partial_cross_audio_corroboration"
    else:
        level = "not_corroborated"
    return {"level": level, "token_f1": overlap, "ordered_common_words": common}


def surrounding_prompt(segments, index, radius=2):
    nearby = []
    for offset in range(max(0, index - radius), min(len(segments), index + radius + 1)):
        if offset == index:
            continue
        text = segments[offset].get("text", "").strip()
        if text:
            nearby.append(text)
    if not nearby:
        return ""
    return "Nearby dialogue for names and vocabulary only: " + " ".join(nearby)


def decode(model, audio, prompt=""):
    segments, info = model.transcribe(
        audio, language="en", vad_filter=False, beam_size=5,
        condition_on_previous_text=False, word_timestamps=False,
        initial_prompt=prompt or None,
    )
    rows = list(segments)
    return {
        "text": " ".join(row.text.strip() for row in rows if row.text.strip()),
        "average_log_probability": (
            sum(row.avg_logprob for row in rows) / len(rows) if rows else None
        ),
        "language_probability": getattr(info, "language_probability", None),
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--labels", type=Path, required=True)
    parser.add_argument("--baseline", type=Path, required=True)
    parser.add_argument("--source-audio", type=Path, required=True)
    parser.add_argument("--separated-audio-dir", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--model", default="large-v2")
    parser.add_argument("--device", default="cpu")
    args = parser.parse_args()

    import numpy as np
    import soundfile as sf
    from faster_whisper import WhisperModel

    labels = json.loads(args.labels.read_text())["labels"]
    baseline = json.loads(args.baseline.read_text())["segments"]
    source, sample_rate = sf.read(args.source_audio, dtype="float32")
    if source.ndim > 1:
        source = np.mean(source, axis=1)
    if sample_rate != 16000:
        raise ValueError(f"Expected 16 kHz source audio, got {sample_rate}")
    model = WhisperModel(
        args.model, device=args.device,
        compute_type="int8" if args.device == "cpu" else "float16",
    )

    results = []
    for position, label in enumerate(labels, 1):
        index = int(label["baseline_index"])
        start, end = float(label["start"]), float(label["end"])
        mixture = source[round(start * sample_rate):round(end * sample_rate)]
        streams = {}
        for stream in label["streams"]:
            path = args.separated_audio_dir / f"segment-{index:04d}-stream-{stream['stream']}.wav"
            wave, rate = sf.read(path, dtype="float32")
            if rate != sample_rate:
                raise ValueError(f"Unexpected sample rate for {path}: {rate}")
            streams[int(stream["stream"])] = wave
        selected = int(label["selected_stream"])
        prompt = surrounding_prompt(baseline, index)
        audio_sources = {"mixture": mixture, **{
            f"stream_{number}": wave for number, wave in streams.items()
        }}
        readings = {}
        for source_name, wave in audio_sources.items():
            readings[source_name] = {
                "unprompted": decode(model, wave),
                "context_prompted": decode(model, wave, prompt),
            }
        selected_name = f"stream_{selected}"
        unprompted = classify_cross_audio(
            readings["mixture"]["unprompted"]["text"],
            readings[selected_name]["unprompted"]["text"],
        )
        prompted = classify_cross_audio(
            readings["mixture"]["context_prompted"]["text"],
            readings[selected_name]["context_prompted"]["text"],
        )
        results.append({
            "review_id": label["review_id"], "baseline_index": index,
            "start": start, "end": end, "human_outcome": label["outcome"],
            "baseline_text": label["baseline_text"], "selected_stream": selected,
            "surrounding_prompt": prompt, "readings": readings,
            "unprompted_cross_audio": unprompted,
            "context_prompted_cross_audio": prompted,
            "automatic_text_insertion": False,
        })
        print(f"{position}/{len(labels)} {label['review_id']}: {unprompted['level']}", flush=True)

    levels = Counter(row["unprompted_cross_audio"]["level"] for row in results)
    useful = {"adds_missing_target_speech", "cleaner_same_target_line", "mixed_but_useful"}
    report = {
        "schema_version": 1,
        "method": "mixture_and_separated_stream_contextual_asr_corroboration",
        "model": args.model,
        "policy": {
            "context_excludes_current_segment": True,
            "cross_audio_agreement_required": True,
            "candidate_words_remain_review_only": True,
            "automatic_text_insertion": False,
        },
        "summary": {
            "segments": len(results), "unprompted_levels": dict(levels),
            "useful_human_labels": sum(row["human_outcome"] in useful for row in results),
        },
        "results": results,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps(report["summary"], indent=2))


if __name__ == "__main__":
    main()
