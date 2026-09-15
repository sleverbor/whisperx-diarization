"""Review doubtful transcript regions without modifying the baseline.

This is a batch companion to review_audio_window.py.  It loads each large
model once, checkpoints transcription/alignment per region, and writes only
supplemental hypotheses.  Selection uses confidence/duration/gaps, never
expected dialogue or speaker-specific video rules.
"""
from __future__ import annotations

import argparse
import gc
import hashlib
import json
import math
from pathlib import Path
import subprocess
import time


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as source:
        for block in iter(lambda: source.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def media_duration(path: Path) -> float:
    value = subprocess.check_output([
        "ffprobe", "-v", "error", "-show_entries", "format=duration",
        "-of", "default=noprint_wrappers=1:nokey=1", str(path)
    ], text=True).strip()
    duration = float(value)
    if not math.isfinite(duration) or duration <= 0:
        raise ValueError("Invalid media duration")
    return duration


def select_review_regions(segments, duration, confidence=0.35,
                          short_seconds=1.0, minimum_gap=5.0,
                          context=3.0, merge_gap=2.0,
                          maximum_window=30.0, overlap=4.0,
                          extra_regions=()):
    """Select and bound review windows. Extra regions are external controls."""
    if not (0 <= confidence <= 1 and short_seconds >= 0 and minimum_gap >= 0
            and context >= 0 and merge_gap >= 0 and maximum_window > 0
            and 0 <= overlap < maximum_window and duration > 0):
        raise ValueError("Invalid region selection settings")
    ordered = sorted(segments, key=lambda row: (float(row["start"]), float(row["end"])))
    candidates = []
    for index, row in enumerate(ordered):
        start, end = float(row["start"]), float(row["end"])
        if not (0 <= start <= end <= duration + 0.5):
            raise ValueError("Baseline contains invalid segment times")
        speaker = row.get("final_speaker", row.get("speaker", "Uncertain"))
        strength = float(row.get("final_confidence", 0.0) or 0.0)
        reasons = []
        if speaker in ("Uncertain", "Unknown", "Unknown_Speaker", None):
            reasons.append("uncertain_speaker")
        if strength < confidence:
            reasons.append("weak_identity_evidence")
        if end - start <= short_seconds:
            reasons.append("short_utterance")
        if reasons:
            candidates.append({"start": max(0, start-context),
                               "end": min(duration, end+context),
                               "reasons": reasons,
                               "baseline_indices": [index]})
    previous = 0.0
    for index, row in enumerate(ordered):
        start = float(row["start"])
        if start - previous >= minimum_gap:
            candidates.append({"start": max(0, previous-context),
                               "end": min(duration, start+context),
                               "reasons": ["transcript_gap"],
                               "baseline_indices": []})
        previous = max(previous, float(row["end"]))
    if duration - previous >= minimum_gap:
        candidates.append({"start": max(0, previous-context), "end": duration,
                           "reasons": ["transcript_gap"], "baseline_indices": []})
    for start, end in extra_regions:
        if not (0 <= start < end <= duration):
            raise ValueError("Extra review region is outside the video")
        candidates.append({"start": start, "end": end,
                           "reasons": ["external_review_control"],
                           "baseline_indices": []})
    candidates.sort(key=lambda row: (row["start"], row["end"]))
    merged = []
    for item in candidates:
        if merged and item["start"] <= merged[-1]["end"] + merge_gap:
            merged[-1]["end"] = max(merged[-1]["end"], item["end"])
            merged[-1]["reasons"] = sorted(set(merged[-1]["reasons"] + item["reasons"]))
            merged[-1]["baseline_indices"] = sorted(set(merged[-1]["baseline_indices"] + item["baseline_indices"]))
        else:
            merged.append(dict(item))
    windows = []
    for item in merged:
        left = item["start"]
        while left < item["end"] - 1e-6:
            right = min(left + maximum_window, item["end"])
            windows.append({"index": len(windows), "start": left, "end": right,
                            "reasons": item["reasons"],
                            "baseline_indices": item["baseline_indices"]})
            if right >= item["end"]:
                break
            left = right - overlap
    return windows


def parse_region(value):
    try:
        start, end = (float(part) for part in value.split(":", 1))
    except Exception as error:
        raise argparse.ArgumentTypeError("Region must be START:END") from error
    if not (math.isfinite(start) and math.isfinite(end) and 0 <= start < end):
        raise argparse.ArgumentTypeError("Region must be finite and increasing")
    return start, end


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--video", type=Path, required=True)
    parser.add_argument("--baseline", type=Path, required=True)
    parser.add_argument("--target-reference", type=Path, required=True)
    parser.add_argument("--other-reference", action="append", default=[], metavar="NAME=PATH")
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--cache-dir", type=Path, required=True)
    parser.add_argument("--extra-region", action="append", type=parse_region, default=[])
    parser.add_argument("--weak-confidence", type=float, default=0.35)
    parser.add_argument("--short-seconds", type=float, default=1.0)
    parser.add_argument("--minimum-gap", type=float, default=5.0)
    parser.add_argument("--context-seconds", type=float, default=3.0)
    parser.add_argument("--maximum-window-seconds", type=float, default=30.0)
    parser.add_argument("--window-overlap-seconds", type=float, default=4.0)
    parser.add_argument("--model", default="large-v2")
    parser.add_argument("--language", default="en")
    parser.add_argument("--device", choices=("auto", "cpu", "cuda"), default="auto")
    parser.add_argument("--threads", type=int, default=4)
    parser.add_argument("--min-similarity", type=float, default=0.25)
    parser.add_argument("--min-margin", type=float, default=0.08)
    parser.add_argument("--speechbrain-cache", type=Path)
    args = parser.parse_args()
    input_paths = [args.video, args.baseline, args.target_reference]
    references = {"Target_Speaker": args.target_reference}
    for item in args.other_reference:
        if "=" not in item:
            parser.error("Other reference must be NAME=PATH")
        name, path = item.split("=", 1)
        if not name or name in references or name in ("Unknown", "Uncertain"):
            parser.error("Reference names must be unique")
        references[name] = Path(path)
        input_paths.append(Path(path))
    input_paths.append(args.baseline)
    for path in input_paths:
        if not path.is_file():
            parser.error(f"Missing input: {path}")
    baseline_bytes = args.baseline.read_bytes()
    baseline_hash = hashlib.sha256(baseline_bytes).hexdigest()
    baseline = json.loads(baseline_bytes)
    if not isinstance(baseline.get("segments"), list):
        parser.error("Baseline must contain a segments list")
    duration = media_duration(args.video)
    windows = select_review_regions(
        baseline["segments"], duration, args.weak_confidence,
        args.short_seconds, args.minimum_gap, args.context_seconds, 2.0,
        args.maximum_window_seconds, args.window_overlap_seconds,
        args.extra_region)
    configuration = {
        "video_sha256": sha256(args.video), "baseline_sha256": baseline_hash,
        "references": {name: sha256(path) for name, path in references.items()},
        "model": args.model, "language": args.language,
        "weak_confidence": args.weak_confidence, "short_seconds": args.short_seconds,
        "minimum_gap": args.minimum_gap, "context_seconds": args.context_seconds,
        "maximum_window_seconds": args.maximum_window_seconds,
        "window_overlap_seconds": args.window_overlap_seconds,
        "extra_regions": args.extra_region, "windows": windows}
    fingerprint = hashlib.sha256(json.dumps(configuration, sort_keys=True).encode()).hexdigest()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    cache = args.cache_dir / fingerprint
    cache.mkdir(parents=True, exist_ok=True)
    (args.output_dir / "selection.json").write_text(json.dumps(configuration, indent=2)+"\n")
    print(f"Selected {len(windows)} windows, {sum(w['end']-w['start'] for w in windows)/60:.1f} decoded minutes")

    import numpy as np
    import torch
    torch.set_num_threads(args.threads)
    device = ("cuda" if torch.cuda.is_available() else "cpu") if args.device == "auto" else args.device
    if device == "cuda" and not torch.cuda.is_available():
        parser.error("CUDA was requested but is unavailable")
    raw = subprocess.check_output(["ffmpeg", "-nostdin", "-hide_banner", "-loglevel", "error",
        "-i", str(args.video), "-vn", "-ar", "16000", "-ac", "1", "-f", "f32le", "pipe:1"])
    audio = np.frombuffer(raw, dtype="<f4").copy()
    if not len(audio) or not np.isfinite(audio).all():
        raise ValueError("Decoded audio is empty or invalid")

    asr_records = {}
    missing = []
    for window in windows:
        path = cache / f"asr-{window['index']:04d}.json"
        if path.exists():
            asr_records[window["index"]] = json.loads(path.read_text())
        else:
            missing.append(window)
    if missing:
        from faster_whisper import WhisperModel
        model = WhisperModel(args.model, device=device,
            compute_type="float16" if device == "cuda" else "int8", cpu_threads=args.threads)
        for count, window in enumerate(missing, 1):
            left, right = window["start"], window["end"]
            decoded, _ = model.transcribe(audio[round(left*16000):round(right*16000)],
                language=args.language, beam_size=5, vad_filter=False,
                condition_on_previous_text=False, word_timestamps=True)
            rows = []
            for segment in decoded:
                rows.append({"start": float(segment.start), "end": float(segment.end),
                    "text": segment.text, "avg_logprob": float(segment.avg_logprob),
                    "no_speech_prob": float(segment.no_speech_prob),
                    "compression_ratio": float(segment.compression_ratio),
                    "words": [{"start": float(word.start), "end": float(word.end),
                               "word": word.word, "probability": float(word.probability)}
                              for word in segment.words or []]})
            record = {"window": window, "segments": rows, "language": args.language}
            path = cache / f"asr-{window['index']:04d}.json"
            path.write_text(json.dumps(record, indent=2)+"\n")
            asr_records[window["index"]] = record
            print(f"Transcribed review window {count}/{len(missing)}", flush=True)
        del model
        gc.collect()
        if device == "cuda": torch.cuda.empty_cache()

    import whisperx
    aligned_records = {}
    missing = []
    for window in windows:
        path = cache / f"alignment-{window['index']:04d}.json"
        if path.exists():
            aligned_records[window["index"]] = json.loads(path.read_text())
        else:
            missing.append(window)
    if missing:
        aligner, metadata = whisperx.load_align_model(language_code=args.language, device=device)
        for count, window in enumerate(missing, 1):
            record = asr_records[window["index"]]
            left, right = window["start"], window["end"]
            if record["segments"]:
                aligned = whisperx.align(record["segments"], aligner, metadata,
                    audio[round(left*16000):round(right*16000)], device,
                    return_char_alignments=False)
            else:
                aligned = {"segments": [], "word_segments": []}
            output = {"window": window, **aligned}
            path = cache / f"alignment-{window['index']:04d}.json"
            path.write_text(json.dumps(output, indent=2)+"\n")
            aligned_records[window["index"]] = output
            print(f"Aligned review window {count}/{len(missing)}", flush=True)
        del aligner
        gc.collect()
        if device == "cuda": torch.cuda.empty_cache()

    from speechbrain.inference.speaker import SpeakerRecognition
    from review_audio_window import (baseline_evidence, decoder_quality_flags,
        decoder_sentence_bounds, resolve_timing_evidence)
    options = {"source": "speechbrain/spkrec-ecapa-voxceleb", "run_opts": {"device": device}}
    if args.speechbrain_cache:
        options["savedir"] = str(args.speechbrain_cache)
    encoder = SpeakerRecognition.from_hparams(**options)

    def unit(value):
        value = np.asarray(value, dtype=np.float32).reshape(-1)
        norm = np.linalg.norm(value)
        if not np.isfinite(value).all() or norm < 1e-8:
            raise ValueError("Invalid voice embedding")
        return value / norm

    profiles = {}
    for name, path in references.items():
        values = np.load(path, allow_pickle=False)
        if values.ndim == 1:
            values = values[None, :]
        if values.ndim != 2 or not len(values):
            raise ValueError("Reference must contain voice embeddings")
        profiles[name] = unit(np.mean([unit(value) for value in values], axis=0))
    rows = []
    for window in windows:
        left = window["start"]
        decoded = asr_records[window["index"]]
        aligned = aligned_records[window["index"]]
        decoder_bounds = decoder_sentence_bounds(decoded["segments"], aligned["segments"])
        for local_index, (segment, bounds) in enumerate(zip(aligned["segments"], decoder_bounds)):
            alignment_bounds = (float(segment["start"]), float(segment["end"]))
            primary = bounds if bounds is not None else alignment_bounds
            variants = []
            for source, crop in (("decoder", bounds), ("alignment", alignment_bounds)):
                if crop is None or crop[1]-crop[0] < 0.4:
                    continue
                if variants and all(abs(crop[i]-variants[0][f"local_{'start' if i == 0 else 'end'}"]) < 1/16000 for i in (0, 1)):
                    continue
                waveform = torch.from_numpy(audio[
                    round((left+crop[0])*16000):round((left+crop[1])*16000)]).unsqueeze(0).to(device)
                with torch.no_grad():
                    voice = unit(encoder.encode_batch(waveform).detach().cpu().numpy())
                scores = {}
                for name, profile in profiles.items():
                    if voice.shape != profile.shape:
                        raise ValueError("Voice reference dimension/model mismatch")
                    scores[name] = float(voice @ profile)
                variants.append({"source": source, "local_start": crop[0], "local_end": crop[1],
                                 "scores": scores})
            hypothesis = resolve_timing_evidence([item["scores"] for item in variants],
                                                  args.min_similarity, args.min_margin)
            quality_flags, quality_signals = decoder_quality_flags(
                decoded["segments"], *(bounds if bounds is not None else alignment_bounds))
            absolute_start, absolute_end = left+primary[0], left+primary[1]
            rows.append({"window_index": window["index"], "local_segment_index": local_index,
                "start": absolute_start, "end": absolute_end, "text": segment["text"],
                "review_speaker_hypothesis": hypothesis, "review_required": True,
                "near_window_boundary": primary[0] <= .25 or primary[1] >= window["end"]-left-.25,
                "evidence": [
                    {"source": "review_selection", "reasons": window["reasons"]},
                    {"source": "baseline_overlap", **baseline_evidence(
                        baseline["segments"], absolute_start, absolute_end)},
                    {"source": "decoder_support", "flags": quality_flags,
                     "signals": quality_signals, "word_probability_is_accuracy": False},
                    {"source": "local_voice", "timing_variants": variants,
                     "variants_are_independent_votes": False,
                     "identity_probability_calibrated": False,
                     "min_similarity": args.min_similarity, "min_margin": args.min_margin}],
                "alignment_words": [{**word,
                    **({"start": left+word["start"]} if "start" in word else {}),
                    **({"end": left+word["end"]} if "end" in word else {})}
                    for word in segment.get("words", [])]})
    assert hashlib.sha256(args.baseline.read_bytes()).hexdigest() == baseline_hash
    result = {"baseline_modified": False, "baseline_sha256": baseline_hash,
              "selection_fingerprint": fingerprint, "segments": rows}
    (args.output_dir / "review_hypotheses.json").write_text(json.dumps(result, indent=2)+"\n")
    (args.output_dir / "review_transcript.txt").write_text("\n".join(
        f"[{row['start']:.2f}-{row['end']:.2f}] {row['review_speaker_hypothesis']}: {row['text']}"
        for row in rows)+"\n")
    counts = {}
    for row in rows:
        counts[row["review_speaker_hypothesis"]] = counts.get(row["review_speaker_hypothesis"], 0)+1
    summary = {"baseline_segment_count": len(baseline["segments"]),
        "baseline_modified": False, "review_window_count": len(windows),
        "decoded_review_minutes": sum(w["end"]-w["start"] for w in windows)/60,
        "review_hypothesis_count": len(rows), "review_label_counts": counts,
        "duplicate_overlap_hypotheses_retained": True,
        "note": "Review hypotheses are supplemental and require comparison; no automatic replacement."}
    (args.output_dir / "comparison_summary.json").write_text(json.dumps(summary, indent=2)+"\n")
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
