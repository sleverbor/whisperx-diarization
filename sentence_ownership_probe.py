"""Probe whether uncertain words belong to the same speaker as target anchors.

This is a supplemental experiment.  Text coherence makes a span eligible for
testing, but contributes no identity evidence.  The probe scores contiguous
word runs and conservative boundary variants against supplied voice profiles,
then reports target, mixed, non-target, or unresolved.  It never edits the
baseline transcript.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import math
from pathlib import Path
import subprocess


def unit(value):
    import numpy as np
    value = np.asarray(value, dtype=np.float32).reshape(-1)
    norm = float(np.linalg.norm(value))
    if not np.isfinite(value).all() or norm < 1e-8:
        raise ValueError("Invalid embedding")
    return value / norm


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as source:
        for block in iter(lambda: source.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def word_runs(words):
    """Return contiguous diarization runs and measured gaps between words."""
    usable = [word for word in words if "start" in word and "end" in word]
    runs = []
    gaps = []
    for index, word in enumerate(usable):
        start, end = float(word["start"]), float(word["end"])
        if not (math.isfinite(start) and math.isfinite(end) and start <= end):
            raise ValueError("Invalid word timing")
        track = word.get("speaker")
        if index:
            gaps.append({"after_word": index - 1, "before_word": index,
                         "seconds": max(0.0, start-float(usable[index-1]["end"]))})
        if runs and runs[-1]["track"] == track:
            runs[-1]["end"] = end
            runs[-1]["word_end"] = index + 1
            runs[-1]["text"] += " " + str(word.get("word", ""))
        else:
            runs.append({"track": track, "start": start, "end": end,
                         "word_start": index, "word_end": index + 1,
                         "text": str(word.get("word", ""))})
    return runs, gaps


def resolve_scores(scores, target_name="Target_Speaker", minimum=.25, margin=.08):
    if target_name not in scores or not scores:
        return "unresolved"
    ranked = sorted(scores.items(), key=lambda item: item[1], reverse=True)
    leader, best = ranked[0]
    runner = ranked[1][1] if len(ranked) > 1 else None
    if best < minimum or (runner is not None and best-runner < margin):
        return "unresolved"
    return "target" if leader == target_name else "non_target"


def stable_resolution(variants, target_name="Target_Speaker", minimum=.25, margin=.08):
    """Boundary shifts are sensitivity checks, not independent votes."""
    labels = [resolve_scores(item["scores"], target_name, minimum, margin)
              for item in variants]
    decided = {label for label in labels if label != "unresolved"}
    if len(decided) != 1:
        return "unresolved"
    label = next(iter(decided))
    # A contrary raw leader is enough to make a short crop unstable even when
    # that variant misses the absolute threshold.
    leaders = {max(item["scores"], key=item["scores"].get)
               for item in variants if item["scores"]}
    expected = target_name if label == "target" else None
    if expected is not None and leaders != {expected}:
        return "unresolved"
    if expected is None and target_name in leaders:
        return "unresolved"
    return label


def conclude_sentence(runs, gaps, target_track, full_resolution,
                      long_gap=.8, confirmed_target_tracks=()):
    """Resolve from acoustic run results and boundary evidence only."""
    target_runs = [run for run in runs if run.get("resolution") == "target"]
    other_runs = [run for run in runs if run.get("resolution") == "non_target"]
    track_target_runs = [run for run in runs if run.get("track") == target_track]
    anchors = target_runs or [run for run in track_target_runs
                              if run.get("track") in confirmed_target_tracks]
    maximum_gap = max((gap["seconds"] for gap in gaps), default=0.0)
    changed_tracks = len({run.get("track") for run in runs}) > 1
    if anchors and other_runs:
        return "mixed"
    if anchors and changed_tracks and maximum_gap >= long_gap:
        return "mixed"
    if full_resolution == "target" and not other_runs and maximum_gap < long_gap:
        return "target"
    if full_resolution == "non_target" and not target_runs:
        return "non_target"
    return "unresolved"


def crop_variants(start, end, media_duration, shifts=(0.0, -0.08, 0.08)):
    variants = []
    for shift in shifts:
        left = max(0.0, start-shift)
        right = min(media_duration, end+shift)
        if right-left >= .4 and (left, right) not in variants:
            variants.append((left, right))
    return variants


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--video", type=Path, required=True)
    parser.add_argument("--baseline", type=Path, required=True)
    parser.add_argument("--segment", action="append", type=int, required=True,
                        help="Baseline segment index; repeat for multiple probes")
    parser.add_argument("--target-reference", type=Path, required=True)
    parser.add_argument("--other-reference", action="append", default=[], metavar="NAME=PATH")
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--device", choices=("auto", "cpu", "cuda"), default="auto")
    parser.add_argument("--threads", type=int, default=4)
    parser.add_argument("--minimum-similarity", type=float, default=.25)
    parser.add_argument("--minimum-margin", type=float, default=.08)
    parser.add_argument("--long-gap", type=float, default=.8)
    parser.add_argument("--speechbrain-cache", type=Path)
    args = parser.parse_args()
    for path in (args.video, args.baseline, args.target_reference):
        if not path.is_file():
            parser.error(f"Missing input: {path}")
    if args.threads < 1 or args.long_gap < 0:
        parser.error("Invalid probe settings")

    references = {"Target_Speaker": args.target_reference}
    for value in args.other_reference:
        if "=" not in value:
            parser.error("Other reference must be NAME=PATH")
        name, raw_path = value.split("=", 1)
        path = Path(raw_path)
        if not name or name in references or not path.is_file():
            parser.error("Invalid other reference")
        references[name] = path

    import numpy as np
    import torch
    torch.set_num_threads(args.threads)
    device = ("cuda" if torch.cuda.is_available() else "cpu") if args.device == "auto" else args.device
    if device == "cuda" and not torch.cuda.is_available():
        parser.error("CUDA is unavailable")
    raw = subprocess.check_output(["ffmpeg", "-nostdin", "-hide_banner", "-loglevel", "error",
        "-i", str(args.video), "-vn", "-ar", "16000", "-ac", "1", "-f", "f32le", "pipe:1"])
    audio = np.frombuffer(raw, dtype="<f4").copy()
    duration = len(audio)/16000
    if not len(audio) or not np.isfinite(audio).all():
        raise ValueError("Could not decode valid audio")

    from speechbrain.inference.speaker import SpeakerRecognition
    options = {"source": "speechbrain/spkrec-ecapa-voxceleb", "run_opts": {"device": device}}
    if args.speechbrain_cache:
        options["savedir"] = str(args.speechbrain_cache)
    encoder = SpeakerRecognition.from_hparams(**options)
    profiles = {}
    for name, path in references.items():
        matrix = np.load(path, allow_pickle=False)
        if matrix.ndim == 1:
            matrix = matrix[None, :]
        profiles[name] = unit(np.mean([unit(row) for row in matrix], axis=0))

    def score_span(start, end):
        rows = []
        for left, right in crop_variants(start, end, duration):
            waveform = torch.from_numpy(audio[round(left*16000):round(right*16000)]).unsqueeze(0).to(device)
            with torch.no_grad():
                embedding = unit(encoder.encode_batch(waveform).detach().cpu().numpy())
            scores = {name: float(embedding @ profile) for name, profile in profiles.items()}
            rows.append({"start": left, "end": right, "scores": scores})
        return rows

    baseline = json.loads(args.baseline.read_text())
    target_track = baseline.get("target_candidate")
    results = []
    for index in args.segment:
        if not 0 <= index < len(baseline.get("segments", [])):
            parser.error(f"Segment index out of range: {index}")
        segment = baseline["segments"][index]
        runs, gaps = word_runs(segment.get("words", []))
        for run in runs:
            run["variants"] = score_span(run["start"], run["end"])
            run["resolution"] = stable_resolution(run["variants"], minimum=args.minimum_similarity,
                                                   margin=args.minimum_margin)
            run["too_short_for_voice"] = not bool(run["variants"])
        full_variants = score_span(float(segment["start"]), float(segment["end"]))
        full_resolution = stable_resolution(full_variants, minimum=args.minimum_similarity,
                                            margin=args.minimum_margin)
        conclusion = conclude_sentence(runs, gaps, target_track, full_resolution,
                                       args.long_gap, (target_track,))
        results.append({"segment_index": index, "start": segment["start"], "end": segment["end"],
            "text": segment["text"], "text_coherence_identity_weight": 0,
            "target_track": target_track, "word_runs": runs, "gaps": gaps,
            "maximum_word_gap": max((gap["seconds"] for gap in gaps), default=0.0),
            "full_span_variants": full_variants, "full_span_resolution": full_resolution,
            "sentence_ownership": conclusion, "baseline_modified": False,
            "review_required": True})
    report = {"experimental": True, "baseline_modified": False,
        "identity_probability_calibrated": False,
        "method": "text defines the candidate span but contributes zero identity weight; contiguous word runs and boundary variants are scored acoustically",
        "provenance": {"video": str(args.video.resolve()), "video_sha256": sha256(args.video),
            "baseline": str(args.baseline.resolve()), "baseline_sha256": sha256(args.baseline),
            "references": {name: {"path": str(path.resolve()), "sha256": sha256(path)}
                           for name, path in references.items()}, "device": device},
        "segments": results}
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2)+"\n")
    for row in results:
        print(f"{row['segment_index']} [{row['start']:.2f}-{row['end']:.2f}] "
              f"{row['sentence_ownership']}: {row['text']}")


if __name__ == "__main__":
    main()
