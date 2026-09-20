"""Evaluate nomo-pVAD on fixed target/non-target/mixed video intervals."""
from __future__ import annotations

import argparse
import json
import math
import os
from pathlib import Path
import subprocess
import sys

import numpy as np

from sentence_ownership_probe import sha256


def decode_audio(path, start=None, end=None):
    command = ["ffmpeg", "-nostdin", "-hide_banner", "-loglevel", "error"]
    if start is not None: command.extend(["-ss", str(start)])
    command.extend(["-i", str(path)])
    if start is not None and end is not None: command.extend(["-t", str(end-start)])
    command.extend(["-vn", "-ar", "16000", "-ac", "1", "-f", "f32le", "pipe:1"])
    wave = np.frombuffer(subprocess.check_output(command), dtype="<f4").copy()
    if not len(wave) or not np.isfinite(wave).all(): raise ValueError("Invalid decoded audio")
    return wave


def chunk_times(window_start, count, seconds=.16):
    return [{"start": window_start+i*seconds, "end": window_start+(i+1)*seconds}
            for i in range(count)]


def interval_probabilities(probabilities, times, start, end):
    selected = [float(p) for p, time in zip(probabilities, times)
                if time["start"] < end and time["end"] > start]
    if not selected: raise ValueError("No pVAD chunks overlap evaluation interval")
    return selected


def summarize(values, thresholds=(.5, .6, .7)):
    values = np.asarray(values, dtype=np.float32)
    return {"chunks": int(len(values)), "mean": float(values.mean()),
        "median": float(np.median(values)), "maximum": float(values.max()),
        "minimum": float(values.min()),
        "fraction_at_or_above": {str(value): float(np.mean(values >= value))
                                 for value in thresholds}}


def active_runs(probabilities, times, threshold=.5, minimum_chunks=2):
    """Return hysteresis-like runs with consecutive target-active chunks."""
    runs = []; current = []
    for probability, time in zip(probabilities, times):
        if float(probability) >= threshold:
            current.append((float(probability), time))
        else:
            if len(current) >= minimum_chunks:
                runs.append({"start": current[0][1]["start"], "end": current[-1][1]["end"],
                    "chunks": len(current), "mean": sum(x[0] for x in current)/len(current),
                    "maximum": max(x[0] for x in current)})
            current = []
    if len(current) >= minimum_chunks:
        runs.append({"start": current[0][1]["start"], "end": current[-1][1]["end"],
            "chunks": len(current), "mean": sum(x[0] for x in current)/len(current),
            "maximum": max(x[0] for x in current)})
    return runs


def confusion(rows, threshold):
    labeled = [row for row in rows if row["label"] in ("target", "non_target")]
    counts = {"true_positive": 0, "false_positive": 0,
              "true_negative": 0, "false_negative": 0}
    for row in labeled:
        predicted = row["summary"]["mean"] >= threshold
        actual = row["label"] == "target"
        key = ("true_positive" if actual and predicted else
               "false_positive" if not actual and predicted else
               "true_negative" if not actual else "false_negative")
        counts[key] += 1
    positive = counts["true_positive"]+counts["false_negative"]
    negative = counts["true_negative"]+counts["false_positive"]
    return {**counts, "target_recall": counts["true_positive"]/max(positive, 1),
            "non_target_rejection": counts["true_negative"]/max(negative, 1)}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--video", type=Path, required=True)
    parser.add_argument("--enrollment", type=Path, required=True)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--nomo-dir", type=Path, required=True)
    parser.add_argument("--eres2netv2-dir", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--context", type=float, default=2.4)
    args = parser.parse_args()
    for path in (args.video, args.enrollment, args.manifest, args.nomo_dir,
                 args.eres2netv2_dir):
        if not path.exists(): parser.error(f"Missing input: {path}")
    sys.path.insert(0, str(args.nomo_dir.resolve()))
    from nomo_pvad import NomoPVAD, NomoPVADSession
    model = NomoPVAD(str(args.nomo_dir/"weights/nomo_pvad.pt"),
                    str(args.eres2netv2_dir), device=args.device)
    session = NomoPVADSession(model)
    enrollment = decode_audio(args.enrollment)
    session.set_enrollment(enrollment)
    manifest = json.loads(args.manifest.read_text())
    results = []
    for case in manifest["cases"]:
        start, end = float(case["start"]), float(case["end"])
        left, right = max(0.0, start-args.context), end+args.context
        wave = decode_audio(args.video, left, right)
        probabilities = session.score_utterance(wave)
        times = chunk_times(left, len(probabilities))
        selected = interval_probabilities(probabilities, times, start, end)
        result = {**case, "window_start": left, "window_end": right,
            "summary": summarize(selected),
            "interval_probabilities": selected,
            "active_runs": {str(value): active_runs(probabilities, times, value)
                            for value in (.5, .6, .7)},
            "timeline": [{**time, "probability": float(probability)}
                         for time, probability in zip(times, probabilities)]}
        results.append(result)
        print(case["case_id"], case["label"],
              f"mean={result['summary']['mean']:.3f}",
              f"max={result['summary']['maximum']:.3f}", flush=True)
    thresholds = {str(value): confusion(results, value) for value in (.5, .6, .7)}
    report = {"experimental": True, "model": "nomo-pVAD release-1.1",
        "probability_calibration_assumed": False, "baseline_modified": False,
        "chunk_seconds": .16, "thresholds_fixed_before_evaluation": [.5, .6, .7],
        "provenance": {"video_sha256": sha256(args.video),
            "enrollment_sha256": sha256(args.enrollment),
            "manifest_sha256": sha256(args.manifest)},
        "threshold_summary": thresholds, "cases": results}
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2)+"\n")
    print(json.dumps(thresholds, indent=2))


if __name__ == "__main__": main()
