"""Compare a supplemental DiaPer RTTM with the preserved baseline evidence.

This evaluator never changes the baseline.  It maps DiaPer's anonymous speaker
labels to baseline tracks using clean, single-speaker time and then measures
whether DiaPer independently corroborates the baseline overlap intervals.
"""

import argparse
from collections import defaultdict
import json
from pathlib import Path


def parse_rttm(path):
    rows = []
    for line in Path(path).read_text().splitlines():
        fields = line.split()
        if not fields or fields[0] != "SPEAKER" or len(fields) < 8:
            continue
        start, duration = float(fields[3]), float(fields[4])
        rows.append({"start": start, "end": start + duration, "speaker": fields[7]})
    return rows


def overlap_duration(left_start, left_end, right_start, right_end):
    return max(0.0, min(left_end, right_end) - max(left_start, right_start))


def overlap_evidence(segment):
    return next((
        item for item in segment.get("evidence", [])
        if item.get("source") == "overlapping_speakers"
        and item.get("details", {}).get("target_and_non_target", False)
    ), None)


def map_speakers(segments, rttm):
    """Greedily map anonymous DiaPer speakers from clean baseline intersections."""
    scores = defaultdict(float)
    for segment in segments:
        if overlap_evidence(segment):
            continue
        baseline = segment.get("baseline", {})
        track = baseline.get("raw_speaker_track")
        if not track:
            continue
        for row in rttm:
            scores[(row["speaker"], track)] += overlap_duration(
                segment["start"], segment["end"], row["start"], row["end"]
            )
    candidates = sorted(
        ((seconds, diaper, track) for (diaper, track), seconds in scores.items()),
        reverse=True,
    )
    mapping, used_tracks = {}, set()
    for seconds, diaper, track in candidates:
        if diaper not in mapping and track not in used_tracks and seconds > 0:
            mapping[diaper] = {"baseline_track": track, "intersection_seconds": seconds}
            used_tracks.add(track)
    return mapping


def activity_summary(start, end, rttm, step=0.02):
    duration = max(end - start, 1e-9)
    frame_count = max(1, int(duration / step + 0.999))
    active_sets = []
    for index in range(frame_count):
        time = min(end, start + (index + 0.5) * duration / frame_count)
        active_sets.append({
            row["speaker"] for row in rttm if row["start"] <= time < row["end"]
        })
    overlap_frames = sum(len(active) >= 2 for active in active_sets)
    return {
        "overlap_fraction": overlap_frames / frame_count,
        "maximum_simultaneous_speakers": max(map(len, active_sets), default=0),
        "active_speakers": sorted(set().union(*active_sets) if active_sets else set()),
    }


def evaluate(baseline, rttm, minimum_overlap_fraction=0.10):
    segments = baseline.get("segments", [])
    mapping = map_speakers(segments, rttm)
    target_track = baseline.get("target_candidate")
    target_diaper = next((
        speaker for speaker, item in mapping.items()
        if item["baseline_track"] == target_track
    ), None)
    rows = []
    for index, segment in enumerate(segments):
        summary = activity_summary(float(segment["start"]), float(segment["end"]), rttm)
        expected = overlap_evidence(segment) is not None
        detected = summary["overlap_fraction"] >= minimum_overlap_fraction
        active_mapped = [mapping.get(speaker, {}).get("baseline_track")
                         for speaker in summary["active_speakers"]]
        rows.append({
            "baseline_index": index,
            "start": segment["start"],
            "end": segment["end"],
            "text": segment.get("text", ""),
            "baseline_speaker": segment.get("final_speaker"),
            "baseline_target_non_target_overlap": expected,
            "diaper_overlap_detected": detected,
            "diaper_target_and_other_active": bool(
                target_diaper in summary["active_speakers"]
                and len(summary["active_speakers"]) >= 2
            ),
            "mapped_active_tracks": sorted(x for x in active_mapped if x),
            **summary,
        })
    expected_rows = [row for row in rows if row["baseline_target_non_target_overlap"]]
    controls = [row for row in rows if not row["baseline_target_non_target_overlap"]]
    summary = {
        "baseline_segments": len(rows),
        "baseline_overlap_segments": len(expected_rows),
        "diaper_corroborated_overlap_segments": sum(
            row["diaper_overlap_detected"] for row in expected_rows
        ),
        "diaper_target_plus_other_segments": sum(
            row["diaper_target_and_other_active"] for row in expected_rows
        ),
        "non_overlap_control_segments": len(controls),
        "diaper_overlap_on_control_segments": sum(
            row["diaper_overlap_detected"] for row in controls
        ),
        "minimum_overlap_fraction": minimum_overlap_fraction,
        "target_baseline_track": target_track,
        "target_diaper_speaker": target_diaper,
        "speaker_mapping": mapping,
        "interpretation": (
            "Agreement is corroborating evidence only; without hand labels it is not accuracy."
        ),
    }
    return {"summary": summary, "segments": rows}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--baseline", type=Path, required=True)
    parser.add_argument("--rttm", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--minimum-overlap-fraction", type=float, default=0.10)
    args = parser.parse_args()
    result = evaluate(
        json.loads(args.baseline.read_text()), parse_rttm(args.rttm),
        args.minimum_overlap_fraction,
    )
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps(result["summary"], indent=2))


if __name__ == "__main__":
    main()
