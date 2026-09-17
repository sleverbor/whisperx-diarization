"""Export a readable transcript while retaining every omitted row for review."""

import argparse
import json
from pathlib import Path


def timestamp(seconds):
    milliseconds = round(float(seconds) * 1000)
    hours, remainder = divmod(milliseconds, 3_600_000)
    minutes, remainder = divmod(remainder, 60_000)
    secs = remainder / 1000
    return f"{hours:02d}:{minutes:02d}:{secs:06.3f}"


def classify(segment, target_minimum=0.35, other_minimum=0.65,
             ambiguous_target_tracks=()):
    speaker = segment.get("final_speaker", "Uncertain")
    confidence = float(segment.get("final_confidence", 0.0))
    if speaker == "Overlapping_Speakers":
        return False, "overlapping_speakers"
    if speaker in ("Uncertain", "Unknown_Speaker", "NonTarget_Unknown"):
        return False, "uncertain_identity"
    if speaker in ambiguous_target_tracks:
        return False, "ambiguous_target_like_track"
    threshold = target_minimum if speaker == "Target_Speaker" else other_minimum
    if confidence < threshold:
        return False, "below_confidence_threshold"
    if not str(segment.get("text", "")).strip():
        return False, "empty_text"
    return True, "included"


def export_transcript(payload, output_dir, target_minimum=0.35,
                      other_minimum=0.65):
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    target_track = payload.get("target_candidate")
    cluster_means = payload.get("cluster_voice_means", {})
    target_mean = cluster_means.get(target_track)
    ambiguous_target_tracks = set()
    if target_mean is not None:
        ambiguous_target_tracks = {
            track for track, mean in cluster_means.items()
            if track != target_track and mean >= 0.18
            and target_mean - mean <= 0.20
        }
    included, review = [], []
    for index, segment in enumerate(payload.get("segments", [])):
        keep, reason = classify(
            segment, target_minimum, other_minimum, ambiguous_target_tracks
        )
        row = {
            "baseline_index": index,
            "start": segment["start"],
            "end": segment["end"],
            "speaker": segment.get("final_speaker", "Uncertain"),
            "confidence": float(segment.get("final_confidence", 0.0)),
            "text": segment.get("text", "").strip(),
            "disposition": reason,
        }
        (included if keep else review).append(row)

    def render(rows, show_reason=False):
        values = []
        for row in rows:
            suffix = f" [{row['disposition']}]" if show_reason else ""
            values.append(
                f"[{timestamp(row['start'])}–{timestamp(row['end'])}] "
                f"{row['speaker']} ({row['confidence']:.2f}){suffix}: "
                f"{row['text']}"
            )
        return "\n".join(values) + ("\n" if values else "")

    (output_dir / "confident_transcript.txt").write_text(render(included))
    (output_dir / "review_transcript.txt").write_text(render(review, True))
    (output_dir / "review_segments.json").write_text(
        json.dumps(review, indent=2, ensure_ascii=False) + "\n"
    )
    counts = {}
    for row in review:
        counts[row["disposition"]] = counts.get(row["disposition"], 0) + 1
    summary = {
        "baseline_modified": False,
        "target_minimum": target_minimum,
        "other_minimum": other_minimum,
        "ambiguous_target_tracks": sorted(ambiguous_target_tracks),
        "total_segments": len(included) + len(review),
        "included_segments": len(included),
        "review_segments": len(review),
        "review_reasons": counts,
        "included_duration_seconds": sum(
            row["end"] - row["start"] for row in included
        ),
        "review_duration_seconds": sum(
            row["end"] - row["start"] for row in review
        ),
    }
    (output_dir / "summary.json").write_text(json.dumps(summary, indent=2) + "\n")
    return summary


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("baseline", type=Path)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--target-minimum", type=float, default=0.35)
    parser.add_argument("--other-minimum", type=float, default=0.65)
    args = parser.parse_args()
    payload = json.loads(args.baseline.read_text())
    summary = export_transcript(
        payload, args.output_dir, args.target_minimum, args.other_minimum
    )
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
