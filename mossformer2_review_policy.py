"""Prepare MossFormer2 overlap inputs and enforce review-only output policy."""

import argparse
import json
from pathlib import Path

from review_overlap_extraction import select_overlap_segments


MINIMUM_TARGET_SIMILARITY = 0.15
MINIMUM_TARGET_MARGIN = 0.08


def prepare_labels(baseline, selection_policy=None):
    """Convert existing overlap-review selections into separator windows."""
    labels = []
    for index, segment, _ in select_overlap_segments(baseline, selection_policy):
        labels.append({
            "exchange_id": f"segment-{index:04d}",
            "baseline_index": index,
            "start": float(segment["start"]),
            "end": float(segment["end"]),
            "baseline_text": segment.get("text", ""),
            "target_words": "",
            "other_words": "",
        })
    return {
        "schema_version": 1,
        "purpose": "mossformer2_review_only_overlap_windows",
        "labels": labels,
    }


def apply_review_policy(report):
    """Reject weak identity evidence and prohibit automatic transcript edits."""
    decisions = []
    for result in report.get("results", []):
        streams = result.get("streams", [])
        if len(streams) != 2:
            raise ValueError(
                f"Expected two MossFormer2 streams for {result.get('exchange_id')}"
            )
        ranked = sorted(
            streams, key=lambda row: float(row.get("target_similarity", -1.0)),
            reverse=True,
        )
        best, second = ranked
        similarity = float(best.get("target_similarity", -1.0))
        margin = similarity - float(second.get("target_similarity", -1.0))
        accepted = (
            similarity >= MINIMUM_TARGET_SIMILARITY
            and margin >= MINIMUM_TARGET_MARGIN
        )
        decisions.append({
            "exchange_id": result.get("exchange_id"),
            "baseline_index": result["baseline_index"],
            "start": result["start"],
            "end": result["end"],
            "disposition": (
                "review_candidate_target_stream"
                if accepted else "rejected_low_target_evidence"
            ),
            "selected_stream": best["stream"] if accepted else None,
            "selected_audio": best.get("audio") if accepted else None,
            "candidate_transcription": (
                best.get("transcription", {}).get("text", "") if accepted else ""
            ),
            "target_similarity": similarity,
            "target_margin": margin,
            "review_required": accepted,
            "mixed_speaker_risk": accepted,
            "automatic_text_insertion": False,
            "speaker_identity_changed": False,
            "baseline_text_changed": False,
            "all_streams": streams,
        })
    return {
        "schema_version": 1,
        "policy": {
            "minimum_target_similarity": MINIMUM_TARGET_SIMILARITY,
            "minimum_target_margin": MINIMUM_TARGET_MARGIN,
            "review_only": True,
            "mixed_streams_never_replace_baseline": True,
        },
        "summary": {
            "segments": len(decisions),
            "review_candidates": sum(
                row["disposition"] == "review_candidate_target_stream"
                for row in decisions
            ),
            "rejected_low_target_evidence": sum(
                row["disposition"] == "rejected_low_target_evidence"
                for row in decisions
            ),
            "automatic_insertions": 0,
        },
        "segments": decisions,
    }


def main():
    parser = argparse.ArgumentParser()
    subparsers = parser.add_subparsers(dest="command", required=True)
    prepare = subparsers.add_parser("prepare")
    prepare.add_argument("--baseline", type=Path, required=True)
    prepare.add_argument("--selection-policy", type=Path)
    prepare.add_argument("--output", type=Path, required=True)
    evaluate = subparsers.add_parser("evaluate")
    evaluate.add_argument("--report", type=Path, required=True)
    evaluate.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()

    if args.command == "prepare":
        result = prepare_labels(
            json.loads(args.baseline.read_text()),
            json.loads(args.selection_policy.read_text())
            if args.selection_policy else None,
        )
    else:
        result = apply_review_policy(json.loads(args.report.read_text()))
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps(
        result.get("summary", {"segments": len(result.get("labels", []))}),
        indent=2,
    ))


if __name__ == "__main__":
    main()
