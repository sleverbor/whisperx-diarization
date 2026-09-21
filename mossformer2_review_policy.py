"""Prepare MossFormer2 overlap inputs and enforce review-only output policy."""

import argparse
import json
import re
from collections import Counter
from pathlib import Path

from review_overlap_extraction import select_overlap_segments


MINIMUM_TARGET_SIMILARITY = 0.15
MINIMUM_TARGET_MARGIN = 0.08
CORROBORATION_MINIMUM_TARGET_SIMILARITY = 0.25
CORROBORATION_MINIMUM_TARGET_MARGIN = 0.20
CORROBORATION_MINIMUM_BASELINE_TOKEN_F1 = 0.60


def token_f1(reference, hypothesis):
    """Measure word overlap without treating word order as ground truth."""
    reference_tokens = re.findall(r"[a-z0-9']+", reference.lower())
    hypothesis_tokens = re.findall(r"[a-z0-9']+", hypothesis.lower())
    if not reference_tokens or not hypothesis_tokens:
        return 0.0
    reference_counts = Counter(reference_tokens)
    hypothesis_counts = Counter(hypothesis_tokens)
    common = sum((reference_counts & hypothesis_counts).values())
    precision = common / len(hypothesis_tokens)
    recall = common / len(reference_tokens)
    return 2 * precision * recall / (precision + recall) if common else 0.0


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
        baseline_text = result.get("baseline_text", "")
        candidate_text = best.get("transcription", {}).get("text", "")
        baseline_overlap = token_f1(baseline_text, candidate_text)
        accepted = (
            similarity >= MINIMUM_TARGET_SIMILARITY
            and margin >= MINIMUM_TARGET_MARGIN
        )
        ownership_corroborated = (
            accepted
            and similarity >= CORROBORATION_MINIMUM_TARGET_SIMILARITY
            and margin >= CORROBORATION_MINIMUM_TARGET_MARGIN
            and baseline_overlap >= CORROBORATION_MINIMUM_BASELINE_TOKEN_F1
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
                candidate_text if accepted else ""
            ),
            # Human evaluation showed that a separated stream can contain
            # useful target audio while fresh ASR on that stream invents or
            # damages words. Similarity and margin validate voice identity;
            # they do not validate the candidate transcription.
            "candidate_transcription_status": (
                "unverified_review_hint" if accepted else "not_selected"
            ),
            "baseline_token_f1": baseline_overlap if accepted else None,
            "transcript_review_class": (
                "baseline_corroboration"
                if accepted and baseline_overlap >= 0.5
                else "novel_or_conflicting_words"
                if accepted else "not_selected"
            ),
            "target_similarity": similarity,
            "target_margin": margin,
            "baseline_ownership_corroborated": ownership_corroborated,
            "review_required": accepted,
            "mixed_speaker_risk": accepted,
            "automatic_text_insertion": False,
            "automatic_candidate_transcription_use": False,
            "speaker_identity_changed": False,
            "baseline_text_changed": False,
            "all_streams": streams,
        })
    return {
        "schema_version": 1,
        "policy": {
            "minimum_target_similarity": MINIMUM_TARGET_SIMILARITY,
            "minimum_target_margin": MINIMUM_TARGET_MARGIN,
            "ownership_corroboration": {
                "minimum_target_similarity": CORROBORATION_MINIMUM_TARGET_SIMILARITY,
                "minimum_target_margin": CORROBORATION_MINIMUM_TARGET_MARGIN,
                "minimum_baseline_token_f1": CORROBORATION_MINIMUM_BASELINE_TOKEN_F1,
                "changes_speaker_identity": False,
                "changes_baseline_text": False,
            },
            "review_only": True,
            "mixed_streams_never_replace_baseline": True,
            "voice_scores_do_not_validate_candidate_words": True,
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
            "candidate_transcriptions_approved": 0,
            "baseline_ownership_corroborated": sum(
                row["baseline_ownership_corroborated"] for row in decisions
            ),
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
