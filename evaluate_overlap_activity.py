"""Compare any overlap-aware RTTM with preserved baseline overlap evidence."""

import argparse
import json
from pathlib import Path

from evaluate_diaper_overlap import activity_summary, overlap_evidence, parse_rttm


def evaluate(baseline, rttm, threshold=0.10):
    rows = []
    for index, segment in enumerate(baseline.get("segments", [])):
        activity = activity_summary(float(segment["start"]), float(segment["end"]), rttm)
        expected = overlap_evidence(segment) is not None
        rows.append({
            "baseline_index": index,
            "start": segment["start"], "end": segment["end"],
            "text": segment.get("text", ""),
            "baseline_speaker": segment.get("final_speaker"),
            "baseline_target_non_target_overlap": expected,
            "system_overlap_detected": activity["overlap_fraction"] >= threshold,
            **activity,
        })
    positives = [row for row in rows if row["baseline_target_non_target_overlap"]]
    controls = [row for row in rows if not row["baseline_target_non_target_overlap"]]
    sweep = []
    for value in (0.01, 0.02, 0.05, 0.10, 0.15, 0.20, 0.30, 0.50):
        tp = sum(row["overlap_fraction"] >= value for row in positives)
        fp = sum(row["overlap_fraction"] >= value for row in controls)
        precision = tp / max(1, tp + fp)
        recall = tp / max(1, len(positives))
        sweep.append({
            "threshold": value, "corroborated": tp, "control_detections": fp,
            "agreement_precision": precision, "agreement_recall": recall,
            "agreement_f1": 2 * precision * recall / max(1e-9, precision + recall),
        })
    chosen = next(item for item in sweep if item["threshold"] == threshold)
    return {
        "summary": {
            "baseline_segments": len(rows),
            "baseline_overlap_segments": len(positives),
            "corroborated_overlap_segments": chosen["corroborated"],
            "non_overlap_control_segments": len(controls),
            "overlap_on_control_segments": chosen["control_detections"],
            "minimum_overlap_fraction": threshold,
            "agreement_precision": chosen["agreement_precision"],
            "agreement_recall": chosen["agreement_recall"],
            "agreement_f1": chosen["agreement_f1"],
            "interpretation": "Agreement between imperfect systems is not measured accuracy.",
        },
        "threshold_sweep": sweep,
        "segments": rows,
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--baseline", type=Path, required=True)
    parser.add_argument("--rttm", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    result = evaluate(json.loads(args.baseline.read_text()), parse_rttm(args.rttm))
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps(result["summary"], indent=2))


if __name__ == "__main__":
    main()
