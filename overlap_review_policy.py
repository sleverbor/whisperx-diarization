"""Create read-only overlap review decisions from supplemental diarizers.

The policy controls expensive review work.  It never changes transcript text,
timing, confidence, or speaker identity.
"""

import argparse
import json
from pathlib import Path


STRONG_SORTFORMER_FRACTION = 0.30
WEAK_SORTFORMER_FRACTION = 0.03
DIAPER_RESCUE_FRACTION = 0.20


def apply_policy(sortformer, diaper):
    diaper_by_index = {row["baseline_index"]: row for row in diaper["segments"]}
    decisions = []
    for sf in sortformer["segments"]:
        dp = diaper_by_index.get(sf["baseline_index"], {})
        sf_fraction = float(sf.get("overlap_fraction", 0.0))
        dp_fraction = float(dp.get("overlap_fraction", 0.0))
        baseline_overlap = bool(sf.get("baseline_target_non_target_overlap", False))
        if baseline_overlap or sf_fraction >= STRONG_SORTFORMER_FRACTION:
            tier = "separation_review"
            reasons = []
            if baseline_overlap:
                reasons.append("baseline_overlap_evidence")
            if sf_fraction >= STRONG_SORTFORMER_FRACTION:
                reasons.append("strong_sortformer_overlap")
        elif (sf_fraction >= WEAK_SORTFORMER_FRACTION
              or dp_fraction >= DIAPER_RESCUE_FRACTION):
            tier = "uncertainty_evidence"
            reasons = []
            if sf_fraction >= WEAK_SORTFORMER_FRACTION:
                reasons.append("weak_sortformer_overlap")
            if dp_fraction >= DIAPER_RESCUE_FRACTION:
                reasons.append("strong_diaper_rescue")
        else:
            tier = "none"
            reasons = []
        decisions.append({
            "baseline_index": sf["baseline_index"],
            "start": sf["start"], "end": sf["end"],
            "text": sf.get("text", ""),
            "baseline_speaker": sf.get("baseline_speaker"),
            "baseline_overlap_evidence": baseline_overlap,
            "tier": tier, "reasons": reasons,
            "sortformer_overlap_fraction": sf_fraction,
            "diaper_overlap_fraction": dp_fraction,
            "speaker_identity_changed": False,
        })
    counts = {tier: sum(row["tier"] == tier for row in decisions)
              for tier in ("separation_review", "uncertainty_evidence", "none")}
    return {
        "policy": {
            "strong_sortformer_fraction": STRONG_SORTFORMER_FRACTION,
            "weak_sortformer_fraction": WEAK_SORTFORMER_FRACTION,
            "diaper_rescue_fraction": DIAPER_RESCUE_FRACTION,
            "speaker_identity_is_read_only": True,
        },
        "summary": {"segments": len(decisions), **counts},
        "segments": decisions,
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--sortformer", type=Path, required=True)
    parser.add_argument("--diaper", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    report = apply_policy(json.loads(args.sortformer.read_text()),
                          json.loads(args.diaper.read_text()))
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps(report["summary"], indent=2))


if __name__ == "__main__":
    main()
