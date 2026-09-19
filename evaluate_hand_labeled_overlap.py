#!/usr/bin/env python3
"""Score overlap activity outputs against an exported overlap-labels.json file."""
import argparse
import json
from pathlib import Path


def confusion(rows, predicate):
    pairs = [(r["overlap_label"] == "true_overlap", predicate(r)) for r in rows]
    tp = sum(y and p for y, p in pairs); fp = sum(not y and p for y, p in pairs)
    fn = sum(y and not p for y, p in pairs); tn = sum(not y and not p for y, p in pairs)
    precision = tp / (tp + fp) if tp + fp else 0.0
    recall = tp / (tp + fn) if tp + fn else 0.0
    f1 = 2 * precision * recall / (precision + recall) if precision + recall else 0.0
    return {"tp": tp, "fp": fp, "fn": fn, "tn": tn,
            "precision": precision, "recall": recall, "f1": f1}


def evaluate(path):
    labels = json.loads(Path(path).read_text())["labels"]
    usable = [r for r in labels if r.get("overlap_label") != "unclear_or_noisy"]
    result = {
        "total": len(labels), "usable": len(usable),
        "true_overlap": sum(r["overlap_label"] == "true_overlap" for r in usable),
        "rapid_turn_boundary": sum(r["overlap_label"] == "rapid_turn_boundary" for r in usable),
        "sortformer_any": confusion(usable, lambda r: r["sortformer_overlap_fraction"] > 0.0),
        "sortformer_high_precision": confusion(usable, lambda r: r["sortformer_overlap_fraction"] > 0.30),
        "diaper_any": confusion(usable, lambda r: r["diaper_overlap_fraction"] > 0.0),
        "sortformer_plus_diaper_rescue": confusion(
            usable, lambda r: r["sortformer_overlap_fraction"] > 0.03
            or r["diaper_overlap_fraction"] > 0.20),
    }
    return result


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("labels", type=Path)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    result = evaluate(args.labels)
    rendered = json.dumps(result, indent=2) + "\n"
    if args.output: args.output.write_text(rendered)
    print(rendered, end="")


if __name__ == "__main__":
    main()
