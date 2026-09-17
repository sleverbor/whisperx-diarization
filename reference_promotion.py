"""Export and promote high-confidence post-run target reference candidates.

Export never changes a reference. Promotion requires an explicit approvals file,
creates a new directory, records provenance, and leaves the parent untouched.
"""

import argparse
import hashlib
import json
from pathlib import Path
import shutil
import subprocess

import numpy as np


DEFAULT_CRITERIA = {
    "minimum_final_confidence": 0.80,
    "minimum_duration_seconds": 1.50,
    "minimum_reference_similarity": 0.50,
    "minimum_local_voice_strength": 0.50,
    "requires_original_target_track": True,
    "requires_no_overlap": True,
}


def unit(value):
    value = np.asarray(value, dtype=np.float32).reshape(-1)
    norm = float(np.linalg.norm(value))
    if not np.isfinite(value).all() or norm <= 0:
        raise ValueError("Embedding must be finite and nonzero")
    return value / norm


def file_hash(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def source_key(video, start, end):
    return str(video), round(float(start), 3), round(float(end), 3)


def promoted_source_keys(metadata):
    """Collect promoted source intervals through the recorded ancestry chain."""
    keys = set()
    if not isinstance(metadata, dict):
        return keys
    review = metadata.get("promotion_review", {})
    for row in review.get("promoted_candidates", []):
        source = row.get("source", {})
        if all(name in source for name in ("video", "start", "end")):
            keys.add(source_key(source["video"], source["start"], source["end"]))
    parent = metadata.get("parent_reference", {}).get("metadata")
    keys.update(promoted_source_keys(parent))
    return keys


def select_candidates(payload, criteria=None, *, source_video=None,
                      excluded_sources=()):
    criteria = dict(DEFAULT_CRITERIA if criteria is None else criteria)
    target_track = payload.get("target_candidate")
    selected = []
    for index, segment in enumerate(payload.get("segments", [])):
        duration = float(segment["end"]) - float(segment["start"])
        voice = next((item for item in segment.get("evidence", [])
                      if item.get("source") == "local_voice"), None)
        similarity = (voice or {}).get("details", {}).get("similarity")
        has_overlap = any(item.get("source") == "overlapping_speakers"
                          for item in segment.get("evidence", []))
        raw_track = segment.get("baseline", {}).get("raw_speaker_track")
        if segment.get("final_speaker") != "Target_Speaker":
            continue
        if float(segment.get("final_confidence", 0)) < criteria["minimum_final_confidence"]:
            continue
        if duration < criteria["minimum_duration_seconds"]:
            continue
        if criteria["requires_original_target_track"] and raw_track != target_track:
            continue
        if criteria["requires_no_overlap"] and has_overlap:
            continue
        if voice is None or float(voice.get("confidence", 0)) < criteria["minimum_local_voice_strength"]:
            continue
        if similarity is None or float(similarity) < criteria["minimum_reference_similarity"]:
            continue
        if source_video is not None and source_key(
                source_video, segment["start"], segment["end"]) in excluded_sources:
            continue
        selected.append({
            "baseline_index": index,
            "start": float(segment["start"]),
            "end": float(segment["end"]),
            "duration": duration,
            "text": str(segment.get("text", "")).strip(),
            "raw_speaker_track": raw_track,
            "final_confidence": float(segment["final_confidence"]),
            "reference_similarity": float(similarity),
            "local_voice_strength": float(voice["confidence"]),
            "review_required": True,
        })
    return selected


def evaluate_reference_update(parent_rows, candidate_rows,
                              minimum_parent_similarity=0.50,
                              minimum_centroid_similarity=0.995,
                              maximum_existing_median_drop=0.01):
    parent = np.stack([unit(row) for row in np.asarray(parent_rows)])
    candidates = np.stack([unit(row) for row in np.asarray(candidate_rows)])
    old_centroid = unit(parent.mean(axis=0))
    new_centroid = unit(np.concatenate([parent, candidates]).mean(axis=0))
    candidate_scores = candidates @ old_centroid
    centroid_similarity = float(np.dot(old_centroid, new_centroid))
    old_existing = parent @ old_centroid
    new_existing = parent @ new_centroid
    median_drop = float(np.median(old_existing) - np.median(new_existing))
    checks = {
        "all_candidates_match_parent": bool(np.min(candidate_scores) >= minimum_parent_similarity),
        "centroid_shift_is_bounded": bool(centroid_similarity >= minimum_centroid_similarity),
        "existing_reference_affinity_is_preserved": bool(median_drop <= maximum_existing_median_drop),
    }
    return {
        "passed": all(checks.values()),
        "checks": checks,
        "thresholds": {
            "minimum_parent_similarity": minimum_parent_similarity,
            "minimum_centroid_similarity": minimum_centroid_similarity,
            "maximum_existing_median_drop": maximum_existing_median_drop,
        },
        "candidate_similarity_to_parent_centroid": candidate_scores.tolist(),
        "old_to_new_centroid_similarity": centroid_similarity,
        "existing_reference_median_affinity_drop": median_drop,
    }


def export_review(args):
    if args.output_dir.exists():
        raise ValueError("Output directory already exists")
    payload = json.loads(args.evidence.read_text())
    source_video = args.source_url or str(args.video.resolve())
    excluded_sources = set()
    if args.reference_metadata is not None:
        excluded_sources = promoted_source_keys(
            json.loads(args.reference_metadata.read_text())
        )
    candidates = select_candidates(
        payload, source_video=source_video,
        excluded_sources=excluded_sources,
    )
    args.output_dir.mkdir(parents=True)
    audio_dir = args.output_dir / "audio"
    audio_dir.mkdir()
    for row in candidates:
        name = (f"candidate-{row['baseline_index']:04d}-"
                f"{row['start']:.3f}-{row['end']:.3f}.wav")
        destination = audio_dir / name
        subprocess.run([
            "ffmpeg", "-nostdin", "-hide_banner", "-loglevel", "error", "-y",
            "-ss", str(row["start"]), "-i", str(args.video),
            "-t", str(row["duration"]), "-vn", "-ar", "16000", "-ac", "1",
            str(destination),
        ], check=True)
        row["audio"] = f"audio/{name}"
        row["audio_sha256"] = file_hash(destination)
        row["source"] = {
            "video": source_video,
            "start": row["start"], "end": row["end"],
        }
    if candidates:
        import soundfile as sf
        parts = []
        silence = np.zeros(round(0.45 * 16000), dtype=np.float32)
        for row in candidates:
            wave, sample_rate = sf.read(
                args.output_dir / row["audio"], dtype="float32"
            )
            if sample_rate != 16000 or wave.ndim != 1:
                raise ValueError("Exported candidate audio must be mono 16 kHz")
            peak = float(np.max(np.abs(wave))) if len(wave) else 0.0
            parts.extend([0.8 * wave / peak if peak else wave, silence])
        sf.write(args.output_dir / "review-montage.wav",
                 np.concatenate(parts[:-1]), 16000)
    manifest = {
        "schema_version": 1,
        "purpose": "post-run target reference promotion review",
        "permanent_reference_modified": False,
        "source_evidence_sha256": file_hash(args.evidence),
        "criteria": DEFAULT_CRITERIA,
        "previously_promoted_sources_excluded": len(excluded_sources),
        "candidates": candidates,
    }
    (args.output_dir / "manifest.json").write_text(
        json.dumps(manifest, indent=2, ensure_ascii=False) + "\n"
    )
    approvals = {
        "manifest_sha256": file_hash(args.output_dir / "manifest.json"),
        "approved_baseline_indices": [],
        "note": "Add only personally reviewed target-only clips.",
    }
    (args.output_dir / "approvals.json").write_text(
        json.dumps(approvals, indent=2) + "\n"
    )
    (args.output_dir / "review.txt").write_text("\n".join(
        f"{row['baseline_index']}: [{row['start']:.3f}-{row['end']:.3f}] "
        f"similarity={row['reference_similarity']:.3f} {row['text']}"
        for row in candidates
    ) + ("\n" if candidates else ""))
    print(json.dumps({"candidates": len(candidates),
                      "review": str(args.output_dir)}, indent=2))


def promote(args):
    if args.output_dir.exists():
        raise ValueError("Output reference directory already exists")
    manifest_path = args.review_dir / "manifest.json"
    manifest = json.loads(manifest_path.read_text())
    approvals = json.loads(args.approvals.read_text())
    if approvals.get("manifest_sha256") != file_hash(manifest_path):
        raise ValueError("Approvals do not match this review manifest")
    approved = set(int(value) for value in approvals.get("approved_baseline_indices", []))
    available = {int(row["baseline_index"]): row for row in manifest["candidates"]}
    if not approved:
        raise ValueError("No candidates were approved")
    if not approved <= set(available):
        raise ValueError("Approvals contain an unknown baseline index")

    import torch
    from speechbrain.inference.speaker import SpeakerRecognition
    import soundfile as sf

    parent_voice_path = args.parent_reference / "voice_embeddings.npy"
    parent_rows = np.load(parent_voice_path, allow_pickle=False)
    device = args.device
    if device == "auto":
        device = "cuda" if torch.cuda.is_available() else "cpu"
    model = SpeakerRecognition.from_hparams(
        source="speechbrain/spkrec-ecapa-voxceleb",
        savedir=str(args.model_dir), run_opts={"device": device},
    )
    vectors, promoted = [], []
    for index in sorted(approved):
        row = dict(available[index])
        audio_path = args.review_dir / row["audio"]
        if file_hash(audio_path) != row["audio_sha256"]:
            raise ValueError(f"Candidate audio hash changed: {index}")
        wave, sample_rate = sf.read(audio_path, dtype="float32")
        if sample_rate != 16000 or wave.ndim != 1:
            raise ValueError(f"Candidate audio must be mono 16 kHz: {index}")
        with torch.no_grad():
            vector = model.encode_batch(
                torch.from_numpy(wave).unsqueeze(0).to(device)
            ).flatten().cpu().numpy()
        vectors.append(unit(vector))
        promoted.append(row)
    vectors = np.stack(vectors)
    validation = evaluate_reference_update(parent_rows, vectors)
    if not validation["passed"]:
        raise ValueError("Reference update failed safety checks: " + json.dumps(validation))

    shutil.copytree(args.parent_reference, args.output_dir)
    combined = np.concatenate([
        np.stack([unit(row) for row in parent_rows]), vectors
    ]).astype(np.float32)
    np.save(args.output_dir / "voice_embeddings.npy", combined)
    np.save(args.output_dir / "voice_embedding.npy", unit(combined.mean(axis=0)))
    # Keep target-conditioned extraction aligned with the promoted embedding
    # profile. The parent enrollment remains first and approved samples are
    # appended with short silences; the parent directory is never modified.
    enrollment_path = args.output_dir / "auditor_enrollment.wav"
    enrollment_parts = []
    if enrollment_path.exists():
        parent_enrollment, enrollment_rate = sf.read(
            enrollment_path, dtype="float32"
        )
        if enrollment_rate != 16000 or parent_enrollment.ndim != 1:
            raise ValueError("Parent enrollment must be mono 16 kHz")
        enrollment_parts.append(parent_enrollment)
    silence = np.zeros(round(0.25 * 16000), dtype=np.float32)
    for row in promoted:
        wave, _ = sf.read(args.review_dir / row["audio"], dtype="float32")
        if enrollment_parts:
            enrollment_parts.append(silence)
        enrollment_parts.append(wave)
    sf.write(enrollment_path, np.concatenate(enrollment_parts), 16000)

    parent_metadata_path = args.parent_reference / "reference.json"
    parent_metadata = (json.loads(parent_metadata_path.read_text())
                       if parent_metadata_path.exists() else None)
    metadata = {
        "schema_version": 1,
        "method": "reviewed post-run promotion; versioned and reversible",
        "parent_reference": {
            "path": str(args.parent_reference),
            "voice_embeddings_sha256": file_hash(parent_voice_path),
            "reference_json_sha256": (file_hash(parent_metadata_path)
                                      if parent_metadata_path.exists() else None),
            "metadata": parent_metadata,
        },
        "promotion_review": {
            "manifest_sha256": file_hash(manifest_path),
            "approvals_sha256": file_hash(args.approvals),
            "promoted_candidates": promoted,
        },
        "validation": validation,
        "voice": {
            "parent_rows": int(len(parent_rows)),
            "promoted_rows": int(len(vectors)),
            "total_rows": int(len(combined)),
            "enrollment_sha256": file_hash(enrollment_path),
        },
    }
    (args.output_dir / "reference.json").write_text(
        json.dumps(metadata, indent=2, ensure_ascii=False) + "\n"
    )
    (args.output_dir / "promotion-validation.json").write_text(
        json.dumps(validation, indent=2) + "\n"
    )
    print(json.dumps({"created": str(args.output_dir),
                      "promoted": sorted(approved),
                      "validation": validation}, indent=2))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest="command", required=True)
    export = commands.add_parser("export")
    export.add_argument("--video", type=Path, required=True)
    export.add_argument("--evidence", type=Path, required=True)
    export.add_argument("--output-dir", type=Path, required=True)
    export.add_argument("--source-url")
    export.add_argument("--reference-metadata", type=Path)
    export.set_defaults(function=export_review)
    accept = commands.add_parser("promote")
    accept.add_argument("--review-dir", type=Path, required=True)
    accept.add_argument("--approvals", type=Path, required=True)
    accept.add_argument("--parent-reference", type=Path, required=True)
    accept.add_argument("--output-dir", type=Path, required=True)
    accept.add_argument("--model-dir", type=Path,
                        default=Path("pretrained_models/spkrec-ecapa-voxceleb"))
    accept.add_argument("--device", default="auto")
    accept.set_defaults(function=promote)
    args = parser.parse_args()
    args.function(args)


if __name__ == "__main__":
    main()
