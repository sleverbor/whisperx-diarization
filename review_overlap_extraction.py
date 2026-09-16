"""Supplemental target-speaker extraction for baseline overlap intervals.

The baseline file is read-only. Results are review candidates and never replace
baseline text, timing, evidence, confidence, or speaker identity.
"""

import argparse
import json
from pathlib import Path
import subprocess

import numpy as np


def select_overlap_segments(baseline):
    selected = []
    for index, segment in enumerate(baseline.get("segments", [])):
        overlap = next(
            (
                item
                for item in segment.get("evidence", [])
                if item.get("source") == "overlapping_speakers"
                and item.get("details", {}).get("target_and_non_target", False)
            ),
            None,
        )
        if overlap is not None:
            selected.append((index, segment, overlap))
    return selected


def classify_extraction(original_similarity, extracted_similarity, energy_retention,
                        transcript):
    """Triage only; every result remains review-required."""
    similarity_gain = extracted_similarity - original_similarity
    if energy_retention < 0.10:
        return "likely_suppressed_residual"
    if transcript.strip() and energy_retention >= 0.10 and similarity_gain >= 0.10:
        return "candidate_target_speech"
    return "unresolved"


def unit(vector):
    vector = np.asarray(vector, dtype=np.float32).reshape(-1)
    return vector / max(float(np.linalg.norm(vector)), 1e-9)


def rms(wave):
    return float(np.sqrt(np.mean(np.asarray(wave, dtype=np.float32) ** 2)))


def transcribe(model, wave):
    segments, _ = model.transcribe(
        wave,
        vad_filter=False,
        condition_on_previous_text=False,
        beam_size=5,
    )
    rows = list(segments)
    return {
        "text": " ".join(row.text.strip() for row in rows if row.text.strip()),
        "average_log_probability": (
            float(np.mean([row.avg_logprob for row in rows])) if rows else None
        ),
        "maximum_no_speech_probability": (
            float(max(row.no_speech_prob for row in rows)) if rows else None
        ),
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--video", type=Path, required=True)
    parser.add_argument("--baseline", type=Path, required=True)
    parser.add_argument("--enrollment", type=Path, required=True)
    parser.add_argument("--voice-priors", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--whisper-model", default="large-v2")
    parser.add_argument("--wesep-model-dir", type=Path)
    parser.add_argument("--maximum-segments", type=int)
    args = parser.parse_args()

    import soundfile as sf
    import torch
    import torchaudio
    import wesep
    from faster_whisper import WhisperModel
    from speechbrain.inference.speaker import SpeakerRecognition

    args.output_dir.mkdir(parents=True, exist_ok=True)
    baseline_bytes = args.baseline.read_bytes()
    baseline = json.loads(baseline_bytes)
    selected = select_overlap_segments(baseline)
    if args.maximum_segments is not None:
        selected = selected[:args.maximum_segments]

    source_audio = args.output_dir / "source-16khz.wav"
    subprocess.run(
        [
            "ffmpeg", "-nostdin", "-hide_banner", "-loglevel", "error", "-y",
            "-i", str(args.video), "-vn", "-ac", "1", "-ar", "16000",
            str(source_audio),
        ],
        check=True,
    )
    full_wave, sample_rate = sf.read(source_audio, dtype="float32")
    if sample_rate != 16000:
        raise RuntimeError(f"Unexpected extracted sample rate: {sample_rate}")

    extractor = (
        wesep.load_model_local(str(args.wesep_model_dir))
        if args.wesep_model_dir
        else wesep.load_model("english")
    )
    extractor.set_device(args.device)
    extractor.set_vad(True)
    # Preserve attenuation so residual noise can be rejected.
    extractor.set_output_norm(False)

    target_rows = np.load(args.voice_priors)
    target = unit(np.mean(np.stack([unit(row) for row in target_rows]), axis=0))
    speaker_dir = Path("pretrained_models/spkrec-ecapa-voxceleb")
    speaker = SpeakerRecognition.from_hparams(
        source=str(speaker_dir),
        savedir=str(speaker_dir),
        run_opts={"device": args.device},
    )
    whisper_device = "cuda" if args.device.startswith("cuda") else "cpu"
    whisper = WhisperModel(
        args.whisper_model,
        device=whisper_device,
        compute_type="float16" if whisper_device == "cuda" else "int8",
    )

    extracted_dir = args.output_dir / "audio"
    extracted_dir.mkdir(exist_ok=True)
    results = []
    for number, (baseline_index, segment, overlap) in enumerate(selected, 1):
        start, end = float(segment["start"]), float(segment["end"])
        left, right = max(0, round(start * sample_rate)), min(
            len(full_wave), round(end * sample_rate)
        )
        original = full_wave[left:right]
        if len(original) < 1:
            continue
        original_path = args.output_dir / f"current-{baseline_index:04d}.wav"
        sf.write(original_path, original, sample_rate)
        extracted_tensor = extractor.extract_speech(
            str(original_path), str(args.enrollment)
        )
        if extracted_tensor is None:
            results.append({
                "baseline_index": baseline_index,
                "start": start,
                "end": end,
                "baseline_text": segment.get("text", ""),
                "status": "extractor_returned_no_speech",
                "review_required": True,
            })
            continue
        extracted = extracted_tensor[0].detach().cpu().numpy()
        extracted_path = extracted_dir / f"overlap-{baseline_index:04d}-target.wav"
        sf.write(extracted_path, extracted, sample_rate)

        def similarity(wave):
            tensor = torch.from_numpy(np.asarray(wave, dtype=np.float32)).unsqueeze(0)
            embedding = unit(
                speaker.encode_batch(tensor).flatten().detach().cpu().numpy()
            )
            return float(np.dot(target, embedding))

        original_similarity = similarity(original)
        extracted_similarity = similarity(extracted)
        retention = rms(extracted) / max(rms(original), 1e-9)
        original_asr = transcribe(whisper, original)
        extracted_asr = transcribe(whisper, extracted)
        status = classify_extraction(
            original_similarity,
            extracted_similarity,
            retention,
            extracted_asr["text"],
        )
        results.append({
            "baseline_index": baseline_index,
            "start": start,
            "end": end,
            "baseline_text": segment.get("text", ""),
            "baseline_speaker": segment.get("final_speaker"),
            "baseline_confidence": segment.get("final_confidence"),
            "overlap": overlap.get("details", {}),
            "original": {
                "target_similarity": original_similarity,
                "rms": rms(original),
                "transcription": original_asr,
            },
            "extracted": {
                "target_similarity": extracted_similarity,
                "similarity_gain": extracted_similarity - original_similarity,
                "rms": rms(extracted),
                "energy_retention": retention,
                "transcription": extracted_asr,
                "audio": str(extracted_path.relative_to(args.output_dir)),
            },
            "status": status,
            "review_required": True,
            "baseline_modified": False,
        })
        print(
            f"Overlap {number}/{len(selected)} at {start:.2f}s: {status}; "
            f"retention={retention:.3f}; similarity={original_similarity:.3f}"
            f"->{extracted_similarity:.3f}; {extracted_asr['text']}",
            flush=True,
        )

    counts = {}
    for row in results:
        counts[row["status"]] = counts.get(row["status"], 0) + 1
    report = {
        "review_required": True,
        "baseline_modified": False,
        "selection": "target/non-target diarization overlap evidence",
        "thresholds_are_provisional": True,
        "triage_thresholds": {
            "suppressed_below_energy_retention": 0.10,
            "candidate_minimum_energy_retention": 0.10,
            "candidate_minimum_similarity_gain": 0.10,
        },
        "summary": {"selected": len(selected), "completed": len(results), "status": counts},
        "segments": results,
    }
    (args.output_dir / "report.json").write_text(json.dumps(report, indent=2) + "\n")
    lines = [
        f"[{row['start']:.2f}-{row['end']:.2f}] {row['status']}: "
        f"{row.get('extracted', {}).get('transcription', {}).get('text', '')}"
        for row in results
    ]
    (args.output_dir / "review.txt").write_text("\n".join(lines) + "\n")
    if args.baseline.read_bytes() != baseline_bytes:
        raise RuntimeError("Baseline changed during supplemental extraction review")


if __name__ == "__main__":
    main()
