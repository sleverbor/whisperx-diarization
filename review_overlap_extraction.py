"""Supplemental target-speaker extraction for baseline overlap intervals.

The baseline file is read-only. Results are review candidates and never replace
baseline text, timing, evidence, confidence, or speaker identity.
"""

import argparse
from collections import Counter
from difflib import SequenceMatcher
import json
from pathlib import Path
import re
import subprocess

import numpy as np


def select_overlap_segments(baseline, policy=None):
    """Select the additive union of baseline and strong supplemental review rows."""
    selected_by_index = {}
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
            selected_by_index[index] = (index, segment, overlap)
    if policy is not None:
        segments = baseline.get("segments", [])
        for row in policy.get("segments", []):
            if row.get("tier") != "separation_review":
                continue
            index = int(row["baseline_index"])
            if not 0 <= index < len(segments):
                raise ValueError(f"Policy baseline index is out of range: {index}")
            segment = segments[index]
            if (abs(float(row["start"]) - float(segment["start"])) > 0.02
                    or abs(float(row["end"]) - float(segment["end"])) > 0.02):
                raise ValueError(f"Policy timing does not match baseline index {index}")
            if index not in selected_by_index:
                selected_by_index[index] = (index, segment, {
                    "source": "supplemental_overlap_policy",
                    "details": {
                        "tier": row["tier"],
                        "reasons": row.get("reasons", []),
                        "sortformer_overlap_fraction": row.get(
                            "sortformer_overlap_fraction", 0.0),
                        "diaper_overlap_fraction": row.get(
                            "diaper_overlap_fraction", 0.0),
                    },
                })
    return [selected_by_index[index] for index in sorted(selected_by_index)]


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


def stereo_metrics(wave):
    """Measure whether stereo contains information beyond duplicated mono."""
    wave = np.asarray(wave, dtype=np.float32)
    if wave.ndim != 2 or wave.shape[1] != 2 or len(wave) < 2:
        return {"available": False, "distinct": False}
    left, right = wave[:, 0], wave[:, 1]
    middle, side = (left + right) / 2, (left - right) / 2
    correlation = float(np.corrcoef(left, right)[0, 1])
    side_to_middle_db = float(20 * np.log10(
        (rms(side) + 1e-12) / (rms(middle) + 1e-12)
    ))
    # Lossy encoders can make duplicated channels differ by tiny amounts. Analyze
    # channels only when the difference is large enough to carry real content.
    distinct = bool(np.isfinite(correlation) and correlation < 0.98
                    and side_to_middle_db >= -25.0)
    return {
        "available": True,
        "distinct": distinct,
        "correlation": correlation,
        "side_to_middle_db": side_to_middle_db,
        "left_to_right_level_db": float(20 * np.log10(
            (rms(left) + 1e-12) / (rms(right) + 1e-12)
        )),
    }


def stereo_signals(wave):
    wave = np.asarray(wave, dtype=np.float32)
    left, right = wave[:, 0], wave[:, 1]
    return {
        "left": left,
        "right": right,
        "middle": (left + right) / 2,
        "difference": (left - right) / 2,
    }


def corroborated_novel_words(transcriptions, baseline_text, minimum_views=2):
    """Return inserted words decoded in independent views.

    Replacement hypotheses such as sue/see or city/scene are transcription
    disagreements, not recovered concurrent speech, and are intentionally
    excluded here.
    """
    tokens = lambda text: re.findall(r"[a-z0-9']+", str(text).casefold())
    baseline_words = tokens(baseline_text)
    support = Counter()
    views = {}
    for name, text in transcriptions.items():
        candidate_words = tokens(text)
        inserted = set()
        for tag, _, _, candidate_start, candidate_end in SequenceMatcher(
                None, baseline_words, candidate_words).get_opcodes():
            if tag == "insert":
                inserted.update(candidate_words[candidate_start:candidate_end])
        for word in inserted:
            support[word] += 1
            views.setdefault(word, []).append(name)
    return [
        {"word": word, "support": support[word], "views": sorted(views[word])}
        for word in sorted(support)
        if support[word] >= minimum_views
    ]


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
    parser.add_argument("--selection-policy", type=Path)
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
    policy = (json.loads(args.selection_policy.read_text())
              if args.selection_policy else None)
    selected = select_overlap_segments(baseline, policy)
    if args.maximum_segments is not None:
        selected = selected[:args.maximum_segments]

    source_stereo = args.output_dir / "source-stereo-16khz.wav"
    subprocess.run(
        [
            "ffmpeg", "-nostdin", "-hide_banner", "-loglevel", "error", "-y",
            "-i", str(args.video), "-vn", "-ac", "2", "-ar", "16000",
            str(source_stereo),
        ],
        check=True,
    )
    full_stereo, sample_rate = sf.read(
        source_stereo, dtype="float32", always_2d=True
    )
    if sample_rate != 16000:
        raise RuntimeError(f"Unexpected extracted sample rate: {sample_rate}")
    full_wave = full_stereo.mean(axis=1)
    source_audio = args.output_dir / "source-16khz.wav"
    sf.write(source_audio, full_wave, sample_rate)

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
        original_stereo = full_stereo[left:right]
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
        channel_metrics = stereo_metrics(original_stereo)
        stereo_review = {
            "analyzed": False,
            "metrics": channel_metrics,
            "status": "channels_not_distinct",
            "review_required": True,
        }
        if channel_metrics.get("distinct", False):
            channel_audio = stereo_signals(original_stereo)
            channel_results = {}
            for name, wave in channel_audio.items():
                channel_results[name] = {
                    "target_similarity": similarity(wave),
                    "rms": rms(wave),
                    "transcription": transcribe(whisper, wave),
                }
            transcriptions = {
                name: value["transcription"]["text"]
                for name, value in channel_results.items()
            }
            novel = corroborated_novel_words(
                transcriptions, segment.get("text", "")
            )
            stereo_review = {
                "analyzed": True,
                "metrics": channel_metrics,
                "signals": channel_results,
                "corroborated_novel_words": novel,
                "status": ("corroborated_words_for_review" if novel
                           else "distinct_channels_no_corroborated_new_words"),
                "speaker": "Uncertain",
                "review_required": True,
                "note": (
                    "Channel decoding is supplemental evidence. Words require "
                    "speaker review and are never inserted into the baseline."
                ),
            }
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
            "stereo": stereo_review,
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
    stereo_counts = {}
    for row in results:
        counts[row["status"]] = counts.get(row["status"], 0) + 1
        stereo_status = row.get("stereo", {}).get("status", "not_available")
        stereo_counts[stereo_status] = stereo_counts.get(stereo_status, 0) + 1
    report = {
        "review_required": True,
        "baseline_modified": False,
        "selection": (
            "additive baseline overlap plus strong supplemental policy"
            if policy is not None
            else "target/non-target diarization overlap evidence"
        ),
        "selection_policy": str(args.selection_policy) if args.selection_policy else None,
        "thresholds_are_provisional": True,
        "triage_thresholds": {
            "suppressed_below_energy_retention": 0.10,
            "candidate_minimum_energy_retention": 0.10,
            "candidate_minimum_similarity_gain": 0.10,
            "stereo_maximum_channel_correlation": 0.98,
            "stereo_minimum_side_to_middle_db": -25.0,
            "stereo_novel_word_minimum_views": 2,
        },
        "summary": {"selected": len(selected), "completed": len(results),
                    "status": counts, "stereo_status": stereo_counts},
        "segments": results,
    }
    (args.output_dir / "report.json").write_text(json.dumps(report, indent=2) + "\n")
    lines = [
        f"[{row['start']:.2f}-{row['end']:.2f}] {row['status']}: "
        f"{row.get('extracted', {}).get('transcription', {}).get('text', '')}; "
        f"stereo={row.get('stereo', {}).get('status', 'not_available')}; "
        f"novel={','.join(item['word'] for item in row.get('stereo', {}).get('corroborated_novel_words', []))}"
        for row in results
    ]
    (args.output_dir / "review.txt").write_text("\n".join(lines) + "\n")
    if args.baseline.read_bytes() != baseline_bytes:
        raise RuntimeError("Baseline changed during supplemental extraction review")


if __name__ == "__main__":
    main()
