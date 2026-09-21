#!/usr/bin/env python3
"""Compare MossFormer2's two blind outputs on the fixed overlap benchmark."""

import argparse
import gc
import json
import os
import re
import subprocess
from pathlib import Path

import numpy as np

from review_overlap_extraction import transcribe, unit
from cloud_runtime import ResumableWorkSet, atomic_json, file_digest


def token_f1(reference, hypothesis):
    """Small self-contained word-overlap diagnostic used only in reports."""
    words = lambda value: re.findall(r"[a-z0-9']+", str(value).casefold())
    reference_words, hypothesis_words = words(reference), words(hypothesis)
    if not reference_words or not hypothesis_words:
        return 0.0
    remaining = list(reference_words)
    hits = 0
    for word in hypothesis_words:
        if word in remaining:
            hits += 1
            remaining.remove(word)
    precision = hits / len(hypothesis_words)
    recall = hits / len(reference_words)
    return 2 * precision * recall / max(precision + recall, 1e-9)


def main():
    parser = argparse.ArgumentParser()
    source_group = parser.add_mutually_exclusive_group(required=True)
    source_group.add_argument("--video", type=Path)
    source_group.add_argument("--audio", type=Path)
    parser.add_argument("--labels", type=Path, required=True)
    parser.add_argument("--voice-priors", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--context", type=float, default=3.0)
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--whisper-model", default="large-v2")
    parser.add_argument("--hf-home", type=Path)
    parser.add_argument("--cache-dir", type=Path)
    parser.add_argument("--snapshot-archive", type=Path)
    parser.add_argument("--snapshot-every", type=int, default=5)
    args = parser.parse_args()
    if args.snapshot_every < 1:
        parser.error("Snapshot interval must be positive")

    # ClearVoice's librosa import uses Numba caching. An explicit writable cache
    # avoids failures when site-packages is read-only or lacks a source locator.
    os.environ.setdefault("NUMBA_CACHE_DIR", "/tmp/numba-clearvoice")
    if args.hf_home:
        os.environ["HF_HOME"] = str(args.hf_home)

    import soundfile as sf
    import torch
    from clearvoice import ClearVoice
    from faster_whisper import WhisperModel
    from speechbrain.inference.speaker import SpeakerRecognition

    args.output_dir.mkdir(parents=True, exist_ok=True)
    audio_dir = args.output_dir / "audio"
    audio_dir.mkdir(exist_ok=True)
    source = args.output_dir / "source-16khz.wav"
    input_path = args.video or args.audio
    command = [
        "ffmpeg", "-nostdin", "-hide_banner", "-loglevel", "error", "-y",
        "-i", str(input_path),
    ]
    if args.video:
        command.append("-vn")
    command.extend(["-ac", "1", "-ar", "16000", str(source)])
    subprocess.run(command, check=True)
    full, sample_rate = sf.read(source, dtype="float32")
    if sample_rate != 16000:
        raise RuntimeError(f"Unexpected sample rate: {sample_rate}")

    labels = json.loads(args.labels.read_text())["labels"]
    fingerprint = {
        "experiment": "mossformer2-full-review-v2",
        "input": file_digest(input_path),
        "labels": file_digest(args.labels),
        "voice_priors": file_digest(args.voice_priors),
        "context": args.context,
        "method": "MossFormer2_SS_16K+ECAPA+faster-whisper",
        "whisper_model": args.whisper_model,
    }
    work = ResumableWorkSet(args.cache_dir, fingerprint)
    newly_completed = 0

    def save_item(stage, exchange_id, value, artifacts=()):
        nonlocal newly_completed
        work.complete(f"{stage}/{exchange_id}", value, artifacts)
        newly_completed += 1
        if args.snapshot_archive and newly_completed % args.snapshot_every == 0:
            work.snapshot(args.snapshot_archive)
    windows = []
    for label in labels:
        start, end = float(label["start"]), float(label["end"])
        window_start = max(0.0, start - args.context)
        window_end = min(len(full) / sample_rate, end + args.context)
        wave = full[round(window_start * sample_rate):round(window_end * sample_rate)]
        windows.append((label, window_start, window_end, wave))

    use_cuda = args.device.startswith("cuda") and torch.cuda.is_available()
    post_gpu_index = 1 if use_cuda and torch.cuda.device_count() > 1 else 0
    gpu_device = f"cuda:{post_gpu_index}" if use_cuda else args.device
    if use_cuda:
        print(
            "GPU allocation: MossFormer2=cuda:0; "
            f"ECAPA/Whisper=cuda:{post_gpu_index}",
            flush=True,
        )
    results = []
    separator = None
    for item_index, (label, window_start, window_end, original) in enumerate(windows):
        exchange_id = label["exchange_id"]
        saved = work.read(f"separation/{exchange_id}")
        if saved is not None:
            results.append(saved)
            print(f"Reusing separation {item_index + 1}/{len(windows)}: {exchange_id}", flush=True)
            continue
        # Decode one short window at a time. MossFormer2 has a large activation
        # footprint; batching all benchmark windows can exhaust a 16 GB laptop.
        if separator is None:
            separator = ClearVoice(
                task="speech_separation", model_names=["MossFormer2_SS_16K"]
            )
        decoded = np.asarray(separator(original[None, :]))
        if decoded.shape[:2] == (2, 1):
            decoded = np.transpose(decoded, (1, 0, 2))
        if decoded.shape[:2] != (1, 2):
            raise RuntimeError(f"Unexpected MossFormer2 output shape: {decoded.shape}")
        start, end = float(label["start"]), float(label["end"])
        left = round((start - window_start) * sample_rate)
        right = round((end - window_start) * sample_rate)
        streams = []
        for stream_index in range(2):
            whole = decoded[0, stream_index, :len(original)]
            cropped = np.asarray(whole[left:right], dtype=np.float32)
            path = audio_dir / f"{label['exchange_id']}-stream-{stream_index + 1}.wav"
            sf.write(path, cropped, sample_rate)
            streams.append({
                "stream": stream_index + 1,
                "audio": str(path.relative_to(args.output_dir)),
            })
        result = {
            "exchange_id": exchange_id,
            "baseline_index": label["baseline_index"],
            "start": start,
            "end": end,
            "context_seconds": args.context,
            "window_start": window_start,
            "window_end": window_end,
            "baseline_text": label.get("baseline_text", ""),
            "target_words": label.get("target_words", ""),
            "other_words": label.get("other_words", ""),
            "streams": streams,
        }
        results.append(result)
        save_item("separation", exchange_id, result,
                  [args.output_dir / row["audio"] for row in streams])
        print(f"Separated {item_index + 1}/{len(windows)}: {exchange_id}", flush=True)
        del decoded

    # Only one large model occupies the GPU at a time. Keeping MossFormer2,
    # ECAPA, and Whisper resident together exceeds a 16 GB T4.
    if separator is not None:
        del separator
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    speaker = None
    target = None
    scored_results = []
    for item_index, result in enumerate(results):
        exchange_id = result["exchange_id"]
        saved = work.read(f"voice/{exchange_id}")
        if saved is not None:
            scored_results.append(saved)
            print(f"Reusing voice score {item_index + 1}/{len(results)}: {exchange_id}", flush=True)
            continue
        if speaker is None:
            speaker_dir = Path(os.environ.get(
                "SPEECHBRAIN_CACHE", "pretrained_models/spkrec-ecapa-voxceleb"
            ))
            speaker = SpeakerRecognition.from_hparams(
                source="speechbrain/spkrec-ecapa-voxceleb", savedir=str(speaker_dir),
                run_opts={"device": gpu_device},
            )
            priors = np.load(args.voice_priors)
            target = unit(np.mean(np.stack([unit(row) for row in priors]), axis=0))
        for stream in result["streams"]:
            wave, _ = sf.read(args.output_dir / stream["audio"], dtype="float32")
            tensor = torch.from_numpy(np.asarray(wave, dtype=np.float32)).unsqueeze(0)
            embedding = unit(
                speaker.encode_batch(tensor).flatten().detach().cpu().numpy()
            )
            stream["target_similarity"] = float(np.dot(target, embedding))
        ranked = sorted(
            result["streams"], key=lambda row: row["target_similarity"], reverse=True
        )
        result["similarity_selected_stream"] = ranked[0]["stream"]
        result["similarity_margin"] = (
            ranked[0]["target_similarity"] - ranked[1]["target_similarity"]
        )
        scored_results.append(result)
        save_item("voice", exchange_id, result,
                  [args.output_dir / row["audio"] for row in result["streams"]])
        print(f"Voice-scored: {exchange_id}", flush=True)
    results = scored_results

    if speaker is not None:
        del speaker
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    whisper = None
    final_results = []
    for item_index, result in enumerate(results):
        exchange_id = result["exchange_id"]
        saved = work.read(f"asr/{exchange_id}")
        if saved is not None:
            final_results.append(saved)
            print(f"Reusing ASR {item_index + 1}/{len(results)}: {exchange_id}", flush=True)
            continue
        if whisper is None:
            whisper = WhisperModel(
                args.whisper_model,
                device="cuda" if use_cuda else "cpu",
                device_index=post_gpu_index if use_cuda else 0,
                compute_type="float16" if use_cuda else "int8",
            )
        for stream in result["streams"]:
            wave, _ = sf.read(args.output_dir / stream["audio"], dtype="float32")
            asr = transcribe(whisper, wave)
            stream["transcription"] = asr
            stream["target_word_f1"] = token_f1(
                result.get("target_words", ""), asr["text"]
            )
            stream["other_word_f1"] = token_f1(
                result.get("other_words", ""), asr["text"]
            )
        final_results.append(result)
        save_item("asr", exchange_id, result,
                  [args.output_dir / row["audio"] for row in result["streams"]])
        print(
            exchange_id,
            "; ".join(
                f"s{row['stream']} sim={row['target_similarity']:.3f}: "
                f"{row['transcription']['text']}" for row in result["streams"]
            ),
            flush=True,
        )
    results = final_results

    report = {
        "schema_version": 1,
        "method": "blind_mossformer2_ss_16k_then_ecapa",
        "labels_used_for_separation": False,
        "context_seconds": args.context,
        "results": results,
    }
    atomic_json(args.output_dir / "report.json", report)
    if args.snapshot_archive:
        work.snapshot(args.snapshot_archive)


if __name__ == "__main__":
    main()
