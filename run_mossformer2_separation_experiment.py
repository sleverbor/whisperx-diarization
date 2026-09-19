#!/usr/bin/env python3
"""Compare MossFormer2's two blind outputs on the fixed overlap benchmark."""

import argparse
import json
import os
import subprocess
from pathlib import Path

import numpy as np

from review_overlap_extraction import transcribe, unit
from run_contextual_wesep_experiment import token_f1


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
    args = parser.parse_args()

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
    windows = []
    for label in labels:
        start, end = float(label["start"]), float(label["end"])
        window_start = max(0.0, start - args.context)
        window_end = min(len(full) / sample_rate, end + args.context)
        wave = full[round(window_start * sample_rate):round(window_end * sample_rate)]
        windows.append((label, window_start, window_end, wave))

    separator = ClearVoice(
        task="speech_separation", model_names=["MossFormer2_SS_16K"]
    )

    speaker_dir = Path("pretrained_models/spkrec-ecapa-voxceleb")
    speaker = SpeakerRecognition.from_hparams(
        source=str(speaker_dir), savedir=str(speaker_dir),
        run_opts={"device": args.device},
    )
    whisper = WhisperModel(
        args.whisper_model,
        device="cuda" if args.device.startswith("cuda") else "cpu",
        compute_type="float16" if args.device.startswith("cuda") else "int8",
    )
    priors = np.load(args.voice_priors)
    target = unit(np.mean(np.stack([unit(row) for row in priors]), axis=0))

    def similarity(wave):
        tensor = torch.from_numpy(np.asarray(wave, dtype=np.float32)).unsqueeze(0)
        embedding = unit(
            speaker.encode_batch(tensor).flatten().detach().cpu().numpy()
        )
        return float(np.dot(target, embedding))

    results = []
    for item_index, (label, window_start, window_end, original) in enumerate(windows):
        # Decode one short window at a time. MossFormer2 has a large activation
        # footprint; batching all benchmark windows can exhaust a 16 GB laptop.
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
            asr = transcribe(whisper, cropped)
            streams.append({
                "stream": stream_index + 1,
                "audio": str(path.relative_to(args.output_dir)),
                "target_similarity": similarity(cropped),
                "transcription": asr,
                "target_word_f1": token_f1(label.get("target_words", ""), asr["text"]),
                "other_word_f1": token_f1(label.get("other_words", ""), asr["text"]),
            })
        ranked = sorted(streams, key=lambda row: row["target_similarity"], reverse=True)
        results.append({
            "exchange_id": label["exchange_id"],
            "baseline_index": label["baseline_index"],
            "start": start,
            "end": end,
            "context_seconds": args.context,
            "window_start": window_start,
            "window_end": window_end,
            "target_words": label.get("target_words", ""),
            "other_words": label.get("other_words", ""),
            "streams": streams,
            "similarity_selected_stream": ranked[0]["stream"],
            "similarity_margin": ranked[0]["target_similarity"] - ranked[1]["target_similarity"],
        })
        print(
            label["exchange_id"],
            "; ".join(
                f"s{row['stream']} sim={row['target_similarity']:.3f}: "
                f"{row['transcription']['text']}" for row in streams
            ),
            flush=True,
        )

    report = {
        "schema_version": 1,
        "method": "blind_mossformer2_ss_16k_then_ecapa",
        "labels_used_for_separation": False,
        "context_seconds": args.context,
        "results": results,
    }
    (args.output_dir / "report.json").write_text(json.dumps(report, indent=2) + "\n")


if __name__ == "__main__":
    main()
