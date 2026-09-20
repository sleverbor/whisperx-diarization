"""Jointly probe wording stability and speaker identity on difficult audio.

Related audio transforms are grouped into evidence families so repeated ASR
errors are not counted as independent confirmation. Results are supplemental,
review-required evidence and never modify a baseline transcript.
"""
from __future__ import annotations

import argparse
from collections import Counter
import json
from pathlib import Path
import re
import subprocess

import numpy as np

from review_audio_window import resolve_voice
from sentence_ownership_probe import sha256, unit


def tokens(text):
    return re.findall(r"[a-z0-9']+", str(text).casefold())


def common_contiguous_spans(left, right, minimum_words=2):
    """Return maximal exact token spans shared by two hypotheses."""
    a, b = tokens(left), tokens(right)
    table = [[0] * (len(b) + 1) for _ in range(len(a) + 1)]
    spans = []
    for i in range(1, len(a) + 1):
        for j in range(1, len(b) + 1):
            if a[i-1] == b[j-1]:
                table[i][j] = table[i-1][j-1] + 1
                length = table[i][j]
                if length >= minimum_words and (i == len(a) or j == len(b)
                        or a[i] != b[j]):
                    spans.append(" ".join(a[i-length:i]))
    # Drop spans contained in a longer result.
    ordered = sorted(set(spans), key=lambda value: (-len(tokens(value)), value))
    kept = []
    for value in ordered:
        needle = tokens(value)
        if not any(any(tokens(other)[i:i+len(needle)] == needle
                       for i in range(len(tokens(other))-len(needle)+1))
                   for other in kept):
            kept.append(value)
    return kept


def cross_family_wording(variants, minimum_words=2):
    """Find exact spans shared by different audio-generation families."""
    support = {}
    for i, left in enumerate(variants):
        for right in variants[i+1:]:
            if left["family"] == right["family"]:
                continue
            for span in common_contiguous_spans(left["text"], right["text"], minimum_words):
                row = support.setdefault(span, {"families": set(), "variants": set()})
                row["families"].update((left["family"], right["family"]))
                row["variants"].update((left["name"], right["name"]))
    rows = [{"text": text, "word_count": len(tokens(text)),
             "families": sorted(value["families"]),
             "variants": sorted(value["variants"])} for text, value in support.items()]
    return sorted(rows, key=lambda row: (-row["word_count"], row["text"]))


def joint_decision(variants, spans, boundary_resolution=None):
    if boundary_resolution == "mixed":
        return "mixed_boundary"
    target = [row for row in variants if row.get("speaker_resolution") == "Target_Speaker"]
    non_target = [row for row in variants if row.get("speaker_resolution") not in
                  (None, "Target_Speaker", "Uncertain")]
    cross_family = bool(spans)
    if target and cross_family and not non_target:
        return "target_wording_candidate"
    if target:
        return "target_identity_only"
    if non_target and not target:
        return "non_target_or_mixed"
    return "unresolved"


def filtered_audio(video, start, end):
    duration = end-start
    command = ["ffmpeg", "-nostdin", "-hide_banner", "-loglevel", "error",
        "-ss", str(start), "-i", str(video), "-t", str(duration), "-vn",
        "-af", "highpass=f=120,lowpass=f=6500,afftdn=nf=-25",
        "-ar", "16000", "-ac", "1", "-f", "f32le", "pipe:1"]
    return np.frombuffer(subprocess.check_output(command), dtype="<f4").copy()


def original_audio(video, start, end):
    command = ["ffmpeg", "-nostdin", "-hide_banner", "-loglevel", "error",
        "-ss", str(start), "-i", str(video), "-t", str(end-start), "-vn",
        "-ar", "16000", "-ac", "1", "-f", "f32le", "pipe:1"]
    return np.frombuffer(subprocess.check_output(command), dtype="<f4").copy()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--video", type=Path, required=True)
    parser.add_argument("--baseline", type=Path, required=True)
    parser.add_argument("--segment", action="append", type=int, required=True)
    parser.add_argument("--target-reference", type=Path, required=True)
    parser.add_argument("--other-reference", action="append", default=[], metavar="NAME=PATH")
    parser.add_argument("--mossformer-report", type=Path)
    parser.add_argument("--mossformer-root", type=Path)
    parser.add_argument("--sentence-ownership-report", type=Path)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--model", default="large-v2")
    parser.add_argument("--device", choices=("cpu", "cuda"), default="cpu")
    parser.add_argument("--threads", type=int, default=4)
    parser.add_argument("--speechbrain-cache", type=Path)
    args = parser.parse_args()
    for path in (args.video, args.baseline, args.target_reference):
        if not path.is_file(): parser.error(f"Missing input: {path}")
    if bool(args.mossformer_report) != bool(args.mossformer_root):
        parser.error("Provide both MossFormer report and root, or neither")

    import torch
    from faster_whisper import WhisperModel
    from speechbrain.inference.speaker import SpeakerRecognition
    torch.set_num_threads(args.threads)
    if args.device == "cuda" and not torch.cuda.is_available():
        parser.error("CUDA unavailable")
    profiles = {"Target_Speaker": args.target_reference}
    for value in args.other_reference:
        if "=" not in value: parser.error("Other reference must be NAME=PATH")
        name, raw = value.split("=", 1); path = Path(raw)
        if not name or name in profiles or not path.is_file(): parser.error("Invalid other reference")
        profiles[name] = path
    profile_vectors = {}
    for name, path in profiles.items():
        matrix = np.load(path, allow_pickle=False)
        if matrix.ndim == 1: matrix = matrix[None, :]
        profile_vectors[name] = unit(np.mean([unit(row) for row in matrix], axis=0))

    options = {"source": "speechbrain/spkrec-ecapa-voxceleb",
               "run_opts": {"device": args.device}}
    if args.speechbrain_cache: options["savedir"] = str(args.speechbrain_cache)
    speaker = SpeakerRecognition.from_hparams(**options)
    whisper = WhisperModel(args.model, device=args.device,
        compute_type="float16" if args.device == "cuda" else "int8",
        cpu_threads=args.threads)

    def voice(wave):
        if len(wave) < 6400: return {}, "Uncertain"
        tensor = torch.from_numpy(np.asarray(wave, dtype=np.float32)).unsqueeze(0).to(args.device)
        with torch.no_grad(): embedding = unit(speaker.encode_batch(tensor).detach().cpu().numpy())
        scores = {name: float(embedding @ profile) for name, profile in profile_vectors.items()}
        return scores, resolve_voice(scores)

    def transcribe(wave):
        decoded, _ = whisper.transcribe(wave, language="en", beam_size=5,
            vad_filter=False, condition_on_previous_text=False)
        rows = list(decoded)
        return " ".join(row.text.strip() for row in rows if row.text.strip())

    moss = {}
    if args.mossformer_report:
        for row in json.loads(args.mossformer_report.read_text()).get("results", []):
            moss[int(row["baseline_index"])] = row
    ownership = {}
    if args.sentence_ownership_report:
        if not args.sentence_ownership_report.is_file():
            parser.error("Missing sentence ownership report")
        for row in json.loads(args.sentence_ownership_report.read_text()).get("segments", []):
            ownership[int(row["segment_index"])] = row.get("sentence_ownership")
    baseline = json.loads(args.baseline.read_text())
    results = []
    for index in args.segment:
        segment = baseline["segments"][index]
        start, end = float(segment["start"]), float(segment["end"])
        variants = []
        for name, wave in (("original", original_audio(args.video, start, end)),
                           ("speech_filtered", filtered_audio(args.video, start, end))):
            scores, resolution = voice(wave)
            variants.append({"name": name, "family": "original_recording",
                "text": transcribe(wave), "voice_scores": scores,
                "speaker_resolution": resolution})
        if index in moss:
            for stream in moss[index].get("streams", []):
                path = args.mossformer_root / stream["audio"]
                import soundfile as sf
                wave, rate = sf.read(path, dtype="float32")
                if rate != 16000: raise ValueError("MossFormer audio must be 16 kHz")
                scores, resolution = voice(wave)
                variants.append({"name": f"mossformer_stream_{stream['stream']}",
                    "family": "mossformer2", "text": stream["transcription"]["text"],
                    "voice_scores": scores, "speaker_resolution": resolution,
                    "audio": str(path.resolve())})
        spans = cross_family_wording(variants)
        results.append({"segment_index": index, "start": start, "end": end,
            "baseline_text": segment.get("text", ""), "variants": variants,
            "cross_family_stable_spans": spans,
            "sentence_boundary_resolution": ownership.get(index),
            "joint_decision": joint_decision(variants, spans, ownership.get(index)),
            "wording_verified": False, "review_required": True,
            "baseline_modified": False})
        print(index, results[-1]["joint_decision"],
              spans[0]["text"] if spans else "(no cross-family wording)", flush=True)
    report = {"experimental": True, "baseline_modified": False,
        "identity_probability_calibrated": False,
        "wording_verified_automatically": False,
        "note": "Related transforms share one evidence family; agreement can repeat the same ASR error.",
        "provenance": {"video_sha256": sha256(args.video),
            "baseline_sha256": sha256(args.baseline), "model": args.model,
            "sentence_ownership_sha256": (sha256(args.sentence_ownership_report)
                if args.sentence_ownership_report else None),
            "references": {name: sha256(path) for name, path in profiles.items()}},
        "segments": results}
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2)+"\n")


if __name__ == "__main__": main()
