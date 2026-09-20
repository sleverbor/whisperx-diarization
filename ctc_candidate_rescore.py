"""Rank competing transcript candidates with an independent CTC acoustic model.

The manifest supplies audio intervals/files and candidate texts. Candidate
origin is excluded from model input and revealed only in the report. Scores are
comparative acoustic evidence, not calibrated correctness probabilities.
"""
from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
import re
import subprocess

import numpy as np

from sentence_ownership_probe import sha256


def normalize_ctc_text(text):
    value = re.sub(r"[^A-Z' ]+", " ", str(text).upper())
    value = re.sub(r"\s+", " ", value).strip()
    return value


def encode_candidate(text, labels):
    normalized = normalize_ctc_text(text)
    table = {label: index for index, label in enumerate(labels)}
    symbols = list(normalized.replace(" ", "|"))
    missing = sorted(set(symbols)-set(table))
    if missing:
        raise ValueError(f"Unsupported CTC symbols: {missing}")
    return normalized, [table[symbol] for symbol in symbols]


def ctc_candidate_loss(log_probs, token_ids, blank=0):
    """Return raw and token-normalized negative CTC log likelihood."""
    import torch
    if not token_ids:
        return math.inf, math.inf
    targets = torch.tensor(token_ids, dtype=torch.long, device=log_probs.device)
    input_lengths = torch.tensor([log_probs.shape[0]], dtype=torch.long)
    target_lengths = torch.tensor([len(token_ids)], dtype=torch.long)
    loss = torch.nn.functional.ctc_loss(
        log_probs.unsqueeze(1), targets, input_lengths, target_lengths,
        blank=blank, reduction="sum", zero_infinity=False)
    raw = float(loss.detach().cpu())
    return raw, raw/len(token_ids)


def rank_scores(rows):
    ordered = sorted(rows, key=lambda row: (row["normalized_ctc_loss"], row["raw_ctc_loss"]))
    for rank, row in enumerate(ordered, 1):
        row["rank"] = rank
        row["loss_from_best"] = row["normalized_ctc_loss"]-ordered[0]["normalized_ctc_loss"]
    return ordered


def greedy_decode(emissions, labels, blank=0):
    ids = emissions.argmax(dim=-1).detach().cpu().tolist()
    collapsed = []
    previous = None
    for index in ids:
        if index != previous and index != blank:
            collapsed.append(labels[index])
        previous = index
    return "".join(collapsed).replace("|", " ").strip()


def decode_video_interval(video, start, end, filtered=False):
    command = ["ffmpeg", "-nostdin", "-hide_banner", "-loglevel", "error",
        "-ss", str(start), "-i", str(video), "-t", str(end-start), "-vn"]
    if filtered:
        command.extend(["-af", "highpass=f=120,lowpass=f=6500,afftdn=nf=-25"])
    command.extend(["-ar", "16000", "-ac", "1", "-f", "f32le", "pipe:1"])
    return np.frombuffer(subprocess.check_output(command), dtype="<f4").copy()


def load_audio(row, video):
    import soundfile as sf
    if "path" in row:
        path = Path(row["path"])
        wave, rate = sf.read(path, dtype="float32")
        if rate != 16000:
            import torch
            wave = torch.nn.functional.interpolate(
                torch.from_numpy(np.asarray(wave)).reshape(1, 1, -1),
                size=round(len(wave)*16000/rate), mode="linear",
                align_corners=False).reshape(-1).numpy()
        return np.asarray(wave, dtype=np.float32), {"path": str(path.resolve()), "sha256": sha256(path)}
    start, end = float(row["start"]), float(row["end"])
    return decode_video_interval(video, start, end, bool(row.get("filtered"))), {
        "video_start": start, "video_end": end, "filtered": bool(row.get("filtered"))}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--video", type=Path, required=True)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--device", choices=("cpu", "cuda"), default="cpu")
    parser.add_argument("--threads", type=int, default=4)
    args = parser.parse_args()
    for path in (args.video, args.manifest):
        if not path.is_file(): parser.error(f"Missing input: {path}")
    import torch
    import torchaudio
    torch.set_num_threads(args.threads)
    if args.device == "cuda" and not torch.cuda.is_available(): parser.error("CUDA unavailable")
    bundle = torchaudio.pipelines.WAV2VEC2_ASR_BASE_960H
    model = bundle.get_model().to(args.device).eval()
    labels = bundle.get_labels()
    manifest = json.loads(args.manifest.read_text())
    results = []
    for case in manifest["cases"]:
        # Hide origins while scoring; restore metadata only after all losses exist.
        candidates = [{"candidate_id": item["candidate_id"], "text": item["text"]}
                      for item in case["candidates"]]
        origins = {item["candidate_id"]: item.get("origin", "unspecified")
                   for item in case["candidates"]}
        audio_results = []
        for audio in case["audio"]:
            wave, provenance = load_audio(audio, args.video)
            if wave.ndim > 1: wave = wave.mean(axis=-1)
            tensor = torch.from_numpy(wave).to(args.device).unsqueeze(0)
            with torch.inference_mode():
                emissions, _ = model(tensor)
                log_probs = torch.log_softmax(emissions[0], dim=-1)
            greedy_text = greedy_decode(emissions[0], labels)
            rows = []
            for candidate in candidates:
                normalized, token_ids = encode_candidate(candidate["text"], labels)
                raw, per_token = ctc_candidate_loss(log_probs, token_ids)
                rows.append({**candidate, "normalized_text": normalized,
                    "raw_ctc_loss": raw, "normalized_ctc_loss": per_token})
            ranked = rank_scores(rows)
            for row in ranked: row["origin"] = origins[row["candidate_id"]]
            audio_results.append({"name": audio["name"], "family": audio["family"],
                "include_in_aggregate": bool(audio.get("include_in_aggregate", True)),
                "speaker_evidence": audio.get("speaker_evidence"),
                "provenance": provenance, "greedy_ctc_transcript": greedy_text,
                "ranking": ranked})
        aggregate = []
        for candidate in candidates:
            scores = [next(row for row in audio["ranking"]
                           if row["candidate_id"] == candidate["candidate_id"])["normalized_ctc_loss"]
                      for audio in audio_results if audio["include_in_aggregate"]]
            if not scores: raise ValueError("Each case needs aggregate-eligible audio")
            aggregate.append({**candidate, "origin": origins[candidate["candidate_id"]],
                "mean_normalized_ctc_loss": float(np.mean(scores)),
                "median_normalized_ctc_loss": float(np.median(scores))})
        aggregate.sort(key=lambda row: row["median_normalized_ctc_loss"])
        for rank, row in enumerate(aggregate, 1): row["rank"] = rank
        results.append({"case_id": case["case_id"], "purpose": case.get("purpose"),
            "audio_results": audio_results, "aggregate_ranking": aggregate,
            "human_answer_used_as_model_input": False,
            "review_required": True})
        print(case["case_id"], "->", aggregate[0]["candidate_id"], aggregate[0]["text"], flush=True)
    report = {"experimental": True, "model": "torchaudio/WAV2VEC2_ASR_BASE_960H",
        "model_family_independent_from_whisper": True,
        "scores_are_calibrated_probabilities": False,
        "baseline_modified": False, "video_sha256": sha256(args.video),
        "manifest_sha256": sha256(args.manifest), "cases": results}
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2)+"\n")


if __name__ == "__main__": main()
