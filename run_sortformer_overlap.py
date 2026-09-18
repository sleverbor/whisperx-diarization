"""Run offline Sortformer in bounded, contextual windows and emit RTTM.

Speaker labels are intentionally scoped to each window.  This experiment uses
Sortformer as an independent overlap detector, so cross-window identity is not
assumed or needed.
"""

import argparse
import json
from pathlib import Path
import subprocess


def parse_segment(row):
    if isinstance(row, str):
        fields = row.replace(",", " ").split()
    elif isinstance(row, (list, tuple)):
        fields = list(row)
    else:
        raise TypeError(f"Unsupported Sortformer segment: {type(row).__name__}: {row!r}")
    if len(fields) < 3:
        raise ValueError(f"Incomplete Sortformer segment: {row!r}")
    return float(fields[0]), float(fields[1]), str(fields[2])


def unwrap(result):
    while isinstance(result, list) and len(result) == 1 and isinstance(result[0], list):
        result = result[0]
    return result


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--audio", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--model", default="nvidia/diar_sortformer_4spk-v1")
    parser.add_argument("--core-seconds", type=float, default=80.0)
    parser.add_argument("--context-seconds", type=float, default=5.0)
    args = parser.parse_args()

    import soundfile as sf
    import torch
    from nemo.collections.asr.models import SortformerEncLabelModel

    if not torch.cuda.is_available():
        raise RuntimeError("Sortformer comparison requires a CUDA GPU")
    args.output_dir.mkdir(parents=True, exist_ok=True)
    crop_dir = args.output_dir / "windows"
    crop_dir.mkdir(exist_ok=True)
    duration = float(sf.info(args.audio).duration)
    model = SortformerEncLabelModel.from_pretrained(args.model)
    model = model.to("cuda").eval()

    rttm_rows, manifest = [], []
    core_start, window_index = 0.0, 0
    while core_start < duration:
        core_end = min(duration, core_start + args.core_seconds)
        crop_start = max(0.0, core_start - args.context_seconds)
        crop_end = min(duration, core_end + args.context_seconds)
        crop = crop_dir / f"window-{window_index:03d}.wav"
        subprocess.run([
            "ffmpeg", "-nostdin", "-hide_banner", "-loglevel", "error", "-y",
            "-ss", str(crop_start), "-i", str(args.audio),
            "-t", str(crop_end - crop_start), "-ac", "1", "-ar", "16000", str(crop),
        ], check=True)
        predicted = unwrap(model.diarize(audio=str(crop), batch_size=1))
        kept = 0
        for row in predicted:
            local_start, local_end, speaker = parse_segment(row)
            absolute_start, absolute_end = local_start + crop_start, local_end + crop_start
            start, end = max(core_start, absolute_start), min(core_end, absolute_end)
            if end <= start:
                continue
            scoped_speaker = f"window{window_index:03d}_{speaker}"
            rttm_rows.append(
                f"SPEAKER full-video 1 {start:.3f} {end-start:.3f} <NA> <NA> {scoped_speaker} <NA> <NA>"
            )
            kept += 1
        manifest.append({
            "window": window_index, "core_start": core_start, "core_end": core_end,
            "crop_start": crop_start, "crop_end": crop_end, "segments_kept": kept,
        })
        print(f"Sortformer window {window_index + 1}: {core_start:.1f}-{core_end:.1f}s, {kept} segments")
        core_start, window_index = core_end, window_index + 1

    (args.output_dir / "sortformer.rttm").write_text("\n".join(rttm_rows) + "\n")
    (args.output_dir / "windows.json").write_text(json.dumps(manifest, indent=2) + "\n")
    print(f"Saved {len(rttm_rows)} RTTM rows across {len(manifest)} windows")


if __name__ == "__main__":
    main()
