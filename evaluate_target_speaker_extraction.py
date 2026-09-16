"""Evaluate reference-conditioned target speech extraction on short overlap crops.

This is an experimental evaluator, not part of the main resolver. It keeps both
the original and extracted results so an extraction failure cannot erase useful
transcript evidence.
"""

import argparse
import json
from pathlib import Path

import numpy as np
import soundfile as sf
import torch
import torchaudio
from faster_whisper import WhisperModel
from speechbrain.inference.speaker import SpeakerRecognition


def unit(vector):
    vector = np.asarray(vector, dtype=np.float32).reshape(-1)
    return vector / max(float(np.linalg.norm(vector)), 1e-9)


def load_audio(path, sample_rate=16000):
    wave, source_rate = torchaudio.load(str(path))
    wave = wave.mean(dim=0, keepdim=True).float()
    if source_rate != sample_rate:
        wave = torchaudio.functional.resample(wave, source_rate, sample_rate)
    return wave


def transcribe(model, wave):
    segments, _ = model.transcribe(
        wave.squeeze(0).numpy(),
        vad_filter=False,
        condition_on_previous_text=False,
        beam_size=5,
    )
    return " ".join(segment.text.strip() for segment in segments if segment.text.strip())


def rms(wave):
    return float(torch.sqrt(torch.mean(wave.float() ** 2)).item())


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--crop", type=Path, action="append", required=True)
    parser.add_argument("--enrollment", type=Path, required=True)
    parser.add_argument("--voice-priors", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--wesep-model-dir", type=Path)
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--whisper-model", default="small")
    parser.add_argument(
        "--normalize-output",
        action="store_true",
        help="Normalize extracted audio. Leave disabled when testing rejection/suppression.",
    )
    args = parser.parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)

    try:
        import wesep
    except ImportError as error:
        raise SystemExit(
            "WeSep is optional. Install requirements-overlap-extraction.txt first."
        ) from error

    if args.wesep_model_dir:
        extractor = wesep.load_model_local(str(args.wesep_model_dir))
    else:
        extractor = wesep.load_model("english")
    extractor.set_device(args.device)
    extractor.set_vad(True)
    extractor.set_output_norm(args.normalize_output)

    priors = np.load(args.voice_priors)
    target = unit(np.mean(np.stack([unit(row) for row in priors]), axis=0))
    speaker_dir = Path("pretrained_models/spkrec-ecapa-voxceleb")
    speaker = SpeakerRecognition.from_hparams(
        source=str(speaker_dir), savedir=str(speaker_dir), run_opts={"device": args.device}
    )
    whisper_device = "cuda" if args.device.startswith("cuda") else "cpu"
    whisper_compute = "float16" if whisper_device == "cuda" else "int8"
    whisper = WhisperModel(
        args.whisper_model, device=whisper_device, compute_type=whisper_compute
    )

    report = []
    for crop in args.crop:
        extracted_path = args.output_dir / f"{crop.stem}-target.wav"
        original = load_audio(crop)
        extracted = extractor.extract_speech(str(crop), str(args.enrollment))
        if extracted is None:
            report.append({"crop": str(crop), "error": "extractor returned no speech"})
            continue
        extracted = extracted[0].detach().cpu().unsqueeze(0)
        sf.write(extracted_path, extracted.squeeze(0).numpy(), 16000)
        original_rms = rms(original)
        extracted_rms = rms(extracted)
        energy_retention = extracted_rms / max(original_rms, 1e-9)

        for kind, path, wave in (
            ("original", crop, original),
            ("conditioned", extracted_path, extracted),
        ):
            embedding = unit(
                speaker.encode_batch(wave).flatten().detach().cpu().numpy()
            )
            report.append(
                {
                    "crop": str(crop),
                    "kind": kind,
                    "target_similarity": float(np.dot(target, embedding)),
                    "rms": rms(wave),
                    "extracted_energy_retention": energy_retention,
                    "transcript": transcribe(whisper, wave),
                    "audio": str(path),
                }
            )

    report_path = args.output_dir / "report.json"
    report_path.write_text(json.dumps(report, indent=2) + "\n")
    for row in report:
        if "error" in row:
            print(f"{Path(row['crop']).name}: {row['error']}")
        else:
            print(
                f"{Path(row['crop']).name} {row['kind']}: "
                f"similarity={row['target_similarity']:.3f}: {row['transcript']}"
            )


if __name__ == "__main__":
    main()
