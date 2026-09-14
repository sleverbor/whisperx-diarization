This experiment compares original audio, a 50% dry / 50% denoised blend, and fully denoised audio on source-video 10:40–11:05. It uses the exact same original 16 kHz mono audio as the preceding Demucs experiment. The current transcript, speaker references, and diarization pipeline are unchanged.

Processing uses Meta's pretrained DNS64 speech denoiser ([denoiser 0.1.5](https://github.com/facebookresearch/denoiser)). The model runs on CUDA when available, otherwise CPU. All outputs preserve the original 400,000 samples at 16 kHz (25 seconds) and are saved as FLOAT without clipping or loudness normalization. The light blend follows the denoiser's dry/wet mixing formula. audio/removed.wav is the original minus the full denoiser output, provided for listening to possible speech loss.

All ASR runs use large-v2, English, beam size five, no voice activity filtering, no previous-text conditioning, and word timestamps. No expected words are supplied as a prompt. JSON contains unmodified decoder output, including confidence measures; transcript text files display full source-video seconds. Confidence alone cannot establish that a decoded word is correct.

The denoiser was installed with --no-deps into an isolated workspace folder, preserving the existing project environment. To reproduce in a fresh test folder using the project virtual environment:

```text
.venv/bin/python -m pip install --no-deps --target pretrained_models/denoiser-tools denoiser==0.1.5
PYTHONPATH=pretrained_models/denoiser-tools .venv/bin/python evaluations/denoiser_test/denoise_audio.py evaluations/denoiser_test/audio/original.wav --output-dir denoiser-review/audio --model-cache pretrained_models/denoiser-cache --wet 0.5
.venv/bin/python evaluations/denoiser_test/compare_denoising.py --original denoiser-review/audio/original.wav --light denoiser-review/audio/light.wav --full denoiser-review/audio/full.wav --output-dir denoiser-review/comparison --source-offset 640
```

The first model use downloads public DNS64 weights. The model hash and input hash are recorded in audio/denoising_summary.json. No Hugging Face token is stored or required by these scripts. The exact mono FLOAT arrays sent to ASR are also saved as *_asr.wav.

Listening copies named listen_*.wav share one common attenuation to prevent playback clipping. ASR input files remain unattenuated and unchanged.
