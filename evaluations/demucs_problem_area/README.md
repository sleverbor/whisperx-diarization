This controlled experiment tests Demucs on source-video time 10:40–11:05: the noisy tattoo, height and weight discussion. It leaves the existing transcript, target reference, and diarization unchanged.

The original 25-second audio was extracted as 44.1 kHz stereo for htdemucs. Demucs uses one shift, a fixed random seed, and CPU on this machine; the reproduction script chooses CUDA when available. The removed-sounds track is the original minus the vocals estimate, not a selected music instrument stem.

All three tracks are downmixed and resampled identically to 16 kHz mono before transcription. They use large-v2, English, beam size five, no voice activity filtering, no previous-text conditioning, and word timestamps. No expected phrases are supplied as a prompt. All decoder output is retained for inspection; an ASR result on residual noise can be a hallucination.

The installed Demucs 4.1.0 CLI writer clamps floating-point WAVs even with clip-mode none. This was detected by reconstruction checks. The final outputs instead use the Demucs Python API and soundfile FLOAT saving, with no clipping or rescaling. Vocals plus residual reconstruct the original; the comparison script also checks equal track lengths and reconstruction after resampling. The initial clipped CLI outputs are not included in the results.

The *_asr.wav files are the exact 16 kHz mono arrays used for ASR and are suitable for listening. The original.wav, vocals.wav and removed_sounds.wav files are the higher-rate source and unclipped separated tracks. Transcript text files display full source-video times; JSON word times are relative to the 25-second excerpt, with source_offset_seconds=640.

Reproduce from the project virtual environment:

```text
ffmpeg -ss 640 -i video.mp4 -t 25 -vn -ar 44100 -ac 2 original.wav
.venv/bin/python separate_audio.py original.wav --output-dir demucs-review
.venv/bin/python compare_audio.py --original original.wav --vocals demucs-review/vocals.wav --residual demucs-review/removed_sounds.wav --output-dir demucs-review --source-offset 640
```

Model downloads may be needed on first use. These scripts contain no Hugging Face token. A production decision should depend on recovered words and retained speech, rather than perceived audio clarity or decoder confidence alone. Speaker matching continues to use original audio.
