These are optional review tools for the existing project environment. They do not replace `chainofrules.py` or change its defaults. No expected dialogue or video-specific speaker rules are used.

`review_audio_window.py` transcribes a complete requested audio window with faster-whisper's native decoder and word timestamps, aligns the complete hypothesis using WhisperX, and matches each utterance against supplied ECAPA voice references. It writes `review_transcript.txt` and structured evidence. The default compares native decoder and WhisperX alignment voice crops. It requires their leading voice identity to agree and at least one crop to pass the voice gates. Conflicting identities remain uncertain; overlapping crops are one evidence family and never accumulate confidence. Native decoder times provide the displayed sentence bounds where available. Add `--timing-source decoder` or `--timing-source alignment` to compare either method alone. Sentence links use only the decoder's own generated text, never a supplied expected transcript. Every hypothesis still requires review; cosine similarity is not a calibrated probability. With only a target reference there is no competing-profile margin test.

Run from the project directory using its existing virtual environment. For example:

```bash
.venv/bin/python review_audio_window.py \
  --video video.mp4 --start 665 --duration 25 \
  --target-reference references/youtube-v1/voice_embeddings.npy \
  --other-reference 'Opening_officer=evaluations/speaker_separation/officer_reference.npy' \
  --baseline full_video_evidence.json \
  --speechbrain-cache pretrained_models/spkrec-ecapa-voxceleb \
  --output-dir review-665s
```

Change the paths to your files. If a baseline uses times relative to a clip, add `--baseline-offset` with that clip's start in the original video. Raw non-target track IDs are retained in baseline evidence and are not merged into an officer identity. The supplied comparison profile is an opening-clip voice reference, not proof that its speaker is a police officer.

`bounded_silero_vad.py` must be next to the runner for the experimental `--asr-engine bounded-whisperx` mode. That mode passed opening coverage but regressed on the height/weight scene and is **not recommended as a default**. Its very short crops produced invented wording. The default native mode does not cut audio with a speech detector. GPU is selected when available; CPU uses int8 ASR and four threads. Public models may download on first use. Any gated model access should use the `HF_TOKEN` environment variable; these tools contain no credential and do not need one for the public ASR/voice models.

`collect_duplicate_evidence.py` finds repeated recordings in a video using normalized, band-limited waveform correlation. It adds `supplemental_evidence` to a separate copy of a transcript. Original text, identities, timings, confidence values and existing evidence are preserved. Source non-target track IDs are scoped to their own transcript; duplicate recordings never become independent voice-reference samples. A high recording-match score does not increase the source's identity confidence.

```bash
.venv/bin/python collect_duplicate_evidence.py \
  --video video.mp4 \
  --source-transcript opening_evidence.json --source-times-local \
  --transcript full_video_evidence.json \
  --source-start 0 --source-duration 30 \
  --query-start 925 --query-duration 60 \
  --source-name reviewed-opening \
  --output full_video_with_duplicate_evidence.json
```

The example timestamps are test selections, not rules embedded in either tool. Exact replays are optional evidence; unrelated videos are not expected to match. Face presence, mouth aperture and question/answer order are not used to overwrite speaker identity.
