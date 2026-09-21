# Auditor reference collector

This local review tool builds a target-speaker reference from clips and images that you explicitly confirm. It exports the ECAPA voice embeddings and InsightFace embeddings consumed by `chainofrules.py` and the Kaggle pipeline.

The collector does not infer identity from a YouTube channel, a phrase, clothing, occupation, or face visibility alone. A prediction can schedule a sample for review, but only your approval can add it to a reference.

## Quick start

From the repository root:

```bash
tools/auditor_reference_collector/run_auditor_collector.sh
```

The launcher uses the repository `.venv`, binds the server to `127.0.0.1`, opens a tokenized local URL, and installs only `yt-dlp` into an isolated `.collector-deps` directory when needed. Keep the terminal open while using the app.

To select another Python environment or keep a new auditor in a separate library:

```bash
AUDITOR_PYTHON=/path/to/python \
AUDITOR_LIBRARY_DIR=/path/to/private/auditor-library \
tools/auditor_reference_collector/run_auditor_collector.sh
```

The Python environment must already provide NumPy, OpenCV, SpeechBrain, Faster Whisper, InsightFace, ONNX Runtime, PyTorch, and FFmpeg. CUDA is used when available; pass `--device cpu` to force CPU.

## Build a reference

1. **Start a fresh library for each auditor.** Never mix identities in one library.
2. Add at least three confirmed face images containing one clear face. Images may be uploaded or saved from a direct image URL.
3. Enter a YouTube video URL with a start time and duration. Use windows where you already know when the auditor speaks. Durations from 0.3 to 600 seconds are accepted.
4. Review each generated candidate. Confirm voice identity separately from face identity. Reject mixed speech, or trim it into a clean candidate and review the trim independently.
5. Correct the words when practical and label normal, raised, quiet, or other speaking style.
6. Collect samples from several videos and recording conditions. Aim for at least three clean voice clips of two seconds or longer and three face samples.
7. Select **Export reference**. The app applies a consistency screen after your approvals and writes a timestamped `reference-*` directory inside the private library.

Short approved clips from 0.3 through 1.6 seconds are kept in a separate phrase-comparison library. They never enter the main voice centroid. Samples between 1.6 and 2 seconds remain in the audit history but enter neither group.

## Exported files

The pipeline inputs are:

- `voice_embeddings.npy`: screened ECAPA reference samples
- `face_embeddings.npy`: screened InsightFace reference samples
- `voice_embedding.npy` and `face_embedding.npy`: centroids
- `reference.json`: screening results and approved-source IDs
- `short_speech_library.json`: confirmed short utterances for optional comparison

Keep the complete library as provenance. `session.json` records decisions, and the `clips`, `seeds`, and `sources` directories contain the reviewed media. Exported short-library entries refer back to this media.

For a local run:

```bash
python chainofrules.py video.mp4 \
  --voice-priors /path/to/reference/voice_embeddings.npy \
  --face-priors /path/to/reference/face_embeddings.npy
```

For Kaggle, create a private input dataset containing the exported reference plus the target enrollment WAV used by speaker-conditioned extraction. Do not put credentials in the reference or notebook.

## New-auditor validation

Keep evaluation videos out of enrollment. A useful generalization test is:

1. Build the reference from known target-only windows in several source videos.
2. Run one initial video to confirm that global voice affinity selects the expected target track.
3. Run a second unseen video with multiple speakers, short replies, noise, and overlap.
4. Keep diarization and MossFormer2 thresholds unchanged.
5. Review every high-confidence separation plus several near-misses before judging precision and recall.

Current MossFormer2 ownership corroboration requires target similarity `>= 0.25`, target margin `>= 0.20`, and baseline token F1 `>= 0.60`. It contributes evidence only: it does not insert separated ASR text or independently change speaker identity.

## Resume and backup

Every decision is saved immediately to `auditor-library/session.json`. Restarting with the same library resumes the session, and rescanning the same interval does not duplicate candidates. Back up the entire library rather than only the exported NumPy files.

The repository ignores `auditor-library`, `.collector-deps`, downloaded media, model caches, and generated embeddings. These may contain biometric data and must remain private unless the subject-specific data is deliberately published with appropriate authorization.

## Tests

The focused tests do not download models:

```bash
python -m unittest discover -s tools/auditor_reference_collector -p 'test_*.py'
```

They cover enrollment separation, consistency screening, scene grouping, URL validation, and bounded image downloading.

## Known limitations

- Face presence does not prove who is speaking.
- The midpoint face sample is not a learned active-speaker or lip-reading model.
- Off-camera and overlapping speakers still require human review.
- Similarity scores are not calibrated identity probabilities.
- YouTube downloads can fail when authentication or extractor behavior changes.
- The app has no deletion or re-review screen; back up `session.json` before manual edits.
