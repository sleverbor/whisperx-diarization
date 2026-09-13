# 30-second transcription and diarization test

This checkpoint records the current runnable test, not a claim of perfect diarization.
All 36 rows in each embedding file are reference samples of the **same target person**.
The script normalizes the valid samples and averages them into target voice and face references.
Other people retain their diarization track IDs. No phrases or police roles identify the target.

## Run in the existing environment

From this folder:

```bash
source .venv/bin/activate
export HF_TOKEN="your_hugging_face_token"
python chainofrules.py
```

The defaults use `short.mp4`, `voice_embeddings.npy`, and `face_embeddings.npy`.
For another video and target references:

```bash
python chainofrules.py another.mp4 --voice-priors target_voice.npy --face-priors target_face.npy
```

Output is printed and saved to `diarization_evidence.json`, which is ignored by Git.
The script reads `HF_TOKEN` or `HUGGINGFACE_TOKEN` from the environment; it does not load `.env` itself.
Use the same embedding models as your references: SpeechBrain ECAPA and InsightFace buffalo_l.
Model weights require the existing cache or downloads and the applicable Hugging Face model access.

## Recorded environment

The test ran under Python 3.12 on CPU. `requirements.txt` records direct dependency versions.
`requirements.lock.txt` records every installed distribution in the tested environment; it is an
installation snapshot, not a portable guarantee across operating systems or hardware.
The existing `.venv` and downloaded model weights are not committed.

## Baseline result

The approximately 30-second clip ran successfully with all 36 reference samples.
Voice affinities were approximately 0.318 for SPEAKER_01 and 0.076 for SPEAKER_00.
`baseline_transcript.txt` records that run for comparison.

Confirmed target lines "I'm not going to go in there" and "Where you took me from was public property"
were attributed to the target. "No" remained uncertain because its aligned interval was 0.10 seconds.
Other short replies and three conflicts also remained uncertain.
Text such as "pat you down to shake" still needs checking against the actual audio.
Confidence values are heuristic strengths, not calibrated probabilities.
The script records face presence and dark clothing context. It does **not** implement police recognition,
lipreading, audio-synchronized active speaker detection, or persistent visual person tracking.

## Safe tuning workflow

The tag `baseline-30s-all-references` marks this state. Before changing attribution rules,
commit the current work and compare the new run against the baseline transcript.

Inspect an older checkpoint without changing the current files:

```bash
git show baseline-30s-all-references:chainofrules.py
```

To work from the checkpoint, first commit any current changes, then create a separate branch:

```bash
git switch -c retry-from-baseline baseline-30s-all-references
```

Keep credentials out of tracked files. Legacy scripts remain on disk but are ignored.
This repository is local; no remote upload is configured.
