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
The script records face presence and short-term target face continuity. It does **not** implement
police recognition, lipreading, or a learned audio-synchronized active speaker detector.

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

## Short-answer tuning

On `tuning-short-answers`, audio crops for very short replies can extend into neighboring timing
gaps, but never into the preceding or following transcript utterance. Voice strength stays low
for these short crops. A direct yes/no question followed immediately by a brief answer can supply
a separate conversational hypothesis when the questioner's assignment is strong and only one
other diarization track is present. This is a low-strength inferred assignment (at most 0.25),
not acoustic verification. Three or more detected speakers leave the answer unresolved.
The question could still be self-answered or addressed to an untracked person; the weak strength
and explicit evidence record reflect that limitation. No police role or specific clip timing is required.

Run behavior checks in the existing environment with `python -m unittest test_short_answers.py`.

The fresh short-answer run attributed "No" to the target with strength 0.20 (context only).
"Yeah, I do" obtained a safe 0.44-second crop, but the voice comparison was weak and
conflicting; all replies shorter than 0.40 seconds now have strength capped at 0.30.
Replaying the fixed baseline evidence changed only "No" and "Yeah, I do" from uncertain
to weak contextual target assignments. A fresh ASR pass produced 23 segments rather than
the baseline's 24, with variation near "OK / All right"; this variation is separate from
the attribution tuning. Three longer baseline conflicts remain uncertain.

## Visual continuity tuning

On `tuning-face-continuity`, the voice correction was committed separately from the visual change.
Independent voice profiles now resolve three previous baseline conflicts to SPEAKER_00.
The visual check uses a half-second identity lead-in, appearance similarity, bounding-box overlap,
and a maximum detection gap to continue a recently recognized target face through a head turn.
Lead-in frames only establish identity; mouth measurements use frames within the utterance.

Normalized 3D mouth-landmark motion is a weak speaking hint, not lipreading or audio-visual synchronization.
It can suggest the target at strength 0.25 only when independent voice profiles are both weak and
closely matched. Face presence alone cannot change identity, and stronger voice profiles are retained.
This heuristic can still mistake expression changes or landmark noise for speech.

The actual video was reanalyzed with the saved 23-utterance transcript and voice evidence, so the
comparison isolates this tuning from fresh-ASR variation. `visual_tuning_transcript.txt` records it.
Three unresolved officer lines changed to SPEAKER_00; the final "Criminal loitering" changed to
a tentative Target_Speaker (0.25). All other speaker labels remained unchanged.
The user identified that final speaker as the target. There were four direct target identity matches
in the visual lead-in/utterance and five tracked mouth measurements during the utterance.
No automatic police role is inferred: SPEAKER_00 is the officer identified in this particular test.
The transcript wording still needs audio verification, and general accuracy remains unproven.

## Full-video cloud test

See [KAGGLE.md](KAGGLE.md) and `kaggle_full_video.ipynb` for the private GPU runner.

## Experimental transcription coverage

See [COVERAGE_TEST.md](COVERAGE_TEST.md) and `diarization_coverage_compare.ipynb` for paired short-clip tests.
