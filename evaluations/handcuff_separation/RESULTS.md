# Handcuff scene: local speaker-separation test

## Outcome: partial success, not ready to integrate

The recovered audio-only wording was kept fixed. Each saved candidate was re-aligned in its local audio crop; the user listening notes and YouTube captions were not model prompts. Voice-reference comparison used the same provisional minimum similarity 0.25 and margin 0.08 as the prior test. All original results remain unchanged.

The longer agreement response matches the target voice. Several key-officer utterances have weak matches to both known references, consistent with another voice, but unknown pairwise similarities also link speech the user assigned to different officers. Connected similarity groups therefore cannot establish a reliable third identity.

One baseline sentence combines the key officer's reflection statement and the target's agreement. The naive sentence-level voice matcher assigns the whole mixed sentence to the target. This is a known error against the user's separate listening reference, not a successful attribution.

## Additional scene-local diarization

A fresh CPU community-1 diarization pass on source 1090–1150 seconds used num_speakers=3, based on the user's scene description. This is a test-specific constraint, not a default for arbitrary videos. It produced nine diarization intervals and three anonymous tracks.

SPEAKER_01 carries much of the replacement-key/arm-on-wall/shackle discussion near 18:53–19:10, consistent with the key officer's later turns in the user's listening notes. SPEAKER_00 covers part of the original officer's cuff-inspection offer, but also some stuck-key speech the user attributed to the key officer. SPEAKER_02 includes the mixed reflection/agreement sentence and is not a clean target track.

The model leaves the longer target agreement response entirely outside its detected intervals, despite the successful direct target voice match. Several other audible turns are also absent. The scene is therefore failing speech/turn detection as well as clustering. Fixed three-speaker count alone is insufficient.

Scene profile identity mapping based on a single mixed sample is unreliable. The experimental output demonstrates this failure and should not replace the baseline. Word-run output also fragments around zero-duration or tied timestamp overlaps; its fine boundaries are provisional.

Recovered text still contains garbled phrases, including an alignment failure on an apparent background phrase. Their attribution is not trustworthy merely because an embedding or track ID exists. No claim of reliable lipreading, visual active-speaker detection or radio identity is made.

## Next practical direction

Retain trustworthy direct voice matches even when diarization misses their turns. For this noisy scene, test speech detection/segmentation in shorter windows before collecting clean multi-sample profiles for unnamed speakers. Mixed-speaker blocks need splitting before their embeddings can be used as identity references. Do not relax identity thresholds to force all three people into labels.
