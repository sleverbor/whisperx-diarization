# Two-minute evaluation at the unchanged checkpoint

Input: first 120 seconds of `video.mp4` (source length approximately 22:44).
It includes encounter footage and studio-style commentary beginning at about 56 seconds.
Code: `2da606e`; no attribution or model changes were made for this test.
Runtime: approximately 23.6 minutes on CPU, using the existing environment and cached weights.

## Result

51 timed utterances: 30 Target_Speaker and 21 SPEAKER_00. No utterances were marked Uncertain.
The commentary remained on the target track. Voice affinities were approximately 0.366
for SPEAKER_01 (9 samples) and 0.100 for SPEAKER_00 (14 samples).

This is an evaluation result, not an accuracy certification. The user-reviewed target
reply "Criminal loitering?" at 28.70–29.28 seconds was incorrectly assigned to SPEAKER_00.
In the 30-second test this had resolved tentatively to the target. The longer run changed
the timing window and voice profiles. The face was recognized and tracked, but the mouth
landmark spread was 0.017, below the heuristic motion threshold, so the visual cue did
not fire. The independent voice profiles weakly favored the target, but the target
reference-affinity score favored non-target. The resolver still returned SPEAKER_00
with strength 0.67, illustrating that its strengths are not calibrated probabilities.
Other short replies and wording still need human review; do not assume the remaining
50 utterances are correct. The alignment transcribed the brief answer as "Nope" rather
than "No"; it still resolved to the target by weak question/answer inference (0.20).

## Files and rerun

`longer_2min.mp4` is the input, `longer_2min_transcript.txt` the readable result,
`longer_2min.srt` the timed speaker captions, and `longer_2min_evidence.json` the full trace.
The labeled preview video is a generated review artifact outside the repository.
From this project directory, with the existing environment activated and HF_TOKEN exported:

```bash
python chainofrules.py longer_2min.mp4 --output longer_2min_evidence.json
```

The input was cut with FFmpeg to 120 seconds and converted to 21.11 fps, H.264 CRF 21,
and AAC 128k. This preserves the constant-frame-rate assumption used by the visual sampler.
Fresh transcription can vary; use the recorded result for comparisons.
SPEAKER_00 is an identity track, not automatic police-role recognition.

## Fingerprints

- test_clip_sha256: `734c457adbb6f03fa3d48dac9334262e4d77069bee73b5327d8a3fd6638a9645`
- script_sha256: `1304a2f0487c6ab9403f50ba05d3ab3195466c46306d66ee67fba27de51da87a`
- voice_references_sha256: `05f81cb80be127bdfee8e8c03f5bfd4c949db914197cf56138f8e11c444438e6`
- face_references_sha256: `317afa985476392c52470cb99c03b1680d6b2b071d1903dbb521e49a4b700774`
