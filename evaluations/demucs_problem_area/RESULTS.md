Demucs did not improve the transcription of this 25-second problem area. Keep the original audio as the preferred source; the separated audio remains a review experiment.

| Item | Original audio decode | Demucs vocals decode |
|---|---|---|
| Height question | How tall are you? | How tall are you? |
| Height answer | 5'10 and a half, 5'11 | 5'9 and a half, 5'11 |
| Later height wording | Tall girls, 5'11, 5'10 | Tall girls, 5'11, 5'9 |
| Weight question, both occurrences | How much do you weigh? | How much do you weigh? |
| Weight response | Sorry? … 160 | Sorry? … 160 |

The original and vocals each produced nine segments, with essentially the same other wording. The new height-number change is a suspected regression: the original and several earlier raw-audio/gap-window decodes consistently produced 5'10. Exact wording should be checked by listening; there is no complete human-verified reference transcript and no calculated accuracy score.

The removed-sounds residual produced “I'll see you in the next video, bye bye.” This is unsupported by the original or vocals decodes and has average log probability -1.30. It is treated as a likely ASR hallucination, not accepted as recovered speech. Residual audio is supplied so speech leakage can be checked by ear.

The original isolated interval already recovered the height and weight questions. This suggests that selecting a suitable interval with enough surrounding audio helped more than Demucs for this example. It does not establish that Demucs is ineffective on other recordings, or that speech-specific denoising cannot help.

The tracks have identical lengths: 400,000 samples each at 16 kHz, exactly 25 seconds. Their reconstruction error after common resampling is 2.53e-7. Separation used unclipped FLOAT saving after detecting and bypassing the installed Demucs CLI WAV writer's peak clipping. No loudness normalization or clipping was applied to the final comparison tracks.

All ASR settings were identical. Full settings, versions and per-run times are recorded in summary.json. The original transcript, target reference and diarization were not changed. This experiment uses source-video 10:40–11:05; all text transcripts display source-video seconds.
