# Complete contextual transcription/alignment test

The original Kaggle results, target reference and diarization tracks were kept unchanged. This optional diagnostic re-aligned the complete audio-only 25-second hypothesis from source 665–690 seconds, including officer/radio speech omitted by the original short-clip decode. No captions or user listening notes were supplied. SpeechBrain voice evidence was then collected from the corrected aligned crops. No final speaker resolver was changed.

| Target response | Old start | Corrected start | Old voice similarity | Corrected similarity |
|---|---:|---:|---:|---:|
| I don't get found guilty of anything. | 665.24s | 677.51s | 0.047 | 0.274 |
| I do have a driver's license. | 667.32s | 680.75s | -0.045 | 0.516 |
| I have a driver's license. | 670.70s | 684.78s | 0.037 | 0.339 |
| I didn't give you any fake information. | 672.73s | 685.74s | -0.057 | 0.301 |

All four corrected starts are within 0.40 seconds of the independent audio-only timing control. Each target voice similarity increases. The strongest driver-license response rises from -0.045 to 0.516. These are relative voice similarities, not calibrated probabilities.

The corrected local transcript retains intervening questions and the officer/radio opening. Speaker labels have deliberately not been inferred from expected wording; the output is marked Speaker not resolved. Saved short-clip diarization still collapses most speech to one track. This test establishes better timing and voice evidence, not complete diarization accuracy.

Next bounded integration: offer contextual decoding/alignment as an optional local repair stage and feed its corrected crops into the existing evidence/resolver process. Preserve old results for comparison. A separate local speaker-separation test remains necessary.

Limits: one noisy 25-second section tested locally on CPU; independent decoder timestamps are a control, not manually annotated ground truth. The final partial sentence at the clip boundary should not be accepted as complete.
