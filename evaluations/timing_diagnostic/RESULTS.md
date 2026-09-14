# Frozen-baseline timing diagnostic

The original transcription, reference and diarization were not modified. Saved checkpoints were audited, then an independent raw-audio 25-second timing control was decoded with large-v2, CPU int8, beam 5, no VAD and no supplied transcript prompt. The control is diagnostic evidence, not an automatic replacement.

| Retained baseline wording | Saved source start | Audio-control start | Shift |
|---|---:|---:|---:|
| I don't get found guilty of anything. | 665.24s | 677.28s | +12.04s |
| I do have a driver's license. | 667.32s | 680.68s | +13.36s |
| I have a driver's license. | 670.70s | 684.38s | +13.68s |
| I didn't give you any fake information. | 672.73s | 685.64s | +12.91s |

The audio control includes the officer/radio lines before these responses, which the saved short-clip ASR omitted. The forced alignment placed the retained responses near the start of that incompletely transcribed block. This accounts for a major part of the observed wrong speaker assignment; it is not evidence of a uniform constant offset across the entire clip. Earlier lines agree with full-video timing.

The saved diarization also collapses almost all speech onto SPEAKER_00; SPEAKER_01 occurs for only about 1.60 seconds in two overlapping intervals. Simple segment or word overlap cannot recover a missing speaker track. For the chest response, target overlap is only about 0.10 seconds, versus 0.78 seconds on the other track. This is an additional diarization issue, separate from the late-block alignment failure.

At the saved wrong crop, the driver-license voice similarities to the target are -0.045 and 0.037. At the full-video aligned crops they are 0.471 and 0.353. These values support the timing explanation but are not calibrated identity probabilities.

Next implementation test: decode bounded contextual windows to retain intervening speech, align complete local hypotheses, and compare timing before collecting identity evidence. Mark disagreement for review rather than silently assigning a speaker. Test scene-local diarization/voice evidence separately because repairing timestamps alone cannot undo the collapsed track. Do not tune identity thresholds to compensate for incorrect audio crops.

All existing model results remain unchanged. No new full-video run was needed.
