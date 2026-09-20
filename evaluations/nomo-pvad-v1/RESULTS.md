# nomo-pVAD target-activity screen v1

This experiment evaluates nomo-pVAD release 1.1 with the existing 19-second
auditor enrollment audio. It scores target activity every 160 ms on four target,
four non-target, and two mixed intervals from the untouched outdoor video.
Thresholds 0.5, 0.6, and 0.7 were fixed before evaluation.

## Short reply

The 0.72-second human-labeled `I'm not panhandling` reply produced five frames
with probabilities 0.46, 0.60, 0.67, 0.62, and 0.60. Four of five frames passed
0.5. The immediately following officer reply averaged 0.213 and had no frame at
or above 0.5. This is the first tested method that adds useful target-activity
evidence on this short phrase.

## Mixed sentence boundaries

For `We can ... test that if you'd like`, the score remained near zero through
the prefix and rose above 0.5 at 419.96 seconds. The earlier word-run probe placed
the target-like suffix at 420.10 seconds. This independently supports treating
the ASR sentence as a mixed turn.

For the public-property sentence, a sustained target-active run began at 426.92
seconds and continued through 428.52. This supports target ownership only for
the later part of the mixed interval, consistent with the joint audio evidence
probe.

## Fixed-threshold interval screen

Using each labeled interval's mean probability, threshold 0.5 detected all four
target intervals and rejected all four non-target intervals. Threshold 0.6
missed the short reply, and threshold 0.7 retained only two of four target
intervals. These eight cases are too small to calibrate a production threshold.

Frame-level behavior was less clean than the interval means. One long officer
passage contained several false target-active runs, including a 1.44-second run
averaging 0.78. Another non-target passage rose late as the next speaker turn
approached. The model therefore cannot be used alone to mute, extract, or assign
words.

## Decision

nomo-pVAD is worth retaining as supplemental temporal evidence. It can propose
target onsets, split suspiciously merged sentences, and prioritize extraction
or ASR review windows. A final attribution should still require agreement with
diarization, local voice evidence, separation, or audiovisual activity. The
baseline transcript was not modified.

The evaluation pins upstream release 1.1 at commit
`cb828468d183424fd574984f00b3ffbcce0be9bc` and uses the upstream bundled pVAD
weights plus its specified ERes2NetV2 enrollment model.
