# Current-reference Kaggle results

## Successful execution

Speech processing used CUDA. All five InsightFace model sessions report CUDA first, with CPU fallback available. Improved embedded reference hashes match the notebook manifest. The full-video baseline contains 445 segments. Gap recovery tested 18 uncovered intervals, produced 32 separate review candidates, and preserved every original segment exactly.

## Useful recovered text

The height and weight questions and answers are recovered. The 12:47 gap includes the arrest-form explanation and the short response exchange. The handcuff gap includes agreement, a stuck-key explanation, offering to inspect cuffs and a replacement-key discussion. Some recovered wording remains garbled, especially under key noise. Other additions include questionable background phrases and incomplete sentence fragments. These are review candidates, not automatically accepted transcript lines or identified speakers.

## Attribution remains unresolved

The opening 30-second test has 11 Target_Speaker lines, 12 non-target lines and one uncertain line. It retains the previously established target responses, including the brief No and final criminal-loitering response. That does not establish transcription accuracy for every word.

The 65-second height/weight test regressed in coverage: only 10 segments compared with the previous local 18; all ten are assigned SPEAKER_00. Its voice affinities distinguish a target track (0.512 versus 0.158), but no final transcript line receives that target label. Reference identification alone is not solving segment-to-track assignment.

The full-video base transcript has exactly the same wording and boundaries as the previous full-video run. Only three final labels changed; the target count is 149 versus 150 previously. It still misattributes known target lines in the opening and date-of-birth exchange. The separate opening clip does substantially better on the same voices. At replay time near 15:35, attribution is also inconsistent with the original recording.

The sentence at 18:10 combines the key officer's reflection statement with the target's agreement under one speaker. Gap recovery retains uncertainty and does not split/attribute all recovered turns.

## Next bounded experiment

Investigate timestamp/diarization-track overlap and mixed-speaker segment boundaries using the saved checkpoints for the height/weight clip, then test local scene windows against global full-video diarization. Keep ASR, reference and existing output fixed during that experiment. Strong heuristic scores are not reliable accuracy percentages and should not be used to hide these errors. No automatic confidence filtering was applied.
