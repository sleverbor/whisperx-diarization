# Contextual WeSep evaluation v1

Five fixed, hand-labeled overlap exchanges were extracted with 3 seconds and 5 seconds of surrounding context. The extracted streams were cropped back to the original labeled intervals before review.

| Human judgment | 3-second context | 5-second context |
|---|---:|---:|
| Target only | 1 | 1 |
| Mostly target | 1 | 1 |
| Mixed speakers | 2 | 2 |
| No intelligible speech | 1 | 0 |
| Unclear | 0 | 1 |

Both settings recovered two useful target streams, left two exchanges mixed, and failed on the other-speaker-only exchange. Five seconds supplied no speaker-separation advantage. Its mean target-word F1 was lower than the 3-second setting (0.339 versus 0.477), and it produced fewer nonempty transcripts (3/5 versus 4/5).

## Decision

Do not add contextual WeSep as an automatic resolver. If retained for supplemental review, use 3 seconds of context and preserve the existing abstention rules. The next separation experiment should use a blind two-output separator and assign its outputs afterward using target-speaker evidence.
