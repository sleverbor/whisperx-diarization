# Multi-offset blind separation evaluation v1

Three exchanges with plausible target recovery were rerun with the three-second context window shifted one second earlier and later. The original center-window human decision was retained from the prior evaluation.

| Exchange | Early target | Center target | Late target | Result |
|---|---|---|---|---|
| 01-0033 | Stream 1 | Stream 1 | Stream 1 | Speaker recovery repeats; words are unstable |
| 04-0157 | Stream 1 | Stream 1 | Stream 1 | Speaker recovery repeats; center words are best |
| 05-0215 | Stream 1 | Neither | Stream 1 | Shifted windows recover target; center unusable |

Human listening identifies a repeatable target-dominant stream more often than the separated-stream ASR indicates. Exact transcriptions vary substantially across offsets, including obvious hallucinations. Stream numbering happened to remain stable here but must not be assumed stable in general.

ECAPA post-separation selection remains unreliable. It selected the human target for both shifted windows of 01-0033 and 05-0215, but selected the other stream for both shifted windows of 04-0157. Similarities remain too low for a general automatic identity decision.

## Decision

Multi-offset separation is useful as corroborating evidence that a target-dominant stream exists, but its short cropped ASR is not reliable enough to recover words automatically. The next focused test should transcribe the full contextual separated stream with word timestamps, then retain only words aligned to the original interval. This gives Whisper linguistic context without allowing surrounding words into the final interval.
