# Blind two-output separation evaluation v1

SepFormer WHAMR separated five fixed overlap exchanges using three seconds of context. Both streams were retained and cropped to the original labeled interval. ECAPA target matching was applied only after separation.

## Human review

| Exchange | Stream 1 | Stream 2 | Human target choice | ECAPA choice |
|---|---|---|---|---|
| 01-0033 | Mostly target | Other only | Stream 1 | Stream 2 |
| 02-0064 | Unclear | Other only | Neither | Stream 2 |
| 03-0118 | Other only | Other only | Neither | Stream 1 |
| 04-0157 | Target only | Unclear | Stream 1 | Stream 2 |
| 05-0215 | Unclear | Mixed speakers | Neither | Stream 2 |

Blind separation produced two useful target streams out of five. Contextual WeSep also produced two useful target streams, but the successful exchange set was partly different. Therefore the methods contain complementary evidence but neither is a reliable automatic resolver.

ECAPA selected the wrong stream for both human-confirmed target recoveries. Its absolute similarities were low on every exchange, and its margins did not distinguish valid target outputs from other-speaker or distorted streams.

## Decision

Do not assign separated streams using the current short-crop ECAPA scores. Keep both blind-separated outputs as review evidence. An automatic path must abstain unless stronger identity evidence is available from a longer clean region of the same separated stream or from cross-window stream consistency. The present outputs must not modify the transcript.
