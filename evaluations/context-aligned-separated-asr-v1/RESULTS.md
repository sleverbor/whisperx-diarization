# Context-aligned separated-stream ASR evaluation v1

Three exchanges were separated at early, center, and late offsets. Whisper large-v2 decoded each complete contextual separated stream with word timestamps. Only words whose timestamp midpoints fell inside the original labeled interval were retained.

## Human review

| Judgment | Count |
|---|---:|
| Correct | 2 |
| Incorrect | 4 |
| Unclear | 3 |

The method consistently failed on exchange 01-0033. It recognized the target phrase in exchange 04-0157, but two of three timestamp-aligned crops pointed to an occurrence one or two seconds too early. Exchange 05-0215 contained repeated instances or fragments of the target phrase, making occurrence selection ambiguous; only the center result was judged correct.

## Decision

Do not use separated-stream Whisper timestamps to insert or replace transcript words automatically. Full-context decoding may generate useful review candidates, but timestamps on artifact-heavy separated audio are not sufficiently reliable to map those words back to the source interval. Preserve the baseline transcript and treat these results as review-only evidence.

The separation experiments establish two distinct limits:

1. Human listeners can sometimes identify a repeatable target-dominant separated stream even when ECAPA cannot.
2. Whisper can sometimes recover the target phrase from that stream, but its words and timing are not stable enough for automatic transcript recovery.
