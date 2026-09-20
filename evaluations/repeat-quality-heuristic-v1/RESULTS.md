# Repeated-perspective quality heuristic

## Question

Can mean Whisper alignment-word score choose the clearer rendition of a
confirmed repeated scene?

The planned second wording review was stopped because it asked the reviewer to
watch the same 19 pairs again. The existing generalization labels already
contained a blinded `Clearer recording` judgment, so they were sufficient to
evaluate the heuristic without another review.

## Result

Among the 19 non-identical transcript pairs:

- Human judgment: 17 `same`, 2 `second`.
- Heuristic prediction: 9 `same`, 5 `first`, 5 `second`.
- Exact agreement: 8/19 (42.1%).
- On the two pairs where the reviewer preferred the second recording, the
  heuristic selected `same` once and `first` once.

The heuristic overstates small score differences and does not identify the
clearer recording. Alignment-word confidence is not a reliable cross-recording
quality measure.

## Decision

Reject automatic donor selection by mean alignment score. Do not ask for a
second full review of these pairs. Repeated-scene evidence remains useful for
surfacing alternative recordings, but choosing or merging wording requires a
specific unresolved transcript question rather than a blanket replay review.
