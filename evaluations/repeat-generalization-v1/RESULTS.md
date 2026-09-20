# Repeated-scene generalization review

## Dataset

The short-window text matcher was applied locally to two completed videos that
were outside the `uxOLBG1OcI0` calibration run. No transcription or diarization
was rerun.

- `lVfKfbFd0SM`: 6 candidates.
- Original 22-minute video: 21 candidates.

The reviewer labeled all 27 paired clips.

## Results

All 21 original-video candidates were the same recorded events shown again.
Nineteen had complete dialogue alignment and two had partial alignment.

All 6 `lVfKfbFd0SM` candidates were similar wording from different events. One
had text similarity 1.0, demonstrating that exact transcript agreement does not
prove a replay. The review's relationship label is authoritative; its
`same_dialogue_partial` selections describe shared words, not shared events.

Overall same-event precision was 21/27 (77.8%). Similarity threshold did not
separate the classes: true repeats ranged down to 0.617, while false event
matches reached 1.0.

## Sequence-support analysis

The existing long-presentation detector produced coherent neighboring anchors
for the original video and none for `lVfKfbFd0SM`.

- Sequence-supported candidates: 21/21 same event.
- Isolated candidates in this generalization set: 0/6 same event.
- The earlier calibration video contained 3/3 real short repeats without long
  sequence support.

Sequence support is therefore a useful high-confidence tier, while isolated
text matches remain ambiguous. Isolated matches must remain review-only rather
than being discarded, because short real replays exist.

## Decision

Keep the current candidate threshold. Add `sequence_supported` and
`isolated_text_match` evidence tiers. Neither tier changes text or speaker
identity automatically. Use sequence-supported pairs first when testing
clearer-recording corroboration; retain isolated pairs for human review.
