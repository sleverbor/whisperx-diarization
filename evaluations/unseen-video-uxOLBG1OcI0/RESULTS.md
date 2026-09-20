# Unseen-video evaluation: `uxOLBG1OcI0`

## Run validity

The run used notebook revision `unseen-video-caption-gap-v29`, the planned video ID, and the unchanged auditor reference hashes. It produced 227 baseline segments over six raw diarization tracks. All supplemental stages preserved the baseline.

## Target voice mapping

The global voice evidence clearly selected `SPEAKER_01`: mean affinity 0.339 versus 0.126 for the next-highest track, with mapping strength 1.0. The opening excerpt also mapped the target cleanly at 0.568 versus 0.074.

Only 9 full-video lines received the final `Target_Speaker` label. Those are concentrated in the first encounter. The targeted review found 16 target hypotheses, including later candidates near 194, 323, 420, 426–428, and 464 seconds. This indicates cross-bodycam target fragmentation: the same target voice is present later, but no longer remains on the globally mapped target track and is not strong enough for the conservative final resolver.

This is evidence for review, not proof that every supplemental candidate is correct. The later candidates include duplicated review windows and need human labeling.

## Face evidence

No segment produced positive target-face visibility evidence. The target is known to be on screen in parts of the video, but the face reference did not cross the configured threshold. A late candidate reached only about 0.274 similarity. Face evidence therefore contributed no useful identity support in this run.

This should be evaluated by reviewing visible target frames before changing a threshold. A lower threshold could create false matches among the many non-target faces.

## Repeated bodycam perspectives

The waveform-based repeat detector found zero groups. The transcript nevertheless contains multiple near-exact repeated exchanges, including:

- “I can't force him to leave” around 212.9 and 239.5 seconds.
- The property-line explanation around 226.8–235.2 and 255.8–263.9 seconds.
- “That's just what he does for a living” around 357.0 and 369.4 seconds.

The existing detector is effective for nearly identical recordings, not the same event captured by different bodycam microphones. This run justifies a new text/time-assisted candidate generator, with human review before using one perspective to fill another.

## Overlap and separation

Twelve overlap intervals were selected. Existing extraction left five unresolved and classified seven as likely suppressed residuals. Stereo analysis found four intervals with corroborated words for review.

MossFormer2 accepted six of twelve intervals as target-like review audio and rejected six. It made no automatic insertions. Notable candidates include the late line near 424.7–428.9 seconds, where the selected stream closely corroborates the public-property response, and a low-score interval near 456.8–463.7 seconds that may instead be non-target speech. Human review is required.

DiaPer corroborated only one of twelve baseline overlap intervals and marked 34 of 215 nominal control segments as overlap. Without labels, this is not an accuracy score, but DiaPer is not a useful primary selector for this video.

## Caption gaps and confidence filtering

Caption-gap detection produced four review candidates after nearby-duplicate suppression. Their approximate source locations are 136–138, 180–182, 382–384, and 473–475 seconds.

The confidence export included 145 segments and sent 82 segments, totaling about 120.65 seconds, to review. Most review items were below the speaker-confidence threshold rather than known errors.

## Reference promotion

One clean early target line met the strict post-run promotion criteria: “I appreciate that, but this is a public sidewalk and I'm going to stay right here.” It remains a review candidate and has not modified the permanent reference.

## Decision

Do not tune thresholds from aggregate counts. Build a focused review covering:

1. Later supplemental target candidates to measure cross-bodycam fragmentation.
2. Visible target frames to diagnose face-reference failure.
3. The four caption gaps.
4. The six MossFormer2 candidates.
5. Representative text-matched repeated bodycam scenes.

The first untouched run is preserved as the comparison baseline.
