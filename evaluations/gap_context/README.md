# Targeted recovery with surrounding audio

This optional pass decodes only gaps in an existing transcript. It never changes existing segments or their speaker assignments. New candidates remain separate, require review, and have an uncertain speaker. Audio context may include existing speech, but candidate words outside the original gap are excluded.

The new `--context-seconds` option controls surrounding audio. Existing defaults remain unchanged (25-second maximum windows, 12-second overlap, 0.5-second context). The current experiment uses 20-second maximum windows, 10-second overlap and 2-second context on raw audio. No expected words or speaker roles are supplied to the model.

From the project directory, using the existing environment:

```sh
.venv/bin/python recover_transcript_gaps.py evaluations/clip_10m25s/clip_10m25s.mp4 --baseline evaluations/clip_10m25s/local_evidence.json --output-dir gap-context-review --minimum-gap 5 --window-seconds 20 --overlap-seconds 10 --context-seconds 2
```

Use your actual video and baseline paths if they differ. The output directory must be new. The baseline JSON must have a `segments` list with start, end, text and any existing speaker/evidence fields. Times are relative to the video provided; optional `source_offset_seconds` adjusts the review display only.

`transcript_with_candidates.json` preserves existing segments and records candidates separately. `review_transcript.txt` interleaves originals with clearly marked REVIEW lines. `window_decodes.json` contains decoder alternatives, including rejected hypotheses. Overlapping agreement is supporting evidence, not verification.
