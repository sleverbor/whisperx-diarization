This pass recovers review candidates inside gaps in an existing transcript, using the same large-v2 speech model. It does not rerun or replace the working transcript.

By default, gaps must be at least two seconds long. Each gap is decoded in eight-second windows with four seconds of overlap and half a second of surrounding context. Voice activity filtering is disabled only for these small windows. There is no topic prompt and no video-specific phrase or identity rule.

Existing segments, word timings, speaker assignments, evidence, and reasons are copied unchanged. Candidate words must fall entirely inside the gap and pass decoder quality checks. Repeated words from distinct overlapping windows are marked, but repeated decoding is not proof that a word is correct. All new speech remains marked for review with an unknown speaker. Single-window candidates are retained separately in the same review view rather than silently discarded.

Run from the project directory using its virtual environment:

```text
.venv/bin/python recover_transcript_gaps.py YOUR_VIDEO.mp4 --baseline YOUR_EVIDENCE.json --output-dir gap-review
```

The baseline's times must be relative to YOUR_VIDEO.mp4. Its optional source_offset_seconds is used only when displaying source-video timestamps. A fresh output directory is required, preventing accidental overwrites. CUDA is used when available, otherwise CPU int8. No Hugging Face token is required for this standalone ASR pass.

Outputs:

- transcript_with_candidates.json: the unchanged baseline plus a separate gap_recovery_candidates list.
- review_transcript.txt: original lines and explicitly marked review additions, ordered by time.
- window_decodes.json: all window decodes, timings and quality measures, including rejected material.

The opening clip check preserves all 24 existing segments and produces no additions because it has no qualifying gaps. The second-clip test uses the prior independent local transcript, not the failed blanket-full-coverage transcript.

Four checks cover gap unions, bounded overlapping windows without tiny tails, distinct-window repetition inside gap boundaries, and preservation of all nested baseline fields.
