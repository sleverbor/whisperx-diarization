# Conversational review

`agent_review.py` is a thin local player and durable command surface for an
agent-led diarization review. The user can use text chat or audio chat without
changing sessions or workflows. The player reports its current media time so
the agent can record “here”, “start”, and “end” without a second microphone or
transcription system. Each observation records whether it arrived through text
or voice, while the confirmed annotation format remains identical.

## Start a session

```bash
python agent_review.py init --session review-sessions/demo \
  --source /path/to/overlap-review-policy.json --video-id lVfKfbFd0SM
python agent_review.py serve --session review-sessions/demo --open
```

Choose the matching local video or audio in the player. Then converse with the
agent by text or voice. The agent uses `status`, `comment`, `play`, `anchor`, `annotate`, and
`next` to conduct a two-pass review:

1. Play the entire contextual clip and record free comments.
2. Resolve each comment with focused replay and one `here` anchor or a
   `start`/`end` pair.
3. Read back the proposed speaker, words, overlap, cue basis, and confidence.
4. Save only after the user confirms or corrects the proposal.

Export with:

```bash
python agent_review.py export --session review-sessions/demo \
  --output review-sessions/demo/export.json
```

The original comment, approximate position, grounding anchors, and confirmed
annotation remain separate in the export. Conversation history is never the
only copy of a label.
