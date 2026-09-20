# Joint audio evidence probe v1

This experiment addresses transcription and attribution together while keeping
their conclusions separate. It compares original audio, a conservatively
speech-filtered version, and available MossFormer2 outputs. Related transforms
of the original recording form one evidence family, so repeated ASR mistakes do
not become independent confirmation. Every result remains review-required.

## Results

### Segment 197 — mixed boundary

The original and filtered whole-interval crops both favored the target voice,
but their wording differed: `We can test that if you buy it` versus `Man, we
could test that if you'd like`. The earlier sentence-ownership probe found a
1.982-second pause and a diarization change before the target-like suffix. That
boundary overrides the whole-crop voice score. The interval remains mixed; no
whole-sentence ownership propagation is allowed.

### Segment 200 — target wording candidate

The original family and the target-like MossFormer2 stream agree on the exact
interior span `doing it right here because this is obviously public property`.
The prefixes disagree (`So you're`, `You're`, and `Now I'm`), so the experiment
does not accept the beginning of the sentence. The separated stream carrying
the stable interior scored 0.392 against the target reference and 0.043 against
the test comparison voice. This makes the interior span a useful target wording
candidate for human review, not verified transcript text.

### Segment 221 — target identity only

The exact crop remained target-like, but the original and filtered ASR outputs
were `I love that video` and `I'm not thinking at all`. Neither supports the
baseline's `So I'm not paying him`, and prior human review instead heard `I'm
not panhandling`. The result therefore preserves target identity evidence while
explicitly withholding wording.

## Decision

The joint method adds useful evidence when it identifies a stable interior span
and binds that span to a target-like audio output. It also prevents a strong
speaker embedding from validating unstable words or crossing an acoustic turn
boundary. It should remain supplemental and abstaining; no baseline text or
speaker assignment was modified.
