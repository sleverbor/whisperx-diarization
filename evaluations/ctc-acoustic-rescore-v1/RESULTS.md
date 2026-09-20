# Independent CTC acoustic rescoring v1

This experiment ranks competing Whisper, baseline, and human-evaluation
wordings with the independently trained wav2vec2 CTC acoustic model bundled in
TorchAudio. Candidate origin is hidden during scoring and restored afterward.
Lower loss means the candidate is a better acoustic fit within the candidate
set; it is not a calibrated probability of correctness.

## Target suffix after the mixed boundary

For the target-like suffix around 420.10–421.12, the aggregate score preferred
`test that if you buy it` over `test that if you'd like`, but the two candidates
were nearly tied on original audio. Greedy CTC produced only `AS THE` and
`AYS THEMBY`. The result is too acoustically degraded to resolve the wording.

## Public-property passage

On the target-like MossFormer2 stream, CTC ranked `not doing it right here
because this is obviously public property` first. Its greedy text began `NO
DOING ... BECAUSE THIS IS OBVIOUSLY ... PROPERTY`, providing independent
support for the baseline prefix rather than the target-separated Whisper prefix
`Now I'm`. This is the one useful result in the three-case screen.

The original and filtered mixtures independently ranked the same baseline
candidate first, but they were excluded from the aggregate because their
speaker evidence was mixed or unresolved.

## Short reply

The exact 0.72-second crop and its filtered counterpart ranked `I love that
video` first. The human-evaluation holdout `I'm not panhandling` ranked third.
Greedy CTC produced only `AA` and `A NOT BE`, showing that the model itself could
not recover a stable phonetic sequence. Tightening the crop did not change the
failure.

## Decision

CTC rescoring may help choose among wordings on longer, cleanly separated target
speech. It does not solve the short, noisy phrases that motivated the test.
Before this evidence can affect a transcript, the scorer needs an abstention
gate based on usable acoustic decoding and validation on a larger hand-labeled
set. It is not added to the automatic pipeline from this experiment.
