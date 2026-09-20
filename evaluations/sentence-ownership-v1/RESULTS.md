# Sentence ownership probe v1

This experiment tests whether uncertain words can inherit target ownership from
confirmed words in the same apparent sentence. Text coherence contributes zero
identity weight. The probe divides each baseline sentence into contiguous
diarization runs, scores each run and three conservative boundary variants with
ECAPA, and checks pauses and track changes before producing a review-only
sentence result.

The first calibration sentence, `We can test that if you'd like`, was classified
as **mixed**. The suffix `test that if you'd like` was target-like, while `We
can` was not, and the two parts were separated by a 1.982-second gap and a track
change. This demonstrates that grammatical completion must not bridge an
acoustic turn boundary.

The second sentence, `Not doing it right here because this is obviously public
property`, remained **unresolved**. The suffix was stably target-like but the
prefix was not, the track changed, and the whole-span voice result was unstable.
This is useful partial evidence but cannot justify assigning the whole sentence
to the target.

The third short sentence, decoded as `So I'm not paying him`, was classified as
**target** for identity: all three boundary variants favored the target over the
comparison voice, there was no internal track change, and the largest word gap
was 0.04 seconds. This does not validate the wording; prior human review suggests
the intended phrase is `I'm not panhandling`.

The result supports adding the probe as supplemental sentence-level evidence.
It should remain review-only until tested on a larger hand-labeled set, and its
identity result must remain separate from ASR wording confidence.

Inputs: unseen-video baseline `diarization-results-20`, the reviewed target
reference v14, and the test-only opening-officer comparison reference. The
baseline was not modified.
