The new YouTube reference is a reproducible candidate, but it does not solve the second clip's diarization errors. Keep the existing reference as the default for now.

| Test | Original target voice affinity | New target voice affinity | Speaker label changes |
|---|---:|---:|---|
| Opening, 0–30 seconds | 0.255 | 0.257 | One officer question improves from uncertain to SPEAKER_01 |
| Second clip, 10:25–11:30 | 0.439 | 0.455 | None |

In the opening clip, “You want to give me a reason why you're here?” at 20.925 seconds is now assigned to the officer. The final target line, “Criminal loitering,” remains incorrectly assigned to the officer in the same-video check with both references.

The second clip still has 16 lines assigned to SPEAKER_00, two uncertain lines, and no final Target_Speaker labels. The target candidate is SPEAKER_01, but that track has only one usable reference signature in the clip. The resolver cannot confidently correct the many target utterances grouped into SPEAKER_00 without independent track support. Face visibility alone does not override speaker identity.

This comparison holds transcription, alignment, diarization, and available voice embeddings fixed. It tests reference matching and final resolution, not speech recovery. Consequently the missing height exchange is unchanged.

The opening upstream checkpoint used a separately encoded excerpt of the same source. Its initial comparison appeared to regress the last target line. Recomputing original-reference visual evidence on the same local video for both changed lines showed that the last-line difference came from the saved visual evidence, rather than the new reference. The reported opening comparison uses that verification and replays the resolver and context stages. Other opening baseline visual evidence remains from the earlier encoding; this is not a wholly fresh paired end-to-end run.

The candidate was built from the three original YouTube source windows, independently of the two evaluation clips. It contains 18 retained voice samples and 33 retained face samples, with raw samples and provenance preserved. The outdoor source uses an explicit face region to exclude the printed face on the target's shirt. The builder's consistency threshold is a screening heuristic, not an identity probability.

Three builder checks passed: invalid/outlier exclusion, refusal of a split enrollment without a majority, and encoder dimension validation. Actual source enrollment and both local clip comparisons also completed.

Project branch: test-target-reference. The original face_embeddings.npy and voice_embeddings.npy have not been replaced.
