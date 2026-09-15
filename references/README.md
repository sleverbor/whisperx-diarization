The new builder enrolls a candidate from three independently recorded YouTube reference windows. Neither evaluation clip supplies reference speech or faces.

The source manifest records the URLs and requested source times. It uses locally downloaded 30-second excerpts, so their local offset is zero. For the outdoor source, the face region ends halfway down the image to exclude the printed face on the shirt. This region is a configurable enrollment choice, not a rule in the diarization pipeline.

Voice windows are four seconds long, do not overlap, and must contain at least 60% detected speech. Windows shorter than two seconds, very quiet windows, and heavily clipped windows are excluded. Speech detection cannot distinguish speakers: enrollment windows must be selected as target-only. The existing ECAPA speaker encoder and buffalo_l face encoder are retained.

Face frames are sampled every 2.5 seconds. The selected region must contain exactly one qualifying face. Both modalities are then checked against the sample with the strongest median agreement. Samples below cosine similarity 0.45 are excluded; the builder refuses to proceed without a consistent majority. This cutoff screens outliers; it is not a probability of correct identity. Raw samples, retained samples, centroids, and provenance are saved for review. A new output directory is required, preventing accidental replacement of a known reference.

The new source-based reference retains 18 voice samples out of 19 and 33 face samples out of 34. The earlier consistency-only candidate is also retained separately; it was made from the existing arrays and is not the source-based reference.

To rebuild from local source clips, run the project virtual environment's Python with:

```text
build_target_reference.py --source-manifest references/source-clips/sources.json --output-dir references/youtube-v2
```

To use the candidate in a normal pipeline run:

```text
chainofrules.py short.mp4 --voice-priors references/youtube-v1/voice_embeddings.npy --face-priors references/youtube-v1/face_embeddings.npy --output candidate_short.json
```

HF_TOKEN remains an environment variable for the diarization pipeline. No token is stored in this builder or the reference.

The controlled comparison freezes transcription, alignment, diarization tracks, and available per-window voice embeddings. Candidate visual evidence is recomputed. The opening checkpoint comes from a separately encoded 0–30-second excerpt of the same source; the second checkpoint matches the exact local video digest. This is a reference comparison, not a new end-to-end transcription run. It does not test recovery of the missing height exchange.
