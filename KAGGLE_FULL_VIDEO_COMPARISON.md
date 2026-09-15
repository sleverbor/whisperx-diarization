# Kaggle full-video comparison

1. Import `kaggle_full_video_comparison.ipynb` into Kaggle.
2. Attach the same private dataset containing `video.mp4`, `voice_embeddings.npy`, `face_embeddings.npy`, and `opening_officer_reference.npy`.
3. Enable Internet and a GPU accelerator. A T4 is sufficient; the pipeline does not currently split one run across two GPUs.
4. Add the Kaggle secret `HF_TOKEN` and grant the notebook access to it.
5. Choose **Run All**.

The notebook first creates the unchanged full-video baseline. It then reviews only uncertain, weak, short, skipped, and explicitly selected evaluation regions. Review hypotheses remain separate from the baseline, so the comparison cannot silently overwrite lines that already work.

At the end, download:

- `diarization-results.zip` for the baseline, supplemental review, and comparison summary.
- `stage-checkpoints.zip` for restartable intermediate results.

If Kaggle stops the session, attach the checkpoint archive as a dataset and run the notebook again. Matching transcription and alignment windows will be reused.

The older gap-only recovery pass and the previously validated height-control run are disabled by default because the broader targeted review covers their purpose without repeating that work.
