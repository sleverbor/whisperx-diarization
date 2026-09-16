# Kaggle full-video comparison

1. Import `kaggle_full_video_comparison.ipynb` into Kaggle.
2. Create or attach a Kaggle dataset containing the supplied `uAtiEviUzGA.mp4` file with that exact filename.
3. Enable Internet and a GPU accelerator. A T4 is sufficient; the pipeline does not currently split one run across two GPUs.
4. Add the Kaggle secret `HF_TOKEN` and grant the notebook access to it.
5. Choose **Run All**.

The notebook uses the attached current video because YouTube blocks downloads from Kaggle's shared network addresses. It embeds the screened auditor reference and the short target enrollment recording used by the extraction model.

It first creates the unchanged full-video baseline. It then records repeated-presentation corroboration, reviews uncertain and short regions, and runs target-conditioned extraction only on intervals where diarization found simultaneous target and non-target speech. Every supplemental result remains separate from the baseline.

At the end, download:

- `diarization-results.zip` for the baseline, repeat candidates, targeted review, overlap-extraction audio and report, logs, and runtime package versions.
- `stage-checkpoints.zip` for restartable intermediate results.

If Kaggle stops the session, attach the checkpoint archive as a dataset and run the notebook again. Matching transcription and alignment windows will be reused.

The notebook runs the first 30 seconds before the whole video and stops if InsightFace is still using the CPU. The older gap-only recovery pass is omitted because the broader targeted review covers its purpose without repeating that work.
