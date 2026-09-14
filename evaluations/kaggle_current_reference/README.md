# Fresh Kaggle test

1. Import kaggle_current_reference_test.ipynb into Kaggle.
2. Attach the existing private dataset containing video.mp4.
3. Enable Internet, GPU T4 ×2, and the HF_TOKEN secret. Keep the notebook private.
4. Run all cells from the top. Download diarization-results.zip and stage-checkpoints.zip in the final cell.

The improved youtube-v1 reference and current pipeline are embedded. Old dataset references are ignored. Opening and height/weight clips are extracted from the full video, then tested before the full run. Face GPU fallback stops the run before the full test. No extra installation repair cells are needed.

Full-video gap recovery is enabled by default, using 20-second windows, 10-second overlap, two-second context and gaps of at least five seconds. This can add runtime. Set RUN_GAP_RECOVERY=False to run only the normal pipeline. Candidates remain separate and speaker-uncertain; original segments are asserted unchanged. Existing transcripts retain speaker labels and heuristic strength values. No rejected alignment-score filtering, denoising or full-audio VAD is applied.

This notebook has been checked locally but has not yet been executed on Kaggle. GPU inference and full-video accuracy require the cloud test. T4 ×2 does not imply the pipeline uses both GPUs.
