# Conservative transcript review

Import kaggle_conservative_transcript.ipynb into Kaggle and add the existing full-video evidence results as an input dataset. Run all cells with CPU; no setup installs, virtual environment, GPU or secrets are needed. The notebook automatically selects the largest evidence transcript unless alternatives tie.

Starting thresholds: mean word alignment 0.40, speaker strength 0.60; omit text when more than half the scored words have alignment below 0.10. These signals are not calibrated accuracy probabilities. Existing labels are preserved on retained text; weak speaker attribution receives a speaker uncertain note. Existing gaps are marked as unavailable, without assuming speech. Recovery candidates are not automatically accepted.

The included local preview retained 353 of 445 segments, omitted 92 and flagged weak speaker attribution on 104 retained segments. Original model outputs remain unchanged. Omitted wording, original evidence and reasons are saved separately. No filtering based on perceived interest or topic is applied.
