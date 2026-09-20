# MossFormer2 cross-audio ASR corroboration

The local screen compared the original mixture with both MossFormer2 streams for all 15 human-reviewed clips. Each source was decoded without a prompt and with surrounding dialogue that excluded the current segment.

Strong unprompted agreement occurred on five clips. Only two were human-labeled useful; three were wrong, garbled, or the wrong speaker. Context prompting made the result worse: two useful and four non-useful clips had strong agreement.

The mixture and separated stream can retain the same interference, and the same ASR model can therefore repeat the same mistake on both. Agreement is not independent corroboration. These outputs remain review-only and cannot add words to the transcript automatically. The result does not justify a large-v2 Kaggle run of this method.
