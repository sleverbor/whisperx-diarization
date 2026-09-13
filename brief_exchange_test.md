Confirmed short-phrase corrections

Saved model evidence was replayed with the updated evidence collector and resolver; models and transcription were not rerun.

At 34.341 seconds, “Oh, okay.” changes from SPEAKER_00 to Target_Speaker, strength 0.20. It follows a strongly attributed statement and the weak independent voice comparison favors the alternative speaker.
At 48.052 seconds, “Yeah.” changes from Target_Speaker to SPEAKER_00, strength 0.20. It answers a brief confirmation question whose target baseline agrees with its resolved identity and whose target face is tracked. The weak independent voice comparison favors the alternative speaker. Questions attributed through conversational inference cannot anchor this rule.

Both corrections agree with user-provided ground truth. All other labels remain identical to the preceding tuning checkpoint, including all labels in the 30-second test. The previous “Criminal loitering?” correction is preserved. No clip timestamps, names, legal terminology, or police role assumptions are encoded as identity rules. These conversational assignments remain hypotheses, not verified active-speaker detection. Hugging Face authentication remains environment-based.
