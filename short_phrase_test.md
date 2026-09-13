Short-phrase tuning validation

The final resolver was replayed against saved model evidence from the 30-second and two-minute tests. No transcription, embeddings, or visual model passes were rerun.

Two-minute test: “Criminal loitering?” at 28.697 seconds changes from SPEAKER_00 to Target_Speaker at weak strength 0.20. This agrees with the user's previously confirmed identity. It requires a short echo question following a strongly attributed statement, exactly one alternative speaker track, tracked target-face presence, weak independent voice profiles favoring the target, and low local voice strength. This is a conversational hypothesis, not verified lipreading or proof that the visible person speaks.

Both tests: “Yeah, I do.” retains Target_Speaker with strength reduced to 0.20; weak padded voice evidence allows tentative direct question/answer attribution. All other speaker labels remain unchanged in both tests.

18 behavior tests passed, including clear voice protection, missing visual evidence, multiple candidate speakers, and weak padded audio. Model setup and environment-based Hugging Face authentication are preserved. Strengths are not calibrated probabilities. New short-phrase ground truth remains needed before changing further labels.
