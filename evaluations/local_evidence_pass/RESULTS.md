# Independent activity, turn hints and clean-profile screening

## Safe evidence collection implemented

collect_activity_evidence.py attaches independent speech coverage as a separate field. It asserts original segments and recovery candidates unchanged, including their text, timing and identity. It merges overlap before calculating coverage, so duplicate windows cannot inflate evidence. The handcuff test output is transcript_with_scoped_activity_evidence.json. Speech activity never identifies a speaker. Spans outside the analyzed audio region have unknown coverage, not zero speech.

## Turn splitting: useful hint, no automatic boundary yet

Sliding 1.2-second ECAPA crops every 0.4 seconds were gated on Silero activity. A crop at 1092.8–1094.0 seconds gives a target similarity of 0.400, versus -0.022 for the opening officer. It localizes target evidence within the mixed reflection/agreement sentence, instead of assigning the entire sentence to the target. This is a hint, not a verified exact word/turn boundary.

Requiring agreement from several short windows eliminates many previously useful whole-utterance hypotheses under key noise. The experimental review_runs output therefore must not replace the prior better whole-utterance result. Existing voice evidence is retained separately; very short windows cannot automatically veto it. The garbled words and missing alignment spans remain unresolved.

## Third speaker: do not promote this profile

The initial three-sample anonymous track fails the minimum 0.40 leave-one-out consistency check: one sample scores 0.385. A second test uses nonoverlapping 2.4-second clean samples, >=80% independent speech, <=10% other-track overlap, then majority-medoid similarity >=0.45 screening. Only two of four samples remain consistent, fewer than the required three and strict majority. No third-speaker reference was promoted.

Sample counts, consistency gates and comparisons are saved. The checks were not lowered to match expected speaker roles. Scene track IDs remain anonymous evidence, not stable people or police labels. Nearby crops from one monologue are not equivalent to independent recordings.

## Status

Original results and core pipeline defaults remain unchanged. Independent activity can now be retained safely. Turn localization is promising but exact splitting and the third-person profile need clearer speech before integration. The current review transcript is deliberately experimental and less complete than earlier whole-utterance matching, not a milestone improvement.
