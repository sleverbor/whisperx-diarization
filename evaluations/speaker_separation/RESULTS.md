# Frozen-text speaker separation test

The corrected 25-second local transcript and alignment were kept fixed. ECAPA voice crops were compared against the improved target reference and a four-sample officer profile from clear opening-clip lines. Reference samples were selected by non-target track, duration and saved strength, not by phrases. The officer role of that opening track comes from earlier user review; the generic matcher sees reference vectors only.

Results: three Target_Speaker hypotheses, three Opening_officer_reference hypotheses, seven uncertain lines. This is an experimental review output, not a measured accuracy score or production resolver change.

The target hypotheses are the two driver-license replies and the fake-information response. The officer hypotheses are the system question, driver-license question and subsequent system statement. The long radio transmission does not match either reference closely and stays uncertain. We cannot assign it a stable separate person identity from one distorted transmission. The brief No and There have insufficient audio for an embedding. The found-guilty response has a target similarity of 0.274 but a margin of only 0.079, just below the provisional 0.08 gate, so it remains uncertain.

Minimum similarity 0.25 and margin 0.08 are provisional experiment gates, not calibrated probabilities. They were set before examining this output and were not adjusted to force expected answers. Good matches are still marked review_required in JSON. There is no semantic question/answer speaker rule in this test.

The final partial sentence is a clip-boundary artifact and remains uncertain; its timing is not verified. Other uncertain lines may be the known officer or another voice. Earlier officer reference quality and recording conditions can limit matches. This demonstrates useful voice-reference separation on corrected crops, not full three-speaker diarization.

All original Kaggle results and the corrected alignment file remain unchanged. No model architecture or default notebook behavior was changed. Next verification: listen to these hypotheses and test another scene before integrating this matcher into the pipeline.
