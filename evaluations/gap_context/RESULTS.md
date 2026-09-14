# Context-window experiment

The second clip's skipped interval was decoded in two raw-audio windows: 14.523–34.523 and 22.193–42.193 seconds relative to the clip (source offset 625 seconds). All 18 existing segments, including their words, speaker labels and evidence, are unchanged. The opening control retains all 24 segments and adds nothing.

Recovered review candidates include “How tall are you?”, “5’10 and a half”, “5’11”, both “How much do you weigh?” questions, “Sorry?” and “160.” The previous 25-second gap crop garbled the weight question; this context-window crop recovers it. This is a useful result for this interval, not an accuracy measurement for arbitrary videos.

Background/radio wording remains questionable, including Caller/Tower numbers and the phrase near “On my ribs.” Overlap support does not confirm correctness. Review candidates can combine multiple turns; no speaker identities are assigned. Zero-duration word timestamps are preserved within phrases rather than dropping words or inventing timing.

Seven helper tests pass. JSON output normalizes decoder numeric/boolean scalar types for reliable saving. Existing 25/12/0.5 defaults are preserved; 20/10/2 is the tested optional configuration.
