The targeted pass recovered the missing height question and answer without changing any existing transcript segment. It still mishears the weight question, so additions remain explicitly marked for review rather than treated as confirmed speech.

Three local ASR comparisons used the same prior 18-segment baseline:

| Gap window strategy | Height question | Height answer | Weight exchange |
|---|---|---|---|
| 8 seconds, 4-second overlap | Missed | Recovered | Question mistranscribed; “Sorry?” and “160” recovered |
| 12 seconds, 6-second overlap | Mistranscribed | Recovered | Question mistranscribed; “Sorry?” and “160” recovered |
| Only the long uncovered gap, 24.67 seconds | “How tall are you?” recovered | “5'10 and a half, 5'11…” recovered | Question garbled; “Sorry?” and “160” recovered |

The successful question/answer candidate spans source-video time 10:51.68–10:57.24 (clip-relative 26.68–32.24 seconds). Some surrounding wording remains inconsistent between decodes; the review file preserves the chosen decode rather than combining different hypotheses. The audio excerpt is supplied for verification.

The original 18 segments, including their text, timestamps, speaker assignments, word data, evidence, and reasons, are identical in all three outputs. No candidate is assigned to the target or officer. New text is stored in a separate gap_recovery_candidates list; the review text file shows these as REVIEW lines among the unchanged original lines.

The opening 30-second clip had no gaps of at least two seconds. Its 24 original segments remain unchanged, with no new candidates and no model decoding needed.

The new default uses up to 25 seconds inside a missing interval. It does not decode blanket chunks across the working transcript. Longer gaps use overlapping windows; eight- or twelve-second windows remain available as options. All candidates require review because decoder confidence and repetition cannot reliably reject hallucinations in this noisy audio.

Five checks passed for gap coverage, avoiding tiny final windows, distinct-window word support within gap boundaries, complete baseline preservation, and keeping a candidate phrase within one decoder hypothesis. The live local comparisons also completed. Candidate selection was replayed from saved window decodes after refining phrase preservation; no ASR text was manually corrected or prompted with expected content.

The height exchange audio starts at source 10:49.00 (clip 24 seconds), lasting 12 seconds. The text file uses full source-video times.
