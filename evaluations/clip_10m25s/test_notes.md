# Local test: 10:25–11:30

Run completed successfully in 8.54 minutes on CPU using code commit 2b5af7d and unchanged attribution functions from the milestone. Input is a 65-second excerpt of video.mp4, starting at source time 625 seconds. The clip begins during an existing utterance. Transcript timestamps refer to the original video; subtitle timings refer to the extracted clip.

The result is not accurate enough: 18 ASR utterances, 16 SPEAKER_00 and 2 Uncertain, no final Target_Speaker. Every assigned utterance has raw track SPEAKER_00. Diarization detected a second track SPEAKER_01, but only at approximately 640.72–640.84 and 680.41–681.76 seconds in the source video. Only its latter interval is long enough for the global voice signature. Its target affinity is 0.439 versus SPEAKER_00 at 0.166. Independent-profile correction cannot validate identity against two sufficiently sampled tracks here.

The local ASR transcript also omits lines appearing in the saved full-video result around source times 642–649 seconds. Audio review is needed to establish coverage and actual speaker identities. This result is saved without new tuning.

local_labeled.mp4 shows the unmodified local labels for review. full_video_comparison.txt and full_video_window_evidence.json retain the cloud result and global IDs for the same review window. Full-video clusters remain available for future evidence replay; this fresh local run performs independent clip diarization, so its track IDs are not interchangeable with global track IDs.
