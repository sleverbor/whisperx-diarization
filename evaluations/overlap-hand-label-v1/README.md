# Hand-labeled overlap evaluation v1

This set contains 35 clips selected to emphasize disagreements among the baseline,
Sortformer, and DiaPer. It is a diagnostic set, not an unbiased estimate of full-video
accuracy. Two unclear/noisy clips are excluded from detector metrics.

Of 33 usable clips, 21 contain intelligible simultaneous speech and 12 are rapid turn
boundaries. Sortformer with any detected overlap recalls 17/21 true overlaps, while
DiaPer recalls 14/21. Requiring Sortformer overlap fraction > 0.30 raises precision to
0.80 but reduces recall to 0.57. Using Sortformer > 0.03 with DiaPer > 0.20 as a rescue
recalls 19/21, but produces 10 false triggers among 12 boundary clips.

The recommended next experiment is a two-tier policy: use Sortformer as the primary
activity detector; send high-fraction detections to separation/review; retain weaker
Sortformer detections and strong DiaPer-only detections as uncertainty evidence. Do not
let either model directly rewrite speaker identity. Validate thresholds on another video
before adopting them globally.

Run `python3 evaluate_hand_labeled_overlap.py evaluations/overlap-hand-label-v1/labels.json`
to reproduce the metrics.
