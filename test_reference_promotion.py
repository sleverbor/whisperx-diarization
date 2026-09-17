import unittest

import numpy as np

from reference_promotion import (evaluate_reference_update,
                                 promoted_source_keys, select_candidates,
                                 source_key)


def segment(index=0, *, target=True, confidence=1.0, duration=2.0,
            similarity=.7, voice_strength=1.0, overlap=False, raw="SPEAKER_04"):
    evidence = [{"source": "local_voice", "confidence": voice_strength,
                 "target_score": .5, "details": {"similarity": similarity}}]
    if overlap:
        evidence.append({"source": "overlapping_speakers", "confidence": 0,
                         "target_score": 0, "details": {}})
    return {"start": float(index * 3), "end": float(index * 3 + duration),
            "text": f"sample {index}",
            "baseline": {"raw_speaker_track": raw, "speaker": raw},
            "final_speaker": "Target_Speaker" if target else raw,
            "final_confidence": confidence, "evidence": evidence}


class ReferencePromotionTests(unittest.TestCase):
    def test_selects_only_clean_strong_original_target_segments(self):
        payload = {"target_candidate": "SPEAKER_04", "segments": [
            segment(0), segment(1, confidence=.79), segment(2, duration=1.0),
            segment(3, similarity=.49), segment(4, overlap=True),
            segment(5, raw="SPEAKER_03"), segment(6, target=False),
        ]}
        self.assertEqual(
            [row["baseline_index"] for row in select_candidates(payload)], [0]
        )

    def test_safe_consistent_update_passes(self):
        parent = np.array([[1, 0], [.99, .1], [.99, -.1]], dtype=np.float32)
        candidates = np.array([[.98, .05]], dtype=np.float32)
        result = evaluate_reference_update(parent, candidates)
        self.assertTrue(result["passed"])

    def test_inconsistent_candidate_fails(self):
        parent = np.array([[1, 0], [.99, .1], [.99, -.1]], dtype=np.float32)
        candidates = np.array([[-1, 0]], dtype=np.float32)
        result = evaluate_reference_update(parent, candidates)
        self.assertFalse(result["passed"])
        self.assertFalse(result["checks"]["all_candidates_match_parent"])

    def test_excludes_already_promoted_source_interval(self):
        metadata = {"promotion_review": {"promoted_candidates": [{
            "source": {"video": "https://example/video", "start": 0, "end": 2}
        }]}}
        excluded = promoted_source_keys(metadata)
        payload = {"target_candidate": "SPEAKER_04", "segments": [
            segment(0), segment(1),
        ]}
        selected = select_candidates(
            payload, source_video="https://example/video",
            excluded_sources=excluded,
        )
        self.assertEqual([row["baseline_index"] for row in selected], [1])
        self.assertIn(source_key("https://example/video", 0, 2), excluded)


if __name__ == "__main__":
    unittest.main()
