"""Detect repeated presentations and propose locally aligned transcript evidence."""

from bisect import bisect_left
from difflib import SequenceMatcher
import re
from statistics import median


STOP_WORDS = {
    "a", "an", "and", "are", "at", "be", "been", "but", "can", "did",
    "do", "does", "for", "from", "had", "has", "have", "he", "her", "here",
    "him", "his", "how", "i", "if", "in", "is", "it", "just", "me", "my",
    "no", "not", "of", "on", "or", "our", "she", "so", "that", "the",
    "their", "them", "there", "they", "this", "to", "was", "we", "were",
    "what", "when", "where", "which", "who", "why", "will", "with", "would",
    "you", "your",
}


def tokens(text):
    return re.findall(r"[a-z']+", text.lower())


def phrase_similarity(left, right):
    """Favor ordered wording while requiring shared meaningful words."""
    left_tokens, right_tokens = tokens(left), tokens(right)
    left_content = set(left_tokens) - STOP_WORDS
    right_content = set(right_tokens) - STOP_WORDS
    shared = left_content & right_content
    if len(shared) < 2:
        return 0.0
    sequence = SequenceMatcher(None, left_tokens, right_tokens).ratio()
    jaccard = len(shared) / max(len(left_content | right_content), 1)
    return 0.65 * sequence + 0.35 * jaccard


def _ordered_unique(candidates):
    """Select a high-scoring, one-to-one monotonic anchor chain."""
    selected, used_left, used_right = [], set(), set()
    for candidate in sorted(candidates, key=lambda item: item["score"], reverse=True):
        if candidate["left"] in used_left or candidate["right"] in used_right:
            continue
        selected.append(candidate)
        used_left.add(candidate["left"])
        used_right.add(candidate["right"])
    selected.sort(key=lambda item: item["left"])
    chain = []
    for candidate in selected:
        position = bisect_left([item["right"] for item in chain], candidate["right"])
        if position == len(chain):
            chain.append(candidate)
        elif candidate["score"] > chain[position]["score"]:
            chain[position] = candidate
    return chain


def find_repeat_groups(segments, minimum_separation=45.0, minimum_score=0.55,
                       offset_tolerance=4.0, minimum_anchors=4,
                       minimum_span=20.0):
    candidates = []
    for left, first in enumerate(segments):
        for right in range(left + 1, len(segments)):
            second = segments[right]
            separation = float(second.start - first.start)
            if separation < minimum_separation:
                continue
            score = phrase_similarity(first.text, second.text)
            if score >= minimum_score:
                candidates.append({
                    "left": left,
                    "right": right,
                    "score": score,
                    "offset": separation,
                })

    groups, remaining = [], candidates[:]
    while remaining:
        # Start with the offset having the largest local support, then refine by median.
        seed = max(
            remaining,
            key=lambda item: sum(
                abs(other["offset"] - item["offset"]) <= offset_tolerance
                for other in remaining
            ),
        )
        nearby = [item for item in remaining
                  if abs(item["offset"] - seed["offset"]) <= offset_tolerance]
        center = median(item["offset"] for item in nearby)
        nearby = [item for item in remaining
                  if abs(item["offset"] - center) <= offset_tolerance]
        chain = _ordered_unique(nearby)
        if len(chain) >= minimum_anchors:
            left_span = segments[chain[-1]["left"]].start - segments[chain[0]["left"]].start
            right_span = segments[chain[-1]["right"]].start - segments[chain[0]["right"]].start
            if left_span >= minimum_span and right_span >= minimum_span:
                group_id = f"repeat_{len(groups) + 1:02d}"
                groups.append({
                    "id": group_id,
                    "offset_seconds": median(item["offset"] for item in chain),
                    "left_start": segments[chain[0]["left"]].start,
                    "left_end": segments[chain[-1]["left"]].end,
                    "right_start": segments[chain[0]["right"]].start,
                    "right_end": segments[chain[-1]["right"]].end,
                    "anchors": chain,
                })
        consumed = set((item["left"], item["right"]) for item in nearby)
        remaining = [item for item in remaining
                     if (item["left"], item["right"]) not in consumed]
    return groups


def _interpolate(value, source, destination):
    if value <= source[0]:
        return destination[0] + value - source[0]
    if value >= source[-1]:
        return destination[-1] + value - source[-1]
    for index in range(1, len(source)):
        if value <= source[index]:
            fraction = (value - source[index - 1]) / max(
                source[index] - source[index - 1], 1e-9
            )
            return destination[index - 1] + fraction * (
                destination[index] - destination[index - 1]
            )
    return destination[-1]


def build_repeat_proposals(segments, groups, maximum_timing_error=3.0):
    """Return donor candidates without changing text or speaker identity."""
    proposals = {index: [] for index in range(len(segments))}
    for group in groups:
        anchors = group["anchors"]
        left_times = [segments[item["left"]].start for item in anchors]
        right_times = [segments[item["right"]].start for item in anchors]
        directions = (
            ("left", "right", left_times, right_times),
            ("right", "left", right_times, left_times),
        )
        for recipient_side, donor_side, source_times, donor_times in directions:
            recipient_indices = [item[recipient_side] for item in anchors]
            lower, upper = min(recipient_indices), max(recipient_indices)
            if lower > 0 and segments[lower].start - segments[lower - 1].end <= 5.0:
                lower -= 1
            if upper + 1 < len(segments) and segments[upper + 1].start - segments[upper].end <= 5.0:
                upper += 1
            donor_pool = sorted({item[donor_side] for item in anchors})
            # Include segments between donor anchors so corrupted or skipped anchor text
            # can still be proposed through the local time map.
            donor_lower, donor_upper = min(donor_pool), max(donor_pool)
            if donor_lower > 0 and segments[donor_lower].start - segments[donor_lower - 1].end <= 5.0:
                donor_lower -= 1
            if (donor_upper + 1 < len(segments)
                    and segments[donor_upper + 1].start - segments[donor_upper].end <= 5.0):
                donor_upper += 1
            donor_pool = list(range(donor_lower, donor_upper + 1))
            for recipient in range(lower, upper + 1):
                midpoint = (segments[recipient].start + segments[recipient].end) / 2
                predicted = _interpolate(midpoint, source_times, donor_times)
                donor = min(
                    donor_pool,
                    key=lambda index: abs(
                        (segments[index].start + segments[index].end) / 2 - predicted
                    ),
                )
                donor_midpoint = (segments[donor].start + segments[donor].end) / 2
                timing_error = abs(donor_midpoint - predicted)
                if timing_error > maximum_timing_error:
                    continue
                anchor_scores = [item["score"] for item in anchors]
                alignment = max(0.0, min(1.0,
                    median(anchor_scores) * (1.0 - timing_error / (maximum_timing_error * 2))))
                proposals[recipient].append({
                    "group_id": group["id"],
                    "donor_start": segments[donor].start,
                    "donor_end": segments[donor].end,
                    "donor_text": segments[donor].text,
                    "donor_final_speaker": segments[donor].final_speaker,
                    "donor_final_confidence": segments[donor].final_confidence,
                    "alignment_confidence": alignment,
                    "timing_error_seconds": timing_error,
                    "note": "Corroboration candidate from an aligned repeated presentation; original text is preserved.",
                })
    return {index: rows for index, rows in proposals.items() if rows}


def repeat_target_corroboration(segment, proposals, minimum_donor_confidence=0.75,
                                minimum_alignment=0.72,
                                minimum_text_similarity=0.85,
                                minimum_local_similarity=0.05):
    """Return one strict target corroboration candidate, without changing text.

    This intentionally handles only nearly identical repeated speech. Partial
    wording, overlap, weak donors, and local acoustic contradictions remain
    review-only.
    """
    if segment.final_speaker in ("Target_Speaker", "Overlapping_Speakers"):
        return None
    voice = next(
        (item for item in segment.evidence if item.source == "local_voice"), None
    )
    local_similarity = (
        voice.details.get("similarity") if voice is not None else None
    )
    if local_similarity is None or local_similarity < minimum_local_similarity:
        return None
    candidates = []
    for proposal in proposals:
        similarity = phrase_similarity(segment.text, proposal["donor_text"])
        if (proposal["donor_final_speaker"] == "Target_Speaker"
                and proposal["donor_final_confidence"] >= minimum_donor_confidence
                and proposal["alignment_confidence"] >= minimum_alignment
                and similarity >= minimum_text_similarity):
            candidates.append({
                **proposal,
                "recipient_text_similarity": similarity,
                "recipient_local_target_similarity": local_similarity,
                "note": "Strict target corroboration from a nearly identical aligned repeated presentation; text is unchanged.",
            })
    if not candidates:
        return None
    return max(candidates, key=lambda item: (
        item["recipient_text_similarity"],
        item["alignment_confidence"],
        item["donor_final_confidence"],
    ))


def resolve_repeat_target_corroboration(segment, details):
    """Second-pass resolver for a candidate produced by the strict gate."""
    if details is None:
        return False
    segment.final_speaker = "Target_Speaker"
    segment.final_confidence = float(min(
        0.55,
        details["donor_final_confidence"],
        details["alignment_confidence"],
        details["recipient_text_similarity"],
    ))
    segment.reasons.append(
        "nearly identical repeated presentation corroborates target identity"
    )
    return True
