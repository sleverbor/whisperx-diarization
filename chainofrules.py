"""Evidence-based refactor of the recovered runnable chainofrules.py.
Run from the existing project directory with HF_TOKEN set in the environment.
Accepts any video and matching target embedding files.
All rows in each embedding file are reference samples of the same target.
Defaults retain short.mp4, large-v2, ECAPA and buffalo_l.
Confidence values are heuristic evidence strengths, not calibrated probabilities.
"""
import os
import argparse
import json
import re
from dataclasses import dataclass, field, asdict
import cv2
import numpy as np
import torch
import torchaudio
import whisperx
from whisperx.diarize import DiarizationPipeline
from speechbrain.inference.speaker import SpeakerRecognition
from insightface.app import FaceAnalysis
from collections import defaultdict


@dataclass(frozen=True)
class Baseline:
    raw_speaker_track: str
    speaker: str

@dataclass(frozen=True)
class Evidence:
    source: str
    target_score: float
    confidence: float
    details: dict = field(default_factory=dict)

@dataclass
class TimelineSegment:
    start: float
    end: float
    text: str
    baseline: Baseline
    words: list = field(default_factory=list)
    evidence: list = field(default_factory=list)
    final_speaker: str = "Uncertain"
    final_confidence: float = 0.0
    reasons: list = field(default_factory=list)


def normalize_vector(value):
    value = np.asarray(value, dtype=np.float32).reshape(-1)
    norm = np.linalg.norm(value)
    if not np.all(np.isfinite(value)) or norm <= 0:
        raise ValueError("Embedding must be finite and nonzero")
    return value / norm


def short_voice_crop(segment, previous=None, following=None, media_duration=None):
    """Recover small timing gaps without including neighboring utterances."""
    start, end = segment.start, segment.end
    if end - start < 0.4:
        lower = previous.end if previous is not None else 0.0
        upper = following.start if following is not None else media_duration
        start = max(lower, start - 0.15)
        end = min(end + 0.15, upper) if upper is not None else end
        # Overlapping transcript boundaries are not safe padding opportunities.
        if start > segment.start or end < segment.end:
            return segment.start, segment.end
    return start, end


def add_question_response_evidence(segment, previous, tracks, target_track, mapping_confidence):
    """A conversational hypothesis, never a police-specific identity rule."""
    if previous is None or segment.end - segment.start > 1.0:
        return
    gap = segment.start - previous.end
    if not 0.0 <= gap <= 0.6 or mapping_confidence < 0.75:
        return
    if previous.final_speaker in ("Uncertain", "NonTarget_Unknown", "Unknown_Speaker"):
        return
    if previous.final_confidence < 0.65:
        return
    question = previous.text.strip().lower()
    # Narrow to addressed yes/no questions; punctuation alone is insufficient.
    direct_question = re.match(
        r"^(?:(?:ok|okay|all right)[.,]?\s+)?"
        r"(?:do you|did you|have you|are you|were you|can you|could you|"
        r"would you|will you|don't you|didn't you|haven't you|aren't you)\b", question)
    if not question.endswith("?") or direct_question is None:
        return
    answer = re.sub(r"[^a-z' ]", " ", segment.text.lower()).split()
    if not answer or len(answer) > 4 or answer[0] not in ("yes", "no", "yeah", "yep", "nope", "nah"):
        return
    question_track = target_track if previous.final_speaker == "Target_Speaker" else previous.final_speaker
    if question_track not in tracks:
        return
    candidates = sorted(track for track in tracks if track != question_track)
    candidate = candidates[0] if len(candidates) == 1 else None
    segment.evidence.append(Evidence("question_response", 0.0, 0.20,
        {"question_start": previous.start, "question_track": question_track,
         "candidate_tracks": candidates, "candidate_track": candidate,
         "gap": gap, "assumption": "Immediate brief answer may be a different speaker; not voice-verified."}))




def add_brief_exchange_evidence(segment, previous, tracks, target_track, mapping_confidence):
    """Tentative acknowledgement or confirmation, with independent voice agreement."""
    if previous is None or mapping_confidence < .75 or segment.end - segment.start >= .4:
        return
    if not 0 <= segment.start - previous.end <= .6:
        return
    words = re.findall(r"[a-z']+", segment.text.lower())
    preceding = re.findall(r"[a-z']+", previous.text.lower())
    acknowledgement = words in (["okay"], ["ok"], ["oh", "okay"], ["oh", "ok"])
    confirmation = (preceding in (["really"], ["seriously"]) and previous.text.strip().endswith("?")
                    and words in (["yes"], ["yeah"], ["yep"], ["no"], ["nope"]))
    if not acknowledgement and not confirmation:
        return
    previous_track = target_track if previous.final_speaker == "Target_Speaker" else previous.final_speaker
    candidates = sorted(set(tracks) - {previous_track})
    if previous_track not in tracks or len(candidates) != 1:
        return
    if acknowledgement and (previous.final_confidence < .65 or previous.text.strip().endswith("?")):
        return
    if confirmation:
        # A weak question is usable only when its baseline agrees, a target face is
        # tracked, and it was not itself attributed through conversational inference.
        face = next((e for e in previous.evidence if e.source == "target_face_visible"), None)
        if (previous.final_confidence < .25 or previous.baseline.raw_speaker_track != previous_track
            or previous_track != target_track or face is None
            or not face.details.get("target_visible_hint", False)
            or any("inference" in reason for reason in previous.reasons)):
            return
    voice = next((e for e in segment.evidence if e.source == "local_voice"), None)
    profiles = voice.details.get("track_similarities", {}) if voice is not None else {}
    if (voice is None or voice.confidence > .30 or len(profiles) < 2
        or max(profiles.values()) >= .30 or voice.details.get("best_track") != candidates[0]):
        return
    segment.evidence.append(Evidence("question_response", 0, .20,
        {"candidate_track": candidates[0], "previous_track": previous_track,
         "assumption": "Brief acknowledgement or confirmation may change speaker; weak voice agrees, not verified."}))


def add_echo_question_evidence(segment, previous, tracks, target_track, mapping_confidence):
    """A brief quoted question can suggest another speaker, never establish one."""
    if previous is None or not segment.text.strip().endswith("?"):
        return
    tokens = lambda text: re.findall(r"[a-z']+", text.lower())
    phrase, statement = tokens(segment.text), tokens(previous.text)
    if not 2 <= len(phrase) <= 5 or statement[-len(phrase):] != phrase:
        return
    if not 0 <= segment.start - previous.end <= 0.8 or segment.end - segment.start > 1.2:
        return
    if previous.final_confidence < 0.65 or mapping_confidence < 0.75:
        return
    previous_track = target_track if previous.final_speaker == "Target_Speaker" else previous.final_speaker
    candidates = sorted(set(tracks) - {previous_track})
    if previous_track not in tracks or len(candidates) != 1:
        return
    segment.evidence.append(Evidence("echo_question", 0, 0.20,
        {"candidate_track": candidates[0], "previous_track": previous_track,
         "assumption": "Brief repeated question may come from the listener; not voice-verified."}))


def bbox_iou(left, right):
    x1, y1 = max(left[0], right[0]), max(left[1], right[1])
    x2, y2 = min(left[2], right[2]), min(left[3], right[3])
    intersection = max(0.0, x2-x1) * max(0.0, y2-y1)
    left_area = max(0.0, left[2]-left[0]) * max(0.0, left[3]-left[1])
    right_area = max(0.0, right[2]-right[0]) * max(0.0, right[3]-right[1])
    return intersection / max(left_area + right_area - intersection, 1e-9)


def collect_visual_evidence(segment, cap, fps, face_analyzer, target_face_centroid):
    """Track a recently recognized face through head turns; mouth motion is only a hint."""
    if not np.isfinite(fps) or fps <= 0:
        segment.evidence.append(Evidence("visual_context", 0, 0, {"reason": "invalid_fps"}))
        return
    frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    # Lead-in frames establish identity; only frames inside speech measure mouth motion.
    times = np.arange(max(0.0, segment.start - 0.5), segment.end, 0.125)
    indices = np.unique(np.rint(times * fps).astype(int))
    best, anchor_best, direct_matches, frames_read = None, None, 0, 0
    anchor = None
    observations, apertures, target_frames = [], [], []
    for index in indices:
        if frame_count > 0 and index >= frame_count:
            continue
        cap.set(cv2.CAP_PROP_POS_FRAMES, int(index))
        ok, frame = cap.read()
        if not ok:
            continue
        frames_read += 1
        time = index / fps
        candidates = []
        for face in face_analyzer.get(frame):
            embedding = getattr(face, "embedding", None)
            if embedding is None:
                continue
            embedding = normalize_vector(embedding)
            similarity = float(np.dot(target_face_centroid, embedding))
            if segment.start <= time <= segment.end:
                best = similarity if best is None else max(best, similarity)
            candidates.append((similarity, face, embedding))
        recognized = [item for item in candidates if item[0] >= 0.40]
        selected, identity_source = None, None
        if recognized:
            selected = max(recognized, key=lambda item: item[0])
            direct_matches += 1
            anchor_best = selected[0] if anchor_best is None else max(anchor_best, selected[0])
            identity_source = "reference_match"
        elif anchor is not None and time - anchor[2] <= 0.30:
            linked = [item for item in candidates
                      if bbox_iou(item[1].bbox, anchor[0]) >= 0.20
                      and float(np.dot(item[2], anchor[1])) >= 0.45]
            if linked:
                selected = max(linked, key=lambda item: float(np.dot(item[2], anchor[1])))
                identity_source = "face_continuity"
        for similarity, face, embedding in candidates:
            observations.append({"frame": int(index), "similarity": similarity,
                                 "bbox": face.bbox.tolist()})
        if selected is None:
            continue
        similarity, face, embedding = selected
        anchor = (face.bbox.copy(), embedding, time)
        if not segment.start <= time <= segment.end:
            continue
        target_frames.append({"frame": int(index), "identity_source": identity_source,
                              "similarity": similarity})
        landmarks = getattr(face, "landmark_3d_68", None)
        if landmarks is not None and np.all(np.isfinite(landmarks)):
            # Standard 68-point inner mouth: aperture / width in 3D landmark coordinates.
            width = float(np.linalg.norm(landmarks[60] - landmarks[64]))
            if width > 1e-6:
                apertures.append(float(np.linalg.norm(landmarks[62] - landmarks[66]) / width))
    spread = float(np.percentile(apertures, 90) - np.percentile(apertures, 10)) if len(apertures) >= 5 else 0.0
    motion_hint = direct_matches >= 2 and len(apertures) >= 5 and spread >= 0.03
    segment.evidence.append(Evidence("target_face_visible", 0, 0,
        {"best_similarity": best, "identity_anchor_similarity": anchor_best,
         "target_visible_hint": bool(target_frames), "tracked_target_frames": target_frames,
         "active_speaker_verified": False}))
    segment.evidence.append(Evidence("visual_context", 0, 0,
        {"frames_read": frames_read, "observations": observations,
         "note": "Face identity and visibility do not identify police or prove speech."}))
    segment.evidence.append(Evidence("target_mouth_motion", 1.0 if motion_hint else 0.0,
        0.20 if motion_hint else 0.0,
        {"direct_identity_matches": direct_matches, "mouth_samples": len(apertures),
         "aperture_spread": spread, "active_speaker_verified": False,
         "note": "Weak landmark motion hint; no lipreading or audio-visual synchronization model."}))


def resolve_segment(segment, target_track, mapping_confidence):
    """Only the resolver assigns final identity; visibility alone cannot flip it."""
    raw = segment.baseline.raw_speaker_track
    known = raw != "Unknown_Speaker"
    prior_weight = 0.55 * mapping_confidence if known else 0.0
    prior = 1.0 if raw == target_track else -1.0
    score, weight = prior * prior_weight, prior_weight
    reasons = [f"baseline={raw}; mapping strength={mapping_confidence:.3f}"]
    voice = None
    response = None
    mouth_motion = None
    echo = None
    visible = None
    for item in segment.evidence:
        # Presence/context describes the scene, not the active speaker.
        if item.source in ("target_face_visible", "visual_context"):
            if item.source == "target_face_visible":
                visible = item
            continue
        if item.source == "echo_question":
            echo = item
            continue
        if item.source == "target_mouth_motion":
            mouth_motion = item
            continue
        if item.source == "question_response":
            response = item
            continue
        contribution = item.target_score * item.confidence
        score += contribution
        weight += item.confidence
        if item.source == "local_voice":
            voice = item
        if item.confidence:
            reasons.append(f"{item.source}: {contribution:+.3f}")
    normalized = score / max(weight, 1e-9)
    # Role phrases cannot establish or contradict identity without acoustic support.
    acoustic_support = prior_weight > 0.08 or (voice is not None and voice.confidence > 0.2)
    strong_conflict = (voice is not None and voice.confidence >= 0.5
                      and voice.target_score * prior < -0.4 and known)
    # A strong reference match plus an independent track match can correct diarization.
    details = voice.details if voice is not None else {}
    matched_track = details.get("best_track")
    verified_correction = (strong_conflict and voice.confidence >= 0.5
                          and abs(voice.target_score) >= 0.4
                          and details.get("track_margin", 0.0) >= 0.10
                          and matched_track is not None
                          and details.get("track_similarities", {}).get(matched_track, -1.0) >= 0.30
                          and ((voice.target_score > 0 and matched_track == target_track)
                               or (voice.target_score < 0 and matched_track != target_track)))
    insufficient_short_audio = (segment.end - segment.start < 0.4
                                and (voice is None or voice.confidence == 0))
    response_track = response.details.get("candidate_track") if response is not None else None
    profiles = details.get("track_similarities", {})
    weak_padded_voice = (segment.end - segment.start < 0.4 and voice is not None
                         and voice.confidence <= 0.30 and len(profiles) >= 2
                         and max(profiles.values()) < 0.30)
    response_inference = ((insufficient_short_audio or weak_padded_voice) and response_track is not None
                          and response.confidence > 0)
    visual_inference = (mouth_motion is not None and mouth_motion.confidence > 0
                        and voice is not None and voice.confidence > 0
                        and mapping_confidence >= 0.75
                        and len(details.get("track_similarities", {})) >= 2
                        and details.get("track_margin", 1.0) < 0.05
                        and max(details["track_similarities"].values()) < 0.35)
    echo_inference = (echo is not None and echo.details["candidate_track"] == target_track
                      and visible is not None and visible.details.get("target_visible_hint", False)
                      and len(profiles) >= 2 and max(profiles.values()) < 0.30
                      and matched_track == target_track and voice.confidence < 0.5)
    if echo_inference:
        final = "Target_Speaker"
        reasons.append("weak repeated-question inference with face continuity and weak supporting voice profile; not voice-verified")
    elif visual_inference:
        final = "Target_Speaker"
        reasons.append("weak visible-target mouth-motion inference; independent voice profiles are ambiguous")
    elif response_inference:
        final = "Target_Speaker" if response_track == target_track else response_track
        reasons.append(f"weak question/answer inference to {response_track}; not voice-verified")
    elif verified_correction:
        final = "Target_Speaker" if matched_track == target_track else matched_track
        reasons.append(f"independent voice profile supports correction to {matched_track}")
    elif insufficient_short_audio or not acoustic_support or abs(normalized) <= 0.18 or strong_conflict:
        final = "Uncertain"
        reasons.append("weak, balanced, or conflicting acoustic evidence")
    elif normalized > 0:
        final = "Target_Speaker"
    else:
        final = raw if known and raw != target_track else "NonTarget_Unknown"
    segment.final_speaker = final
    # Avoid baseline-only certainty and account for weak global separation.
    segment.final_confidence = float(min(abs(normalized), weight / 1.55))
    if verified_correction:
        segment.final_confidence = float(min(voice.confidence, abs(voice.target_score),
                                             details["track_margin"] / 0.20))
    if echo_inference:
        segment.final_confidence = 0.20
    elif visual_inference:
        segment.final_confidence = 0.25
    elif response_inference:
        segment.final_confidence = min(0.25, response.confidence)
    elif insufficient_short_audio:
        segment.final_confidence = 0.0
        reasons.append("short utterance has no usable local voice evidence")
    if segment.end - segment.start < 0.4:
        segment.final_confidence = min(segment.final_confidence, 0.30)
    segment.reasons = reasons


def main():
    parser = argparse.ArgumentParser(description="Resolve a supplied target voice in any video.")
    parser.add_argument("video", nargs="?", default="short.mp4")
    parser.add_argument("--voice-priors", default="voice_embeddings.npy")
    parser.add_argument("--face-priors", default="face_embeddings.npy")
    parser.add_argument("--output", default="diarization_evidence.json")
    args = parser.parse_args()
    HF_TOKEN = os.environ.get("HF_TOKEN") or os.environ.get("HUGGINGFACE_TOKEN")
    if not HF_TOKEN:
        raise RuntimeError("Set HF_TOKEN (or HUGGINGFACE_TOKEN) before running.")
    device = "cuda" if torch.cuda.is_available() else "cpu"
    compute_type = "float16" if torch.cuda.is_available() else "int8"

    # =====================================================================
    # 1. CORE PIPELINE INITIALIZATION
    # =====================================================================
    print("⏳ Initializing Core Tracking Engines...")
    whisper_model = whisperx.load_model("large-v2", device, compute_type=compute_type)
    diarize_model = DiarizationPipeline(token=HF_TOKEN, device=device)
    embedding_model = SpeakerRecognition.from_hparams(
        source="speechbrain/spkrec-ecapa-voxceleb", savedir="pretrained_models/spkrec-ecapa-voxceleb"
    )
    face_analyzer = FaceAnalysis(name="buffalo_l", providers=["CPUExecutionProvider"])
    face_analyzer.prepare(ctx_id=0, det_size=(640, 640))

    # Load Priors Matrix
    voice_priors = np.load(args.voice_priors, allow_pickle=False)
    face_priors = np.load(args.face_priors, allow_pickle=False)
    if voice_priors.ndim == 1: voice_priors = np.expand_dims(voice_priors, axis=0)
    if face_priors.ndim == 1: face_priors = np.expand_dims(face_priors, axis=0)

    def reference_centroid(samples, label):
        if samples.ndim != 2:
            raise ValueError(f"{label}: expected a vector or matrix of target samples")
        norms = np.linalg.norm(samples, axis=1)
        valid = np.all(np.isfinite(samples), axis=1) & (norms > 0)
        if not np.any(valid):
            raise ValueError(f"{label}: no valid target samples")
        normalized = samples[valid] / norms[valid, None]
        print(f"{label}: using {len(normalized)} of {len(samples)} target reference samples")
        return normalize_vector(np.mean(normalized, axis=0))

    target_voice_vector = reference_centroid(voice_priors, "Voice references")
    target_face_centroid = reference_centroid(face_priors, "Face references")

    # =====================================================================
    # 2. AUDIO & VIDEO DATA PREP
    # =====================================================================
    print("⏳ Preparing media data tracks...")
    video_path = args.video
    audio_loaded = whisperx.load_audio(video_path)
    cap = cv2.VideoCapture(video_path)
    fps = cap.get(cv2.CAP_PROP_FPS)

    waveform, sample_rate = torchaudio.load(video_path)
    if sample_rate != 16000:
        resampler = torchaudio.transforms.Resample(orig_freq=sample_rate, new_freq=16000)
        waveform = resampler(waveform)
    waveform = torch.mean(waveform, dim=0, keepdim=True)
    total_samples = waveform.shape[1]

    # =====================================================================
    # 3. GENERAL TRANSCRIPTION & LAYERING
    # =====================================================================
    print("⏳ Processing WhisperX Text Script...")
    asr_result = whisper_model.transcribe(audio_loaded, batch_size=16)
    alignment_model, metadata = whisperx.load_align_model(language_code=asr_result["language"], device=device)
    aligned_result = whisperx.align(asr_result["segments"], alignment_model, metadata, audio_loaded, device, return_char_alignments=False)

    print("⏳ Generating Unsupervised Voice Tracks...")
    diarize_segments = diarize_model(audio_loaded)


    target_voice_vector = normalize_vector(target_voice_vector)
    target_face_centroid = normalize_vector(target_face_centroid)
    embedding_cache = {}

    def audio_embedding(start, end):
        key = (float(start), float(end))
        if key in embedding_cache:
            return embedding_cache[key]
        left = max(0, int(start * 16000))
        right = min(total_samples, int(end * 16000))
        result = None
        if right - left >= 6400:
            try:
                with torch.no_grad():
                    result = normalize_vector(embedding_model.encode_batch(
                        waveform[:, left:right]).flatten().cpu().numpy())
                if result.shape != target_voice_vector.shape:
                    raise ValueError("Voice prior dimensions do not match ECAPA output")
            except ValueError:
                raise
            except Exception as exc:
                print(f"Voice embedding unavailable at {start:.2f}-{end:.2f}: {exc}")
        embedding_cache[key] = result
        return result

    cluster_scores = defaultdict(list)
    cluster_embeddings = defaultdict(list)
    for _, row in diarize_segments.iterrows():
        start, end = float(row["start"]), float(row["end"])
        if end - start < 0.6:
            continue
        emb = audio_embedding(start, end)
        if emb is not None:
            track = str(row["speaker"])
            cluster_scores[track].append(float(np.dot(target_voice_vector, emb)))
            cluster_embeddings[track].append((start, end, emb))
    means = {track: float(np.mean(scores)) for track, scores in cluster_scores.items()}
    ranked = sorted(means, key=means.get, reverse=True)
    target_track = ranked[0] if ranked else None
    target_mean = means[target_track] if ranked else 0.0
    # No invented competitor when only one cluster has usable speech.
    other_mean = means[ranked[1]] if len(ranked) > 1 else None
    separation = target_mean - other_mean if other_mean is not None else 0.0
    mapping_confidence = min(1.0, max(0.0, separation / 0.15))
    print("\n--- Baseline voice affinity ---")
    for track in ranked:
        print(f"{track}: {means[track]:.3f} ({len(cluster_scores[track])} samples)")
    print(f"Target candidate: {target_track}; mapping strength={mapping_confidence:.3f}")

    assigned = whisperx.assign_word_speakers(diarize_segments, aligned_result)
    timeline = []
    for source in assigned["segments"]:
        raw = str(source.get("speaker") or "Unknown_Speaker")
        base = "Target_Speaker" if raw == target_track else ("Unknown" if raw == "Unknown_Speaker" else raw)
        timeline.append(TimelineSegment(float(source["start"]), float(source["end"]),
                        source["text"].strip(), Baseline(raw, base), source.get("words", [])))

    def collect_voice(segment, previous=None, following=None):
        crop_start, crop_end = short_voice_crop(segment, previous, following, total_samples / 16000)
        emb = audio_embedding(crop_start, crop_end)
        similarity = float(np.dot(target_voice_vector, emb)) if emb is not None else None
        strength = min(1.0, max(0.0, (segment.end - segment.start) / 1.2)) * mapping_confidence
        if segment.end - segment.start < 0.4:
            strength = min(strength, 0.30)
        target_score = 0.0
        if similarity is not None and separation >= 0.03:
            target_score = float(np.clip((similarity - (target_mean + other_mean) / 2) / separation, -1, 1))
        else:
            strength = 0.0
        # Exclude intersecting speech from profiles so a segment cannot validate itself.
        track_similarities = {}
        profile_counts = {}
        if emb is not None:
            for track, samples in cluster_embeddings.items():
                independent = [vector for start, end, vector in samples
                               if end <= crop_start or start >= crop_end]
                if len(independent) < 2:
                    continue
                centroid = normalize_vector(np.mean(independent, axis=0))
                track_similarities[track] = float(np.dot(emb, centroid))
                profile_counts[track] = len(independent)
        candidates = sorted(track_similarities, key=track_similarities.get, reverse=True)
        best_track = candidates[0] if len(candidates) >= 2 else None
        margin = (track_similarities[candidates[0]] - track_similarities[candidates[1]]
                  if len(candidates) >= 2 else 0.0)
        segment.evidence.append(Evidence("local_voice", target_score, strength,
            {"similarity": similarity, "crop_start": crop_start, "crop_end": crop_end,
             "target_mean": target_mean, "competitor_mean": other_mean,
             "track_similarities": track_similarities, "independent_profile_counts": profile_counts,
             "best_track": best_track, "track_margin": margin}))

    def collect_visual(segment):
        collect_visual_evidence(segment, cap, fps, face_analyzer, target_face_centroid)

    def collect_semantic(segment):
        # Without an explicit role-to-identity mapping, words cannot identify a person.
        # Keep semantic/context observations neutral for arbitrary videos and targets.
        segment.evidence.append(Evidence("semantic_context", 0.0, 0.0,
            {"identity_mapping": None,
             "note": "No role or phrase is assumed to identify the supplied target."}))

    try:
        all_tracks = {str(track) for track in diarize_segments["speaker"].dropna().unique()}
        for index, segment in enumerate(timeline):
            previous = timeline[index - 1] if index else None
            following = timeline[index + 1] if index + 1 < len(timeline) else None
            collect_voice(segment, previous, following)
            collect_visual(segment)
            collect_semantic(segment)
            add_question_response_evidence(segment, previous, all_tracks, target_track, mapping_confidence)
            add_brief_exchange_evidence(segment, previous, all_tracks, target_track, mapping_confidence)
            add_echo_question_evidence(segment, previous, all_tracks, target_track, mapping_confidence)
            resolve_segment(segment, target_track, mapping_confidence)
        print("\n--- Evidence-Based Speaker Resolution ---")
        for segment in timeline:
            print(f"[{segment.start:.2f}s - {segment.end:.2f}s] {segment.final_speaker} "
                  f"(strength={segment.final_confidence:.2f}): {segment.text}")
            print(f"    baseline={segment.baseline.speaker}; raw={segment.baseline.raw_speaker_track}")
            for item in segment.evidence:
                print(f"    {item.source}: score={item.target_score:+.2f}, strength={item.confidence:.2f}, {item.details}")
        with open(args.output, "w", encoding="utf-8") as output:
            json.dump({"target_candidate": target_track, "cluster_voice_means": means,
                       "mapping_strength": mapping_confidence,
                       "confidence_is_calibrated": False,
                       "segments": [asdict(segment) for segment in timeline]}, output, indent=2, ensure_ascii=False)
    finally:
        cap.release()


if __name__ == "__main__":
    main()
