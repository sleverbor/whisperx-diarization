"""Optional WhisperX VAD: retain speech islands and enforce decoder chunk limits.

Does not identify speakers. Uses faster-whisper's installed Silero model; no
TorchHub download. Deliberately does not join speech islands across silence.
"""
import math


def bounded_chunks(segments, chunk_size):
    """Return ordered, non-overlapping decoder crops, each <= chunk_size.

    Overlapping context pads are united first. Disjoint islands remain separate.
    Long islands are split without duplicated context. Sentence-boundary repair
    and uncertainty belong to the caller; splitting is not a diarization claim.
    """
    if not math.isfinite(chunk_size) or chunk_size <= 0:
        raise ValueError("chunk_size must be positive and finite")
    intervals = []
    for segment in segments:
        left, right = float(segment.start), float(segment.end)
        if not math.isfinite(left) or not math.isfinite(right) or left < 0 or right < left:
            raise ValueError("Invalid speech interval")
        if left != right:
            intervals.append((left, right))
    islands = []
    for left, right in sorted(intervals):
        if islands and left <= islands[-1][1]:
            islands[-1][1] = max(islands[-1][1], right)
        else:
            islands.append([left, right])
    crops = []
    for left, end in islands:
        while left < end:
            right = min(left + chunk_size, end)
            crops.append({"start": left, "end": right, "segments": [(left, right)]})
            left = right
    return crops


def make_vad(threshold=0.5, context=0.5):
    """Construct lazily so the interval helper can be tested without models."""
    import numpy as np
    from faster_whisper.vad import get_speech_timestamps, VadOptions
    from whisperx.diarize import Segment
    from whisperx.vads.vad import Vad

    if not 0 < threshold < 1 or not math.isfinite(context) or context < 0:
        raise ValueError("Invalid threshold or context")

    class BoundedSilero(Vad):
        def __init__(self):
            super().__init__(threshold)

        @staticmethod
        def preprocess_audio(audio):
            return audio

        def __call__(self, audio, **kwargs):
            if audio['sample_rate'] != 16000:
                raise ValueError("Expected 16 kHz audio")
            waveform = np.asarray(audio['waveform'], dtype=np.float32).reshape(-1)
            duration = len(waveform) / 16000
            timestamps = get_speech_timestamps(waveform, sampling_rate=16000,
                vad_options=VadOptions(threshold=threshold,
                    min_speech_duration_ms=100, max_speech_duration_s=20,
                    min_silence_duration_ms=200, speech_pad_ms=0))
            return [Segment(max(0, r['start']/16000-context),
                            min(duration, r['end']/16000+context), 'UNKNOWN')
                    for r in timestamps]

        @staticmethod
        def merge_chunks(segments, chunk_size, onset=.5, offset=None):
            return bounded_chunks(segments, chunk_size)

    return BoundedSilero()
