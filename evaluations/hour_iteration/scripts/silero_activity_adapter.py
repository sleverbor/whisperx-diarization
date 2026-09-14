"""WhisperX-compatible Silero activity using the installed faster-whisper model."""
import numpy as np
from faster_whisper.vad import get_speech_timestamps,VadOptions
from whisperx.diarize import Segment
from whisperx.vads.vad import Vad

class SileroActivity(Vad):
    def __init__(self,threshold=.5,context=.5):
        super().__init__(threshold)
        self.threshold=threshold;self.context=context
    @staticmethod
    def preprocess_audio(audio):return audio
    def __call__(self,audio,**kwargs):
        if audio['sample_rate']!=16000:raise ValueError('Silero adapter needs 16 kHz audio')
        waveform=np.asarray(audio['waveform'],dtype=np.float32).reshape(-1)
        timestamps=get_speech_timestamps(waveform,vad_options=VadOptions(threshold=self.threshold,min_speech_duration_ms=100,max_speech_duration_s=20,min_silence_duration_ms=200,speech_pad_ms=0),sampling_rate=16000)
        duration=len(waveform)/16000;spans=[]
        for r in timestamps:
            left=max(0,r['start']/16000-self.context);right=min(duration,r['end']/16000+self.context)
            # Keep contiguous detected speech intact, then cap batches in merge_chunks.
            if spans and left<=spans[-1][1] and right-spans[-1][0]<=20:spans[-1][1]=max(spans[-1][1],right)
            else:spans.append([left,right])
        return [Segment(a,b,'UNKNOWN') for a,b in spans]
    @staticmethod
    def merge_chunks(segments,chunk_size,onset=.5,offset=None):
        if not segments:return []
        return Vad.merge_chunks(segments,chunk_size,onset,offset)
