# Two-clip coverage experiment

Import diarization_coverage_compare.ipynb as a new private Kaggle notebook. Attach the existing
private full-video dataset, enable Internet and GPU T4 x2, and enable HF_TOKEN secret access.
Run from the top. No new dataset or repair cells are needed.

The notebook extracts 0:00–0:30 and 10:25–11:30 from video.mp4 and runs each twice:
normal VAD coverage and experimental full-audio coverage. It never runs the full video.
Both modes retain the same whole-clip diarization checkpoint; voice and visual checkpoints
are reusable when their sample/time windows match. ASR and alignment are cached separately
by mode. Embedded source includes virtualenv bootstrap, wrapt inside the venv, CUDA 12
ONNX Runtime 1.23.2, removal of overlapping CPU/GPU distributions, and Agg for subprocesses.

CLI: python chainofrules.py clip.mp4 --transcription-coverage full --cache-dir .cloud-cache
Normal behavior remains --transcription-coverage vad (the default). Speaker rules are unchanged.
Full coverage uses contiguous windows no longer than WhisperX's requested chunk size (default
30 seconds), including noise and silence. It is a controlled experiment, not a verified speech
detector or automatic missing-word repair. It may produce incorrect words or split speech at
window boundaries. Compare both transcripts against audio, especially noisy passages.

Download diarization-results.zip and stage-checkpoints.zip in the final notebook cell or
from Kaggle's output files pane. The four transcripts/evidence files and comparison summary
are retained, with transcript timestamps converted to original source times.

Validation: all 26 local behavior/integration tests passed. Coverage adapter covers every sample
and honors requested chunk size. Integration checks demonstrate ASR/alignment separation while
retaining identical diarization and successful voice/visual caches across modes. Notebook code
compiles; paired-run orchestration was tested with stubs. The original WhisperX decoder already
recovered the known height/weight exchanges when forced to cover raw audio in the diagnostic
experiment. This new four-run notebook has not yet been executed live on Kaggle.
