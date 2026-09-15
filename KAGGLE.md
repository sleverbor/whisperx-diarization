# Private full-video Kaggle test

This branch preserves every attribution function from `milestone-two-minute-diarization`.
WhisperX/diarization and SpeechBrain use CUDA when PyTorch detects it. InsightFace requests
CUDA when ONNX Runtime exposes its CUDA provider, with CPU fallback and actual session
providers printed. CPU remains supported. GPU execution still requires a cloud smoke test.

## Setup

1. Build the private dataset upload archive with `python prepare_kaggle_bundle.py --output /path/to/kaggle_diarization_bundle.zip`.
2. Create a **private** Kaggle Dataset from that archive and let Kaggle extract it.
3. Import `kaggle_full_video.ipynb` into a **private** Kaggle Notebook. Attach the dataset,
   enable Internet, and choose an NVIDIA GPU accelerator.
4. Add `HF_TOKEN` through Add-ons → Secrets and enable access for the notebook. Use the
   Hugging Face account with diarization-model access already accepted. No token is included.
5. Run setup and the two-minute compatibility test. Check actual face providers and transcript.
6. Run the separate full-video cell. Download results and stage-checkpoints archives afterwards.

The notebook installs the recorded direct dependency versions into an isolated environment,
substituting `onnxruntime-gpu` for `onnxruntime`. CUDA libraries are exposed to CTranslate2
and ONNX Runtime. This installation has not been tested on Kaggle yet; runtime/package details
are exported with the result. The local environment is not modified.

## Runtime and recovery

ASR, alignment, and whole-video diarization models load one stage at a time and are released
before the next. SpeechBrain and face analysis load after diarization. The cloud notebook starts
with batch size 4 to leave memory headroom; retry with 1 after a GPU memory error. Batch size
changes invalidate checkpoints, and fresh GPU ASR may vary from the CPU milestone.

`--cache-dir` enables atomic JSON checkpoints: completed transcription, alignment, whole-video
tracks, each successful voice embedding, and each completed segment's visual evidence.
Inputs, source/helper code, device, batch size, direct model package versions and advertised
ONNX providers fingerprint the cache. Attribution is always recomputed from retained evidence.
An interrupted stage reruns. No independent chunk diarization is introduced.

Kaggle session files are temporary unless saved/downloaded. Download `stage-checkpoints.zip`
and attach it as a private dataset in a later session to resume matching completed stages.
Checkpoint archives contain transcript and identity evidence; keep them private.
Model weights are not included and may need downloading again. Package/cache compatibility
can change across cloud sessions; incompatible fingerprints cause recomputation.

Local use remains `python chainofrules.py video.mp4 --cache-dir .cloud-cache` with HF_TOKEN
in the environment. Without `--cache-dir`, behavior is unchanged except model scheduling/device.
Tests: `python -m unittest test_cloud_runtime test_short_answers`.

Sources: https://www.kaggle.com/docs/efficient-gpu-usage ;
https://www.kaggle.com/product-feedback/114053 ;
https://onnxruntime.ai/docs/execution-providers/CUDA-ExecutionProvider.html ;
https://speechbrain.readthedocs.io/en/latest/API/speechbrain.utils.run_opts.html
