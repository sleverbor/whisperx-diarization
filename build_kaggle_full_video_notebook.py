"""Build the self-contained Kaggle notebook used for the current full-video test."""

import argparse
import base64
import json
from pathlib import Path
import pprint


ROOT = Path(__file__).resolve().parent


def lines(text):
    return text.splitlines(keepends=True)


def cell(kind, text):
    return {"cell_type": kind, "metadata": {}, "source": lines(text),
            **({"execution_count": None, "outputs": []} if kind == "code" else {})}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--reference-dir", type=Path, required=True)
    parser.add_argument("--enrollment", type=Path, required=True)
    parser.add_argument("--opening-reference", type=Path, required=True)
    parser.add_argument("--output", type=Path,
                        default=ROOT / "kaggle_full_video_comparison.ipynb")
    args = parser.parse_args()

    source_names = [
        "chainofrules.py", "cloud_runtime.py", "repeat_evidence.py",
        "recover_transcript_gaps.py", "review_audio_window.py",
        "review_transcript_regions.py", "review_overlap_extraction.py",
        "test_cloud_runtime.py", "test_short_answers.py",
        "test_transcript_gaps.py", "test_window_review.py",
        "test_review_regions.py", "test_repeat_evidence.py",
        "test_overlap_resolution.py", "test_overlap_extraction_review.py",
        "test_single_speaker_mapping.py",
    ]
    embedded = {name: (ROOT / name).read_text() for name in source_names}
    binary_paths = {
        "face_embeddings.npy": args.reference_dir / "face_embeddings.npy",
        "voice_embeddings.npy": args.reference_dir / "voice_embeddings.npy",
        "reference.json": args.reference_dir / "reference.json",
        "opening_officer_reference.npy": args.opening_reference,
        "auditor_enrollment.wav": args.enrollment,
    }
    encoded = {
        name: base64.b64encode(path.read_bytes()).decode("ascii")
        for name, path in binary_paths.items()
    }

    config = '''from pathlib import Path
import subprocess, sys, os, json, shutil, time, zipfile

VIDEO_URL = 'https://www.youtube.com/watch?v=uAtiEviUzGA'
RUN_FULL_VIDEO = True
RUN_TARGETED_REVIEW = True
RUN_OVERLAP_EXTRACTION = True
REVIEW_WEAK_CONFIDENCE = 0.35
REVIEW_SHORT_SECONDS = 1.0
BATCH_SIZE = 4

ON_KAGGLE = Path('/kaggle/input').exists()
if ON_KAGGLE:
    probe = subprocess.run(['nvidia-smi', '-L'], capture_output=True, text=True) if shutil.which('nvidia-smi') else None
    if probe is None or probe.returncode or 'GPU ' not in probe.stdout:
        raise RuntimeError('Enable a GPU accelerator in Kaggle Settings, then rerun this cell.')
    print(probe.stdout)
    BASE = Path('/kaggle/working')
else:
    BASE = Path.cwd()/'diarization-run'
BASE.mkdir(parents=True, exist_ok=True)
WORK = BASE/'diarization'; WORK.mkdir(exist_ok=True)
RESULTS = BASE/'results'; RESULTS.mkdir(exist_ok=True)
CACHE = BASE/'stage-cache'
VENV = BASE/'diarization-venv'
PYTHON = str(VENV/('Scripts/python.exe' if os.name == 'nt' else 'bin/python'))
VIDEO = WORK/'video.mp4'
REFERENCE = WORK/'target-reference'; REFERENCE.mkdir(exist_ok=True)
print('Video URL:', VIDEO_URL)
print('Results folder:', RESULTS)
print('Setup will use:', PYTHON)
'''

    setup = f'''EMBEDDED_FILES = {pprint.pformat(embedded, width=100)}
for name, source in EMBEDDED_FILES.items():
    (WORK/name).write_text(source)

ENV = os.environ.copy()
ENV['PYTHONUNBUFFERED'] = '1'
ENV['MPLCONFIGDIR'] = str(BASE/'matplotlib-cache')
def checked(command, **kwargs):
    return subprocess.run(command, env=ENV, check=True, **kwargs)

print('1/4: Creating/checking the isolated environment', flush=True)
ready = False
if Path(PYTHON).exists():
    ready = subprocess.run([PYTHON, '-m', 'pip', '--version'], env=ENV,
        stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL).returncode == 0
if not ready:
    bootstrap = BASE/'bootstrap-tools'
    checked([sys.executable, '-m', 'pip', 'install', '--target', str(bootstrap),
             'virtualenv>=20.26,<21', 'wrapt'])
    bootstrap_env = ENV.copy(); bootstrap_env['PYTHONPATH'] = str(bootstrap)
    subprocess.run([sys.executable, '-m', 'virtualenv', '--no-download', str(VENV)],
                   env=bootstrap_env, check=True)

print('2/4: Installing the pipeline and extraction model', flush=True)
requirements = [
    'whisperx==3.8.6', 'speechbrain==1.1.1', 'insightface==2.0',
    'torch==2.8.0', 'torchaudio==2.8.0', 'numpy==2.5.3',
    'opencv-python==5.0.0.93', 'onnxruntime-gpu==1.23.2', 'wrapt',
    'yt-dlp', 'soundfile',
]
checked([PYTHON, '-m', 'pip', 'install', '--upgrade', 'pip'])
checked([PYTHON, '-m', 'pip', 'install', *requirements])
checked([PYTHON, '-m', 'pip', 'install',
         'git+https://github.com/wenet-e2e/wesep.git',
         'git+https://github.com/wenet-e2e/wespeaker.git'])

print('3/4: Selecting the CUDA ONNX runtime', flush=True)
checked([PYTHON, '-m', 'pip', 'uninstall', '-y', 'onnxruntime', 'onnxruntime-gpu'])
checked([PYTHON, '-m', 'pip', 'install', '--no-deps', '--force-reinstall',
         'onnxruntime-gpu==1.23.2'])
if shutil.which('ffmpeg') is None:
    raise RuntimeError('ffmpeg is required. Kaggle normally includes it.')
library_dirs = subprocess.check_output([PYTHON, '-c',
    "import site,pathlib; print(':'.join(str(p) for d in site.getsitepackages() for p in pathlib.Path(d).glob('nvidia/*/lib')))"], env=ENV, text=True).strip()
ENV['LD_LIBRARY_PATH'] = library_dirs + ':' + ENV.get('LD_LIBRARY_PATH', '')

print('4/4: Verifying GPU imports and focused behavior tests', flush=True)
verification = """import torch,onnxruntime as ort,wrapt,wesep
import chainofrules,repeat_evidence
print('Torch:', torch.__version__, 'CUDA build:', torch.version.cuda)
print('GPU:', torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'NONE (local CPU)')
print('ONNX providers:', ort.get_available_providers())
"""
if ON_KAGGLE:
    verification += "assert torch.cuda.is_available(), 'Kaggle GPU is unavailable'\\nassert 'CUDAExecutionProvider' in ort.get_available_providers(), 'GPU ONNX runtime is unavailable'\\n"
checked([PYTHON, '-c', verification], cwd=WORK)
checked([PYTHON, '-m', 'unittest', 'test_cloud_runtime', 'test_short_answers',
         'test_repeat_evidence', 'test_overlap_resolution',
         'test_overlap_extraction_review'], cwd=WORK)

import hashlib
REFERENCE_FILES = {pprint.pformat(encoded, width=100)}
reference_hashes = {{}}
for name, value in REFERENCE_FILES.items():
    content = base64.b64decode(value)
    (REFERENCE/name).write_bytes(content)
    reference_hashes[name] = hashlib.sha256(content).hexdigest()
(RESULTS/'reference-hashes.json').write_text(json.dumps(reference_hashes, indent=2))
print('Reference bundle ready:', reference_hashes)
'''

    credentials = '''if ON_KAGGLE:
    from kaggle_secrets import UserSecretsClient
    ENV['HF_TOKEN'] = UserSecretsClient().get_secret('HF_TOKEN')
else:
    import getpass
    ENV['HF_TOKEN'] = os.environ.get('HF_TOKEN') or os.environ.get('HUGGINGFACE_TOKEN') or getpass.getpass('Hugging Face token: ')
if not ENV['HF_TOKEN']:
    raise RuntimeError('A Hugging Face token with diarization-model access is required.')
if ON_KAGGLE:
    for archive in Path('/kaggle/input').rglob('stage-checkpoints.zip'):
        with zipfile.ZipFile(archive) as zipped:
            for member in zipped.infolist():
                target = (BASE/member.filename).resolve()
                if not target.is_relative_to(CACHE.resolve()):
                    raise RuntimeError('Unexpected checkpoint archive path')
            zipped.extractall(BASE)
if not VIDEO.exists():
    checked([PYTHON, '-m', 'yt_dlp', '-f', 'bv*[height<=720]+ba/b[height<=720]',
             '--merge-output-format', 'mp4', '-o', str(VIDEO), VIDEO_URL])
print('Credentials configured; token not displayed.')
print('Video ready:', VIDEO, VIDEO.stat().st_size, 'bytes')
'''

    functions = '''def export_checkpoints():
    if CACHE.exists():
        shutil.make_archive(str(BASE/'stage-checkpoints'), 'zip', CACHE.parent, CACHE.name)

def stream(command, log_name, failure):
    with (RESULTS/log_name).open('w') as log:
        process = subprocess.Popen(command, cwd=WORK, env=ENV, stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT, text=True, bufsize=1)
        for line in process.stdout:
            line = line.replace(ENV['HF_TOKEN'], '[REDACTED]')
            log.write(line); log.flush()
            print(line if len(line) < 1000 else line[:1000]+' ... [full line saved]\\n', end='')
        if process.wait() != 0:
            raise RuntimeError(failure)

def run_test(video, stem, batch_size=4):
    output = RESULTS/(stem+'_evidence.json')
    command = [PYTHON, str(WORK/'chainofrules.py'), str(video),
        '--voice-priors', str(REFERENCE/'voice_embeddings.npy'),
        '--face-priors', str(REFERENCE/'face_embeddings.npy'),
        '--output', str(output), '--cache-dir', str(CACHE), '--batch-size', str(batch_size)]
    started = time.monotonic()
    try:
        stream(command, stem+'.log', 'Pipeline failed; see the saved log. Retry with BATCH_SIZE=1 for CUDA memory errors.')
    finally:
        export_checkpoints()
    result = json.loads(output.read_text())
    transcript = '\\n'.join(f"[{s['start']:.2f}-{s['end']:.2f}] {s['final_speaker']} (strength={s['final_confidence']:.2f}): {s['text']}" for s in result['segments'])
    (RESULTS/(stem+'_transcript.txt')).write_text(transcript+'\\n')
    repeat_rows = []
    for segment in result['segments']:
        for evidence in segment.get('evidence', []):
            if evidence.get('source') == 'repeated_presentation':
                d = evidence.get('details', {})
                repeat_rows.append(f"[{segment['start']:.2f}] {segment['text']}  <=>  [{d.get('matched_start', 0):.2f}] {d.get('matched_text', '')}")
    (RESULTS/(stem+'_repeat_candidates.txt')).write_text('\\n'.join(repeat_rows)+'\\n')
    print(f'Elapsed: {(time.monotonic()-started)/60:.1f} minutes')
    print('Repeated-presentation groups:', len(result.get('repeated_presentations', [])))
    return result

def extract_clip(start, duration, name):
    clip = WORK/name
    checked(['ffmpeg', '-nostdin', '-hide_banner', '-loglevel', 'error', '-y',
             '-ss', str(start), '-i', str(VIDEO), '-t', str(duration),
             '-c:v', 'libx264', '-preset', 'fast', '-crf', '20', '-c:a', 'aac', str(clip)])
    return clip

def run_targeted_review():
    output_dir = RESULTS/'targeted-review'
    command = [PYTHON, str(WORK/'review_transcript_regions.py'), '--video', str(VIDEO),
        '--baseline', str(RESULTS/'full_video_evidence.json'),
        '--target-reference', str(REFERENCE/'voice_embeddings.npy'),
        '--other-reference', 'Opening_officer='+str(REFERENCE/'opening_officer_reference.npy'),
        '--output-dir', str(output_dir), '--cache-dir', str(CACHE/'targeted-review'),
        '--weak-confidence', str(REVIEW_WEAK_CONFIDENCE), '--short-seconds', str(REVIEW_SHORT_SECONDS),
        '--minimum-gap', '5', '--context-seconds', '3', '--maximum-window-seconds', '30',
        '--window-overlap-seconds', '4', '--device', 'cuda' if ON_KAGGLE else 'auto']
    before = (RESULTS/'full_video_evidence.json').read_bytes()
    try:
        stream(command, 'targeted-review.log', 'Targeted review failed; see the saved log.')
    finally:
        export_checkpoints()
    assert (RESULTS/'full_video_evidence.json').read_bytes() == before
    return json.loads((output_dir/'comparison_summary.json').read_text())

def run_overlap_extraction():
    output_dir = RESULTS/'overlap-extraction'
    command = [PYTHON, str(WORK/'review_overlap_extraction.py'), '--video', str(VIDEO),
        '--baseline', str(RESULTS/'full_video_evidence.json'),
        '--enrollment', str(REFERENCE/'auditor_enrollment.wav'),
        '--voice-priors', str(REFERENCE/'voice_embeddings.npy'),
        '--output-dir', str(output_dir), '--device', 'cuda' if ON_KAGGLE else 'cpu',
        '--whisper-model', 'large-v2']
    before = (RESULTS/'full_video_evidence.json').read_bytes()
    try:
        stream(command, 'overlap-extraction.log', 'Overlap extraction failed; baseline results remain valid.')
    finally:
        export_checkpoints()
    assert (RESULTS/'full_video_evidence.json').read_bytes() == before
    return json.loads((output_dir/'report.json').read_text())
'''

    run = '''opening_clip = extract_clip(0, 30, 'opening_30s.mp4')
opening = run_test(opening_clip, 'opening', BATCH_SIZE)
providers = opening.get('runtime', {}).get('face_providers', {})
if ON_KAGGLE and (not providers or any('CUDAExecutionProvider' not in p for p in providers.values())):
    raise RuntimeError('Face models are still on CPU. Review opening.log before starting the full video.')
print((RESULTS/'opening_transcript.txt').read_text())

if RUN_FULL_VIDEO:
    full_video = run_test(VIDEO, 'full_video', BATCH_SIZE)
    if RUN_TARGETED_REVIEW:
        targeted_review = run_targeted_review()
        print('Targeted review:', json.dumps(targeted_review, indent=2))
    if RUN_OVERLAP_EXTRACTION:
        overlap_review = run_overlap_extraction()
        print('Overlap extraction:', json.dumps(overlap_review['summary'], indent=2))
else:
    print('Full video disabled. Review the opening result, then set RUN_FULL_VIDEO=True.')
'''

    save = '''from IPython.display import FileLink, display
export_checkpoints()
with (RESULTS/'runtime-packages.txt').open('w') as packages:
    checked([PYTHON, '-m', 'pip', 'freeze'], stdout=packages)
shutil.make_archive(str(BASE/'diarization-results'), 'zip', RESULTS)
display(FileLink(str(BASE/'diarization-results.zip')))
display(FileLink(str(BASE/'stage-checkpoints.zip')))
print('Saved in:', BASE)
'''

    notebook = {
        "cells": [
            cell("markdown", "# Current-video full diarization test\n\nThis notebook downloads `uAtiEviUzGA`, runs the unchanged evidence-based baseline, finds repeated presentations, and evaluates uncertain overlap intervals with target-conditioned extraction. Supplemental stages never overwrite the baseline.\n"),
            cell("code", config),
            cell("markdown", "## Install and verify\n\nEnable Internet and a GPU before running. The setup uses an isolated environment and verifies CUDA before the full video starts.\n"),
            cell("code", "import base64\n" + setup),
            cell("markdown", "## Credentials, checkpoint restore, and video download\n\nCreate a private Kaggle secret named `HF_TOKEN`. The token is read from the environment and is never embedded or printed.\n"),
            cell("code", credentials),
            cell("code", functions),
            cell("markdown", "## Run the opening check and whole video\n\nThe opening check confirms that face analysis is actually using CUDA. Then the full baseline, targeted review, repeat evidence, and overlap extraction run.\n"),
            cell("code", run),
            cell("markdown", "## Download results\n\n`diarization-results.zip` contains the baseline transcript and evidence, repeat candidates, targeted review, overlap extraction audio and report, logs, and package versions. `stage-checkpoints.zip` can restart expensive baseline stages.\n"),
            cell("code", save),
        ],
        "metadata": {"kernelspec": {"display_name": "Python 3", "language": "python", "name": "python3"},
                     "language_info": {"name": "python", "version": "3.12"}},
        "nbformat": 4, "nbformat_minor": 5,
    }
    args.output.write_text(json.dumps(notebook, indent=1) + "\n")
    print(args.output)


if __name__ == "__main__":
    main()
