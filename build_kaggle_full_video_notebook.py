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
        "run_mossformer2_separation_experiment.py", "mossformer2_review_policy.py",
        "evaluate_diaper_overlap.py",
        "export_confident_transcript.py", "reference_promotion.py",
        "test_cloud_runtime.py", "test_short_answers.py",
        "test_transcript_gaps.py", "test_window_review.py",
        "test_review_regions.py", "test_repeat_evidence.py",
        "test_overlap_resolution.py", "test_overlap_extraction_review.py",
        "test_mossformer2_review_policy.py",
        "test_single_speaker_mapping.py", "test_confident_transcript.py",
        "test_reference_promotion.py",
        "test_diaper_overlap.py",
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
from urllib.parse import urlparse, parse_qs

VIDEO_URL = 'https://www.youtube.com/watch?v=lVfKfbFd0SM'
NOTEBOOK_REVISION = 'mossformer2-review-only-evidence-v26'
REQUIRE_OVERLAP_POLICY = True
RUN_FULL_VIDEO = True
RUN_TARGETED_REVIEW = True
RUN_OVERLAP_EXTRACTION = True
RUN_MOSSFORMER2_REVIEW = True
RUN_DIAPER_OVERLAP = True
REVIEW_WEAK_CONFIDENCE = 0.35
REVIEW_SHORT_SECONDS = 1.0
BATCH_SIZE = 4

parsed_video_url = urlparse(VIDEO_URL)
VIDEO_ID = (parsed_video_url.path.strip('/') if parsed_video_url.netloc == 'youtu.be'
            else parse_qs(parsed_video_url.query).get('v', [''])[0])
if not VIDEO_ID or any(character not in 'abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789_-' for character in VIDEO_ID):
    raise ValueError(f'Could not derive a safe YouTube video ID from {VIDEO_URL!r}')

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
RESULTS = BASE/('results-'+VIDEO_ID); RESULTS.mkdir(exist_ok=True)
CACHE = BASE/'stage-cache'/VIDEO_ID
VENV = BASE/'diarization-venv'
PYTHON = str(VENV/('Scripts/python.exe' if os.name == 'nt' else 'bin/python'))
VIDEO = WORK/('video-'+VIDEO_ID+'-h264.mp4')
REFERENCE = WORK/'target-reference'; REFERENCE.mkdir(exist_ok=True)
print('Video URL:', VIDEO_URL)
print('Video ID:', VIDEO_ID)
print('Notebook revision:', NOTEBOOK_REVISION)
print('Results folder:', RESULTS)
print('Setup will use:', PYTHON)
'''

    setup = f'''EMBEDDED_FILES = {pprint.pformat(embedded, width=100)}
for name, source in EMBEDDED_FILES.items():
    (WORK/name).write_text(source)

ENV = os.environ.copy()
ENV['PYTHONUNBUFFERED'] = '1'
ENV['MPLCONFIGDIR'] = str(BASE/'matplotlib-cache')
ENV['MPLBACKEND'] = 'Agg'
ENV['NUMBA_CACHE_DIR'] = str(BASE/'numba-cache')
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
    'yt-dlp', 'soundfile', 'safe-gpu', 'yamlargparse==1.31.1',
    'decorator', 'h5py', 'matplotlib', 'librosa', 'scikit-learn', 'tensorboard',
]
checked([PYTHON, '-m', 'pip', 'install', '--upgrade', 'pip'])
checked([PYTHON, '-m', 'pip', 'install', *requirements])
checked([PYTHON, '-m', 'pip', 'install', '--no-deps', 'clearvoice==0.1.2'])
checked([PYTHON, '-m', 'pip', 'install', 'gdown', 'librosa==0.10.2.post1',
         'rotary-embedding-torch==0.8.3', 'scenedetect==0.6.6',
         'python-speech-features==0.6', 'torchinfo', 'pydub'])
checked([PYTHON, '-m', 'pip', 'install',
         'git+https://github.com/wenet-e2e/wespeaker.git'])
# WeSep's current package metadata omits its namespace-style wesep/utils
# directory. Keep the checkout and put it first on PYTHONPATH so the complete
# source tree is used, while pip still installs all declared dependencies.
WESEP_SOURCE = WORK/'vendor'/'wesep'
if not (WESEP_SOURCE/'wesep'/'utils'/'utils.py').is_file():
    WESEP_SOURCE.parent.mkdir(parents=True, exist_ok=True)
    checked(['git', 'clone', '--depth', '1',
             'https://github.com/wenet-e2e/wesep.git', str(WESEP_SOURCE)])
checked([PYTHON, '-m', 'pip', 'install', str(WESEP_SOURCE)])
# The upstream wheel omits wesep/utils because that directory has no
# __init__.py. Overlay the complete checkout onto site-packages so imports do
# not depend on PYTHONPATH or notebook process state.
site_packages = Path(subprocess.check_output(
    [PYTHON, '-c', 'import site; print(site.getsitepackages()[0])'],
    env=ENV, text=True).strip())
installed_wesep = site_packages/'wesep'
shutil.copytree(WESEP_SOURCE/'wesep', installed_wesep, dirs_exist_ok=True)
missing_utility = installed_wesep/'utils'/'utils.py'
if not missing_utility.is_file():
    raise RuntimeError(f'WeSep repair failed; missing {{missing_utility}}')
# These upstream source directories contain Python modules but omit package
# markers, which is also why they disappear from the built wheel.
for directory in [installed_wesep/'utils', installed_wesep/'dataset', installed_wesep/'bin']:
    if directory.is_dir():
        (directory/'__init__.py').touch()
ENV['PYTHONPATH'] = str(WORK) + os.pathsep + ENV.get('PYTHONPATH', '')

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
print('WeSep repaired package:', wesep.__file__)
"""
if ON_KAGGLE:
    verification += "assert torch.cuda.is_available(), 'Kaggle GPU is unavailable'\\nassert 'CUDAExecutionProvider' in ort.get_available_providers(), 'GPU ONNX runtime is unavailable'\\n"
checked([PYTHON, '-c', verification], cwd=WORK)
checked([PYTHON, '-m', 'unittest', 'test_cloud_runtime', 'test_short_answers',
         'test_repeat_evidence', 'test_overlap_resolution',
         'test_overlap_extraction_review', 'test_confident_transcript',
         'test_mossformer2_review_policy',
         'test_reference_promotion', 'test_diaper_overlap'], cwd=WORK)

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
    for archive in Path('/kaggle/input').rglob('stage-checkpoints*.zip'):
        with zipfile.ZipFile(archive) as zipped:
            for member in zipped.infolist():
                target = (BASE/member.filename).resolve()
                if not target.is_relative_to(CACHE.resolve()):
                    raise RuntimeError('Unexpected checkpoint archive path')
            zipped.extractall(BASE)
    expanded_checkpoints = [path for path in Path('/kaggle/input').rglob(VIDEO_ID)
                            if path.is_dir() and path.parent.name == 'stage-cache']
    if len(expanded_checkpoints) > 1:
        raise RuntimeError('Found more than one expanded checkpoint dataset for this video')
    if expanded_checkpoints:
        shutil.copytree(expanded_checkpoints[0], CACHE, dirs_exist_ok=True)
        print('Restored expanded stage checkpoints:', expanded_checkpoints[0])
OVERLAP_POLICY = None
policy_matches = []
search_root = Path('/kaggle/input') if ON_KAGGLE else Path.cwd()
for head_path in search_root.rglob('head-to-head.json'):
    try:
        head = json.loads(head_path.read_text())
    except Exception:
        continue
    candidate = head_path.parent/'overlap-review-policy.json'
    if (head.get('video_id') == VIDEO_ID
            and head.get('revision') == 'sortformer-additive-two-tier-policy-v5'
            and candidate.is_file()):
        policy_matches.append(candidate)
# Kaggle normally expands dataset archives, but accept a retained ZIP too.
for archive in search_root.rglob('*.zip'):
    try:
        with zipfile.ZipFile(archive) as zipped:
            names = set(zipped.namelist())
            for head_name in [name for name in names if name.endswith('head-to-head.json')]:
                head = json.loads(zipped.read(head_name))
                policy_name = str(Path(head_name).parent/'overlap-review-policy.json')
                if (head.get('video_id') == VIDEO_ID
                        and head.get('revision') == 'sortformer-additive-two-tier-policy-v5'
                        and policy_name in names):
                    extracted = WORK/'attached-overlap-review-policy.json'
                    extracted.write_bytes(zipped.read(policy_name))
                    policy_matches.append(extracted)
    except (zipfile.BadZipFile, KeyError, json.JSONDecodeError):
        continue
unique_policies = []
for candidate in policy_matches:
    if not any(candidate.read_bytes() == existing.read_bytes() for existing in unique_policies):
        unique_policies.append(candidate)
if len(unique_policies) == 1:
    OVERLAP_POLICY = unique_policies[0]
    print('Found required additive Sortformer policy:', OVERLAP_POLICY)
elif len(unique_policies) > 1:
    raise RuntimeError('Found multiple different matching v5 overlap policies')
elif REQUIRE_OVERLAP_POLICY:
    raise RuntimeError(
        'Attach the Kaggle dataset created from sortformer-comparison-results(4).zip. '
        'No matching v5 overlap policy was found; stopping before the full run.')
if not VIDEO.exists():
    if ON_KAGGLE:
        accepted_names = {VIDEO_ID+'.mp4', VIDEO_ID+'_full480.mp4'}
        matches = [path for path in Path('/kaggle/input').rglob('*.mp4')
                   if path.name in accepted_names]
        if len(matches) != 1:
            raise RuntimeError(
                'Attach a Kaggle dataset containing exactly one of '
                f'{sorted(accepted_names)}. YouTube blocks downloads from Kaggle.')
        source_video = matches[0]
    else:
        local_video = Path.cwd()/(VIDEO_ID+'.mp4')
        if not local_video.is_file():
            raise RuntimeError(f'Missing local video: {{local_video}}')
        source_video = local_video
    # The supplied YouTube video uses AV1, which Kaggle's OpenCV build cannot
    # decode. Normalize it once so the full visual pass actually reads frames.
    checked(['ffmpeg', '-nostdin', '-hide_banner', '-loglevel', 'error', '-y',
             '-i', str(source_video), '-map', '0:v:0', '-map', '0:a:0',
             '-c:v', 'libx264', '-preset', 'fast', '-crf', '22',
             '-pix_fmt', 'yuv420p', '-c:a', 'aac', '-b:a', '160k', str(VIDEO)])
video_codec = subprocess.check_output(
    ['ffprobe', '-v', 'error', '-select_streams', 'v:0',
     '-show_entries', 'stream=codec_name', '-of', 'default=nw=1:nk=1', str(VIDEO)],
    env=ENV, text=True).strip()
if video_codec != 'h264':
    raise RuntimeError(f'Expected normalized H.264 video, found {{video_codec!r}}')
print('Normalized video codec:', video_codec)
print('Credentials configured; token not displayed.')
print('Video ready:', VIDEO, VIDEO.stat().st_size, 'bytes')
(RESULTS/'run-input.json').write_text(json.dumps({
    'video_url': VIDEO_URL,
    'video_id': VIDEO_ID,
    'normalized_video_sha256': hashlib.sha256(VIDEO.read_bytes()).hexdigest(),
    'notebook_revision': NOTEBOOK_REVISION,
}, indent=2))
'''

    functions = '''def export_checkpoints():
    if CACHE.exists():
        shutil.make_archive(str(BASE/'stage-checkpoints'), 'zip', BASE,
                            str(CACHE.relative_to(BASE)))

def stream(command, log_name, failure):
    log_path = RESULTS/log_name
    recent_lines = []
    with log_path.open('w') as log:
        process = subprocess.Popen(command, cwd=WORK, env=ENV, stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT, text=True, bufsize=1)
        for line in process.stdout:
            line = line.replace(ENV['HF_TOKEN'], '[REDACTED]')
            log.write(line); log.flush()
            recent_lines.append(line.rstrip())
            recent_lines = recent_lines[-20:]
            print(line if len(line) < 1000 else line[:1000]+' ... [full line saved]\\n', end='')
        if process.wait() != 0:
            tail = '\\n'.join(recent_lines)
            raise RuntimeError(f"{failure}\\nLog: {log_path}\\nLast output:\\n{tail}")

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
                repeat_rows.append(f"[{segment['start']:.2f}] {segment['text']}  <=>  [{d.get('donor_start', 0):.2f}] {d.get('donor_text', '')}")
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
    if OVERLAP_POLICY is not None:
        command += ['--selection-policy', str(OVERLAP_POLICY)]
        print('Using additive Sortformer policy:', OVERLAP_POLICY)
    before = (RESULTS/'full_video_evidence.json').read_bytes()
    try:
        stream(command, 'overlap-extraction.log', 'Overlap extraction failed; baseline results remain valid.')
    finally:
        export_checkpoints()
    assert (RESULTS/'full_video_evidence.json').read_bytes() == before
    return json.loads((output_dir/'report.json').read_text())

def run_mossformer2_review():
    """Separate overlap candidates without modifying baseline text or identity."""
    output_dir = RESULTS/'mossformer2-review'
    inference_dir = output_dir/'inference'
    labels = output_dir/'selected-overlaps.json'
    output_dir.mkdir(parents=True, exist_ok=True)
    prepare = [PYTHON, str(WORK/'mossformer2_review_policy.py'), 'prepare',
        '--baseline', str(RESULTS/'full_video_evidence.json'), '--output', str(labels)]
    if OVERLAP_POLICY is not None:
        prepare += ['--selection-policy', str(OVERLAP_POLICY)]
    checked(prepare, cwd=WORK)
    command = [PYTHON, '-B', str(WORK/'run_mossformer2_separation_experiment.py'),
        '--video', str(VIDEO), '--labels', str(labels),
        '--voice-priors', str(REFERENCE/'voice_embeddings.npy'),
        '--output-dir', str(inference_dir), '--context', '3',
        '--device', 'cuda' if ON_KAGGLE else 'cpu',
        '--hf-home', str(BASE/'huggingface-cache')]
    ENV['SPEECHBRAIN_CACHE'] = str(
        BASE/'speechbrain-cache'/'spkrec-ecapa-voxceleb')
    before = (RESULTS/'full_video_evidence.json').read_bytes()
    try:
        stream(command, 'mossformer2-review.log',
               'MossFormer2 review failed; baseline results remain valid.')
    finally:
        export_checkpoints()
    report = output_dir/'review-evidence.json'
    checked([PYTHON, str(WORK/'mossformer2_review_policy.py'), 'evaluate',
             '--report', str(inference_dir/'report.json'), '--output', str(report)],
            cwd=WORK)
    assert (RESULTS/'full_video_evidence.json').read_bytes() == before
    return json.loads(report.read_text())

def run_diaper_overlap():
    """Run official DiaPer on full audio, then compare without changing baseline."""
    source = WORK/'vendor'/'DiaPer'
    checkpoint = source/'models'/'10attractors'/'SC_LibriSpeech_2spk_adapted1-10'/'models'/'checkpoint_100.tar'
    infer_config = source/'examples'/'infer_16k_10attractors.yaml'
    if not checkpoint.is_file():
        if source.exists():
            shutil.rmtree(source)
        source.parent.mkdir(parents=True, exist_ok=True)
        checked(['git', 'clone', '--depth', '1', '--filter=blob:none', '--no-checkout',
                 'https://github.com/BUTSpeechFIT/DiaPer.git', str(source)])
        checked(['git', '-C', str(source), 'sparse-checkout', 'init', '--no-cone'])
        checked(['git', '-C', str(source), 'sparse-checkout', 'set',
                 '/diaper/', '/examples/infer_16k_10attractors.yaml',
                 '/models/10attractors/SC_LibriSpeech_2spk_adapted1-10/models/checkpoint_100.tar'])
        checked(['git', '-C', str(source), 'checkout'])
    # DiaPer relies on a small Perceiver change from the authors' Transformers
    # fork. Install it into a private overlay so the established pipeline keeps
    # its own dependency set.
    transformer_overlay = source/'python-overlay'
    if not (transformer_overlay/'transformers').is_dir():
        checked([PYTHON, '-m', 'pip', 'install', '--no-deps', '--target',
                 str(transformer_overlay),
                 'git+https://github.com/fnlandini/transformers.git@b830ec2245139b157576153cfd8999e1da24a82c'])
    # DiaPer imports only the Perceiver model and does not use tokenization.
    # Keep the host pipeline's current tokenizers build and disable only this
    # irrelevant upper-bound check inside DiaPer's private overlay.
    dependency_check = transformer_overlay/'transformers'/'dependency_versions_check.py'
    dependency_text = dependency_check.read_text()
    runtime_loop = 'for pkg in pkgs_to_check_at_runtime:\\n'
    skip_marker = '    if pkg == "tokenizers":  # unused by DiaPer\\n        continue\\n'
    if skip_marker not in dependency_text:
        if runtime_loop not in dependency_text:
            raise RuntimeError('Could not patch DiaPer Transformers dependency checks')
        dependency_text = dependency_text.replace(
            runtime_loop, runtime_loop + skip_marker, 1)
    dependency_check.write_text(dependency_text)
    # The official 2023 script's GPU check treats GPU index 0 as CPU and asks
    # safe_gpu to allocate devices. Kaggle already assigned CUDA_VISIBLE_DEVICES,
    # so use that allocation directly.
    infer_script = source/'diaper'/'infer_single_file.py'
    infer_text = infer_script.read_text()
    infer_text = infer_text.replace(
        "if args.gpu >= 1:",
        "if args.gpu >= 0 and torch.cuda.is_available():")
    infer_text = infer_text.replace(
        "        safe_gpu.claim_gpus(nb_gpus=args.gpu)\\n", "")
    infer_text = infer_text.replace(
        "librosa.get_duration(filename=filepath)", "sf.info(filepath).duration")
    infer_script.write_text(infer_text)
    models_script = source/'diaper'/'backend'/'models.py'
    models_text = models_script.read_text().replace(
        "map_location=args.device)", "map_location=args.device, weights_only=False)").replace(
        "map_location=device)", "map_location=device, weights_only=False)")
    models_script.write_text(models_text)
    # Librosa 0.10+ made mel-filter arguments keyword-only. Retain DiaPer's
    # published feature settings while adapting the call syntax.
    features_script = source/'diaper'/'common_utils'/'features.py'
    features_text = features_script.read_text()
    legacy_mel_call = 'librosa.filters.mel(sampling_rate, n_fft, feature_dim)'
    current_mel_call = 'librosa.filters.mel(sr=sampling_rate, n_fft=n_fft, n_mels=feature_dim)'
    if legacy_mel_call in features_text:
        features_text = features_text.replace(legacy_mel_call, current_mel_call)
    if current_mel_call not in features_text:
        raise RuntimeError('Could not patch DiaPer for the current Librosa API')
    features_script.write_text(features_text)

    audio_dir = RESULTS/'diaper-overlap'/'input'
    audio_dir.mkdir(parents=True, exist_ok=True)
    audio = audio_dir/'full-video.wav'
    if not audio.is_file():
        checked(['ffmpeg', '-nostdin', '-hide_banner', '-loglevel', 'error', '-y',
                 '-i', str(VIDEO), '-vn', '-ac', '1', '-ar', '16000', str(audio)])
    output_dir = RESULTS/'diaper-overlap'/'inference'
    command = [PYTHON, str(infer_script), '-c', str(infer_config),
        '--wav-dir', str(audio_dir), '--wav-name', 'full-video',
        '--models-path', str(checkpoint.parent), '--epochs', '100',
        '--rttms-dir', str(output_dir), '--gpu', '0']
    prior_pythonpath = ENV.get('PYTHONPATH', '')
    ENV['PYTHONPATH'] = str(transformer_overlay) + os.pathsep + prior_pythonpath
    try:
        stream(command, 'diaper-overlap.log',
               'DiaPer inference failed; baseline and existing overlap results remain valid.')
    finally:
        ENV['PYTHONPATH'] = prior_pythonpath
    rttms = list(output_dir.rglob('full-video.rttm'))
    if len(rttms) != 1:
        raise RuntimeError(f'Expected one DiaPer RTTM, found {{len(rttms)}}')
    report_path = RESULTS/'diaper-overlap'/'comparison.json'
    checked([PYTHON, str(WORK/'evaluate_diaper_overlap.py'),
             '--baseline', str(RESULTS/'full_video_evidence.json'),
             '--rttm', str(rttms[0]), '--output', str(report_path)], cwd=WORK)
    return json.loads(report_path.read_text())

def export_reference_promotion_review():
    output_dir = RESULTS/'reference-promotion-review'
    manifest_path = output_dir/'manifest.json'
    if manifest_path.is_file():
        print('Reusing completed reference-promotion review:', manifest_path)
        return json.loads(manifest_path.read_text())
    if output_dir.exists():
        print('Removing incomplete reference-promotion review:', output_dir)
        shutil.rmtree(output_dir)
    command = [PYTHON, str(WORK/'reference_promotion.py'), 'export',
        '--video', str(VIDEO), '--evidence', str(RESULTS/'full_video_evidence.json'),
        '--output-dir', str(output_dir), '--source-url', VIDEO_URL,
        '--reference-metadata', str(REFERENCE/'reference.json')]
    checked(command, cwd=WORK)
    return json.loads(manifest_path.read_text())
'''

    run = '''opening_clip = extract_clip(0, 30, 'opening_30s.mp4')
opening = run_test(opening_clip, 'opening', BATCH_SIZE)
providers = opening.get('runtime', {}).get('face_providers', {})
if ON_KAGGLE and (not providers or any('CUDAExecutionProvider' not in p for p in providers.values())):
    raise RuntimeError('Face models are still on CPU. Review opening.log before starting the full video.')
print((RESULTS/'opening_transcript.txt').read_text())

if RUN_FULL_VIDEO:
    full_video = run_test(VIDEO, 'full_video', BATCH_SIZE)
    decoded_visual_segments = sum(
        1 for segment in full_video['segments']
        for evidence in segment.get('evidence', [])
        if evidence.get('source') == 'visual_context'
        and evidence.get('details', {}).get('frames_read', 0) > 0)
    if decoded_visual_segments == 0:
        raise RuntimeError('No full-video frames were decoded; supplemental reviews were not started.')
    print('Full-video segments with decoded visual frames:', decoded_visual_segments)
    confident_dir = RESULTS/'confidence-filtered-transcript'
    checked([PYTHON, str(WORK/'export_confident_transcript.py'),
             str(RESULTS/'full_video_evidence.json'),
             '--output-dir', str(confident_dir),
             '--target-minimum', '0.35', '--other-minimum', '0.65'], cwd=WORK)
    print('Confidence-filtered transcript:', confident_dir/'confident_transcript.txt')
    if RUN_TARGETED_REVIEW:
        targeted_review = run_targeted_review()
        print('Targeted review:', json.dumps(targeted_review, indent=2))
    if RUN_OVERLAP_EXTRACTION:
        overlap_review = run_overlap_extraction()
        print('Overlap extraction:', json.dumps(overlap_review['summary'], indent=2))
    if RUN_MOSSFORMER2_REVIEW:
        mossformer2_review = run_mossformer2_review()
        print('MossFormer2 review evidence:',
              json.dumps(mossformer2_review['summary'], indent=2))
    if RUN_DIAPER_OVERLAP:
        diaper_review = run_diaper_overlap()
        print('DiaPer overlap comparison:', json.dumps(diaper_review['summary'], indent=2))
    promotion_review = export_reference_promotion_review()
    print('Reference promotion candidates:', len(promotion_review['candidates']))
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
            cell("markdown", "# Current-video full diarization and additive overlap extraction\n\nAttach a Kaggle dataset containing `lVfKfbFd0SM_full480.mp4` (the existing local filename) or `lVfKfbFd0SM.mp4`. To process the additional Sortformer candidates, also attach a dataset made from the v5 `sortformer-comparison-results.zip`. The notebook preserves the evidence-based baseline and uses the additive policy only to select supplemental speaker-conditioned extraction. Supplemental stages never overwrite the baseline.\n"),
            cell("code", config),
            cell("markdown", "## Install and verify\n\nEnable Internet and a GPU before running. The setup uses an isolated environment and verifies CUDA before the full video starts.\n"),
            cell("code", "import base64\n" + setup),
            cell("markdown", "## Credentials, checkpoint restore, and attached video\n\nCreate a private Kaggle secret named `HF_TOKEN`. The token is read from the environment and is never embedded or printed. The current video is copied from the attached dataset because YouTube blocks Kaggle's shared addresses.\n"),
            cell("code", credentials),
            cell("code", functions),
            cell("markdown", "## Run the opening check, whole video, and additive overlap review\n\nThe opening check confirms that face analysis is actually using CUDA. The established stages run first. When the matching v5 Sortformer result dataset is attached, speaker-conditioned extraction processes the union of existing baseline overlap intervals and strong Sortformer additions. MossFormer2 then separates those same review candidates, rejects weak target matches, and exports review-only evidence; it never inserts text or changes speaker identity. DiaPer remains a read-only comparison.\n"),
            cell("code", run),
            cell("markdown", "## Download results\n\n`diarization-results.zip` contains the preserved baseline, existing supplemental reviews, MossFormer2 review-only evidence, DiaPer RTTM and comparison report, logs, and package versions. `stage-checkpoints.zip` can restart expensive baseline stages.\n"),
            cell("code", save),
        ],
        "metadata": {"kernelspec": {"display_name": "Python 3", "language": "python", "name": "python3"},
                     "language_info": {"name": "python", "version": "3.12"}},
        "nbformat": 4, "nbformat_minor": 5,
    }
    for index, notebook_cell in enumerate(notebook["cells"]):
        if notebook_cell["cell_type"] == "code":
            compile("".join(notebook_cell["source"]), f"notebook-cell-{index}", "exec")
    args.output.write_text(json.dumps(notebook, indent=1) + "\n")
    print(args.output)


if __name__ == "__main__":
    main()
