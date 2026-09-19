"""Build the focused five-exchange MossFormer2 comparison notebook."""

import base64
import json
from pathlib import Path
import pprint


ROOT = Path(__file__).resolve().parent


def cell(kind, source):
    row = {"cell_type": kind, "metadata": {}, "source": source.splitlines(keepends=True)}
    if kind == "code":
        row.update(execution_count=None, outputs=[])
    return row


def main():
    embedded = {
        name: (ROOT / name).read_text()
        for name in (
            "run_mossformer2_separation_experiment.py",
            "review_overlap_extraction.py",
            "run_contextual_wesep_experiment.py",
        )
    }
    labels = (ROOT / "evaluations/separation-hand-label-v1/labels.json").read_text()
    priors = base64.b64encode(
        (ROOT / "references/auditor-reviewed-v14/voice_embeddings.npy").read_bytes()
    ).decode("ascii")

    config = f'''from pathlib import Path
import base64, json, os, shutil, subprocess, sys

REVISION = "mossformer2-five-exchange-v1"
BASE = Path("/kaggle/working")
WORK = BASE/"mossformer2-comparison"
OUTPUT = BASE/"mossformer2-results"
VENV = BASE/"diarization-venv"
PYTHON = str(VENV/"bin"/"python")
for directory in (WORK, OUTPUT): directory.mkdir(parents=True, exist_ok=True)
print("Revision:", REVISION)
print("Output:", OUTPUT)
'''
    setup = f'''EMBEDDED_FILES = {pprint.pformat(embedded, width=100)}
for name, source in EMBEDDED_FILES.items():
    (WORK/name).write_text(source)
(WORK/"labels.json").write_text({labels!r})
(WORK/"voice_embeddings.npy").write_bytes(base64.b64decode({priors!r}))

ENV = os.environ.copy()
ENV["PYTHONUNBUFFERED"] = "1"
ENV["NUMBA_CACHE_DIR"] = str(BASE/"numba-clearvoice")
ENV["HF_HOME"] = str(BASE/"huggingface-cache")
def checked(command, **kwargs):
    return subprocess.run(command, env=ENV, check=True, **kwargs)

if not Path(PYTHON).is_file():
    bootstrap = BASE/"virtualenv-bootstrap"
    checked([sys.executable, "-m", "pip", "install", "--target", str(bootstrap),
             "virtualenv>=20.26,<21"])
    bootstrap_env = ENV.copy(); bootstrap_env["PYTHONPATH"] = str(bootstrap)
    subprocess.run([sys.executable, "-m", "virtualenv", "--system-site-packages",
                    "--no-download", str(VENV)], env=bootstrap_env, check=True)

# Install ClearVoice without its old strict NumPy/OpenCV pins. Kaggle's current
# Torch/WhisperX numerical stack remains untouched.
checked([PYTHON, "-m", "pip", "install", "--no-deps", "clearvoice==0.1.2"])
checked([PYTHON, "-m", "pip", "install", "gdown", "librosa==0.10.2.post1",
         "rotary-embedding-torch==0.8.3", "scenedetect==0.6.6",
         "python-speech-features==0.6", "yamlargparse", "torchinfo", "pydub"])
checked([PYTHON, "-c", "import torch,clearvoice,whisperx,speechbrain; "
         "print('GPU:',torch.cuda.get_device_name(0) if torch.cuda.is_available() else None); "
         "assert torch.cuda.is_available(), 'Enable a Kaggle GPU accelerator'"])
'''
    locate = '''import soundfile as sf
input_root = Path("/kaggle/input")
candidates = []
for path in input_root.rglob("*.wav"):
    try:
        info = sf.info(path)
    except Exception:
        continue
    duration = info.frames / info.samplerate
    if 430 <= duration <= 450:
        candidates.append((0 if path.name == "full-video.wav" else 1, path, duration))
if not candidates:
    raise RuntimeError("Attach the stage-checkpoints dataset containing the 7:18 full-video.wav.")
_, AUDIO, duration = sorted(candidates)[0]
print("Using:", AUDIO)
print("Duration:", round(duration, 2), "seconds")
'''
    run = '''command = [PYTHON, "-B", str(WORK/"run_mossformer2_separation_experiment.py"),
           "--audio", str(AUDIO),
           "--labels", str(WORK/"labels.json"),
           "--voice-priors", str(WORK/"voice_embeddings.npy"),
           "--output-dir", str(OUTPUT),
           "--context", "3", "--device", "cuda",
           "--hf-home", ENV["HF_HOME"]]
log_path = OUTPUT/"run.log"
with log_path.open("w") as log:
    process = subprocess.Popen(command, cwd=WORK, env=ENV, stdout=subprocess.PIPE,
                               stderr=subprocess.STDOUT, text=True, bufsize=1)
    recent = []
    for line in process.stdout:
        log.write(line); log.flush(); print(line, end="")
        recent = (recent + [line.rstrip()])[-40:]
    if process.wait() != 0:
        raise RuntimeError("MossFormer2 failed. Last output:\\n" + "\\n".join(recent))
print(json.dumps(json.loads((OUTPUT/"report.json").read_text()), indent=2)[:12000])
'''
    save = '''with (OUTPUT/"runtime-packages.txt").open("w") as packages:
    checked([PYTHON, "-m", "pip", "freeze"], stdout=packages)
shutil.make_archive(str(BASE/"mossformer2-comparison-results"), "zip", OUTPUT)
from IPython.display import FileLink, display
display(FileLink(str(BASE/"mossformer2-comparison-results.zip")))
'''
    notebook = {
        "cells": [
            cell("markdown", "# MossFormer2 speech-separation comparison\n\nThis runs the same five hand-labeled overlapping exchanges used for the SepFormer tests. It processes one short window at a time to bound memory use, preserves both blind separator outputs, and applies ECAPA only after separation. Enable Internet and a GPU, and attach the stage-checkpoints dataset containing `full-video.wav`.\n"),
            cell("code", config),
            cell("markdown", "## Install the isolated runtime\n"),
            cell("code", setup),
            cell("markdown", "## Find the preserved full-video audio\n"),
            cell("code", locate),
            cell("markdown", "## Run the five-exchange test\n"),
            cell("code", run),
            cell("markdown", "## Download the results\n"),
            cell("code", save),
        ],
        "metadata": {
            "kernelspec": {"display_name": "Python 3", "language": "python", "name": "python3"},
            "language_info": {"name": "python", "version": "3.12"},
        },
        "nbformat": 4,
        "nbformat_minor": 5,
    }
    for index, row in enumerate(notebook["cells"]):
        if row["cell_type"] == "code":
            compile("".join(row["source"]), f"notebook-cell-{index}", "exec")
    output = ROOT / "kaggle_mossformer2_comparison.ipynb"
    output.write_text(json.dumps(notebook, indent=1) + "\n")
    print(output)


if __name__ == "__main__":
    main()
