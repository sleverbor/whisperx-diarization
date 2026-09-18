"""Build a focused Kaggle notebook comparing Sortformer with saved DiaPer results."""

import json
from pathlib import Path
import pprint


ROOT = Path(__file__).resolve().parent
NEMO_COMMIT = "429e2ac4b69398d3cefb0ed15dbf9c043f290367"


def cell(kind, source):
    result = {"cell_type": kind, "metadata": {}, "source": source.splitlines(keepends=True)}
    if kind == "code":
        result.update(execution_count=None, outputs=[])
    return result


def main():
    embedded = {
        name: (ROOT / name).read_text()
        for name in (
            "evaluate_diaper_overlap.py", "evaluate_overlap_activity.py",
            "run_sortformer_overlap.py", "test_overlap_activity.py",
            "test_sortformer_overlap.py",
        )
    }
    config = f'''from pathlib import Path
import json, os, shutil, subprocess, sys, zipfile

REVISION = "sortformer-diaPer-head-to-head-v1"
VIDEO_ID = "lVfKfbFd0SM"
NEMO_COMMIT = "{NEMO_COMMIT}"
BASE = Path("/kaggle/working") if Path("/kaggle/input").exists() else Path.cwd()/"sortformer-run"
WORK = BASE/"sortformer-comparison"
PRIOR = BASE/"prior-diarization-results"
OUTPUT = BASE/"sortformer-results"
VENV = BASE/"sortformer-venv"
PYTHON = str(VENV/"bin"/"python")
for directory in (WORK, PRIOR, OUTPUT): directory.mkdir(parents=True, exist_ok=True)
print("Revision:", REVISION)
print("Output:", OUTPUT)
'''
    setup = f'''EMBEDDED_FILES = {pprint.pformat(embedded, width=100)}
for name, source in EMBEDDED_FILES.items():
    (WORK/name).write_text(source)

ENV = os.environ.copy()
ENV["PYTHONUNBUFFERED"] = "1"
ENV["MPLBACKEND"] = "Agg"
ENV["HF_HOME"] = str(BASE/"huggingface-cache")
def checked(command, **kwargs):
    return subprocess.run(command, env=ENV, check=True, **kwargs)

if not Path(PYTHON).is_file():
    bootstrap = BASE/"sortformer-bootstrap"
    checked([sys.executable, "-m", "pip", "install", "--target", str(bootstrap),
             "virtualenv>=20.26,<21"])
    bootstrap_env = ENV.copy(); bootstrap_env["PYTHONPATH"] = str(bootstrap)
    subprocess.run([sys.executable, "-m", "virtualenv", "--system-site-packages",
                    "--no-download", str(VENV)], env=bootstrap_env, check=True)

checked([PYTHON, "-m", "pip", "install", "--upgrade", "pip", "setuptools", "wheel"])
checked([PYTHON, "-m", "pip", "install", "Cython", "packaging", "soundfile"])
checked([PYTHON, "-m", "pip", "install",
         "git+https://github.com/NVIDIA-NeMo/NeMo.git@"+NEMO_COMMIT+"#egg=nemo_toolkit[asr]"])
checked([PYTHON, "-c", "import torch; from nemo.collections.asr.models import SortformerEncLabelModel; print('Torch',torch.__version__,'GPU',torch.cuda.get_device_name(0) if torch.cuda.is_available() else None); assert torch.cuda.is_available()"])
checked([PYTHON, "-m", "unittest", "test_overlap_activity", "test_sortformer_overlap"], cwd=WORK)
'''
    restore = '''archives = sorted(Path("/kaggle/input").rglob("diarization-results*.zip")) if Path("/kaggle/input").exists() else []
if not archives:
    local = Path.cwd()/"diarization-results(15).zip"
    if local.is_file(): archives = [local]
if not archives:
    raise RuntimeError("Attach the completed diarization-results(15).zip as a Kaggle dataset.")

selected = None
for archive in archives:
    with zipfile.ZipFile(archive) as zipped:
        names = set(zipped.namelist())
        run_names = [name for name in names if name.endswith("run-input.json")]
        comparison_names = [name for name in names if name.endswith("diaper-overlap/comparison.json")]
        if len(run_names) != 1 or len(comparison_names) != 1:
            continue
        run_input = json.loads(zipped.read(run_names[0]))
        if run_input.get("video_id") == VIDEO_ID and run_input.get("notebook_revision") == "diaper-librosa-keywords-v23":
            selected = archive
            break
if selected is None:
    raise RuntimeError("No attached result ZIP contains the completed v23 DiaPer run.")
if PRIOR.exists(): shutil.rmtree(PRIOR)
PRIOR.mkdir()
with zipfile.ZipFile(selected) as zipped:
    for member in zipped.infolist():
        target = (PRIOR/member.filename).resolve()
        if not target.is_relative_to(PRIOR.resolve()):
            raise RuntimeError("Unsafe path in result ZIP")
    zipped.extractall(PRIOR)

BASELINE = next(PRIOR.rglob("full_video_evidence.json"))
AUDIO = next(PRIOR.rglob("diaper-overlap/input/full-video.wav"))
DIAPER = next(PRIOR.rglob("diaper-overlap/comparison.json"))
print("Using:", selected)
print("Baseline:", BASELINE)
print("Audio:", AUDIO)
'''
    run = '''log_path = OUTPUT/"sortformer.log"
command = [PYTHON, str(WORK/"run_sortformer_overlap.py"),
           "--audio", str(AUDIO), "--output-dir", str(OUTPUT/"inference")]
with log_path.open("w") as log:
    process = subprocess.Popen(command, cwd=WORK, env=ENV, stdout=subprocess.PIPE,
                               stderr=subprocess.STDOUT, text=True, bufsize=1)
    recent = []
    for line in process.stdout:
        log.write(line); log.flush(); print(line, end="")
        recent = (recent + [line.rstrip()])[-30:]
    if process.wait() != 0:
        raise RuntimeError("Sortformer failed. Last output:\\n" + "\\n".join(recent))

comparison_path = OUTPUT/"comparison.json"
checked([PYTHON, str(WORK/"evaluate_overlap_activity.py"),
         "--baseline", str(BASELINE),
         "--rttm", str(OUTPUT/"inference"/"sortformer.rttm"),
         "--output", str(comparison_path)], cwd=WORK)
sortformer = json.loads(comparison_path.read_text())
diaper = json.loads(DIAPER.read_text())
head_to_head = {
    "revision": REVISION,
    "video_id": VIDEO_ID,
    "sortformer": sortformer["summary"],
    "diaper": diaper["summary"],
    "note": "Agreement with baseline overlap evidence is not ground-truth accuracy."
}
(OUTPUT/"head-to-head.json").write_text(json.dumps(head_to_head, indent=2)+"\\n")
print(json.dumps(head_to_head, indent=2))
'''
    save = '''with (OUTPUT/"runtime-packages.txt").open("w") as packages:
    checked([PYTHON, "-m", "pip", "freeze"], stdout=packages)
shutil.make_archive(str(BASE/"sortformer-comparison-results"), "zip", OUTPUT)
from IPython.display import FileLink, display
display(FileLink(str(BASE/"sortformer-comparison-results.zip")))
'''
    notebook = {
        "cells": [
            cell("markdown", "# Sortformer versus DiaPer overlap test\n\nAttach the completed `diarization-results(15).zip`. This notebook reuses its preserved baseline and full-video audio, runs NVIDIA's offline four-speaker Sortformer in bounded contextual windows, and applies the same overlap/control comparison used for DiaPer. It does not rerun or modify the transcription pipeline. The public model is licensed CC-BY-NC-4.0.\n"),
            cell("code", config),
            cell("markdown", "## Install the isolated Sortformer runtime\n\nEnable Internet and a GPU. NeMo is pinned to the recorded commit for reproducibility.\n"),
            cell("code", setup),
            cell("markdown", "## Restore the completed DiaPer result bundle\n"),
            cell("code", restore),
            cell("markdown", "## Run the head-to-head comparison\n\nThe 7:18 audio is divided into 80-second evaluation cores with five seconds of context on either side. Speaker labels remain local to each window; this test measures overlap activity rather than cross-window identity.\n"),
            cell("code", run),
            cell("markdown", "## Download the comparison\n"),
            cell("code", save),
        ],
        "metadata": {"kernelspec": {"display_name": "Python 3", "language": "python", "name": "python3"},
                     "language_info": {"name": "python", "version": "3.12"}},
        "nbformat": 4, "nbformat_minor": 5,
    }
    for index, item in enumerate(notebook["cells"]):
        if item["cell_type"] == "code":
            compile("".join(item["source"]), f"notebook-cell-{index}", "exec")
    output = ROOT/"kaggle_sortformer_comparison.ipynb"
    output.write_text(json.dumps(notebook, indent=1)+"\n")
    print(output)


if __name__ == "__main__":
    main()
