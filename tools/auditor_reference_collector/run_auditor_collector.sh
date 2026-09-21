#!/usr/bin/env bash
set -e
TASK_COLLECTOR_DIR="$(cd -- "$(dirname -- "$0")" && pwd)"
TASK_COLLECTOR_PY="${AUDITOR_PYTHON:-}"
if [ -z "$TASK_COLLECTOR_PY" ]; then
  for TASK_COLLECTOR_CANDIDATE in "$TASK_COLLECTOR_DIR/../../.venv/bin/python" "$TASK_COLLECTOR_DIR/.venv/bin/python"; do
    if [ -x "$TASK_COLLECTOR_CANDIDATE" ]; then TASK_COLLECTOR_PY="$TASK_COLLECTOR_CANDIDATE"; break; fi
  done
fi
TASK_COLLECTOR_PY="${TASK_COLLECTOR_PY:-python3}"
"$TASK_COLLECTOR_PY" -c 'import numpy,cv2,speechbrain,faster_whisper,insightface' || { echo 'Use your existing diarization Python environment. See AUDITOR_COLLECTOR.md.'; exit 1; }
export PYTHONPATH="$TASK_COLLECTOR_DIR/.collector-deps${PYTHONPATH:+:$PYTHONPATH}"
if ! "$TASK_COLLECTOR_PY" -c 'import yt_dlp' >/dev/null 2>&1; then
  "$TASK_COLLECTOR_PY" -m pip install --target "$TASK_COLLECTOR_DIR/.collector-deps" -r "$TASK_COLLECTOR_DIR/requirements-collector.txt"
fi
TASK_COLLECTOR_LIBRARY="${AUDITOR_LIBRARY_DIR:-$TASK_COLLECTOR_DIR/auditor-library}"
exec "$TASK_COLLECTOR_PY" "$TASK_COLLECTOR_DIR/auditor_collector.py" --library-dir "$TASK_COLLECTOR_LIBRARY" --open-browser "$@"
