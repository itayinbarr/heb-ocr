#!/bin/bash
# Launch a training run fully detached from the calling shell.
#
# Two failures this guards against, both of which cost hours:
#
#   nohup only blocks SIGHUP. The process stays in the caller's process group,
#   so a session teardown still reaches it. setsid puts it in its own session,
#   where SID equals its own PID and nothing upstream can signal it.
#
#   A launch that dies on an argparse error writes one lowercase line and
#   exits 2. A log that looks quiet is not the same as a run that is healthy,
#   so this checks explicitly instead of assuming.
#
# Usage: scripts/launch_training.sh <logfile> [train.py args...]
set -euo pipefail

LOG="$1"; shift
REPO="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$REPO"

rm -f "$LOG"
setsid "$REPO/.venv/bin/python" -m hebocr.train "$@" > "$LOG" 2>&1 < /dev/null &
sleep 20

if grep -iqE "traceback|error:|invalid choice|usage:|unrecognized arguments" "$LOG"; then
    echo "LAUNCH FAILED:" >&2
    grep -iE "traceback|error:|invalid choice|usage:|unrecognized" "$LOG" | head -5 >&2
    exit 1
fi

PID="$(pgrep -f "hebocr[.]t""rain.*${LOG##*/}" | head -1 || true)"
[ -z "$PID" ] && PID="$(pgrep -f "hebocr[.]t""rain" | head -1 || true)"
if [ -z "$PID" ]; then
    echo "LAUNCH FAILED: no training process found" >&2
    tail -5 "$LOG" >&2
    exit 1
fi

echo "running detached: PID=$PID SID=$(ps -o sid= -p "$PID" | tr -d ' ')"
tail -2 "$LOG"
