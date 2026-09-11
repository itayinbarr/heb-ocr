#!/usr/bin/env bash
# Wait for stage B, then produce RESULTS.md. Detached, cache-only, no network.
set -u
cd "$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
export HF_HUB_OFFLINE=1 PYTHONPATH="$PWD"

echo "[eval] waiting for stage B to exist ($(date '+%F %T'))"
while [ ! -f runs/stage_b.log ]; do sleep 60; done
echo "[eval] stage B started, waiting for it to finish"
while pgrep -f "hebocr[.]train --out runs/stage_b" >/dev/null; do sleep 60; done

if [ ! -f runs/stage_b/best.pt ]; then
    echo "[eval] no stage B checkpoint; nothing to evaluate"
    exit 1
fi
echo "[eval] scoring ($(date '+%F %T'))"
.venv/bin/python scripts/make_results.py runs/stage_b/best.pt \
    --beam-width 12 --lm-weight 0.4 \
    --out RESULTS.md --json runs/results.json > runs/final_eval.log 2>&1
echo "[eval] done ($(date '+%F %T'))"
tail -20 runs/final_eval.log
