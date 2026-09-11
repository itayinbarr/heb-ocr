#!/usr/bin/env bash
# Wait for stage A to finish, then start stage B from its weights.
#
# Exists so the pipeline does not depend on anyone being connected when stage A
# ends. Detached with nohup and needing no network, it survives a dropped
# session, a closed laptop lid on the controlling machine, or a lost wifi link.
set -u
cd "$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"

# Cached datasets only. A network blip during the benchmark evaluation would
# otherwise stall a run that has no other reason to touch the network.
export HF_HUB_OFFLINE=1

echo "[chain] waiting for stage A ($(date '+%F %T'))"
while pgrep -f "hebocr[.]train --out runs/stage_a" >/dev/null; do sleep 60; done

if [ ! -f runs/stage_a/best.pt ]; then
    echo "[chain] stage A produced no checkpoint; not starting stage B"
    exit 1
fi
echo "[chain] stage A done, starting stage B ($(date '+%F %T'))"

rm -rf runs/stage_b
.venv/bin/python -m hebocr.train \
    --out runs/stage_b --arch htrvt --size large --epochs 10 \
    --pixel-budget 24000 --concat-prob 0.35 --num-workers 10 \
    --lr 1.5e-4 --seed 0 --glyph-lines 25000 --ema 0.9995 \
    --real-ink all --init-from runs/stage_a/best.pt \
    > runs/stage_b.log 2>&1

echo "[chain] stage B finished ($(date '+%F %T'))"
