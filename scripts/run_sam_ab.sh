#!/usr/bin/env bash
# Does Sharpness-Aware Minimization earn its cost on this problem?
#
# HTR-VT, the architecture this project follows, uses SAM. This project never
# has: `use_sam` was False in every run including both stages of the shipped
# model. The argument for it is that flat minima transfer better out of
# distribution, and the distribution gap to real paper is this project's entire
# remaining error. The argument against is that it costs two forward/backward
# passes per step, so it has to beat twice as many ordinary steps.
#
# That is the comparison this script makes, and it is why the two arms run a
# different number of epochs rather than the same number. Equal epochs would
# flatter SAM by handing it twice the compute. Equal wall clock is the decision
# actually being made when the long run is configured.
#
# Both arms start from the stage A pretrain, not from the shipped checkpoint.
# Starting from a converged model would measure whether SAM helps a short
# fine-tune out of an already-sharp basin, which is not the question and would
# handicap the thing being tested.
#
# Detached and cache-only: nothing here needs the network, and a dropped link
# must not take the run with it.
set -u
cd "$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
export HF_HUB_OFFLINE=1 PYTHONPATH="$PWD"

PY=.venv/bin/python
INIT=runs/stage_a/best.pt

# A mixture in the same proportions as the full recipe but small enough that an
# epoch is about 25 minutes, so the cheaper arm gets eight measurements rather
# than two. Real Hebrew is NOT capped: there are only 10,219 lines of it, and it
# is the new variable, so every arm should see all of it.
COMMON=(
    --size large
    --init-from "$INIT"
    --train-limit 40000
    --val-limit 2000
    --glyph-lines 20000
    --real-hebrew all
    --real-ink all
    --real-ink-cap 8000
    --pixel-budget 24000
    --concat-prob 0.35
    --num-workers 10
    --lr 1.5e-4
    --ema 0.9995
    --seed 0
)

if [ ! -f "$INIT" ]; then
    echo "missing $INIT; the A/B starts from the stage A pretrain" >&2
    exit 1
fi

echo "[ab] arm A, no SAM, 8 epochs ($(date '+%F %T'))"
rm -rf runs/ab_nosam
$PY -m hebocr.train --out runs/ab_nosam --epochs 8 "${COMMON[@]}" \
    > runs/ab_nosam.log 2>&1
echo "[ab] arm A finished ($(date '+%F %T'))"

echo "[ab] arm B, SAM, 4 epochs ($(date '+%F %T'))"
rm -rf runs/ab_sam
$PY -m hebocr.train --out runs/ab_sam --epochs 4 --sam "${COMMON[@]}" \
    > runs/ab_sam.log 2>&1
echo "[ab] arm B finished ($(date '+%F %T'))"

echo "[ab] both arms done ($(date '+%F %T'))"
$PY scripts/compare_sam.py || true
