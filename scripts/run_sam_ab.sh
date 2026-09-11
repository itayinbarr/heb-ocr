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
# epoch is about 20 minutes, so the cheaper arm gets eight measurements rather
# than two and the whole comparison fits in a working day. Measured on this card
# at this budget: 2.5 iterations a second, so roughly 3,000 steps an epoch
# without SAM and half the rate with it.
#
# Real Hebrew is NOT capped. There are only 10,219 lines of it, it is the new
# variable, and capping the one source being introduced to make room for the
# ones already known to work would answer a question nobody asked.
COMMON=(
    --size large
    --init-from "$INIT"
    --train-limit 20000
    --val-limit 2000
    --glyph-lines 10000
    --real-hebrew all
    --real-ink all
    --real-ink-cap 4000
    # 22000 rather than the 24000 the shipped run used: a couple of hundred
    # megabytes of headroom is cheap, and losing a multi-day run at hour thirty
    # to a transient allocation is not.
    --pixel-budget 22000
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

# Refuse to start if something else holds the card. The first attempt at this
# A/B died forty seconds in because mogli-asr.service is set to restart and had
# come back between freeing the GPU and launching, leaving 1.1 GB gone and the
# pixel budget, which was tuned against a free card, no longer affordable. An
# out-of-memory error an hour into a run reads like a bad hyperparameter; it is
# cheaper to refuse up front and say what is holding the memory.
BUSY="$(nvidia-smi --query-compute-apps=pid,used_memory,process_name --format=csv,noheader 2>/dev/null)"
if [ -n "$BUSY" ]; then
    echo "the GPU is not free, refusing to start:" >&2
    echo "$BUSY" >&2
    echo "stop those first (mogli-asr.service and llama-server are the usual two)" >&2
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
