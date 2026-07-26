#!/bin/bash
# P3 queue (2026-07-26) — everything the analysis paper still needs from the GPU.
# Chains behind whatever is running now (the LOLv2-Real no-early-stop control).
#
# 1. NFE sweep {2,4,8,16} on the r54 best EMA — the efficiency/accuracy trade
#    curve. Inference only, ~10 min total.
# 2. r55 x1 endpoint ablation (raw-LQ bridge boundary) — the mechanism ablation:
#    does the illumination-lifted endpoint drive the phase transition, or would
#    a raw-LQ endpoint behave the same? ~15 h.
#
# Launch: nohup bash run_paper_p3_queue.sh > p3_queue.log 2>&1 &
set -o pipefail
cd "$(dirname "$0")"
PY=$HOME/anaconda3/envs/Retinexformer/bin/python

echo "waiting for the GPU to drain..."
while pgrep -f "basicsr\.train" > /dev/null; do sleep 120; done
echo "GPU free at $(date '+%F %T')"

R54_OPT=Options/ISB_ecaformer_r54_repro_r52b.yml
R54_BEST=experiments/ISB_ecaformer_r54_repro_r52b/best_psnr_21.87_21000.pth

if [ -f "$R54_BEST" ]; then
    echo "=== NFE sweep on $R54_BEST ==="
    for NFE in 2 4 8 16; do
        "$PY" tools/eval_lol.py --opt "$R54_OPT" --ckpt "$R54_BEST" \
            --param-key params_ema --inference-steps "$NFE" \
            --tag "r54_best_nfe${NFE}_lolv1" --out paper_pack/metrics/eval.csv \
            2>&1 | grep -E "EVAL_RESULT|Traceback"
    done
else
    echo "WARNING: $R54_BEST missing, skipping NFE sweep"
fi

echo "=== r55 x1 endpoint ablation ==="
"$PY" -m basicsr.train --opt Options/ISB_ecaformer_r55_ablate_x1_rawlq.yml \
    2>&1 | tee train_ISB_ecaformer_r55_ablate_x1_rawlq.log

echo "P3 queue complete: $(date '+%F %T')"
