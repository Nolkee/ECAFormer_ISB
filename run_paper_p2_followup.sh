#!/bin/bash
# P2 follow-up: waits for the P1 queue to drain, then
#   (a) NFE sweep {2,4,8,16} on the recovered r54 best checkpoint (LOLv1)
#   (b) cross-domain re-probe with the recovered peak (completeness check)
#   (c) x1 endpoint ablation training (raw-LQ bridge boundary) — needed in
#       every paper branch (method AND analysis)
# Launch: nohup bash run_paper_p2_followup.sh > p2_followup.log 2>&1 &
set -o pipefail
cd "$(dirname "$0")"
PY=$HOME/anaconda3/envs/Retinexformer/bin/python

echo "waiting for P1 queue to drain..."
while pgrep -f "run_paper_p1_queue.sh" > /dev/null; do sleep 300; done
while pgrep -f "basicsr.train" > /dev/null; do sleep 300; done
echo "P1 queue drained at $(date)"

BEST=$(ls -t experiments/ISB_ecaformer_r54_repro_r52b/best_psnr_*.pth 2>/dev/null | head -1)
if [ -n "$BEST" ]; then
    echo "NFE sweep on $BEST"
    for NFE in 2 4 16; do
        "$PY" tools/eval_lol.py --opt Options/ISB_ecaformer_r54_repro_r52b.yml \
            --ckpt "$BEST" --param-key params_ema --inference-steps "$NFE" \
            --tag "r54_best_nfe${NFE}_lolv1" --out paper_pack/metrics/eval.csv 2>&1 | grep EVAL_RESULT
    done
    "$PY" tools/eval_lol.py --opt Options/ISB_ecaformer_r54_repro_r52b.yml \
        --ckpt "$BEST" --param-key params_ema \
        --tag r54_best_nfe8_lolv1 --out paper_pack/metrics/eval.csv 2>&1 | grep EVAL_RESULT
    "$PY" tools/eval_lol.py --opt Options/ISB_ecaformer_r54_repro_r52b.yml \
        --ckpt "$BEST" --param-key params_ema \
        --dataroot-lq data/LOLv2/Synthetic/Test/Low --dataroot-gt data/LOLv2/Synthetic/Test/Normal \
        --dataset-name LOLv2Syn_zeroshot --tag r54_best_zeroshot_lolv2syn \
        --out paper_pack/metrics/eval.csv 2>&1 | grep EVAL_RESULT
else
    echo "WARNING: no r54 best checkpoint found; skipping NFE sweep"
fi

echo "starting x1 endpoint ablation"
"$PY" -m basicsr.train --opt Options/ISB_ecaformer_r55_ablate_x1_rawlq.yml 2>&1 | tee train_ISB_ecaformer_r55_ablate_x1_rawlq.log
echo "P2 follow-up complete: $(date)"
