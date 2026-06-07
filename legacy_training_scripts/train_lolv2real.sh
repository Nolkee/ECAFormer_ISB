#!/bin/bash
# LOLv2 Real: R38c (best LOLv1) on LOLv2 Real dataset
# Data path: data/LOLv2Real/

set -e

CONFIG="Options/ISB_ecaformer_r38c_lolv2real.yml"
name=$(basename "$CONFIG" .yml)

echo "=========================================="
echo "LOLv2 Real: R38c on LOLv2Real"
echo "=========================================="
echo "Training: $name"
echo "Config:   $CONFIG"
echo "Time:     $(date)"
echo "------------------------------------------"

python -m basicsr.train --opt "$CONFIG" 2>&1 | tee "train_${name}.log"

echo "Finished: $name at $(date)"
echo "=========================================="
