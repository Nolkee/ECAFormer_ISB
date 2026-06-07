#!/bin/bash
# Quick 8K iteration diagnostic training to test warmup hypothesis
# Usage: bash quick_test_warmup.sh

set -e

echo "========================================"
echo "Quick Warmup Test (8K iterations)"
echo "========================================"
echo ""
echo "Testing r43a_warmup config to verify:"
echo "  - Identity scale warmup prevents 3500-6000 instability"
echo "  - PSNR curve stays smooth throughout"
echo ""

CONFIG="Options/ISB_ecaformer_r43a_warmup.yml"

# Create temporary 8K config
TEMP_CONFIG="Options/ISB_ecaformer_r43a_warmup_8k.yml"
cp "$CONFIG" "$TEMP_CONFIG"

# Modify to 8K iterations
sed -i.bak 's/total_iter: 24000/total_iter: 8000/' "$TEMP_CONFIG"
sed -i.bak 's/periods: \[24000\]/periods: [8000]/' "$TEMP_CONFIG"
echo "✓ Created 8K iteration test config: $TEMP_CONFIG"

# Run training
echo ""
echo "Starting training..."
python -m basicsr.train --opt "$TEMP_CONFIG" 2>&1 | tee train_warmup_8k.log

echo ""
echo "========================================"
echo "Training completed!"
echo "========================================"
echo ""
echo "Next steps:"
echo "  1. Check train_warmup_8k.log for PSNR @ 5K-8K iters"
echo "  2. If stable (no drop), extend to 24K"
echo "  3. If unstable, hypothesis is wrong - try hybrid/learnable"
echo ""
echo "View training curve:"
echo "  tensorboard --logdir experiments/ISB_ecaformer_r43a_warmup_8k/tb_logger"
