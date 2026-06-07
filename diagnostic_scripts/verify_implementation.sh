#!/bin/bash
# 诊断实施验证清单

echo "========================================"
echo "R43a/R38c 不稳定性诊断 - 实施验证"
echo "========================================"
echo ""

# 1. 代码修改验证
echo "1. 代码修改验证"
echo "----------------------------------------"
python -m py_compile basicsr/models/archs/ECAFormer_ISB_arch.py && echo "  ✓ ECAFormer_ISB_arch.py 语法正确" || echo "  ✗ 语法错误"
python -m py_compile basicsr/models/image_isb_model.py && echo "  ✓ image_isb_model.py 语法正确" || echo "  ✗ 语法错误"
python -m py_compile tools/diagnose_checkpoint.py && echo "  ✓ diagnose_checkpoint.py 语法正确" || echo "  ✗ 语法错误"
echo ""

# 2. 配置文件验证
echo "2. 配置文件验证"
echo "----------------------------------------"
for config in Options/ISB_ecaformer_r43a_warmup.yml \
              Options/ISB_ecaformer_r43_hybrid.yml \
              Options/ISB_ecaformer_r43a_learnable.yml; do
    if [ -f "$config" ]; then
        echo "  ✓ $config"
    else
        echo "  ✗ 缺失: $config"
    fi
done
echo ""

# 3. 工具文件验证
echo "3. 工具文件验证"
echo "----------------------------------------"
for tool in tools/diagnose_checkpoint.py \
            quick_test_warmup.sh \
            DIAGNOSTIC_README.md; do
    if [ -f "$tool" ]; then
        echo "  ✓ $tool"
    else
        echo "  ✗ 缺失: $tool"
    fi
done
echo ""

# 4. 关键修改点检查
echo "4. 关键修改点检查"
echo "----------------------------------------"
if grep -q "identity_scale_warmup_iters" basicsr/models/archs/ECAFormer_ISB_arch.py; then
    echo "  ✓ identity_scale_warmup_iters 参数已添加"
else
    echo "  ✗ 缺失 warmup 参数"
fi

if grep -q "identity_scale_start" basicsr/models/archs/ECAFormer_ISB_arch.py; then
    echo "  ✓ warmup buffers 已创建"
else
    echo "  ✗ 缺失 warmup buffers"
fi

if grep -q "_current_iter" basicsr/models/image_isb_model.py; then
    echo "  ✓ current_iter 传递已实现"
else
    echo "  ✗ 缺失 current_iter 传递"
fi

if grep -q "current_scale = (1 - alpha)" basicsr/models/archs/ECAFormer_ISB_arch.py; then
    echo "  ✓ warmup 插值逻辑已实现"
else
    echo "  ✗ 缺失 warmup 逻辑"
fi
echo ""

# 5. 下一步提示
echo "========================================"
echo "实施验证完成！"
echo "========================================"
echo ""
echo "下一步行动:"
echo "  1. 运行快速测试: bash quick_test_warmup.sh"
echo "  2. 监控训练日志: tail -f train_warmup_8k.log"
echo "  3. 查看 TensorBoard: tensorboard --logdir experiments --port 6006"
echo "  4. 阅读详细说明: cat DIAGNOSTIC_README.md"
echo ""
