#!/bin/bash
# Custom Overfit 训练脚本

# 设置环境变量，忽略 DeprecationWarning
export PYTHONWARNINGS="ignore::DeprecationWarning"

# 指定使用空闲的 GPU（避免与其他进程冲突）
# 如果 GPU 0 已被占用，使用 GPU 1
export CUDA_VISIBLE_DEVICES=1

# 训练配置
LOG_DIR="./logs/custom_overfit/$(date +%Y%m%d_%H%M%S)"
CONFIG_PATH="./configs/custom_overfit.yaml"

echo "=========================================="
echo "启动 Custom Overfit 训练"
echo "=========================================="
echo "配置文件: $CONFIG_PATH"
echo "日志目录: $LOG_DIR"
echo "使用 GPU: $CUDA_VISIBLE_DEVICES"
echo "=========================================="

# 运行训练
# 注意：使用 python -m scripts.run 而不是 python scripts/run.py
# 原因：
# 1. -m 方式更可靠，自动将项目根目录加入 Python 路径
# 2. 避免 PYTHONPATH 配置问题
# 3. 与项目的包结构更一致
python -m scripts.run \
    "$LOG_DIR" \
    --config-path "$CONFIG_PATH" \
    --wandb-offline \
    "$@"

echo ""
echo "=========================================="
echo "训练脚本已启动"
echo "=========================================="
echo "提示："
echo "  - 日志保存在: $LOG_DIR"
echo "  - 检查点保存在: $LOG_DIR/ckpts/"
echo "  - 可视化结果在: $LOG_DIR/vis/"
echo "=========================================="
echo ""
echo "如果遇到 ModuleNotFoundError 错误，可以尝试："
echo "  1. 安装项目: pip install -e ."
echo "  2. 或者使用: python scripts/run.py 代替 python -m scripts.run"
echo "=========================================="
