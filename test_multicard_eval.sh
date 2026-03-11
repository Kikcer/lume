#!/usr/bin/env bash
# 测试多卡评测的调试脚本

set -x  # 打印所有命令

export PYTHONUNBUFFERED=1
export NCCL_DEBUG=INFO  # 打开NCCL调试信息
export NCCL_DEBUG_SUBSYS=ALL

MODALITIES=image \
USE_COCONUT_LATENT_REASONING=False \
CHECKPOINT=/home/share/yty_model/UME-R1/2B/UME-R1/2B \
CUDA_VISIBLE_DEVICES=0,1 \
NPROC_PER_NODE=2 \
DISC_BATCH_SIZE=8 \
DATASET_NAMES=ImageNet-1K \
OUTPUT_BASEDIR=output/Eval/debug-multicard \
bash src/eval/VLM2Vec/experiments/public/eval/eval_coconut_all_modalities.sh
