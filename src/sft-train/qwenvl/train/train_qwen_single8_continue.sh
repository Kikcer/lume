#!/usr/bin/env bash
set -euo pipefail

# Single-node (8 GPU) SFT continuation launcher ("方式B": new run from existing weights)
# Usage:
#   bash src/sft-train/qwenvl/train/train_qwen_single8_continue.sh
# Override example:
#   MODEL_PATH=/path/to/checkpoint DATASETS=mmeb_v2_group MAX_STEPS=3000 bash src/sft-train/qwenvl/train/train_qwen_single8_continue.sh

PREFIX="${PREFIX:-/home/guohaiyun/yangtianyu}"
WORK_DIR="${WORK_DIR:-${PREFIX}/UME-R1}"

# Base model / checkpoint to continue from
MODEL_PATH="${MODEL_PATH:-/home/share/yty_model/UME-R1/2B/UME-R1/2B}"

RUN_NAME="${RUN_NAME:-UME-2B-continue-1node8gpu-$(date +%Y-%m-%d-%H-%M-%S)}"
OUTPUT_DIR="${OUTPUT_DIR:-${WORK_DIR}/output/${RUN_NAME}}"
LOG_DIR="${LOG_DIR:-${WORK_DIR}/log}"
LOG_FILE="${LOG_FILE:-${LOG_DIR}/${RUN_NAME}.log}"

DEEPSPEED_CFG="${DEEPSPEED_CFG:-${WORK_DIR}/src/sft-train/scripts/zero3.json}"
ENTRY_FILE="${ENTRY_FILE:-${WORK_DIR}/src/sft-train/qwenvl/train/train_qwen.py}"

# Keep data root consistent with train_qwen_coconut_gc_largebatch.sh
ANNOTATION_PATH="${ANNOTATION_PATH:-/home/share/yty_data/UME_R1_train/UME-sft-train.jsonl}"
MEDIA_ROOT="${MEDIA_ROOT:-/home/share/yty_data/vlm2vec_train}"
BLANK_IMAGE_PATH="${BLANK_IMAGE_PATH:-/home/share/yty_data/UME_R1_train/images/blank.jpg}"

# open_r1 is required by qwenvl/train/trainer.py
R1_SRC_PATH="${R1_SRC_PATH:-${WORK_DIR}/src/r1-train/src}"
export PYTHONPATH="${R1_SRC_PATH}:${PYTHONPATH:-}"

# Data / train hyperparameters
DATASETS="${DATASETS:-mmeb_v2_group}"
LR="${LR:-1e-5}"
BATCH_SIZE="${BATCH_SIZE:-4}"
GRAD_ACC_STEPS="${GRAD_ACC_STEPS:-4}"
MAX_STEPS="${MAX_STEPS:-2000}"
SAVE_STEPS="${SAVE_STEPS:-500}"
LOGGING_STEPS="${LOGGING_STEPS:-10}"
MAX_LEN="${MAX_LEN:-12288}"
MAX_PIXELS="${MAX_PIXELS:-2359296}"
MIN_PIXELS="${MIN_PIXELS:-768}"

NPROC_PER_NODE="${NPROC_PER_NODE:-8}"
CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0,1,2,3,4,5,6,7}"
TORCHRUN_BIN="${TORCHRUN_BIN:-torchrun}"

mkdir -p "${OUTPUT_DIR}" "${LOG_DIR}"
cd "${WORK_DIR}"

# Build MMEB-train links expected by data_qwen.py (consistent with coconut data roots)
mkdir -p "MMEB-train/v2" "MMEB-train/images"
ln -sfn "${ANNOTATION_PATH}" "MMEB-train/v2/UME-R1-SFT.json"

for p in "${MEDIA_ROOT}/MMEB-train"/*; do
  b="$(basename "$p")"
  [ "$b" = "v2" ] && continue
  [ "$b" = "images" ] && continue
  ln -sfn "$p" "MMEB-train/$b"
done

for p in "${MEDIA_ROOT}/MMEB-train/images"/*; do
  b="$(basename "$p")"
  ln -sfn "$p" "MMEB-train/images/$b"
done

ln -sfn "${BLANK_IMAGE_PATH}" "MMEB-train/images/blank.jpg"
ln -sfn . "MMEB-train/MMEB-train"

echo "[SFT-CONTINUE] work_dir=${WORK_DIR}"
echo "[SFT-CONTINUE] model_path=${MODEL_PATH}"
echo "[SFT-CONTINUE] annotation_path=${ANNOTATION_PATH}"
echo "[SFT-CONTINUE] media_root=${MEDIA_ROOT}"
echo "[SFT-CONTINUE] output_dir=${OUTPUT_DIR}"
echo "[SFT-CONTINUE] log_file=${LOG_FILE}"
echo "[SFT-CONTINUE] nproc_per_node=${NPROC_PER_NODE}, batch=${BATCH_SIZE}, grad_acc=${GRAD_ACC_STEPS}, max_steps=${MAX_STEPS}"

CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES}" WANDB_MODE="disabled" \
"${TORCHRUN_BIN}" --standalone --nproc_per_node="${NPROC_PER_NODE}" \
  "${ENTRY_FILE}" \
  --deepspeed "${DEEPSPEED_CFG}" \
  --model_name_or_path "${MODEL_PATH}" \
  --dataset_use "${DATASETS}" \
  --data_flatten False \
  --tune_mm_vision False \
  --tune_mm_mlp True \
  --tune_mm_llm True \
  --output_dir "${OUTPUT_DIR}" \
  --max_steps "${MAX_STEPS}" \
  --data_group True \
  --bf16 \
  --per_device_train_batch_size "${BATCH_SIZE}" \
  --per_device_eval_batch_size "$((BATCH_SIZE*2))" \
  --gradient_accumulation_steps "${GRAD_ACC_STEPS}" \
  --max_pixels "${MAX_PIXELS}" \
  --min_pixels "${MIN_PIXELS}" \
  --eval_strategy no \
  --save_strategy steps \
  --save_steps "${SAVE_STEPS}" \
  --save_total_limit 10 \
  --learning_rate "${LR}" \
  --weight_decay 0 \
  --warmup_ratio 0.03 \
  --max_grad_norm 1 \
  --lr_scheduler_type cosine \
  --logging_steps "${LOGGING_STEPS}" \
  --model_max_length "${MAX_LEN}" \
  --gradient_checkpointing True \
  --dataloader_num_workers 4 \
  --run_name "${RUN_NAME}" \
  --report_to none \
  >> "${LOG_FILE}" 2>&1

echo "[SFT-CONTINUE] done. log: ${LOG_FILE}"