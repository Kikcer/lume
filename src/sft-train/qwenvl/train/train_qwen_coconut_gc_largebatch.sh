#!/usr/bin/env bash
set -euo pipefail

# Large-batch launcher for latent-reasoning + manual gradient checkpointing.
# Override any variable inline, e.g.:
#   PER_DEVICE_BS=12 GRAD_ACC=8 bash src/sft-train/qwenvl/train/train_qwen_coconut_gc_largebatch.sh

NPROC_PER_NODE="${NPROC_PER_NODE:-8}"
CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0,1,2,3,4,5,6,7}"
TORCHRUN_BIN="${TORCHRUN_BIN:-torchrun}"
WORK_DIR="${WORK_DIR:-/home/guohaiyun/yangtianyu/UME-R1}"

MODEL_PATH="${MODEL_PATH:-/output/UME-R1-2B-Coconut-Fulldata-8node-2026-03-08-21-00-10/checkpoint-2862}"
ANNOTATION_PATH="${ANNOTATION_PATH:-/home/share/yty_data/UME_R1_train/UME-sft-train.jsonl}"
MEDIA_ROOT="${MEDIA_ROOT:-/home/share/yty_data/vlm2vec_train}"
SUBSET_FILTER="${SUBSET_FILTER:-InfographicsVQA}"
MAX_PIXELS="${MAX_PIXELS:-2359296}"                  # 28*28*576
MIN_PIXELS="${MIN_PIXELS:-768}"     
PER_DEVICE_BS="${PER_DEVICE_BS:-16}"
GRAD_ACC="${GRAD_ACC:-1}"
LR="${LR:-2e-6}"
EPOCHS="${EPOCHS:-3}"
MAX_LEN="${MAX_LEN:-12288}"
WARMUP_RATIO="${WARMUP_RATIO:-0.03}"
LR_SCHEDULER="${LR_SCHEDULER:-cosine}"
WEIGHT_DECAY="${WEIGHT_DECAY:-0}"

LORA_R="${LORA_R:-16}"
LORA_ALPHA="${LORA_ALPHA:-32}"
LORA_TARGET_MODULES="${LORA_TARGET_MODULES:-q_proj,v_proj,k_proj,o_proj,gate_proj,up_proj,down_proj}"

THINK_SEGMENTS="${THINK_SEGMENTS:-4}"
CT_PER_SEG="${CT_PER_SEG:-1}"

GEN_CONTRASTIVE_W="${GEN_CONTRASTIVE_W:-1.0}"
DISC_CONTRASTIVE_W="${DISC_CONTRASTIVE_W:-1.0}"
CONTRASTIVE_LOGIT_SCALE="${CONTRASTIVE_LOGIT_SCALE:-50.0}"
DEBUG_DISC_ORACLE_POS_FROM_QRY="${DEBUG_DISC_ORACLE_POS_FROM_QRY:-False}"

SAVE_STEPS="${SAVE_STEPS:-50}"
LOG_STEPS="${LOG_STEPS:-1}"
WANDB_MODE="${WANDB_MODE:-disabled}"
COCONUT_ENABLE_OOM_PRECHECK="${COCONUT_ENABLE_OOM_PRECHECK:-False}"
COCONUT_OOM_PRECHECK_BATCHES="${COCONUT_OOM_PRECHECK_BATCHES:-1}"

OUTPUT_DIR="${OUTPUT_DIR:-${WORK_DIR}/output/UME-R1-2B-Coconut-GC-Info-Continue-$(date +%Y-%m-%d-%H-%M-%S)}"

# NCCL stability knobs (aligned with multinode launcher defaults).
export NCCL_DEBUG="${NCCL_DEBUG:-WARN}"
export NCCL_ASYNC_ERROR_HANDLING="${NCCL_ASYNC_ERROR_HANDLING:-1}"
export TORCH_NCCL_BLOCKING_WAIT="${TORCH_NCCL_BLOCKING_WAIT:-1}"
export NCCL_TIMEOUT="${NCCL_TIMEOUT:-3600}"
export NCCL_NVLS_ENABLE="${NCCL_NVLS_ENABLE:-0}"
export NCCL_CUMEM_ENABLE="${NCCL_CUMEM_ENABLE:-0}"
export NCCL_MIN_NCHANNELS="${NCCL_MIN_NCHANNELS:-1}"
export NCCL_MAX_NCHANNELS="${NCCL_MAX_NCHANNELS:-4}"
export TORCH_DISTRIBUTED_DEBUG="${TORCH_DISTRIBUTED_DEBUG:-OFF}"

GLOBAL_BATCH=$(( PER_DEVICE_BS * GRAD_ACC * NPROC_PER_NODE ))
echo "[COCONUT-GC-LAUNCH] nproc=${NPROC_PER_NODE}, per_device_bs=${PER_DEVICE_BS}, grad_acc=${GRAD_ACC}, effective_global_batch=${GLOBAL_BATCH}"
echo "[COCONUT-GC-LAUNCH] output_dir=${OUTPUT_DIR}"
echo "[COCONUT-GC-LAUNCH] work_dir=${WORK_DIR}"
echo "[COCONUT-GC-LAUNCH] pwd(before cd)=$(pwd)"

cd "${WORK_DIR}"
mkdir -p "${OUTPUT_DIR}"
echo "[COCONUT-GC-LAUNCH] pwd(after cd)=$(pwd)"

CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES}" WANDB_MODE="${WANDB_MODE}" \
"${TORCHRUN_BIN}" --nproc_per_node="${NPROC_PER_NODE}" \
  src/sft-train/qwenvl/train/train_qwen_coconut_gc.py \
  --model_name_or_path "${MODEL_PATH}" \
  --output_dir "${OUTPUT_DIR}" \
  --attn_implementation flash_attention_2 \
  --bf16 \
  --learning_rate "${LR}" \
  --warmup_ratio "${WARMUP_RATIO}" \
  --lr_scheduler_type "${LR_SCHEDULER}" \
  --weight_decay "${WEIGHT_DECAY}" \
  --per_device_train_batch_size "${PER_DEVICE_BS}" \
  --gradient_accumulation_steps "${GRAD_ACC}" \
  --num_train_epochs "${EPOCHS}" \
  --save_steps "${SAVE_STEPS}" \
  --logging_steps "${LOG_STEPS}" \
  --model_max_length "${MAX_LEN}" \
  --gradient_checkpointing True \
  --use_lora False \
  --lora_r "${LORA_R}" \
  --lora_alpha "${LORA_ALPHA}" \
  --lora_use_dora False \
  --lora_target_modules "${LORA_TARGET_MODULES}" \
  --tune_mm_llm True \
  --tune_mm_mlp True \
  --max_pixels "${MAX_PIXELS}" \
  --min_pixels "${MIN_PIXELS}" \
  --tune_mm_vision False \
  --coconut_annotation_path "${ANNOTATION_PATH}" \
  --coconut_subset_filter "${SUBSET_FILTER}" \
  --coconut_media_root "${MEDIA_ROOT}" \
  --coconut_use_qry True \
  --coconut_use_pos True \
  --coconut_curriculum_stages 1.0 \
  --coconut_think_segments "${THINK_SEGMENTS}" \
  --coconut_ct_tokens_per_segment "${CT_PER_SEG}" \
  --coconut_include_gen_emb_loss True \
  --coconut_gen_contrastive_weight "${GEN_CONTRASTIVE_W}" \
  --coconut_disc_contrastive_weight "${DISC_CONTRASTIVE_W}" \
  --coconut_contrastive_logit_scale "${CONTRASTIVE_LOGIT_SCALE}" \
  --coconut_contrastive_cross_device True \
  --coconut_contrastive_local_loss True \
  --coconut_latent_answer_in_final_half False \
  --coconut_debug_disc_oracle_pos_from_qry "${DEBUG_DISC_ORACLE_POS_FROM_QRY}" \
  --coconut_oom_precheck_batches "${COCONUT_OOM_PRECHECK_BATCHES}" \
  --coconut_enable_oom_precheck "${COCONUT_ENABLE_OOM_PRECHECK}"
  # --coconut_force_reinit_all_tokens True


