#!/usr/bin/env bash
set -euo pipefail

# Peak-memory probe for the large-batch 7B ZeRO-3 setup.
# This does NOT train on real data. It builds a synthetic max-length batch and
# runs a single forward/backward pass to validate whether the current config OOMs.
#
# Example:
#   bash src/sft-train/qwenvl/train/probe_qwen_coconut_gc_largebatch_peak.sh
#   PER_DEVICE_BS=2 MAX_LEN=11288 bash src/sft-train/qwenvl/train/probe_qwen_coconut_gc_largebatch_peak.sh

NPROC_PER_NODE="${NPROC_PER_NODE:-8}"
CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0,1,2,3,4,5,6,7}"
TORCHRUN_BIN="${TORCHRUN_BIN:-torchrun}"
WORK_DIR="${WORK_DIR:-/home/guohaiyun/yangtianyu/UME-R1}"

MODEL_PATH="${MODEL_PATH:-/home/share/yty_model/UME-R1/7B}"
MAX_PIXELS="${MAX_PIXELS:-2359296}"
MIN_PIXELS="${MIN_PIXELS:-768}"
PER_DEVICE_BS="${PER_DEVICE_BS:-4}"
GRAD_ACC="${GRAD_ACC:-4}"
MAX_LEN="${MAX_LEN:-11288}"

LATENT_MOE_ENABLE="${LATENT_MOE_ENABLE:-True}"
LATENT_MOE_NUM_EXPERTS="${LATENT_MOE_NUM_EXPERTS:-4}"
LATENT_MOE_TOP_K="${LATENT_MOE_TOP_K:-2}"
LATENT_MOE_USE_SHARED_EXPERT="${LATENT_MOE_USE_SHARED_EXPERT:-True}"
LATENT_MOE_BALANCE_LOSS_WEIGHT="${LATENT_MOE_BALANCE_LOSS_WEIGHT:-0.1}"
LATENT_MOE_STEP_EMBED_MAX_STEPS="${LATENT_MOE_STEP_EMBED_MAX_STEPS:-32}"
LATENT_MOE_CONTEXT_TYPE="${LATENT_MOE_CONTEXT_TYPE:-none}"

THINK_SEGMENTS="${THINK_SEGMENTS:-4}"
CT_PER_SEG="${CT_PER_SEG:-1}"
GEN_CONTRASTIVE_W="${GEN_CONTRASTIVE_W:-1.0}"
DISC_CONTRASTIVE_W="${DISC_CONTRASTIVE_W:-1.0}"
CONTRASTIVE_LOGIT_SCALE="${CONTRASTIVE_LOGIT_SCALE:-50.0}"

OUTPUT_DIR="${OUTPUT_DIR:-${WORK_DIR}/output/test/peak-probe-$(date +%Y-%m-%d-%H-%M-%S)}"
USE_DEEPSPEED="${USE_DEEPSPEED:-1}"
DEEPSPEED_CFG="${DEEPSPEED_CFG:-${WORK_DIR}/src/sft-train/scripts/zero3.json}"
PROBE_VISION_MODE="${PROBE_VISION_MODE:-both}"
PROBE_VIDEO_FRAMES="${PROBE_VIDEO_FRAMES:-4}"

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
echo "[PEAK-PROBE-LAUNCH] nproc=${NPROC_PER_NODE}, per_device_bs=${PER_DEVICE_BS}, grad_acc=${GRAD_ACC}, effective_global_batch=${GLOBAL_BATCH}"
echo "[PEAK-PROBE-LAUNCH] model=${MODEL_PATH}, max_len=${MAX_LEN}, max_pixels=${MAX_PIXELS}, vision_mode=${PROBE_VISION_MODE}"
echo "[PEAK-PROBE-LAUNCH] latent_moe enable=${LATENT_MOE_ENABLE}, experts=${LATENT_MOE_NUM_EXPERTS}, top_k=${LATENT_MOE_TOP_K}, ctx=${LATENT_MOE_CONTEXT_TYPE}"
echo "[PEAK-PROBE-LAUNCH] deepspeed use=${USE_DEEPSPEED}, cfg=${DEEPSPEED_CFG}"
echo "[PEAK-PROBE-LAUNCH] output_dir=${OUTPUT_DIR}"

cd "${WORK_DIR}"
mkdir -p "${OUTPUT_DIR}"

DEEPSPEED_ARGS=()
case "${USE_DEEPSPEED,,}" in
  1|true|yes|on)
    DEEPSPEED_ARGS=(--deepspeed "${DEEPSPEED_CFG}")
    ;;
esac

CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES}" \
"${TORCHRUN_BIN}" --nproc_per_node="${NPROC_PER_NODE}" \
  src/sft-train/qwenvl/train/probe_coconut_peak_mem.py \
  "${DEEPSPEED_ARGS[@]}" \
  --model_name_or_path "${MODEL_PATH}" \
  --output_dir "${OUTPUT_DIR}" \
  --attn_implementation flash_attention_2 \
  --bf16 \
  --per_device_train_batch_size "${PER_DEVICE_BS}" \
  --gradient_accumulation_steps "${GRAD_ACC}" \
  --logging_steps 1 \
  --save_steps 999999 \
  --model_max_length "${MAX_LEN}" \
  --gradient_checkpointing True \
  --use_lora False \
  --latent_moe_enable "${LATENT_MOE_ENABLE}" \
  --latent_moe_num_experts "${LATENT_MOE_NUM_EXPERTS}" \
  --latent_moe_top_k "${LATENT_MOE_TOP_K}" \
  --latent_moe_use_shared_expert "${LATENT_MOE_USE_SHARED_EXPERT}" \
  --latent_moe_balance_loss_weight "${LATENT_MOE_BALANCE_LOSS_WEIGHT}" \
  --latent_moe_step_embed_max_steps "${LATENT_MOE_STEP_EMBED_MAX_STEPS}" \
  --latent_moe_context_type "${LATENT_MOE_CONTEXT_TYPE}" \
  --tune_mm_llm True \
  --tune_mm_mlp True \
  --tune_mm_vision False \
  --max_pixels "${MAX_PIXELS}" \
  --min_pixels "${MIN_PIXELS}" \
  --coconut_curriculum_stages 0 \
  --coconut_think_segments "${THINK_SEGMENTS}" \
  --coconut_ct_tokens_per_segment "${CT_PER_SEG}" \
  --coconut_include_gen_emb_loss True \
  --coconut_gen_contrastive_weight "${GEN_CONTRASTIVE_W}" \
  --coconut_disc_contrastive_weight "${DISC_CONTRASTIVE_W}" \
  --coconut_contrastive_logit_scale "${CONTRASTIVE_LOGIT_SCALE}" \
  --coconut_contrastive_cross_device True \
  --coconut_contrastive_local_loss True \
  --probe_vision_mode "${PROBE_VISION_MODE}" \
  --probe_video_frames "${PROBE_VIDEO_FRAMES}"
