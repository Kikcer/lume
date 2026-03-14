#!/usr/bin/env bash
set -euo pipefail

# ============================================================================
# Multi-node launcher for COCONUT latent-reasoning training (8 nodes × 8 GPUs)
#
# Usage:
#   On EACH node, run this script with the correct NODE_RANK:
#     NODE_RANK=0 bash src/sft-train/qwenvl/train/train_coconut_multinode.sh   # on 113 (master)
#     NODE_RANK=1 bash src/sft-train/qwenvl/train/train_coconut_multinode.sh   # on 114
#     ...
#     NODE_RANK=7 bash src/sft-train/qwenvl/train/train_coconut_multinode.sh   # on 127
#
#   Or use launch_multinode.sh to launch all 8 nodes from 113 via SSH.
# ============================================================================
# ---------- Cluster topology ----------
MASTER_ADDR="${MASTER_ADDR:-192.168.100.113}"
MASTER_PORT="${MASTER_PORT:-29500}"
NNODES="${NNODES:-4}"
NODE_RANK="${NODE_RANK:-0}"
NPROC_PER_NODE="${NPROC_PER_NODE:-8}"

# ---------- Paths (GPFS shared) ----------
WORK_DIR="/home/guohaiyun/yangtianyu/UME-R1"
MODEL_PATH="${MODEL_PATH:-/home/share/yty_model/UME-R1/2B/UME-R1/2B}"
ANNOTATION_PATH="${ANNOTATION_PATH:-/home/share/yty_data/UME_R1_train/UME-sft-train.jsonl}"
MEDIA_ROOT="${MEDIA_ROOT:-/home/share/yty_data/vlm2vec_train}"
# SUBSET_FILTER="${SUBSET_FILTER:-CIRR,MSCOCO_t2i,WebQA,ImageNet-1K,RefCOCO,InfographicsVQA}"  # empty = ALL datasets
SUBSET_FILTER="${SUBSET_FILTER:-InfographicsVQA}"
# ---------- Training hyperparams ----------
# 8 nodes × 8 GPUs = 64 GPUs
# effective_global_batch = 2 * 4 * 64 = 512
# contrastive_batch (per step) = 2 * 64 = 128
PER_DEVICE_BS="${PER_DEVICE_BS:-8}"
GRAD_ACC="${GRAD_ACC:-4}"
LR="${LR:-5e-5}"
EPOCHS="${EPOCHS:-5}"
MAX_LEN="${MAX_LEN:-12288}"
WARMUP_RATIO="${WARMUP_RATIO:-0.03}"
LR_SCHEDULER="${LR_SCHEDULER:-cosine}"
WEIGHT_DECAY="${WEIGHT_DECAY:-0}"
# Visual resolution controls (same semantics as DataArguments)
MAX_PIXELS="${MAX_PIXELS:-2359296}"                  # 28*28*576
MIN_PIXELS="${MIN_PIXELS:-768}"                  # 28*28*16 (match original)
VIDEO_MAX_FRAME_PIXELS="${VIDEO_MAX_FRAME_PIXELS:-2359296}"  # 32*28*28
VIDEO_MIN_FRAME_PIXELS="${VIDEO_MIN_FRAME_PIXELS:-784}"   # 4*28*28

LORA_R="${LORA_R:-16}"
LORA_ALPHA="${LORA_ALPHA:-32}"
LORA_TARGET_MODULES="${LORA_TARGET_MODULES:-q_proj,v_proj,k_proj,o_proj,gate_proj,up_proj,down_proj}"
LATENT_MOE_ENABLE="${LATENT_MOE_ENABLE:-True}"
LATENT_MOE_NUM_EXPERTS="${LATENT_MOE_NUM_EXPERTS:-4}"
LATENT_MOE_TOP_K="${LATENT_MOE_TOP_K:-2}"
LATENT_MOE_USE_SHARED_EXPERT="${LATENT_MOE_USE_SHARED_EXPERT:-True}"
LATENT_MOE_BALANCE_LOSS_WEIGHT="${LATENT_MOE_BALANCE_LOSS_WEIGHT:-0.1}"
LATENT_MOE_STEP_EMBED_MAX_STEPS="${LATENT_MOE_STEP_EMBED_MAX_STEPS:-32}"
LATENT_MOE_CONTEXT_TYPE="${LATENT_MOE_CONTEXT_TYPE:-disc}"

THINK_SEGMENTS="${THINK_SEGMENTS:-4}"
CT_PER_SEG="${CT_PER_SEG:-1}"
SAMPLING_STRATEGY="${SAMPLING_STRATEGY:-subset_balanced}"
FINAL_STAGE_PORTION="${FINAL_STAGE_PORTION:-0.5}"
LATENT_ANSWER_IN_FINAL_HALF="${LATENT_ANSWER_IN_FINAL_HALF:-True}"
FINAL_STAGE_ANSWER_PORTION="${FINAL_STAGE_ANSWER_PORTION:-0.5}"

GEN_CONTRASTIVE_W="${GEN_CONTRASTIVE_W:-1.0}"
DISC_CONTRASTIVE_W="${DISC_CONTRASTIVE_W:-1.0}"
CONTRASTIVE_LOGIT_SCALE="${CONTRASTIVE_LOGIT_SCALE:-50.0}"

SAVE_STEPS="${SAVE_STEPS:-200}"
LOG_STEPS="${LOG_STEPS:-10}"
WANDB_MODE="${WANDB_MODE:-disabled}"

OUTPUT_DIR="${OUTPUT_DIR:-output/UME-R1-2B-Coconut-FullData-8node-$(date +%Y-%m-%d-%H-%M-%S)}"

# ---------- NCCL tuning ----------
export NCCL_IB_DISABLE="${NCCL_IB_DISABLE:-1}"       # disable IB, use TCP socket (IB has connectivity issues across nodes)
export NCCL_SOCKET_IFNAME="${NCCL_SOCKET_IFNAME:-bond0}"  # TCP via bond0 (192.168.100.x)
export NCCL_DEBUG="${NCCL_DEBUG:-WARN}"
export NCCL_ASYNC_ERROR_HANDLING="${NCCL_ASYNC_ERROR_HANDLING:-1}"
export TORCH_NCCL_BLOCKING_WAIT="${TORCH_NCCL_BLOCKING_WAIT:-1}"
export NCCL_TIMEOUT="${NCCL_TIMEOUT:-3600}"
# H20 + NCCL 2.27 may occasionally hang on NVLS/CUMEM paths in large multi-node setups.
export NCCL_NVLS_ENABLE="${NCCL_NVLS_ENABLE:-0}"
export NCCL_CUMEM_ENABLE="${NCCL_CUMEM_ENABLE:-0}"
# Reduce channel fanout to lower socket/collective complexity for 64-rank bring-up.
export NCCL_MIN_NCHANNELS="${NCCL_MIN_NCHANNELS:-1}"
export NCCL_MAX_NCHANNELS="${NCCL_MAX_NCHANNELS:-4}"
# Multi-thread / multi-socket to reduce per-connection pressure over TCP
export NCCL_SOCKET_NTHREADS="${NCCL_SOCKET_NTHREADS:-4}"
export NCCL_NSOCKS_PERTHREAD="${NCCL_NSOCKS_PERTHREAD:-4}"
export NCCL_BUFFSIZE="${NCCL_BUFFSIZE:-8388608}"       # 8MB buffer, fewer small packets
export TORCH_DISTRIBUTED_DEBUG="${TORCH_DISTRIBUTED_DEBUG:-OFF}"
# Force Gloo (used by DDP for CPU collectives / monitoredBarrier) to use bond0 only
export GLOO_SOCKET_IFNAME="${GLOO_SOCKET_IFNAME:-bond0}"
export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0,1,2,3,4,5,6,7}"

# TCP keepalive: detect dead peers faster instead of hanging forever
sysctl -w net.ipv4.tcp_keepalive_time=60  2>/dev/null || true
sysctl -w net.ipv4.tcp_keepalive_intvl=10 2>/dev/null || true
sysctl -w net.ipv4.tcp_keepalive_probes=5 2>/dev/null || true

# ---------- Print info ----------
TOTAL_GPUS=$(( NNODES * NPROC_PER_NODE ))
GLOBAL_BATCH=$(( PER_DEVICE_BS * GRAD_ACC * TOTAL_GPUS ))
CONTRASTIVE_BATCH=$(( PER_DEVICE_BS * TOTAL_GPUS ))
echo "============================================="
echo "[MULTINODE] node_rank=${NODE_RANK}/${NNODES}, master=${MASTER_ADDR}:${MASTER_PORT}"
echo "[MULTINODE] total_gpus=${TOTAL_GPUS}, per_device_bs=${PER_DEVICE_BS}, grad_acc=${GRAD_ACC}"
echo "[MULTINODE] effective_global_batch=${GLOBAL_BATCH}, contrastive_batch=${CONTRASTIVE_BATCH}"
echo "[MULTINODE] pixels image=${MIN_PIXELS}~${MAX_PIXELS}, video_frame=${VIDEO_MIN_FRAME_PIXELS}~${VIDEO_MAX_FRAME_PIXELS}"
echo "[MULTINODE] latent_moe enable=${LATENT_MOE_ENABLE}, experts=${LATENT_MOE_NUM_EXPERTS}, top_k=${LATENT_MOE_TOP_K}, ctx=${LATENT_MOE_CONTEXT_TYPE}, balance_w=${LATENT_MOE_BALANCE_LOSS_WEIGHT}"
echo "[MULTINODE] output_dir=${OUTPUT_DIR}"
echo "============================================="

cd "${WORK_DIR}"

# Activate conda env (needed for non-interactive SSH sessions)
CONDA_BASE="${CONDA_BASE:-/home/guohaiyun/anaconda3}"
if [ -f "${CONDA_BASE}/etc/profile.d/conda.sh" ]; then
    source "${CONDA_BASE}/etc/profile.d/conda.sh"
    conda activate ume-r1
fi

WANDB_MODE="${WANDB_MODE}" \
torchrun \
  --nnodes="${NNODES}" \
  --nproc_per_node="${NPROC_PER_NODE}" \
  --node_rank="${NODE_RANK}" \
  --master_addr="${MASTER_ADDR}" \
  --master_port="${MASTER_PORT}" \
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
  --max_pixels "${MAX_PIXELS}" \
  --min_pixels "${MIN_PIXELS}" \
  --video_max_frame_pixels "${VIDEO_MAX_FRAME_PIXELS}" \
  --video_min_frame_pixels "${VIDEO_MIN_FRAME_PIXELS}" \
  --gradient_checkpointing True \
  --max_grad_norm 1 \
  --dataloader_num_workers 0 \
  --save_total_limit 10 \
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
  --coconut_annotation_path "${ANNOTATION_PATH}" \
  --coconut_subset_filter "${SUBSET_FILTER}" \
  --coconut_media_root "${MEDIA_ROOT}" \
  --coconut_use_qry True \
  --coconut_use_pos True \
  --coconut_sampling_strategy "${SAMPLING_STRATEGY}" \
  --coconut_curriculum_stages 0.25,0.5,0.75,1.0 \
  --coconut_final_stage_portion "${FINAL_STAGE_PORTION}" \
  --coconut_latent_answer_in_final_half "${LATENT_ANSWER_IN_FINAL_HALF}" \
  --coconut_final_stage_answer_portion "${FINAL_STAGE_ANSWER_PORTION}" \
  --coconut_think_segments "${THINK_SEGMENTS}" \
  --coconut_ct_tokens_per_segment "${CT_PER_SEG}" \
  --coconut_include_gen_emb_loss True \
  --coconut_gen_contrastive_weight "${GEN_CONTRASTIVE_W}" \
  --coconut_disc_contrastive_weight "${DISC_CONTRASTIVE_W}" \
  --coconut_contrastive_logit_scale "${CONTRASTIVE_LOGIT_SCALE}" \
  --coconut_contrastive_cross_device True \
  --coconut_contrastive_local_loss True \
  --coconut_oom_precheck_batches 1 \
  --coconut_enable_oom_precheck False \
  --coconut_oom_precheck_subsets "K700,Video-MME,YouCook2" \
  --coconut_oom_precheck_batches 2 \
  --ddp_timeout 3600 \
  ${RESUME_CKPT:+--resume_from_checkpoint "$RESUME_CKPT"}
# for node in 192.168.100.113 192.168.100.24 192.168.100.28 192.168.100.33 192.168.100.34 192.168.100.115 192.168.100.116 192.168.100.127; do
#   echo "Killing on $node..."
#   ssh "$node" "pkill -f train_qwen_coconut_gc.py; pkill -f torchrun" 2>/dev/null &
# done