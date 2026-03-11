#!/usr/bin/env bash
set -euo pipefail

# ============================================================================
# 监控训练进程，完成后自动启动评测
# Usage: bash monitor_and_eval.sh
# ============================================================================

NODES=("192.168.100.34" "192.168.100.114" "192.168.100.115" "192.168.100.116" "192.168.100.24" "192.168.100.28" "192.168.100.33" "192.168.100.127")
WORK_DIR="/home/guohaiyun/yangtianyu/UME-R1"
CONDA_ENV="ume-r1"
CONDA_BASE="/home/guohaiyun/anaconda3"

# 评测节点（选前三个）
EVAL_NODES=("${NODES[0]}" "${NODES[1]}" "${NODES[2]}")

CHECK_INTERVAL=60  # 检查间隔（秒）

# 检查节点GPU是否空闲（无训练进程）
check_gpu_idle() {
    local ip=$1
    # 检查是否有 python 进程占用 GPU
    local gpu_procs
    gpu_procs=$(ssh -o StrictHostKeyChecking=no -o ConnectTimeout=10 "guohaiyun@${ip}" \
        "nvidia-smi --query-compute-apps=pid --format=csv,noheader 2>/dev/null | wc -l" 2>/dev/null || echo "-1")

    if [[ "$gpu_procs" == "-1" ]]; then
        echo "error"
    elif [[ "$gpu_procs" == "0" ]]; then
        echo "idle"
    else
        echo "busy"
    fi
}

# 检查所有节点是否都空闲
all_nodes_idle() {
    for ip in "${NODES[@]}"; do
        status=$(check_gpu_idle "$ip")
        if [[ "$status" != "idle" ]]; then
            return 1
        fi
    done
    return 0
}

echo "============================================="
echo "[MONITOR] 开始监控训练进程..."
echo "[MONITOR] 检查间隔: ${CHECK_INTERVAL}秒"
echo "[MONITOR] 监控节点: ${NODES[*]}"
echo "[MONITOR] 评测节点: ${EVAL_NODES[*]}"
echo "============================================="

# 监控循环
while true; do
    echo ""
    echo "[MONITOR] $(date '+%Y-%m-%d %H:%M:%S') 检查各节点GPU状态..."

    all_idle=true
    for i in "${!NODES[@]}"; do
        ip="${NODES[$i]}"
        status=$(check_gpu_idle "$ip")
        echo "  节点${i} (${ip}): ${status}"
        if [[ "$status" != "idle" ]]; then
            all_idle=false
        fi
    done

    if $all_idle; then
        echo ""
        echo "[MONITOR] ✓ 所有节点GPU已空闲，训练完成！"
        break
    fi

    echo "[MONITOR] 训练仍在进行，${CHECK_INTERVAL}秒后再次检查..."
    sleep $CHECK_INTERVAL
done

echo ""
echo "============================================="
echo "[EVAL] 开始启动评测任务..."
echo "============================================="

# 评测指令
EVAL_CMDS=(
    "MODALITIES=image bash src/eval/VLM2Vec/experiments/public/eval/eval_coconut_all_modalities.sh"
    "MODALITIES=image DATASET_NAMES=\"TextVQA,VisDial,CIRR,VisualNews_t2i,VisualNews_i2t,MSCOCO_t2i,MSCOCO_i2t,NIGHTS,WebQA,FashionIQ,Wiki-SS-NQ,OVEN,EDIS,MSCOCO,RefCOCO,RefCOCO-Matching,Visual7W-Pointing\" bash src/eval/VLM2Vec/experiments/public/eval/eval_coconut_all_modalities.sh"
    "MODALITIES=visdoc bash src/eval/VLM2Vec/experiments/public/eval/eval_coconut_all_modalities.sh"
)

EVAL_NAMES=("image_all" "image_subset" "visdoc")

# 在三个节点上启动评测
for i in {0..2}; do
    ip="${EVAL_NODES[$i]}"
    cmd="${EVAL_CMDS[$i]}"
    name="${EVAL_NAMES[$i]}"
    log_file="${WORK_DIR}/output/eval_${name}_$(date +%Y%m%d_%H%M%S).log"

    echo ""
    echo "[EVAL] 节点${i} (${ip}): 启动 ${name} 评测..."
    echo "[EVAL] 日志: ${log_file}"

    ssh -f -o StrictHostKeyChecking=no "guohaiyun@${ip}" \
        "source ${CONDA_BASE}/etc/profile.d/conda.sh && \
         conda activate ${CONDA_ENV} && \
         cd ${WORK_DIR} && \
         nohup bash -c '${cmd}' > ${log_file} 2>&1 &"

    echo "[EVAL] ✓ ${name} 评测已启动"
done

echo ""
echo "============================================="
echo "[DONE] 所有评测任务已启动！"
echo "[DONE] 可通过以下命令查看日志:"
echo "  tail -f output/eval_*.log"
echo "============================================="
