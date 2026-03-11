#!/bin/bash

PASSWORD="Z3VvaGFpeXVuCg=="
SERVERS=(17 24 28 33 34 113 114 115 116 127)
AVAILABLE=()

for x in "${SERVERS[@]}"; do
    HOST="192.168.100.${x}"
    echo "========== 检查 ${HOST} =========="

    # SSH 连接并获取 GPU 使用情况
    GPU_INFO=$(sshpass -p "${PASSWORD}" ssh -o StrictHostKeyChecking=no -o ConnectTimeout=5 guohaiyun@${HOST} \
        "nvidia-smi --query-gpu=index,name,memory.used,memory.total,utilization.gpu --format=csv,noheader,nounits" 2>/dev/null)

    if [ $? -ne 0 ] || [ -z "${GPU_INFO}" ]; then
        echo "  ❌ 无法连接或无法获取 GPU 信息"
        continue
    fi

    HAS_FREE=false
    while IFS= read -r line; do
        # 解析: index, name, memory_used(MiB), memory_total(MiB), gpu_util(%)
        IDX=$(echo "$line" | awk -F', ' '{print $1}')
        NAME=$(echo "$line" | awk -F', ' '{print $2}')
        MEM_USED=$(echo "$line" | awk -F', ' '{print $3}')
        MEM_TOTAL=$(echo "$line" | awk -F', ' '{print $4}')
        GPU_UTIL=$(echo "$line" | awk -F', ' '{print $5}')

        # 判断空闲: 显存使用 < 500MiB 且 GPU利用率 < 5%
        if [ "${MEM_USED}" -lt 500 ] 2>/dev/null && [ "${GPU_UTIL}" -lt 5 ] 2>/dev/null; then
            echo "  ✅ GPU ${IDX} (${NAME}) 空闲 — 显存 ${MEM_USED}/${MEM_TOTAL} MiB, 利用率 ${GPU_UTIL}%"
            HAS_FREE=true
        else
            echo "  🔴 GPU ${IDX} (${NAME}) 占用 — 显存 ${MEM_USED}/${MEM_TOTAL} MiB, 利用率 ${GPU_UTIL}%"
        fi
    done <<< "${GPU_INFO}"

    if [ "${HAS_FREE}" = true ]; then
        AVAILABLE+=("${HOST}")
    fi
done

echo ""
echo "======================================"
echo "📊 检查完毕，结果汇总："
echo "======================================"
if [ ${#AVAILABLE[@]} -eq 0 ]; then
    echo "😢 所有服务器均无空闲 GPU"
else
    echo "🎉 以下服务器有空闲 GPU："
    for srv in "${AVAILABLE[@]}"; do
        echo "  → ${srv}"
    done
fi
