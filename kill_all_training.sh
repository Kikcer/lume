#!/bin/bash
NODES=("192.168.100.116" "192.168.100.114" "192.168.100.113""192.168.100.33" )  # 修改为你的节点名
for node in "${NODES[@]}"; do
    echo "Killing processes on $node..."
    ssh "guohaiyun@${node}" "pkill -9 -f train_qwen_coconut; pkill -9 torchrun" &
done
wait
echo "All training processes killed."
