#!/usr/bin/env python3
"""统计训练集每个子数据集的 CoT 和 Answer 平均长度"""

import json
import re
from collections import defaultdict
from pathlib import Path

def extract_cot_and_answer(text: str):
    """从 gpt 回复中提取 <think>...</think> 和之后的 answer 部分"""
    # 提取 CoT (think 标签内的内容)
    think_match = re.search(r'<think>(.*?)</think>', text, re.DOTALL)
    cot = think_match.group(1).strip() if think_match else ""

    # 提取 Answer (</think> 之后到 <gen_emb> 之前的内容)
    answer_match = re.search(r'</think>(.*?)(?:<gen_emb>|$)', text, re.DOTALL)
    answer = answer_match.group(1).strip() if answer_match else ""

    return cot, answer

def main():
    data_path = "/home/share/yty_data/UME_R1_train/UME-sft-train.jsonl"

    # 统计数据结构: {dataset_name: {"qry_cot": [], "qry_ans": [], "pos_cot": [], "pos_ans": []}}
    stats = defaultdict(lambda: {"qry_cot": [], "qry_ans": [], "pos_cot": [], "pos_ans": []})

    print(f"Reading {data_path}...")
    with open(data_path, 'r', encoding='utf-8') as f:
        for line_num, line in enumerate(f, 1):
            if not line.strip():
                continue
            try:
                data = json.loads(line)
            except json.JSONDecodeError as e:
                print(f"Line {line_num}: JSON decode error: {e}")
                continue

            dataset_name = data.get("dataset_name", "unknown")

            # 处理 qry
            if "qry" in data and "conversations" in data["qry"]:
                for conv in data["qry"]["conversations"]:
                    if conv.get("from") == "gpt":
                        cot, answer = extract_cot_and_answer(conv["value"])
                        if cot:
                            stats[dataset_name]["qry_cot"].append(len(cot))
                        if answer:
                            stats[dataset_name]["qry_ans"].append(len(answer))

            # 处理 pos
            if "pos" in data and "conversations" in data["pos"]:
                for conv in data["pos"]["conversations"]:
                    if conv.get("from") == "gpt":
                        cot, answer = extract_cot_and_answer(conv["value"])
                        if cot:
                            stats[dataset_name]["pos_cot"].append(len(cot))
                        if answer:
                            stats[dataset_name]["pos_ans"].append(len(answer))

            if line_num % 50000 == 0:
                print(f"Processed {line_num} lines...")

    # 输出统计结果
    print("\n" + "=" * 120)
    print(f"{'Dataset':<35} {'Count':>8} {'QryCOT':>10} {'QryAns':>10} {'PosCOT':>10} {'PosAns':>10}")
    print("=" * 120)

    total_count = 0
    all_qry_cot, all_qry_ans, all_pos_cot, all_pos_ans = [], [], [], []

    for dataset_name in sorted(stats.keys()):
        s = stats[dataset_name]
        count = max(len(s["qry_cot"]), len(s["pos_cot"]), 1)

        qry_cot_avg = sum(s["qry_cot"]) / len(s["qry_cot"]) if s["qry_cot"] else 0
        qry_ans_avg = sum(s["qry_ans"]) / len(s["qry_ans"]) if s["qry_ans"] else 0
        pos_cot_avg = sum(s["pos_cot"]) / len(s["pos_cot"]) if s["pos_cot"] else 0
        pos_ans_avg = sum(s["pos_ans"]) / len(s["pos_ans"]) if s["pos_ans"] else 0

        print(f"{dataset_name:<35} {count:>8} {qry_cot_avg:>10.1f} {qry_ans_avg:>10.1f} {pos_cot_avg:>10.1f} {pos_ans_avg:>10.1f}")

        total_count += count
        all_qry_cot.extend(s["qry_cot"])
        all_qry_ans.extend(s["qry_ans"])
        all_pos_cot.extend(s["pos_cot"])
        all_pos_ans.extend(s["pos_ans"])

    print("=" * 120)
    overall_qry_cot = sum(all_qry_cot) / len(all_qry_cot) if all_qry_cot else 0
    overall_qry_ans = sum(all_qry_ans) / len(all_qry_ans) if all_qry_ans else 0
    overall_pos_cot = sum(all_pos_cot) / len(all_pos_cot) if all_pos_cot else 0
    overall_pos_ans = sum(all_pos_ans) / len(all_pos_ans) if all_pos_ans else 0
    print(f"{'TOTAL':<35} {total_count:>8} {overall_qry_cot:>10.1f} {overall_qry_ans:>10.1f} {overall_pos_cot:>10.1f} {overall_pos_ans:>10.1f}")
    print("\n(长度单位: 字符数)")

    # 保存结果到 JSON 文件
    output_path = Path(__file__).parent / "cot_length_stats.json"
    results = {}
    for dataset_name in sorted(stats.keys()):
        s = stats[dataset_name]
        results[dataset_name] = {
            "count": max(len(s["qry_cot"]), len(s["pos_cot"]), 1),
            "qry_cot_avg": round(sum(s["qry_cot"]) / len(s["qry_cot"]), 1) if s["qry_cot"] else 0,
            "qry_ans_avg": round(sum(s["qry_ans"]) / len(s["qry_ans"]), 1) if s["qry_ans"] else 0,
            "pos_cot_avg": round(sum(s["pos_cot"]) / len(s["pos_cot"]), 1) if s["pos_cot"] else 0,
            "pos_ans_avg": round(sum(s["pos_ans"]) / len(s["pos_ans"]), 1) if s["pos_ans"] else 0,
        }
    results["TOTAL"] = {
        "count": total_count,
        "qry_cot_avg": round(overall_qry_cot, 1),
        "qry_ans_avg": round(overall_qry_ans, 1),
        "pos_cot_avg": round(overall_pos_cot, 1),
        "pos_ans_avg": round(overall_pos_ans, 1),
    }
    with open(output_path, 'w', encoding='utf-8') as f:
        json.dump(results, f, indent=2, ensure_ascii=False)
    print(f"\n结果已保存到: {output_path}")

if __name__ == "__main__":
    main()
