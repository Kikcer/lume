#!/usr/bin/env python3
"""
从训练日志中提取并绘制 loss 曲线。

支持曲线：
- total loss (`loss`)
- ce loss (`ce_loss`)
- gen contrastive loss (`gen_contrastive_loss`)
- disc contrastive loss (`disc_contrastive_loss`)

用法示例：
python src/sft-train/qwenvl/train/plot_train_losses.py \
  --log_path output/XXX/train_rank0.log

python src/sft-train/qwenvl/train/plot_train_losses.py \
  --log_path output/UME-R1-2B-Coconut-fewData-4node-2026-03-02-18-06-56/train_rank0.log \
  --output output/UME-R1-2B-Coconut-fewData-4node-2026-03-02-18-06-56/loss_curves.png \
"""

from __future__ import annotations

import argparse
import ast
import re
from pathlib import Path
from typing import Dict, List

import matplotlib.pyplot as plt


def moving_average(values: List[float], window: int) -> List[float]:
    if window <= 1:
        return list(values)
    out: List[float] = []
    q: List[float] = []
    running = 0.0
    for v in values:
        q.append(v)
        running += v
        if len(q) > window:
            running -= q.pop(0)
        out.append(running / len(q))
    return out


def parse_log_dicts(log_path: Path) -> List[Dict]:
    """提取日志中的 Python 字典行并解析。"""
    pattern = re.compile(r"\{.*?\}")
    parsed: List[Dict] = []

    with log_path.open("r", encoding="utf-8", errors="ignore") as f:
        for raw_line in f:
            line = raw_line.strip()
            if "{" not in line or "}" not in line:
                continue
            for block in pattern.findall(line):
                try:
                    obj = ast.literal_eval(block)
                except Exception:
                    continue
                if not isinstance(obj, dict):
                    continue
                # 只保留训练损失日志（避免把其他不相关 dict 混进来）
                if "loss" not in obj:
                    continue
                parsed.append(obj)

    return parsed


def build_series(records: List[Dict], key: str) -> List[float]:
    vals: List[float] = []
    for r in records:
        if key not in r:
            continue
        v = r[key]
        if isinstance(v, (int, float)):
            vals.append(float(v))
    return vals


def main() -> None:
    parser = argparse.ArgumentParser(description="从 train_rank 日志绘制损失曲线")
    parser.add_argument("--log_path", required=True, help="训练日志路径，如 train_rank0.log")
    parser.add_argument(
        "--output",
        default="",
        help="输出图片路径，默认与日志同目录，文件名为 <log_stem>_loss_curves.png",
    )
    parser.add_argument(
        "--smooth_window",
        type=int,
        default=30,
        help="滑动平均窗口大小（默认 30）",
    )
    parser.add_argument(
        "--no_raw",
        action="store_true",
        help="不绘制原始曲线，只绘制平滑曲线",
    )

    args = parser.parse_args()
    log_path = Path(args.log_path)
    if not log_path.exists():
        raise FileNotFoundError(f"日志不存在: {log_path}")

    output_path = Path(args.output) if args.output else log_path.with_name(f"{log_path.stem}_loss_curves.png")

    records = parse_log_dicts(log_path)
    if not records:
        raise RuntimeError("未从日志中解析到包含 'loss' 的训练记录")

    total_loss = build_series(records, "loss")
    ce_loss = build_series(records, "ce_loss")
    gen_loss = build_series(records, "gen_contrastive_loss")
    disc_loss = build_series(records, "disc_contrastive_loss")

    # 用 total loss 的点位作为横轴基准
    x = list(range(len(total_loss)))

    def align(series: List[float], target_len: int) -> List[float]:
        # 当前日志格式中这些 key 基本同频出现；这里做防御性对齐
        if len(series) >= target_len:
            return series[:target_len]
        if not series:
            return [float("nan")] * target_len
        pad = [series[-1]] * (target_len - len(series))
        return series + pad

    ce_loss = align(ce_loss, len(total_loss))
    gen_loss = align(gen_loss, len(total_loss))
    disc_loss = align(disc_loss, len(total_loss))

    plt.figure(figsize=(12, 6))

    if not args.no_raw:
        plt.plot(x, total_loss, alpha=0.22, linewidth=1.0, label="total loss (raw)")
        plt.plot(x, ce_loss, alpha=0.22, linewidth=1.0, label="ce loss (raw)")
        plt.plot(x, gen_loss, alpha=0.22, linewidth=1.0, label="gen loss (raw)")
        plt.plot(x, disc_loss, alpha=0.22, linewidth=1.0, label="disc loss (raw)")

    sw = max(1, int(args.smooth_window))
    plt.plot(x, moving_average(total_loss, sw), linewidth=2.2, label=f"total loss (ma{sw})")
    plt.plot(x, moving_average(ce_loss, sw), linewidth=2.0, label=f"ce loss (ma{sw})")
    plt.plot(x, moving_average(gen_loss, sw), linewidth=2.0, label=f"gen loss (ma{sw})")
    plt.plot(x, moving_average(disc_loss, sw), linewidth=2.0, label=f"disc loss (ma{sw})")

    plt.title(f"Training Loss Curves\n{log_path.name}")
    plt.xlabel("log index")
    plt.ylabel("loss")
    plt.grid(alpha=0.3)
    plt.legend(ncol=2)
    plt.tight_layout()

    output_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(output_path, dpi=180)

    print(f"[OK] records={len(records)}")
    print(f"[OK] saved={output_path}")


if __name__ == "__main__":
    main()
