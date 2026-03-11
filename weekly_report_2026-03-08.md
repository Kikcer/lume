# 周报 — Latent reasoning for UME

> 日期：2026-03-08

---

## 一、本周进展

基于 UME-R1 多模态检索/表征框架，引入 COCONUT（Chain of Continuous Thought）方法，将原有的显式 Chain-of-Thought 推理过程替换为隐式 latent token 推理，目标是在保持表征质量的前提下加速推理。

主要完成内容：

- 实现了 COCONUT 三段式训练流程（prefix → latent loop → suffix teacher forcing）
- 引入多阶段课程学习（Curriculum Learning），逐步将 `<think>` 中的显式推理文本替换为 `<bot>...<ct>...<eot>` 隐式 latent token
- 同时保留生成式（`<gen_emb>`）与判别式（`<disc_emb>`）双路对比学习
- 完成 MMEB V2 全训练集训练与评测

### 实验结果（MMEB V2 全训练集）

| 指标 | UME-R1 原始 | COCONUT (Ours) | 差值 |
|:---|:---:|:---:|:---:|
| overall hit@1 avg | 0.666 | 0.630 | -0.036 |
| I-CLS hit@1 avg (n=10) | 0.648 | 0.605 | -0.043 |
| I-QA hit@1 avg (n=10) | 0.628 | 0.579 | -0.049 |
| I-RET hit@1 avg (n=12) | 0.676 | 0.654 | -0.022 |
| I-VG hit@1 avg (n=4) | 0.772 | 0.747 | -0.025 |

当前 COCONUT 隐式推理方案相比原 UME-R1 整体下降约 3 个点。其中检索（I-RET）和视觉定位（I-VG）下降相对较小，分类（I-CLS）和问答（I-QA）下降较明显。

---

## 二、下一步计划

1. **Answer 部分也替换为隐式 token**
   - 当前方案仅将 `<think>` 推理过程替换为 latent token，`<answer>` 部分仍保留显式文本生成
   - 下一步计划将 `<answer>` 也替换为隐式表示，使模型在推理时完全不需要回到文本空间，直接从 latent 状态输出最终 embedding

2. **对显式 Answer 文本进行强化学习（RL）**
   - 在当前保留 `<answer>` 显式文本的版本上，引入 RL 训练
   - 通过奖励信号优化生成的 answer 质量，提升最终 embedding 表征效果

---

## 三、已知问题

1. **课程学习阶段的数据分布不均匀**
   - 当前数据集按顺序排布，多阶段课程学习导致不同数据子集只经历了部分课程阶段：某些数据集仅在课程阶段 1 被训练，另一些仅在阶段 2 被训练，未能在所有阶段上均匀覆盖。这可能是当前性能下降的原因之一，后续数据采样策略。

2. **计算资源紧张，训练开销大**

   - 多机多卡（4 节点 × 8 GPU）训练耗时较长，卡资源紧张
