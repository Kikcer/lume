# COCONUT 原版对齐方案

## 目标

保留当前的对比损失（gen_emb, disc_emb），同时将 latent 部分与原版 COCONUT 对齐。

---

## 方案概述

### 当前实现 vs 目标实现

| 方面 | 当前实现 | 目标实现（原版对齐） |
|------|----------|---------------------|
| 数据格式 | Prefix/Suffix 分离 | 单一序列，含 latent 占位符 |
| Latent 处理 | 追加式（递增位置） | 替换式（固定位置） |
| Hidden 来源 | 序列最后位置 | 前一个位置 |
| Batch 处理 | 逐样本串行 | 可 batch 并行（纯文本） |

---

## 实现方案

### 方案 A：最小改动（推荐）

只修改 `_single_sample_loss` 中的 latent loop 逻辑，保持数据格式不变。

**核心改动**：让所有 latent step 共享同一个 position_id（模拟原版的固定位置）

```python
# 修改前（当前实现）
for _ in range(latent_steps):
    step_position_ids = torch.full((3, 1, 1), processed_len, ...)  # 递增
    processed_len += 1

# 修改后（原版对齐）
latent_position = processed_len  # 固定位置
for _ in range(latent_steps):
    step_position_ids = torch.full((3, 1, 1), latent_position, ...)  # 固定
    # processed_len 不变
```

**优点**：改动最小，风险低
**缺点**：KV Cache 仍然追加，与原版不完全一致

---

### 方案 B：完整对齐（推荐用于追求最佳效果）

重构数据格式和 latent loop，完全对齐原版。

#### Step 1: 修改数据格式

**文件**: `data_coconut.py`

将 prefix/suffix 分离改为单一序列：

```python
# 当前格式
{
    "prefix_input_ids": [Q, img, <disc_emb>, <bot>],
    "suffix_input_ids": [<eot>, Answer, <gen_emb>],
    "coconut_latent_steps": N,
}

# 目标格式
{
    "input_ids": [Q, img, <disc_emb>, <bot>, <lat>, <lat>, ..., <eot>, Answer, <gen_emb>],
    "labels": [-100, ..., -100, answer_tokens, ...],
    "latent_positions": [pos_bot+1, pos_bot+2, ...],  # latent token 的位置
}
```

#### Step 2: 修改 Latent Loop

**文件**: `train_qwen_coconut.py`

```python
def _coconut_forward_aligned(
    self,
    model,
    input_ids,        # [1, seq_len] 完整序列
    attention_mask,
    position_ids,
    labels,
    latent_positions, # latent token 的位置列表
    pixel_values,
    image_grid_thw,
    ...
):
    device = input_ids.device
    lm_head = _get_lm_head(model)
    backbone = _get_backbone_model(model)

    # 1. 获取 embedding
    inputs_embeds = model.get_input_embeddings()(input_ids)  # [1, seq, D]

    # 2. 处理视觉输入（如果有）
    if pixel_values is not None:
        # Qwen2-VL 的视觉 embedding 注入
        inputs_embeds = self._inject_visual_embeddings(
            inputs_embeds, input_ids, pixel_values, image_grid_thw
        )

    n_latent = len(latent_positions)
    if n_latent == 0:
        # 无 latent，直接前向
        outputs = model(inputs_embeds=inputs_embeds, attention_mask=attention_mask, ...)
        return self._compute_loss_and_reps(outputs, input_ids, labels)

    # 3. 分段前向（原版逻辑）
    first_latent_pos = latent_positions[0]
    next_compute_range = (0, first_latent_pos)
    kv_cache = None
    all_logits = []

    for pass_idx in range(n_latent):
        if kv_cache is None:
            # 第一次前向：计算到第一个 latent 之前
            outputs = backbone(
                inputs_embeds=inputs_embeds[:, next_compute_range[0]:next_compute_range[1], :],
                attention_mask=attention_mask[:, :next_compute_range[1]],
                position_ids=position_ids[:, next_compute_range[0]:next_compute_range[1]],
                output_hidden_states=True,
                use_cache=True,
            )
            hidden_offset = 0
        else:
            # 后续前向：复用 KV Cache
            past_kv = [(k[:, :, :next_compute_range[0], :],
                        v[:, :, :next_compute_range[0], :]) for k, v in kv_cache]
            outputs = backbone(
                inputs_embeds=inputs_embeds[:, next_compute_range[0]:next_compute_range[1], :],
                attention_mask=attention_mask[:, :next_compute_range[1]],
                position_ids=position_ids[:, next_compute_range[0]:next_compute_range[1]],
                past_key_values=past_kv,
                output_hidden_states=True,
                use_cache=True,
            )
            hidden_offset = next_compute_range[0]

        all_logits.append(lm_head(outputs.hidden_states[-1]))
        kv_cache = outputs.past_key_values
        hidden_states = outputs.hidden_states[-1]

        # 4. 关键：用前一个位置的 hidden 替换 latent embedding
        latent_pos = latent_positions[pass_idx]
        # hidden_states 的索引需要减去 hidden_offset
        prev_hidden = hidden_states[0, latent_pos - 1 - hidden_offset, :]

        # 替换 inputs_embeds 中 latent 位置的 embedding
        inputs_embeds = inputs_embeds.clone()
        inputs_embeds[0, latent_pos, :] = prev_hidden

        # 更新下一次计算范围
        if pass_idx + 1 < n_latent:
            next_compute_range = (next_compute_range[1], next_compute_range[1] + 1)
        else:
            next_compute_range = (next_compute_range[1], input_ids.shape[1])

    # 5. 最后一次前向：计算剩余部分
    past_kv = [(k[:, :, :next_compute_range[0], :],
                v[:, :, :next_compute_range[0], :]) for k, v in kv_cache]
    outputs = backbone(
        inputs_embeds=inputs_embeds[:, next_compute_range[0]:next_compute_range[1], :],
        attention_mask=attention_mask[:, :next_compute_range[1]],
        position_ids=position_ids[:, next_compute_range[0]:next_compute_range[1]],
        past_key_values=past_kv,
        output_hidden_states=True,
    )
    all_logits.append(lm_head(outputs.hidden_states[-1]))

    # 6. 拼接 logits，计算 loss
    logits = torch.cat(all_logits, dim=1)

    # 7. 提取对比学习表示
    # disc_emb 在 prefix 部分，gen_emb 在 suffix 部分
    disc_rep = self._extract_rep_from_hidden(outputs, input_ids, self.disc_emb_token_id)
    gen_rep = self._extract_rep_from_hidden(outputs, input_ids, self.gen_emb_token_id)

    # 8. 计算 CE Loss
    shift_logits = logits[:, :-1, :].contiguous()
    shift_labels = labels[:, 1:].contiguous()
    loss = F.cross_entropy(
        shift_logits.view(-1, shift_logits.size(-1)),
        shift_labels.view(-1),
        ignore_index=IGNORE_INDEX,
    )

    return loss, gen_rep, disc_rep
```

#### Step 3: 修改数据 Collator

支持 batch 内不同 latent 数量的对齐（左 padding）：

```python
def collate_coconut_aligned(features, tokenizer, latent_token_id):
    # 找到 batch 中最早的 latent 位置
    earliest_latent = [
        f["input_ids"].tolist().index(latent_token_id)
        for f in features
        if latent_token_id in f["input_ids"]
    ]

    if earliest_latent:
        latest_earliest = max(earliest_latent)
        # 左 padding 对齐
        for f in features:
            if latent_token_id in f["input_ids"]:
                n_pad = latest_earliest - f["input_ids"].tolist().index(latent_token_id)
            else:
                n_pad = 0
            if n_pad > 0:
                f["input_ids"] = torch.cat([
                    torch.full((n_pad,), tokenizer.pad_token_id),
                    f["input_ids"]
                ])
                f["position_ids"] = torch.cat([
                    torch.zeros(n_pad, dtype=torch.long),
                    f["position_ids"] + n_pad  # 注意：position 需要偏移
                ])
                # ... 同样处理 labels, attention_mask

    # 右 padding 到相同长度
    return pad_sequence(features, ...)
```

---

## 视觉输入的特殊处理

Qwen2-VL 的视觉输入会展开 `<image_pad>` token，导致每个样本的序列长度不同。

### 解决方案

1. **纯文本样本**：可以 batch 并行
2. **视觉样本**：仍需逐样本处理，但 latent loop 逻辑对齐原版

```python
def _run_side_batch_aligned(self, model, side_inputs):
    # 分离纯文本和视觉样本
    text_only_indices = []
    visual_indices = []

    for idx in range(batch_size):
        if side_inputs["pixel_values"][idx] is None:
            text_only_indices.append(idx)
        else:
            visual_indices.append(idx)

    results = {}

    # 纯文本样本：batch 并行
    if text_only_indices:
        text_batch = self._extract_batch(side_inputs, text_only_indices)
        text_results = self._coconut_forward_aligned_batch(model, text_batch)
        results.update(text_results)

    # 视觉样本：逐样本处理
    for idx in visual_indices:
        single = self._extract_single(side_inputs, idx)
        single_result = self._coconut_forward_aligned(model, **single)
        results[idx] = single_result

    return self._merge_results(results)
```

---

## 预期效果

| 指标 | 当前实现 | 对齐后 |
|------|----------|--------|
| 纯文本训练速度 | 1x | ~3-4x |
| 视觉样本训练速度 | 1x | ~1.2x |
| 位置编码 | 递增 | 固定（原版） |
| 语义对齐 | 部分 | 完全 |

---

## 实施建议

1. **第一步**：先实现方案 A（最小改动），验证位置编码固定的效果
2. **第二步**：如果效果提升明显，再实现方案 B（完整对齐）
3. **第三步**：针对纯文本样本启用 batch 并行优化

---

## 代码修改清单

### 方案 A（最小改动）

- [ ] `train_qwen_coconut.py`: 修改 `_single_sample_loss` 中的 position_id 逻辑

### 方案 B（完整对齐）

- [ ] `data_coconut.py`: 修改 `_build_side` 返回单一序列格式
- [ ] `data_coconut.py`: 修改 collator 支持左 padding 对齐
- [ ] `train_qwen_coconut.py`: 新增 `_coconut_forward_aligned` 方法
- [ ] `train_qwen_coconut.py`: 修改 `_run_side_batch` 调用新方法
- [ ] `train_qwen_coconut.py`: 添加纯文本 batch 并行优化
