#!/usr/bin/env python3
"""
COCONUT Latent 实现对比实验

比较原论文实现 vs UME-R1 实现的数值差异。

实验设置：
- 纯文本输入（无视觉）
- 关闭 dropout
- 冻结模型
- 比较每个 latent step 的 hidden、suffix logits、final loss
"""

import torch
import torch.nn.functional as F
from transformers import AutoModelForCausalLM, AutoTokenizer, AutoConfig
from transformers.cache_utils import DynamicCache
import argparse


def setup_model(model_path: str, device: str = "cuda"):
    """加载模型并关闭 dropout"""
    print(f"Loading model from {model_path}...")

    tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True)

    config = AutoConfig.from_pretrained(model_path, trust_remote_code=True)
    config_class_name = config.__class__.__name__

    if "Qwen2VL" in config_class_name:
        from transformers import Qwen2VLForConditionalGeneration
        model = Qwen2VLForConditionalGeneration.from_pretrained(
            model_path,
            torch_dtype=torch.bfloat16,
            trust_remote_code=True,
            attn_implementation="flash_attention_2",
        ).to(device)
    else:
        model = AutoModelForCausalLM.from_pretrained(
            model_path,
            torch_dtype=torch.bfloat16,
            trust_remote_code=True,
            attn_implementation="flash_attention_2",
        ).to(device)

    model.eval()
    for module in model.modules():
        if hasattr(module, 'dropout'):
            module.dropout = 0.0
        if hasattr(module, 'attention_dropout'):
            module.attention_dropout = 0.0

    for param in model.parameters():
        param.requires_grad = False

    print(f"Model loaded. dtype={next(model.parameters()).dtype}")
    return model, tokenizer


def get_special_tokens(tokenizer):
    """获取特殊 token ID"""
    vocab = tokenizer.get_vocab()

    bot_candidates = ["<bot>", "<|bot|>", "[BOT]", "<think>"]
    eot_candidates = ["<eot>", "<|eot|>", "[EOT]", "</think>"]
    latent_candidates = ["<latent>", "<|latent|>", "[LATENT]", "<thought>"]

    def find_token(candidates):
        for c in candidates:
            if c in vocab:
                return vocab[c]
        return None

    bot_id = find_token(bot_candidates)
    eot_id = find_token(eot_candidates)
    latent_id = find_token(latent_candidates)

    if bot_id is None:
        bot_id = 151643
    if eot_id is None:
        eot_id = 151644
    if latent_id is None:
        latent_id = 151645

    print(f"Special tokens: bot_id={bot_id}, eot_id={eot_id}, latent_id={latent_id}")
    return bot_id, eot_id, latent_id


def make_position_ids(seq_len: int, device: str):
    """创建 Qwen2-VL 兼容的 position_ids (3D)"""
    pos = torch.arange(seq_len, device=device)
    return pos.view(1, 1, -1).expand(3, 1, -1)


def slice_kv_cache(past_key_values, end_pos: int):
    """截断 KV Cache 到指定位置"""
    if past_key_values is None:
        return None

    if isinstance(past_key_values, DynamicCache):
        new_cache = DynamicCache()
        for layer_idx in range(len(past_key_values.key_cache)):
            k = past_key_values.key_cache[layer_idx][:, :, :end_pos, :]
            v = past_key_values.value_cache[layer_idx][:, :, :end_pos, :]
            new_cache.update(k, v, layer_idx)
        return new_cache
    else:
        return tuple(
            (k[:, :, :end_pos, :], v[:, :, :end_pos, :])
            for k, v in past_key_values
        )


def get_cache_len(past_key_values):
    """获取 KV Cache 长度"""
    if past_key_values is None:
        return 0
    if isinstance(past_key_values, DynamicCache):
        return past_key_values.get_seq_length()
    else:
        return past_key_values[0][0].shape[2]


def original_coconut_forward(
    model,
    input_ids: torch.LongTensor,
    latent_positions: list,
    device: str = "cuda",
):
    """
    原论文实现：替换式 latent loop

    输入: [Q] <bot> <latent> <latent> <eot> [suffix]
    每一步把 latent slot embedding 替换成前一位置 hidden
    """
    backbone = model.model
    embed_tokens = backbone.embed_tokens
    lm_head = model.lm_head

    inputs_embeds = embed_tokens(input_ids)  # [1, seq_len, D]
    seq_len = input_ids.shape[1]

    n_latent = len(latent_positions)
    if n_latent == 0:
        outputs = backbone(inputs_embeds=inputs_embeds, output_hidden_states=True)
        logits = lm_head(outputs.last_hidden_state)
        return {
            "latent_input_embeds": [],
            "latent_output_hiddens": [],
            "all_logits": logits,
            "final_hidden": outputs.last_hidden_state,
        }

    first_latent_pos = latent_positions[0]
    next_compute_range = (0, first_latent_pos)
    kv_cache = None
    all_logits = []
    latent_input_embeds = []      # 每个 latent step 的输入 embedding
    latent_output_hiddens = []    # 每个 latent step 的输出 hidden

    for pass_idx in range(n_latent):
        start, end = next_compute_range
        position_ids = make_position_ids(end, device)[:, :, start:end]

        if kv_cache is None:
            outputs = backbone(
                inputs_embeds=inputs_embeds[:, start:end, :],
                position_ids=position_ids,
                output_hidden_states=True,
                use_cache=True,
            )
            hidden_offset = 0
        else:
            past_kv = slice_kv_cache(kv_cache, start)
            outputs = backbone(
                inputs_embeds=inputs_embeds[:, start:end, :],
                position_ids=position_ids,
                past_key_values=past_kv,
                output_hidden_states=True,
                use_cache=True,
            )
            hidden_offset = start

        all_logits.append(lm_head(outputs.hidden_states[-1]))
        kv_cache = outputs.past_key_values
        hidden_states = outputs.hidden_states[-1]

        # 用前一个位置的 hidden 替换 latent embedding
        latent_pos = latent_positions[pass_idx]
        prev_hidden = hidden_states[0, latent_pos - 1 - hidden_offset, :]

        # 记录：这个 prev_hidden 将作为下一步 latent 的输入
        latent_input_embeds.append(prev_hidden.clone())

        # 替换 inputs_embeds
        inputs_embeds = inputs_embeds.clone()
        inputs_embeds[0, latent_pos, :] = prev_hidden

        # 更新下一次计算范围
        if pass_idx + 1 < n_latent:
            next_compute_range = (end, end + 1)
        else:
            next_compute_range = (end, seq_len)

    # 最后一次前向：计算剩余部分（包括最后一个 latent 和 suffix）
    start, end = next_compute_range
    position_ids = make_position_ids(end, device)[:, :, start:end]
    past_kv = slice_kv_cache(kv_cache, start)

    outputs = backbone(
        inputs_embeds=inputs_embeds[:, start:end, :],
        position_ids=position_ids,
        past_key_values=past_kv,
        output_hidden_states=True,
    )
    all_logits.append(lm_head(outputs.hidden_states[-1]))
    final_hidden_states = outputs.hidden_states[-1]

    # 提取每个 latent 位置的输出 hidden
    # 最后一次前向包含了最后一个 latent 和 suffix
    # 需要从之前的 outputs 和最后的 outputs 中提取
    # 简化：重新计算一次完整序列来获取所有 latent 输出
    # 但这样效率低，改为在循环中记录

    # 实际上，latent_i 的输出 hidden 在 pass_idx=i 的 outputs 中
    # 但由于我们是分段计算，需要特殊处理
    # 这里简化：用最后一次完整前向来获取所有位置的 hidden

    # 重新做一次完整前向（用替换后的 inputs_embeds）
    full_position_ids = make_position_ids(seq_len, device)
    full_outputs = backbone(
        inputs_embeds=inputs_embeds,
        position_ids=full_position_ids,
        output_hidden_states=True,
    )
    full_hidden = full_outputs.hidden_states[-1]

    # 提取每个 latent 位置的输出
    for latent_pos in latent_positions:
        latent_output_hiddens.append(full_hidden[0, latent_pos, :].clone())

    logits = torch.cat(all_logits, dim=1)

    return {
        "latent_input_embeds": latent_input_embeds,    # 替换进去的 embedding
        "latent_output_hiddens": latent_output_hiddens,  # latent 位置的输出 hidden
        "all_logits": lm_head(full_hidden),  # 用完整前向的 logits
        "final_hidden": full_hidden,
    }


def umer1_coconut_forward(
    model,
    prefix_ids: torch.LongTensor,
    suffix_ids: torch.LongTensor,
    latent_steps: int,
    device: str = "cuda",
):
    """
    UME-R1 实现：追加式 latent loop

    prefix -> latent loop -> suffix
    每步用 last_hidden 当 inputs_embeds
    """
    backbone = model.model
    lm_head = model.lm_head

    prefix_len = prefix_ids.shape[1]
    suffix_len = suffix_ids.shape[1]

    # 1. Prefix forward
    prefix_position_ids = make_position_ids(prefix_len, device)
    prefix_out = backbone(
        input_ids=prefix_ids,
        position_ids=prefix_position_ids,
        output_hidden_states=True,
        use_cache=True,
    )
    past_key_values = prefix_out.past_key_values
    last_hidden = prefix_out.hidden_states[-1][:, -1, :]  # [1, D]

    processed_len = get_cache_len(past_key_values)

    # 2. Latent loop
    latent_input_embeds = []      # 每个 latent step 的输入
    latent_output_hiddens = []    # 每个 latent step 的输出

    for step in range(latent_steps):
        # 记录输入
        latent_input_embeds.append(last_hidden.clone())

        step_inputs_embeds = last_hidden.unsqueeze(1)  # [1, 1, D]
        step_position_ids = torch.full((3, 1, 1), processed_len, dtype=torch.long, device=device)

        step_out = backbone(
            inputs_embeds=step_inputs_embeds,
            position_ids=step_position_ids,
            past_key_values=past_key_values,
            use_cache=True,
            output_hidden_states=True,
        )

        past_key_values = step_out.past_key_values
        last_hidden = step_out.hidden_states[-1][:, -1, :]

        # 记录输出
        latent_output_hiddens.append(last_hidden.clone())
        processed_len += 1

    # 3. Suffix forward
    suffix_position_ids = make_position_ids(processed_len + suffix_len, device)[:, :, processed_len:]

    suffix_out = backbone(
        input_ids=suffix_ids,
        position_ids=suffix_position_ids,
        past_key_values=past_key_values,
        use_cache=True,
        output_hidden_states=True,
    )

    # 4. 计算 logits
    # last_hidden 预测 suffix 第一个 token
    first_logits = lm_head(last_hidden.unsqueeze(1))  # [1, 1, V]
    suffix_logits = lm_head(suffix_out.hidden_states[-1])

    # suffix_logits 对应 suffix_ids 的预测
    # 完整的 suffix logits: [first_logits, suffix_logits]
    # first_logits 预测 suffix[0]
    # suffix_logits[i] 预测 suffix[i+1]

    return {
        "latent_input_embeds": latent_input_embeds,
        "latent_output_hiddens": latent_output_hiddens,
        "first_logits": first_logits,           # 预测 suffix[0]
        "suffix_logits": suffix_logits,         # suffix_logits[i] 预测 suffix[i+1]
        "final_hidden": suffix_out.hidden_states[-1],
    }


def run_comparison(
    model,
    tokenizer,
    question: str = "What is 2 + 3?",
    answer: str = "The answer is 5.",
    n_latent: int = 3,
    device: str = "cuda",
):
    """运行对比实验"""
    print("\n" + "=" * 60)
    print("COCONUT Latent 实现对比实验")
    print("=" * 60)

    bot_id, eot_id, latent_id = get_special_tokens(tokenizer)

    q_tokens = tokenizer.encode(question, add_special_tokens=False)
    a_tokens = tokenizer.encode(answer, add_special_tokens=False)

    print(f"\nQuestion tokens: {len(q_tokens)}")
    print(f"Answer tokens: {len(a_tokens)}")
    print(f"Latent steps: {n_latent}")

    # ========== 方案 A: 原论文实现 ==========
    print("\n" + "-" * 40)
    print("方案 A: 原论文实现（替换式）")
    print("-" * 40)

    # 构造序列: [Q] <bot> <latent>... <eot> [suffix]
    original_ids = (
        q_tokens +
        [bot_id] +
        [latent_id] * n_latent +
        [eot_id] +
        a_tokens
    )
    original_ids = torch.tensor([original_ids], dtype=torch.long, device=device)

    bot_pos = len(q_tokens)
    latent_positions = [bot_pos + 1 + i for i in range(n_latent)]
    eot_pos = bot_pos + 1 + n_latent

    print(f"Sequence length: {original_ids.shape[1]}")
    print(f"Latent positions: {latent_positions}")
    print(f"EOT position: {eot_pos}")

    with torch.no_grad():
        result_a = original_coconut_forward(
            model, original_ids, latent_positions, device
        )

    # ========== 方案 B: UME-R1 实现 ==========
    print("\n" + "-" * 40)
    print("方案 B: UME-R1 实现（追加式）")
    print("-" * 40)

    prefix_ids = q_tokens + [bot_id]
    prefix_ids = torch.tensor([prefix_ids], dtype=torch.long, device=device)

    suffix_ids = [eot_id] + a_tokens
    suffix_ids = torch.tensor([suffix_ids], dtype=torch.long, device=device)

    print(f"Prefix length: {prefix_ids.shape[1]}")
    print(f"Suffix length: {suffix_ids.shape[1]}")

    with torch.no_grad():
        result_b = umer1_coconut_forward(
            model, prefix_ids, suffix_ids, n_latent, device
        )

    # ========== 比较结果 ==========
    print("\n" + "=" * 60)
    print("比较结果")
    print("=" * 60)

    # 1. 比较 latent 输入 embedding
    print("\n1. Latent 输入 Embedding 比较:")
    print("   A: 前一位置的 hidden (替换进 latent slot)")
    print("   B: 上一步的输出 hidden (作为当前步输入)")

    for i in range(n_latent):
        h_a = result_a["latent_input_embeds"][i]
        h_b = result_b["latent_input_embeds"][i]

        diff = (h_a - h_b).abs()
        max_diff = diff.max().item()
        mean_diff = diff.mean().item()
        cos_sim = F.cosine_similarity(h_a.unsqueeze(0), h_b.unsqueeze(0), dim=-1).item()

        print(f"  Latent {i} 输入:")
        print(f"    max_diff={max_diff:.6e}, mean_diff={mean_diff:.6e}")
        print(f"    A_norm={h_a.norm().item():.4f}, B_norm={h_b.norm().item():.4f}")
        print(f"    cosine_sim={cos_sim:.6f}")

    # 2. 比较 latent 输出 hidden
    print("\n2. Latent 输出 Hidden 比较:")

    for i in range(n_latent):
        h_a = result_a["latent_output_hiddens"][i]
        h_b = result_b["latent_output_hiddens"][i]

        diff = (h_a - h_b).abs()
        max_diff = diff.max().item()
        mean_diff = diff.mean().item()
        cos_sim = F.cosine_similarity(h_a.unsqueeze(0), h_b.unsqueeze(0), dim=-1).item()

        print(f"  Latent {i} 输出:")
        print(f"    max_diff={max_diff:.6e}, mean_diff={mean_diff:.6e}")
        print(f"    A_norm={h_a.norm().item():.4f}, B_norm={h_b.norm().item():.4f}")
        print(f"    cosine_sim={cos_sim:.6f}")

    # 3. 比较 suffix logits
    print("\n3. Suffix Logits 比较:")

    # A: all_logits 从 eot_pos-1 开始（因为 eot_pos-1 的 hidden 预测 eot_pos）
    # 实际上，logits[i] 预测 token[i+1]
    # 所以 logits[eot_pos-1] 预测 token[eot_pos] = <eot>
    # logits[eot_pos] 预测 token[eot_pos+1] = answer[0]
    # 我们关心的是预测 answer 的 logits，即从 logits[eot_pos] 开始

    suffix_logits_a = result_a["all_logits"][:, eot_pos:, :]  # 预测 answer

    # B: first_logits 预测 suffix[0] = <eot>
    # suffix_logits[0] 预测 suffix[1] = answer[0]
    # 所以预测 answer 的是 suffix_logits[:, :-1, :]（最后一个预测的是 answer 之后的）
    # 但 suffix = [<eot>] + answer，所以 suffix_logits 预测的是 answer + next
    # suffix_logits[i] 预测 suffix[i+1]
    # suffix_logits[0] 预测 suffix[1] = answer[0]
    # suffix_logits[-1] 预测 suffix[-1+1] = 超出范围

    # 对齐：
    # A: logits[eot_pos + i] 预测 answer[i]
    # B: suffix_logits[i] 预测 suffix[i+1] = answer[i] (因为 suffix = [eot] + answer)

    suffix_logits_b = result_b["suffix_logits"]  # [1, suffix_len, V]

    # 取相同长度
    min_len = min(suffix_logits_a.shape[1], suffix_logits_b.shape[1])
    suffix_logits_a_cmp = suffix_logits_a[:, :min_len, :]
    suffix_logits_b_cmp = suffix_logits_b[:, :min_len, :]

    diff = (suffix_logits_a_cmp - suffix_logits_b_cmp).abs()
    max_diff = diff.max().item()
    mean_diff = diff.mean().item()

    pred_a = suffix_logits_a_cmp.argmax(dim=-1)
    pred_b = suffix_logits_b_cmp.argmax(dim=-1)
    pred_match = (pred_a == pred_b).float().mean().item()

    print(f"  max_diff={max_diff:.6e}, mean_diff={mean_diff:.6e}")
    print(f"  Shape A: {suffix_logits_a_cmp.shape}, Shape B: {suffix_logits_b_cmp.shape}")
    print(f"  Top-1 prediction match rate: {pred_match:.2%}")

    # 4. 比较 loss（不做 shift，直接对齐）
    print("\n4. Loss 比较:")

    # A: logits[eot_pos + i] 预测 answer[i]
    # 构造 labels: answer tokens
    answer_tokens = torch.tensor([a_tokens], dtype=torch.long, device=device)

    # A 的 loss
    logits_for_answer_a = suffix_logits_a[:, :len(a_tokens), :]
    loss_a = F.cross_entropy(
        logits_for_answer_a.view(-1, logits_for_answer_a.size(-1)),
        answer_tokens.view(-1),
    )

    # B 的 loss
    # suffix_logits[i] 预测 suffix[i+1] = answer[i]
    logits_for_answer_b = suffix_logits_b[:, :len(a_tokens), :]
    loss_b = F.cross_entropy(
        logits_for_answer_b.view(-1, logits_for_answer_b.size(-1)),
        answer_tokens.view(-1),
    )

    loss_diff = abs(loss_a.item() - loss_b.item())

    print(f"  Loss A (原论文): {loss_a.item():.6f}")
    print(f"  Loss B (UME-R1): {loss_b.item():.6f}")
    print(f"  Difference: {loss_diff:.6e}")
    print(f"  Relative diff: {loss_diff / (loss_a.item() + 1e-8):.2%}")

    # 5. 总结
    print("\n" + "=" * 60)
    print("总结")
    print("=" * 60)

    threshold = 1e-4

    # 检查输入 embedding 差异
    input_max_diffs = [
        (result_a["latent_input_embeds"][i] - result_b["latent_input_embeds"][i]).abs().max().item()
        for i in range(n_latent)
    ]
    max_input_diff = max(input_max_diffs)

    # 检查输出 hidden 差异
    output_max_diffs = [
        (result_a["latent_output_hiddens"][i] - result_b["latent_output_hiddens"][i]).abs().max().item()
        for i in range(n_latent)
    ]
    max_output_diff = max(output_max_diffs)

    print(f"\n  Latent 输入 max diff: {max_input_diff:.6e}")
    print(f"  Latent 输出 max diff: {max_output_diff:.6e}")
    print(f"  Suffix logits max diff: {max_diff:.6e}")
    print(f"  Loss diff: {loss_diff:.6e}")

    if max_input_diff < threshold and max_output_diff < threshold and loss_diff < threshold:
        print(f"\n✓ 所有差异在 {threshold} 量级内，两种实现数值上等价")
    else:
        print(f"\n✗ 存在显著差异")
        print("\n差异来源分析:")
        if max_input_diff >= threshold:
            print(f"  - Latent 输入差异大: 原版用 pos[i-1] 的 hidden，UME-R1 用上一步输出")
        if max_output_diff >= threshold:
            print(f"  - Latent 输出差异大: 可能由输入差异传播，或位置编码不同")
        print("  - 位置编码: 原版固定位置，UME-R1 递增位置")
        print("  - KV Cache: 原版截断复用，UME-R1 持续追加")

    return {
        "latent_input_max_diff": max_input_diff,
        "latent_output_max_diff": max_output_diff,
        "suffix_logits_max_diff": max_diff,
        "loss_a": loss_a.item(),
        "loss_b": loss_b.item(),
        "loss_diff": loss_diff,
        "pred_match_rate": pred_match,
    }


def main():
    parser = argparse.ArgumentParser(description="COCONUT 实现对比实验")
    parser.add_argument(
        "--model_path",
        type=str,
        default="Qwen/Qwen2.5-0.5B",
        help="模型路径",
    )
    parser.add_argument(
        "--n_latent",
        type=int,
        default=3,
        help="Latent steps 数量",
    )
    parser.add_argument(
        "--device",
        type=str,
        default="cuda",
        help="设备",
    )
    args = parser.parse_args()

    model, tokenizer = setup_model(args.model_path, args.device)

    results = run_comparison(
        model,
        tokenizer,
        question="What is the capital of France?",
        answer="The capital of France is Paris.",
        n_latent=args.n_latent,
        device=args.device,
    )

    print("\n" + "=" * 60)
    print("实验完成")
    print("=" * 60)


if __name__ == "__main__":
    main()
