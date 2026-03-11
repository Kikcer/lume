#!/usr/bin/env python3
"""
Verification script: compare <disc_emb> hidden state between
  (A) full-sequence forward  (same as original UME-R1 trainer)
  (B) prefix-only forward    (same as coconut trainer)

If the two hidden states are identical, the disc_contrastive_loss difference
must come from data ordering / batch composition, not from the forward path.
If they differ, there is a real bug in the coconut prefix forward.

Usage:
  python src/sft-train/qwenvl/train/debug_disc_emb_verify.py \
      --model_path /home/share/yty_model/UME-R1/2B/UME-R1/2B \
      --data_path  /home/share/yty_data/UME_R1_train/UME-sft-train.jsonl \
      --num_samples 8
"""
import argparse
import json
import os
import sys
import copy

import torch
import torch.nn.functional as F
from transformers import AutoTokenizer, AutoProcessor

# ── paths ──────────────────────────────────────────────────────────────
PROJ_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", "..", ".."))
sys.path.insert(0, os.path.join(PROJ_ROOT, "src", "sft-train"))


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--model_path", type=str,
                   default="/home/share/yty_model/UME-R1/2B/UME-R1/2B")
    p.add_argument("--data_path", type=str,
                   default="/home/share/yty_data/UME_R1_train/UME-sft-train.jsonl")
    p.add_argument("--num_samples", type=int, default=8)
    p.add_argument("--max_seq_len", type=int, default=2048)
    return p.parse_args()


# ── helpers ────────────────────────────────────────────────────────────
CHAT_TEMPLATE = (
    "{% for message in messages %}"
    "{{'<|im_start|>' + message['role'] + '\\n' + message['content'] + '<|im_end|>' + '\\n'}}"
    "{% endfor %}"
    "{% if add_generation_prompt %}{{ '<|im_start|>assistant\\n' }}{% endif %}"
)

ROLES = {"human": "user", "gpt": "assistant"}


def tokenize_sample_full(sample, tokenizer):
    """Tokenize a full conversation (user + assistant) into a single sequence.
    Returns input_ids as a 1-D LongTensor.
    Text-only (no image/video processing)."""
    tok = copy.deepcopy(tokenizer)
    tok.chat_template = CHAT_TEMPLATE
    system_message = "You are a helpful assistant."

    convs = sample.get("conversations", [])
    first_role = convs[0].get("from", convs[0].get("role", ""))
    first_role = ROLES.get(first_role, first_role)
    if first_role != "user":
        convs = convs[1:]

    ids = tok.apply_chat_template([{"role": "system", "content": system_message}])
    for conv in convs:
        role = conv.get("from", conv.get("role", ""))
        role = ROLES.get(role, role)
        content = conv.get("value", conv.get("content", ""))
        # Strip image/video placeholders for text-only forward
        content = content.replace("<image>", "").replace("<video>", "")
        encoded = tok.apply_chat_template([{"role": role, "content": content}])
        ids += encoded

    return torch.tensor(ids, dtype=torch.long)


def find_token_pos(ids_1d, token_id):
    """Return list of positions where token_id appears."""
    return (ids_1d == token_id).nonzero(as_tuple=False).flatten().tolist()


def main():
    args = parse_args()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    dtype = torch.bfloat16

    # ── load model & tokenizer ─────────────────────────────────────────
    print(f"Loading model from {args.model_path} ...")
    from transformers import Qwen2VLForConditionalGeneration
    model = Qwen2VLForConditionalGeneration.from_pretrained(
        args.model_path,
        torch_dtype=dtype,
        attn_implementation="flash_attention_2",
    ).to(device).eval()

    tokenizer = AutoTokenizer.from_pretrained(args.model_path, use_fast=False)

    # Add coconut tokens if not present
    existing = set(tokenizer.get_vocab().keys())
    new_tokens = [t for t in ["<gen_emb>", "<disc_emb>", "<bot>", "<eot>", "<ct>"]
                  if t not in existing]
    if new_tokens:
        tokenizer.add_tokens(new_tokens)
        model.resize_token_embeddings(len(tokenizer))
        print(f"Added tokens: {new_tokens}")

    DISC_EMB_ID = tokenizer.convert_tokens_to_ids("<disc_emb>")
    GEN_EMB_ID = tokenizer.convert_tokens_to_ids("<gen_emb>")
    BOT_ID = tokenizer.convert_tokens_to_ids("<bot>")
    EOT_ID = tokenizer.convert_tokens_to_ids("<eot>")
    print(f"Token IDs: disc_emb={DISC_EMB_ID}, gen_emb={GEN_EMB_ID}, bot={BOT_ID}, eot={EOT_ID}")

    backbone = model.model  # Qwen2VLModel (no lm_head)

    # ── load data ──────────────────────────────────────────────────────
    print(f"Loading data from {args.data_path} ...")
    with open(args.data_path) as f:
        all_data = json.load(f)

    # Data structure: each item has "qry" and "pos", each with "conversations"
    candidates = []
    for item in all_data:
        for side in ["qry", "pos"]:
            s = item.get(side, {})
            convs = s.get("conversations", [])
            text = json.dumps(convs)
            if "<disc_emb>" in text and "<gen_emb>" in text:
                candidates.append({"conversations": convs})
                break
        if len(candidates) >= args.num_samples:
            break

    print(f"Found {len(candidates)} candidate samples, using first {args.num_samples}")
    candidates = candidates[:args.num_samples]

    # ── run comparison ─────────────────────────────────────────────────
    print("\n" + "=" * 80)
    print("COMPARISON: full-sequence forward vs prefix-only forward")
    print("  + BATCH forward vs single-sample forward")
    print("=" * 80)

    results = []
    # Collect all tokenized sequences first for batch test
    all_tokenized = []
    for si, sample in enumerate(candidates):
        ids = tokenize_sample_full(sample, tokenizer)
        if len(ids) > args.max_seq_len:
            ids = ids[:args.max_seq_len]

        disc_positions = find_token_pos(ids, DISC_EMB_ID)
        gen_positions = find_token_pos(ids, GEN_EMB_ID)
        bot_positions = find_token_pos(ids, BOT_ID)

        if not disc_positions:
            print(f"\n[Sample {si}] SKIP: no <disc_emb> found")
            continue

        disc_pos = disc_positions[-1]

        # Find <bot> position for prefix split
        if bot_positions:
            bot_pos = bot_positions[-1]
            prefix_end = bot_pos + 1  # include <bot>
        else:
            prefix_end = len(ids)

        if disc_pos >= prefix_end:
            print(f"\n[Sample {si}] SKIP: disc_emb at {disc_pos} >= prefix_end {prefix_end}")
            continue

        all_tokenized.append({
            "si": si, "ids": ids, "disc_pos": disc_pos,
            "prefix_end": prefix_end, "bot_positions": bot_positions,
            "gen_positions": gen_positions,
        })

    # --- Test 1: single-sample full vs prefix ---
    for t in all_tokenized:
        si, ids, disc_pos, prefix_end = t["si"], t["ids"], t["disc_pos"], t["prefix_end"]
        full_ids = ids.unsqueeze(0).to(device)
        prefix_ids = ids[:prefix_end].unsqueeze(0).to(device)

        with torch.no_grad():
            full_out = backbone(input_ids=full_ids, output_hidden_states=True, return_dict=True)
            full_disc_rep = full_out.hidden_states[-1][0, disc_pos, :]

            prefix_out = backbone(input_ids=prefix_ids, output_hidden_states=True, return_dict=True)
            prefix_disc_rep = prefix_out.hidden_states[-1][0, disc_pos, :]

        cos_sim = F.cosine_similarity(
            full_disc_rep.unsqueeze(0).float(), prefix_disc_rep.unsqueeze(0).float(), dim=-1
        ).item()
        l2_diff = (full_disc_rep.float() - prefix_disc_rep.float()).norm().item()

        gen_in_prefix = [p for p in t["gen_positions"] if p < prefix_end]
        gen_info = ""
        if gen_in_prefix:
            gp = gen_in_prefix[-1]
            full_gen = full_out.hidden_states[-1][0, gp, :]
            prefix_gen = prefix_out.hidden_states[-1][0, gp, :]
            gen_cos = F.cosine_similarity(
                full_gen.unsqueeze(0).float(), prefix_gen.unsqueeze(0).float(), dim=-1
            ).item()
            gen_info = f"  gen_emb_in_prefix@{gp}: cos={gen_cos:.8f}"

        print(f"\n[Sample {si}] seq_len={len(ids)}, prefix_len={prefix_end}, "
              f"disc_emb@{disc_pos}, bot@{t['bot_positions']}")
        print(f"  [single] full_vs_prefix: cos={cos_sim:.8f}, l2={l2_diff:.6f}")
        if gen_info:
            print(gen_info)

        t["single_full_disc"] = full_disc_rep
        t["single_prefix_disc"] = prefix_disc_rep
        results.append({"sample": si, "single_cos": cos_sim, "single_l2": l2_diff})

    # --- Test 2: batch forward vs single forward ---
    # Pad all full sequences to same length and do batch forward
    if len(all_tokenized) >= 2:
        print("\n" + "-" * 60)
        print("BATCH vs SINGLE forward comparison")
        print("-" * 60)

        max_len = max(len(t["ids"]) for t in all_tokenized)
        batch_ids = torch.full((len(all_tokenized), max_len), tokenizer.pad_token_id,
                               dtype=torch.long, device=device)
        batch_mask = torch.zeros((len(all_tokenized), max_len), dtype=torch.long, device=device)

        for i, t in enumerate(all_tokenized):
            seq_len = len(t["ids"])
            batch_ids[i, :seq_len] = t["ids"].to(device)
            batch_mask[i, :seq_len] = 1

        with torch.no_grad():
            batch_out = backbone(
                input_ids=batch_ids,
                attention_mask=batch_mask,
                output_hidden_states=True,
                return_dict=True,
            )
            batch_hidden = batch_out.hidden_states[-1]

        for i, t in enumerate(all_tokenized):
            disc_pos = t["disc_pos"]
            batch_disc_rep = batch_hidden[i, disc_pos, :]
            single_disc_rep = t["single_full_disc"]

            cos_batch_vs_single = F.cosine_similarity(
                batch_disc_rep.unsqueeze(0).float(),
                single_disc_rep.unsqueeze(0).float(), dim=-1
            ).item()
            l2_batch_vs_single = (batch_disc_rep.float() - single_disc_rep.float()).norm().item()

            print(f"  [Sample {t['si']}] batch_vs_single: "
                  f"cos={cos_batch_vs_single:.8f}, l2={l2_batch_vs_single:.6f}")

            results[i]["batch_cos"] = cos_batch_vs_single
            results[i]["batch_l2"] = l2_batch_vs_single

    # ── summary ────────────────────────────────────────────────────────
    print("\n" + "=" * 80)
    print("SUMMARY")
    print("=" * 80)
    if results:
        cos_vals = [r["single_cos"] for r in results]
        print(f"Samples tested: {len(results)}")
        print(f"\n[Test 1] Full-seq vs Prefix-only (single sample):")
        print(f"  Cosine: min={min(cos_vals):.8f}, max={max(cos_vals):.8f}, "
              f"mean={sum(cos_vals)/len(cos_vals):.8f}")
        if all(c > 0.9999 for c in cos_vals):
            print("  ✅ Prefix-only forward matches full-sequence forward.")
        else:
            print("  ❌ Prefix-only forward differs from full-sequence forward!")

        batch_cos = [r.get("batch_cos") for r in results if "batch_cos" in r]
        if batch_cos:
            print(f"\n[Test 2] Batch forward vs Single forward (full-seq):")
            print(f"  Cosine: min={min(batch_cos):.8f}, max={max(batch_cos):.8f}, "
                  f"mean={sum(batch_cos)/len(batch_cos):.8f}")
            batch_l2 = [r["batch_l2"] for r in results if "batch_l2" in r]
            print(f"  L2:     min={min(batch_l2):.6f}, max={max(batch_l2):.6f}")
            if all(c > 0.9999 for c in batch_cos):
                print("  ✅ Batch forward matches single forward.")
            else:
                print("  ❌ Batch forward DIFFERS from single forward!")
                print("     => Padding + flash_attention causes different disc_emb hidden states!")
                print("     => This explains the disc_contrastive_loss difference.")
    else:
        print("No valid samples found.")


if __name__ == "__main__":
    main()
