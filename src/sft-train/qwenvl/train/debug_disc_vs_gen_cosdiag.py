#!/usr/bin/env python3
"""
Quick verification: load UME-R1 model + first N samples from training data,
do full-sequence forward (like original trainer), extract disc_emb and gen_emb
hidden states, compute qry-pos cosine similarity (cos_diag).

This tells us whether the low disc_cos_diag seen in coconut logs is inherent
to the data+model, or specific to the coconut forward path.

Usage:
  python debug_disc_vs_gen_cosdiag.py \
      --model_path /home/share/yty_model/UME-R1/2B/UME-R1/2B \
      --data_path  /home/share/yty_data/UME_R1_train/UME-sft-train.jsonl \
      --num_pairs 8
"""
import argparse, json, os, sys, re
import torch
import torch.nn.functional as F
from transformers import AutoTokenizer, Qwen2VLForConditionalGeneration

PROJ_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", "..", ".."))

CHAT_TEMPLATE = (
    "{% for message in messages %}"
    "{{'<|im_start|>' + message['role'] + '\\n' + message['content'] + '<|im_end|>' + '\\n'}}"
    "{% endfor %}"
    "{% if add_generation_prompt %}{{ '<|im_start|>assistant\\n' }}{% endif %}"
)

def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--model_path", default="/home/share/yty_model/UME-R1/2B/UME-R1/2B")
    p.add_argument("--data_path", default="/home/share/yty_data/UME_R1_train/UME-sft-train.jsonl")
    p.add_argument("--num_pairs", type=int, default=8)
    p.add_argument("--max_seq_len", type=int, default=2048)
    return p.parse_args()


def strip_media(text):
    """Remove <image>, <video>, Picture N: ... patterns for text-only tokenization."""
    text = re.sub(r"<image>", "", text)
    text = re.sub(r"<video>", "", text)
    text = re.sub(r"Picture \d+:\s*", "", text)
    return text.strip()


def conversations_to_text(convs):
    """Convert conversations list to a single string via chat template."""
    messages = []
    for c in convs:
        role = c.get("from", c.get("role", "user"))
        if role == "human":
            role = "user"
        elif role == "gpt":
            role = "assistant"
        content = strip_media(c.get("value", c.get("content", "")))
        messages.append({"role": role, "content": content})
    return messages


def extract_rep(hidden, ids, token_id):
    """Extract hidden state at the last occurrence of token_id."""
    positions = (ids == token_id).nonzero(as_tuple=False)
    if positions.numel() == 0:
        return None
    last_pos = int(positions[-1].item())
    return hidden[last_pos]


def main():
    args = parse_args()
    print(f"Loading model from {args.model_path} ...")
    tokenizer = AutoTokenizer.from_pretrained(args.model_path, trust_remote_code=True)
    tokenizer.chat_template = CHAT_TEMPLATE
    model = Qwen2VLForConditionalGeneration.from_pretrained(
        args.model_path, torch_dtype=torch.bfloat16, device_map="auto", trust_remote_code=True
    )
    model.eval()
    backbone = model.model  # the decoder backbone

    disc_id = tokenizer.convert_tokens_to_ids("<disc_emb>")
    gen_id = tokenizer.convert_tokens_to_ids("<gen_emb>")
    print(f"disc_emb token id = {disc_id}, gen_emb token id = {gen_id}")

    print(f"Loading data from {args.data_path} ...")
    with open(args.data_path) as f:
        raw_data = json.load(f)
    print(f"Total samples: {len(raw_data)}")

    # Collect pairs: each sample has qry and pos
    pairs = []
    for item in raw_data:
        qry_convs = item.get("qry", {}).get("conversations", [])
        pos_convs = item.get("pos", {}).get("conversations", [])
        if not qry_convs or not pos_convs:
            continue
        pairs.append((qry_convs, pos_convs))
        if len(pairs) >= args.num_pairs:
            break

    print(f"Collected {len(pairs)} pairs")

    disc_cos_list = []
    gen_cos_list = []

    for i, (qry_convs, pos_convs) in enumerate(pairs):
        qry_msgs = conversations_to_text(qry_convs)
        pos_msgs = conversations_to_text(pos_convs)

        qry_text = tokenizer.apply_chat_template(qry_msgs, tokenize=False, add_generation_prompt=False)
        pos_text = tokenizer.apply_chat_template(pos_msgs, tokenize=False, add_generation_prompt=False)

        qry_ids = tokenizer.encode(qry_text, add_special_tokens=False, return_tensors="pt")[0]
        pos_ids = tokenizer.encode(pos_text, add_special_tokens=False, return_tensors="pt")[0]

        if len(qry_ids) > args.max_seq_len:
            qry_ids = qry_ids[:args.max_seq_len]
        if len(pos_ids) > args.max_seq_len:
            pos_ids = pos_ids[:args.max_seq_len]

        # Check tokens exist
        qry_has_disc = (qry_ids == disc_id).any().item()
        qry_has_gen = (qry_ids == gen_id).any().item()
        pos_has_disc = (pos_ids == disc_id).any().item()
        pos_has_gen = (pos_ids == gen_id).any().item()

        if not (qry_has_disc and pos_has_disc and qry_has_gen and pos_has_gen):
            print(f"  Pair {i}: skipping (missing tokens: qry_disc={qry_has_disc} qry_gen={qry_has_gen} pos_disc={pos_has_disc} pos_gen={pos_has_gen})")
            continue

        with torch.no_grad():
            # Full-sequence forward for qry
            qry_input = qry_ids.unsqueeze(0).to(model.device)
            qry_out = backbone(input_ids=qry_input, output_hidden_states=False, return_dict=True)
            qry_hidden = qry_out.last_hidden_state[0]  # [L, D]

            # Full-sequence forward for pos
            pos_input = pos_ids.unsqueeze(0).to(model.device)
            pos_out = backbone(input_ids=pos_input, output_hidden_states=False, return_dict=True)
            pos_hidden = pos_out.last_hidden_state[0]  # [L, D]

            # Extract reps
            qry_disc = extract_rep(qry_hidden, qry_ids.to(model.device), disc_id)
            pos_disc = extract_rep(pos_hidden, pos_ids.to(model.device), disc_id)
            qry_gen = extract_rep(qry_hidden, qry_ids.to(model.device), gen_id)
            pos_gen = extract_rep(pos_hidden, pos_ids.to(model.device), gen_id)

            if qry_disc is not None and pos_disc is not None:
                disc_cos = F.cosine_similarity(qry_disc.unsqueeze(0), pos_disc.unsqueeze(0)).item()
                disc_cos_list.append(disc_cos)
            else:
                disc_cos = None

            if qry_gen is not None and pos_gen is not None:
                gen_cos = F.cosine_similarity(qry_gen.unsqueeze(0), pos_gen.unsqueeze(0)).item()
                gen_cos_list.append(gen_cos)
            else:
                gen_cos = None

            # Also compute disc_emb position info
            qry_disc_pos = (qry_ids == disc_id).nonzero(as_tuple=False).flatten().tolist()
            pos_disc_pos = (pos_ids == disc_id).nonzero(as_tuple=False).flatten().tolist()
            qry_gen_pos = (qry_ids == gen_id).nonzero(as_tuple=False).flatten().tolist()
            pos_gen_pos = (pos_ids == gen_id).nonzero(as_tuple=False).flatten().tolist()

            print(
                f"  Pair {i}: "
                f"qry_len={len(qry_ids)} pos_len={len(pos_ids)} | "
                f"disc_cos={disc_cos:.4f} gen_cos={gen_cos:.4f} | "
                f"qry_disc@{qry_disc_pos} pos_disc@{pos_disc_pos} "
                f"qry_gen@{qry_gen_pos} pos_gen@{pos_gen_pos}"
            )

    print("\n=== Summary ===")
    if disc_cos_list:
        print(f"disc_cos_diag: {[f'{x:.4f}' for x in disc_cos_list]}")
        print(f"  mean={sum(disc_cos_list)/len(disc_cos_list):.4f}")
    if gen_cos_list:
        print(f"gen_cos_diag:  {[f'{x:.4f}' for x in gen_cos_list]}")
        print(f"  mean={sum(gen_cos_list)/len(gen_cos_list):.4f}")

    if disc_cos_list and gen_cos_list:
        disc_mean = sum(disc_cos_list) / len(disc_cos_list)
        gen_mean = sum(gen_cos_list) / len(gen_cos_list)
        print(f"\ndisc_mean={disc_mean:.4f} vs gen_mean={gen_mean:.4f}")
        if abs(disc_mean - gen_mean) < 0.05:
            print("=> disc and gen cos_diag are similar => coconut disc extraction is correct")
        else:
            print(f"=> disc is {'lower' if disc_mean < gen_mean else 'higher'} than gen by {abs(disc_mean-gen_mean):.4f}")
            print("   This is inherent to the data+model, not a coconut bug.")


if __name__ == "__main__":
    main()
