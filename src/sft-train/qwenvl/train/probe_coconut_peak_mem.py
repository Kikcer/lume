import copy
import math
import os
import re
import sys
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np
import torch
import torch.distributed as dist
import transformers
from PIL import Image
from torch.utils.data import Dataset
from transformers import AutoProcessor, Qwen2VLImageProcessor

project_root = Path(__file__).parent.parent.parent
sys.path.insert(0, str(project_root))

import qwenvl.train.train_qwen_coconut as coconut_base
from qwenvl.data.data_coconut import (
    _build_latent_assistant_text,
    _pad_position_ids,
    preprocess_qwen_2_visual,
)
from qwenvl.data.rope2d import get_rope_index_2, get_rope_index_25
from qwenvl.train.argument import DataArguments, ModelArguments, TrainingArguments
from qwenvl.train.train_qwen_coconut_gc import CoconutGradientCheckpointTrainer

local_rank = 0


@dataclass
class ProbeArguments:
    probe_vision_mode: str = field(
        default="both",
        metadata={"help": "One of: none, image, video, both"},
    )
    probe_video_frames: int = field(
        default=4,
        metadata={"help": "Synthetic video frame count when vision_mode uses video."},
    )
    probe_text_token_target: int = field(
        default=0,
        metadata={"help": "Optional manual token target for assistant text. 0 means auto-fill to model_max_length."},
    )
    probe_seed: int = field(default=1234)


class SyntheticProbeDataset(Dataset):
    def __init__(self, length: int):
        self.length = max(int(length), 1)

    def __len__(self) -> int:
        return self.length

    def __getitem__(self, idx: int) -> int:
        return idx


def _clone_batch_value(value):
    if torch.is_tensor(value):
        return value.clone()
    if isinstance(value, list):
        return [_clone_batch_value(v) for v in value]
    if isinstance(value, dict):
        return {k: _clone_batch_value(v) for k, v in value.items()}
    return value


class SyntheticProbeCollator:
    def __init__(self, batch: Dict[str, object]):
        self.batch = batch

    def __call__(self, instances) -> Dict[str, object]:
        return _clone_batch_value(self.batch)


def rank0_print(msg: str) -> None:
    if (not dist.is_available()) or (not dist.is_initialized()) or dist.get_rank() == 0:
        print(msg, flush=True)


def _init_probe_log(output_dir: str, local_rank_value: int) -> None:
    os.makedirs(output_dir, exist_ok=True)
    log_file = os.path.join(output_dir, f"probe_rank{local_rank_value}.log")

    class _Tee:
        def __init__(self, *files):
            self.files = files

        def write(self, data):
            for f in self.files:
                try:
                    f.write(data)
                    f.flush()
                except Exception:
                    pass

        def flush(self):
            for f in self.files:
                try:
                    f.flush()
                except Exception:
                    pass

    try:
        fh = open(log_file, "a", buffering=1)
        sys.stdout = _Tee(sys.__stdout__, fh)
        sys.stderr = _Tee(sys.__stderr__, fh)
    except Exception as e:
        print(f"[PEAK-PROBE] failed to init log file {log_file}: {e}", flush=True)


def _log_local_cuda_mem(tag: str, **kwargs) -> None:
    if not torch.cuda.is_available():
        return
    device = torch.device("cuda", torch.cuda.current_device())
    free_bytes, total_bytes = torch.cuda.mem_get_info(device)
    alloc_gb = torch.cuda.memory_allocated(device) / (1024 ** 3)
    reserved_gb = torch.cuda.memory_reserved(device) / (1024 ** 3)
    peak_alloc_gb = torch.cuda.max_memory_allocated(device) / (1024 ** 3)
    peak_reserved_gb = torch.cuda.max_memory_reserved(device) / (1024 ** 3)
    free_gb = free_bytes / (1024 ** 3)
    total_gb = total_bytes / (1024 ** 3)
    rank = dist.get_rank() if dist.is_available() and dist.is_initialized() else local_rank
    extra = " ".join(f"{k}={v}" for k, v in kwargs.items())
    print(
        f"[PEAK-PROBE][MEM][rank{rank}][{tag}] "
        f"alloc={alloc_gb:.2f}GB reserved={reserved_gb:.2f}GB "
        f"peak_alloc={peak_alloc_gb:.2f}GB peak_reserved={peak_reserved_gb:.2f}GB "
        f"free={free_gb:.2f}GB total={total_gb:.2f}GB"
        + (f" {extra}" if extra else ""),
        flush=True,
    )


def _parse_oom_allocation_requirement(message: str) -> Tuple[Optional[float], Optional[float], Optional[float]]:
    tried_match = re.search(r"Tried to allocate ([0-9.]+) (MiB|GiB)", message)
    in_use_match = re.search(r"this process has ([0-9.]+) GiB memory in use", message)
    total_match = re.search(r"GPU \d+ has a total capacity of ([0-9.]+) GiB", message)

    tried_gb = None
    if tried_match:
        value = float(tried_match.group(1))
        unit = tried_match.group(2)
        tried_gb = value / 1024.0 if unit == "MiB" else value

    in_use_gb = float(in_use_match.group(1)) if in_use_match else None
    total_gb = float(total_match.group(1)) if total_match else None
    return tried_gb, in_use_gb, total_gb


def _to_bool_str(x: str) -> bool:
    return str(x).strip().lower() in {"1", "true", "yes", "on"}


def _make_square_image(target_pixels: int, seed: int) -> Image.Image:
    side = max(28, int(math.sqrt(max(int(target_pixels), 28 * 28))))
    rng = np.random.default_rng(seed)
    arr = rng.integers(0, 256, size=(side, side, 3), dtype=np.uint8)
    return Image.fromarray(arr, mode="RGB")


def _build_visual_payloads(
    data_args: DataArguments,
    vision_mode: str,
    seed: int,
    video_frames: int,
) -> Tuple[Optional[torch.Tensor], Optional[torch.Tensor], Optional[torch.Tensor], Optional[torch.Tensor], List[int], List[int]]:
    image_tensor = None
    image_grid_thw = None
    video_tensor = None
    video_grid_thw = None
    image_grid_thw_merged: List[int] = []
    video_grid_thw_merged: List[int] = []

    mode = str(vision_mode).strip().lower()
    merge_size_sq = int(data_args.image_processor.merge_size) ** 2

    if mode in {"image", "both"}:
        processor = copy.deepcopy(data_args.image_processor)
        processor.max_pixels = data_args.max_pixels
        processor.min_pixels = data_args.max_pixels
        processor.size["longest_edge"] = data_args.max_pixels
        processor.size["shortest_edge"] = data_args.max_pixels
        image = _make_square_image(data_args.max_pixels, seed)
        visual_processed = processor.preprocess(image, return_tensors="pt")
        image_tensor = visual_processed["pixel_values"]
        if isinstance(image_tensor, list):
            image_tensor = image_tensor[0]
        image_grid_thw = visual_processed["image_grid_thw"]
        if image_grid_thw.ndim == 1:
            image_grid_thw = image_grid_thw.unsqueeze(0)
        image_grid_thw_merged = [
            int(x) for x in (image_grid_thw.prod(dim=1) // merge_size_sq).tolist()
        ]

    if mode in {"video", "both"}:
        processor = copy.deepcopy(data_args.image_processor)
        processor.max_pixels = data_args.video_max_frame_pixels
        processor.min_pixels = data_args.video_max_frame_pixels
        processor.size["longest_edge"] = data_args.video_max_frame_pixels
        processor.size["shortest_edge"] = data_args.video_max_frame_pixels
        frames = [
            np.array(_make_square_image(data_args.video_max_frame_pixels, seed + idx + 17))
            for idx in range(max(int(video_frames), 1))
        ]
        visual_processed = processor.preprocess(images=None, videos=frames, return_tensors="pt")
        video_tensor = visual_processed["pixel_values_videos"]
        video_grid_thw = visual_processed["video_grid_thw"]
        if video_grid_thw.ndim == 1:
            video_grid_thw = video_grid_thw.unsqueeze(0)
        video_grid_thw_merged = [
            int(x) for x in (video_grid_thw.prod(dim=1) // merge_size_sq).tolist()
        ]

    return (
        image_tensor,
        image_grid_thw,
        video_tensor,
        video_grid_thw,
        image_grid_thw_merged,
        video_grid_thw_merged,
    )


def _make_user_prompt(vision_mode: str) -> str:
    parts = ["Synthetic peak-memory probe.", "<disc_emb>"]
    mode = str(vision_mode).strip().lower()
    if mode in {"image", "both"}:
        parts.append("<image>")
    if mode in {"video", "both"}:
        parts.append("<video>")
    parts.append("Please answer after thinking.")
    return "\n".join(parts)


def _make_assistant_text(units: int, think_segments: int, ct_per_segment: int) -> str:
    think_body = (" reasoning_step" * max(units, 1)).strip()
    answer_body = (" answer_token" * max(units // 2, 1)).strip()
    raw_text = f"<think>{think_body}</think><answer>{answer_body}</answer>"
    return _build_latent_assistant_text(
        raw_text=raw_text,
        latent_ct_tokens=0,
        think_segments=think_segments,
        ct_per_segment=ct_per_segment,
        drop_answer_text=False,
    )


def _tokenize_messages(
    tokenizer,
    user_text: str,
    assistant_text: str,
    image_grid_thw_merged: List[int],
    video_grid_thw_merged: List[int],
):
    original_model_max_length = getattr(tokenizer, "model_max_length", None)
    if original_model_max_length is not None:
        tokenizer.model_max_length = max(int(original_model_max_length), 1_000_000)
    messages = [[
        {"from": "human", "value": user_text},
        {"from": "gpt", "value": assistant_text},
    ]]
    try:
        return preprocess_qwen_2_visual(
            messages,
            tokenizer,
            grid_thw_image=image_grid_thw_merged or None,
            grid_thw_video=video_grid_thw_merged or None,
        )
    finally:
        if original_model_max_length is not None:
            tokenizer.model_max_length = original_model_max_length


def _get_position_ids(
    model_name_or_path: str,
    image_processor,
    input_ids: torch.Tensor,
    image_grid_thw: Optional[torch.Tensor],
    video_grid_thw: Optional[torch.Tensor],
) -> torch.Tensor:
    if image_grid_thw is None and video_grid_thw is None:
        seq_len = input_ids.size(1)
        return torch.arange(seq_len, dtype=torch.long).view(1, 1, -1).expand(3, 1, -1)

    rope_fn = get_rope_index_25 if "qwen2.5" in model_name_or_path.lower() else get_rope_index_2
    position_ids, _ = rope_fn(
        image_processor.merge_size,
        input_ids,
        image_grid_thw=image_grid_thw,
        video_grid_thw=video_grid_thw,
    )
    return position_ids


def _truncate_sequence_if_needed(
    seq_ids: torch.Tensor,
    seq_labels: torch.Tensor,
    seq_pos: torch.Tensor,
    max_seq_length: int,
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    if max_seq_length <= 0:
        return seq_ids, seq_labels, seq_pos
    seq_len = int(seq_ids.shape[0])
    if seq_len <= max_seq_length:
        return seq_ids, seq_labels, seq_pos

    cut = seq_len - max_seq_length
    vision_end_id = 151653
    vision_end_positions = torch.nonzero(seq_ids == vision_end_id, as_tuple=False).flatten()
    safe_head = int(vision_end_positions[-1].item()) + 1 if vision_end_positions.numel() > 0 else 0
    safe_tail = 256
    available_middle = seq_len - safe_head - safe_tail

    if available_middle >= cut and safe_head > 0:
        mid_start = safe_head
        mid_end = mid_start + cut
        seq_ids = torch.cat([seq_ids[:mid_start], seq_ids[mid_end:]])
        seq_labels = torch.cat([seq_labels[:mid_start], seq_labels[mid_end:]])
        seq_pos = torch.cat([seq_pos[:, :mid_start], seq_pos[:, mid_end:]], dim=1)
    else:
        seq_ids = seq_ids[cut:]
        seq_labels = seq_labels[cut:]
        seq_pos = seq_pos[:, cut:]
    return seq_ids, seq_labels, seq_pos


def _build_side_instance(
    tokenizer,
    model_args: ModelArguments,
    data_args: DataArguments,
    training_args: TrainingArguments,
    probe_args: ProbeArguments,
) -> Dict[str, torch.Tensor]:
    (
        image_tensor,
        image_grid_thw,
        video_tensor,
        video_grid_thw,
        image_grid_thw_merged,
        video_grid_thw_merged,
    ) = _build_visual_payloads(
        data_args=data_args,
        vision_mode=probe_args.probe_vision_mode,
        seed=probe_args.probe_seed,
        video_frames=probe_args.probe_video_frames,
    )

    user_text = _make_user_prompt(probe_args.probe_vision_mode)
    think_segments = max(int(getattr(data_args, "coconut_think_segments", 4)), 1)
    ct_per_segment = max(int(getattr(data_args, "coconut_ct_tokens_per_segment", 1)), 1)

    target_len = int(
        probe_args.probe_text_token_target
        if int(probe_args.probe_text_token_target) > 0
        else getattr(training_args, "model_max_length", 0)
    )
    if target_len <= 0:
        raise ValueError("model_max_length must be positive for peak-memory probing.")

    low, high = 1, max(target_len * 8, 1024)
    best = None
    while low <= high:
        mid = (low + high) // 2
        assistant_text = _make_assistant_text(mid, think_segments, ct_per_segment)
        tokenized = _tokenize_messages(
            tokenizer,
            user_text,
            assistant_text,
            image_grid_thw_merged,
            video_grid_thw_merged,
        )
        seq_len = int(tokenized["input_ids"].shape[1])
        if seq_len <= target_len:
            best = (assistant_text, tokenized)
            low = mid + 1
        else:
            high = mid - 1

    if best is None:
        raise RuntimeError("Failed to synthesize a sequence within model_max_length.")

    assistant_text, tokenized = best
    input_ids = tokenized["input_ids"]
    labels = tokenized["labels"]
    position_ids = _get_position_ids(
        model_name_or_path=model_args.model_name_or_path,
        image_processor=data_args.image_processor,
        input_ids=input_ids,
        image_grid_thw=image_grid_thw,
        video_grid_thw=video_grid_thw,
    )

    seq_ids = input_ids[0]
    seq_labels = labels[0]
    seq_pos = position_ids[:, 0, :]
    seq_ids, seq_labels, seq_pos = _truncate_sequence_if_needed(
        seq_ids,
        seq_labels,
        seq_pos,
        target_len,
    )

    bot_positions = torch.nonzero(seq_ids == tokenizer.convert_tokens_to_ids("<bot>"), as_tuple=False).flatten()
    eot_positions = torch.nonzero(seq_ids == tokenizer.convert_tokens_to_ids("<eot>"), as_tuple=False).flatten()
    if bot_positions.numel() == 0 or eot_positions.numel() == 0:
        raise ValueError("Synthetic sample cannot find <bot>/<eot> markers.")
    bot_pos = int(bot_positions[-1].item())
    eot_after_bot = eot_positions[eot_positions > bot_pos]
    if eot_after_bot.numel() == 0:
        raise ValueError("Synthetic sample cannot find <eot> after <bot>.")
    eot_pos = int(eot_after_bot[0].item())

    latent_steps = int(max(eot_pos - bot_pos - 1, 0))
    prefix_ids = seq_ids[: bot_pos + 1]
    prefix_pos = seq_pos[:, : bot_pos + 1]
    suffix_ids = seq_ids[eot_pos:]
    suffix_pos = seq_pos[:, eot_pos:]
    suffix_labels = seq_labels[eot_pos:].clone()

    if not bool(getattr(data_args, "coconut_include_gen_emb_loss", True)):
        gen_emb_id = tokenizer.convert_tokens_to_ids("<gen_emb>")
        suffix_labels[suffix_ids == gen_emb_id] = -100

    return {
        "prefix_input_ids": prefix_ids,
        "prefix_attention_mask": torch.ones_like(prefix_ids, dtype=torch.long),
        "prefix_position_ids": prefix_pos.long(),
        "suffix_input_ids": suffix_ids,
        "suffix_attention_mask": torch.ones_like(suffix_ids, dtype=torch.long),
        "suffix_position_ids": suffix_pos.long(),
        "suffix_labels": suffix_labels.long(),
        "coconut_latent_steps": torch.tensor(latent_steps, dtype=torch.long),
        "pixel_values": image_tensor,
        "image_grid_thw": image_grid_thw,
        "pixel_values_videos": video_tensor,
        "video_grid_thw": video_grid_thw,
        "_stats": {
            "seq_len": int(seq_ids.shape[0]),
            "prefix_len": int(prefix_ids.shape[0]),
            "suffix_len": int(suffix_ids.shape[0]),
            "latent_steps": latent_steps,
            "assistant_chars": len(assistant_text),
            "image_tokens": int(image_grid_thw_merged[0]) if image_grid_thw_merged else 0,
            "video_tokens": int(video_grid_thw_merged[0]) if video_grid_thw_merged else 0,
        },
    }


def _repeat_side_instance(side_instance: Dict[str, torch.Tensor], batch_size: int) -> Dict[str, object]:
    side_instances = []
    for _ in range(batch_size):
        cloned = {}
        for key, value in side_instance.items():
            if key == "_stats":
                continue
            if torch.is_tensor(value):
                cloned[key] = value.clone()
            else:
                cloned[key] = value
        side_instances.append(cloned)

    pad_id = 0
    prefix_ids_list = [x["prefix_input_ids"] for x in side_instances]
    prefix_pos_list = [x["prefix_position_ids"] for x in side_instances]
    suffix_ids_list = [x["suffix_input_ids"] for x in side_instances]
    suffix_pos_list = [x["suffix_position_ids"] for x in side_instances]
    suffix_labels_list = [x["suffix_labels"] for x in side_instances]

    prefix_max_len = max(int(x.shape[0]) for x in prefix_ids_list)
    suffix_max_len = max(int(x.shape[0]) for x in suffix_ids_list)

    prefix_input_ids = torch.full((batch_size, prefix_max_len), pad_id, dtype=torch.long)
    suffix_input_ids = torch.full((batch_size, suffix_max_len), pad_id, dtype=torch.long)
    suffix_labels = torch.full((batch_size, suffix_max_len), -100, dtype=torch.long)

    for idx, (p_ids, s_ids, s_labels) in enumerate(zip(prefix_ids_list, suffix_ids_list, suffix_labels_list)):
        prefix_input_ids[idx, : p_ids.shape[0]] = p_ids
        suffix_input_ids[idx, : s_ids.shape[0]] = s_ids
        suffix_labels[idx, : s_labels.shape[0]] = s_labels

    return {
        "prefix_input_ids": prefix_input_ids,
        "prefix_attention_mask": prefix_input_ids.ne(pad_id).long(),
        "prefix_position_ids": _pad_position_ids(prefix_pos_list, prefix_max_len),
        "suffix_input_ids": suffix_input_ids,
        "suffix_attention_mask": suffix_input_ids.ne(pad_id).long(),
        "suffix_position_ids": _pad_position_ids(suffix_pos_list, suffix_max_len),
        "suffix_labels": suffix_labels,
        "coconut_latent_steps": torch.stack([x["coconut_latent_steps"] for x in side_instances], dim=0),
        "pixel_values": [x.get("pixel_values", None) for x in side_instances],
        "image_grid_thw": [x.get("image_grid_thw", None) for x in side_instances],
        "pixel_values_videos": [x.get("pixel_values_videos", None) for x in side_instances],
        "video_grid_thw": [x.get("video_grid_thw", None) for x in side_instances],
    }


def train_probe():
    global local_rank
    parser = transformers.HfArgumentParser(
        (ModelArguments, DataArguments, TrainingArguments, ProbeArguments)
    )
    model_args, data_args, training_args, probe_args = parser.parse_args_into_dataclasses()
    local_rank = int(getattr(training_args, "local_rank", 0))
    _init_probe_log(training_args.output_dir, int(getattr(training_args, "local_rank", 0)))
    training_args.max_steps = 1
    training_args.save_steps = max(int(getattr(training_args, "save_steps", 999999)), 999999)
    os.environ["COCONUT_DEBUG_MEMORY"] = os.environ.get("COCONUT_DEBUG_MEMORY", "1")

    torch.manual_seed(probe_args.probe_seed)
    np.random.seed(probe_args.probe_seed)

    training_args.remove_unused_columns = False
    manual_gc_enabled = bool(training_args.gradient_checkpointing)
    manual_gc_use_reentrant = False
    gc_kwargs = getattr(training_args, "gradient_checkpointing_kwargs", None)
    if isinstance(gc_kwargs, dict):
        manual_gc_use_reentrant = bool(gc_kwargs.get("use_reentrant", False))
    training_args.gradient_checkpointing = False
    training_args.gradient_checkpointing_kwargs = {}

    attn_implementation = coconut_base.resolve_attn_implementation(model_args.attn_implementation)
    model, attn_implementation = coconut_base.load_qwen_model_with_fallback(
        model_name_or_path=model_args.model_name_or_path,
        cache_dir=training_args.cache_dir,
        attn_implementation=attn_implementation,
        torch_dtype=(torch.bfloat16 if training_args.bf16 else None),
    )
    rank0_print(f"[PEAK-PROBE] attn_implementation={attn_implementation}")
    _log_local_cuda_mem("after_model_load")

    if "qwen2.5" in model_args.model_name_or_path.lower():
        data_args.image_processor = AutoProcessor.from_pretrained(
            model_args.model_name_or_path
        ).image_processor
    else:
        data_args.image_processor = Qwen2VLImageProcessor.from_pretrained(
            model_args.model_name_or_path
        )
    data_args.image_processor.max_pixels = data_args.max_pixels
    data_args.image_processor.min_pixels = data_args.min_pixels
    data_args.image_processor.size["longest_edge"] = data_args.max_pixels
    data_args.image_processor.size["shortest_edge"] = data_args.min_pixels

    tokenizer = transformers.AutoTokenizer.from_pretrained(
        model_args.model_name_or_path,
        cache_dir=training_args.cache_dir,
        model_max_length=training_args.model_max_length,
        padding_side="right",
        use_fast=False,
    )

    coconut_base.initialize_new_tokens(
        model,
        tokenizer,
        model_args.model_name_or_path,
        force_reinit_all=model_args.coconut_force_reinit_all_tokens,
    )
    coconut_base.set_model(model_args, model)
    model.config.use_cache = True
    coconut_base.initialize_latent_moe(model, model_args)
    _log_local_cuda_mem("after_model_setup")

    gen_emb_token_id = tokenizer.convert_tokens_to_ids("<gen_emb>")
    disc_emb_token_id = tokenizer.convert_tokens_to_ids("<disc_emb>")
    trainer = CoconutGradientCheckpointTrainer(
        model=model,
        processing_class=tokenizer,
        args=training_args,
        gen_emb_token_id=gen_emb_token_id,
        disc_emb_token_id=disc_emb_token_id,
        gen_contrastive_weight=data_args.coconut_gen_contrastive_weight,
        disc_contrastive_weight=data_args.coconut_disc_contrastive_weight,
        contrastive_logit_scale=data_args.coconut_contrastive_logit_scale,
        contrastive_cross_device=data_args.coconut_contrastive_cross_device,
        contrastive_local_loss=data_args.coconut_contrastive_local_loss,
        debug_disc_oracle_pos_from_qry=data_args.coconut_debug_disc_oracle_pos_from_qry,
        latent_moe_enable=model_args.latent_moe_enable,
        latent_moe_balance_loss_weight=model_args.latent_moe_balance_loss_weight,
        latent_moe_context_type=model_args.latent_moe_context_type,
        enable_manual_gradient_checkpointing=manual_gc_enabled,
        manual_gc_use_reentrant=manual_gc_use_reentrant,
        train_dataset=SyntheticProbeDataset(
            max(
                int(training_args.per_device_train_batch_size)
                * max(1, int(training_args.gradient_accumulation_steps)),
                1,
            )
        ),
    )
    _log_local_cuda_mem("after_trainer_init")

    synthetic_side = _build_side_instance(tokenizer, model_args, data_args, training_args, probe_args)
    batch_size = int(training_args.per_device_train_batch_size)
    synthetic_batch = {
        "qry": _repeat_side_instance(synthetic_side, batch_size),
        "pos": _repeat_side_instance(synthetic_side, batch_size),
        "coconut_real_pair": torch.ones((batch_size,), dtype=torch.bool),
    }
    trainer.data_collator = SyntheticProbeCollator(synthetic_batch)
    stats = synthetic_side["_stats"]
    rank0_print(
        "[PEAK-PROBE] synthetic sample "
        f"vision_mode={probe_args.probe_vision_mode}, seq_len={stats['seq_len']}, "
        f"prefix_len={stats['prefix_len']}, suffix_len={stats['suffix_len']}, "
        f"latent_steps={stats['latent_steps']}, image_tokens={stats['image_tokens']}, "
        f"video_tokens={stats['video_tokens']}, per_device_batch={batch_size}"
    )
    _log_local_cuda_mem(
        "after_synthetic_batch",
        prefix_len=stats["prefix_len"],
        suffix_len=stats["suffix_len"],
        image_tokens=stats["image_tokens"],
        video_tokens=stats["video_tokens"],
    )

    model.zero_grad(set_to_none=True)
    torch.cuda.empty_cache()
    torch.cuda.reset_peak_memory_stats()

    try:
        trainer.train()
        peak_alloc = torch.cuda.max_memory_allocated() / (1024 ** 3)
        peak_reserved = torch.cuda.max_memory_reserved() / (1024 ** 3)
        rank0_print(
            f"[PEAK-PROBE] local peak_alloc={peak_alloc:.2f}GB peak_reserved={peak_reserved:.2f}GB"
        )
        _log_local_cuda_mem("after_probe_train")

        if dist.is_available() and dist.is_initialized():
            device = torch.device("cuda", torch.cuda.current_device())
            peak_pair = torch.tensor([peak_alloc, peak_reserved], dtype=torch.float64, device=device)
            dist.all_reduce(peak_pair, op=dist.ReduceOp.MAX)
            if dist.get_rank() == 0:
                rank0_print(
                    f"[PEAK-PROBE] global max peak_alloc={peak_pair[0].item():.2f}GB "
                    f"peak_reserved={peak_pair[1].item():.2f}GB"
                )
    except torch.OutOfMemoryError as e:
        peak_alloc = torch.cuda.max_memory_allocated() / (1024 ** 3)
        peak_reserved = torch.cuda.max_memory_reserved() / (1024 ** 3)
        tried_gb, in_use_gb, total_gb = _parse_oom_allocation_requirement(str(e))
        estimated_min_gb = None
        recommended_gb = None
        if tried_gb is not None and in_use_gb is not None:
            estimated_min_gb = in_use_gb + tried_gb
            recommended_gb = estimated_min_gb + 1.0
        elif tried_gb is not None:
            estimated_min_gb = peak_reserved + tried_gb
            recommended_gb = estimated_min_gb + 1.0
        estimated_min_str = f"{estimated_min_gb:.2f}GB" if estimated_min_gb is not None else "unknown"
        recommended_str = f"{recommended_gb:.2f}GB" if recommended_gb is not None else "unknown"
        total_str = f"{total_gb:.2f}GB" if total_gb is not None else "unknown"
        rank0_print(
            "[PEAK-PROBE] OOM "
            f"peak_alloc={peak_alloc:.2f}GB peak_reserved={peak_reserved:.2f}GB "
            f"prefix_len={stats['prefix_len']} suffix_len={stats['suffix_len']} "
            f"image_tokens={stats['image_tokens']} video_tokens={stats['video_tokens']} "
            f"estimated_min_required={estimated_min_str} "
            f"recommended_required={recommended_str} "
            f"gpu_total={total_str}"
        )
        _log_local_cuda_mem("oom_catch")
        try:
            print(torch.cuda.memory_summary(device=torch.cuda.current_device(), abbreviated=False), flush=True)
        except Exception as summary_e:
            print(f"[PEAK-PROBE] failed to dump memory_summary: {summary_e}", flush=True)
        raise
    finally:
        model.zero_grad(set_to_none=True)
        torch.cuda.empty_cache()


if __name__ == "__main__":
    train_probe()
