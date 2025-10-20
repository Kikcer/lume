# Adopted from https://github.com/lm-sys/FastChat. Below is the original copyright:
# Adopted from tatsu-lab@stanford_alpaca. Below is the original copyright:
#    Copyright 2023 Rohan Taori, Ishaan Gulrajani, Tianyi Zhang, Yann Dubois, Xuechen Li
#
#    Licensed under the Apache License, Version 2.0 (the "License");
#    you may not use this file except in compliance with the License.
#    You may obtain a copy of the License at
#
#        http://www.apache.org/licenses/LICENSE-2.0
#
#    Unless required by applicable law or agreed to in writing, software
#    distributed under the License is distributed on an "AS IS" BASIS,
#    WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
#    See the License for the specific language governing permissions and
#    limitations under the License.

import os
import logging
import pathlib
import torch
import torch.distributed as dist
import transformers
import json
from typing import Dict
import shutil
import sys
from pathlib import Path

# from deepspeed.runtime.zero.config import ZeroStageEnum
from deepspeed.runtime.zero.config import ZeroStageEnum
from deepspeed.runtime.fp16.loss_scaler import LossScaler
from deepspeed.runtime.zero.stage_1_and_2 import DeepSpeedZeroOptimizer
from deepspeed.runtime.zero.stage3 import DeepSpeedZeroOptimizer_Stage3


torch.serialization.add_safe_globals([
    ZeroStageEnum,
    LossScaler,
    DeepSpeedZeroOptimizer,
    DeepSpeedZeroOptimizer_Stage3
    ])
    
project_root = Path(__file__).parent.parent.parent
sys.path.append(str(project_root))

import qwenvl.train.trainer
from trainer import replace_qwen2_vl_attention_class

from transformers import (
    Qwen2VLForConditionalGeneration,
    Qwen2_5_VLForConditionalGeneration,
)
from qwenvl.data.data_qwen import make_supervised_data_module

from qwenvl.train.argument import (
    ModelArguments,
    DataArguments,
    TrainingArguments,
)
from transformers import AutoTokenizer, AutoProcessor, Qwen2VLImageProcessor, Trainer
from peft import LoraConfig, get_peft_model
from deepspeed.runtime.zero.partition_parameters import GatheredParameters

from deepspeed.runtime.zero.config import ZeroStageEnum
# import torch.serialization

# torch.serialization.add_safe_globals({'ZeroStageEnum': ZeroStageEnum})

local_rank = None


def rank0_print(*args):
    if local_rank == 0:
        print(*args)


def safe_save_model_for_hf_trainer(trainer: transformers.Trainer, output_dir: str):
    """Collects the state dict and dump to disk."""

    if trainer.deepspeed:
        torch.cuda.synchronize()
        trainer.save_model(output_dir)
        return

    state_dict = trainer.model.state_dict()
    if trainer.args.should_save:
        cpu_state_dict = {key: value.cpu() for key, value in state_dict.items()}
        del state_dict
        trainer._save(output_dir, state_dict=cpu_state_dict)  # noqa


def set_model(model_args, model):
    if model_args.tune_mm_vision:
        for n, p in model.visual.named_parameters():
            p.requires_grad = True
    else:
        for n, p in model.visual.named_parameters():
            p.requires_grad = False

    if model_args.tune_mm_mlp:
        for n, p in model.visual.merger.named_parameters():
            p.requires_grad = True
    else:
        for n, p in model.visual.merger.named_parameters():
            p.requires_grad = False

    if model_args.tune_mm_llm:
        for n, p in model.model.named_parameters():
            p.requires_grad = True
        model.lm_head.requires_grad = True
    else:
        for n, p in model.model.named_parameters():
            p.requires_grad = False
        model.lm_head.requires_grad = False


def train(attn_implementation="flash_attention_2"):
    global local_rank

    parser = transformers.HfArgumentParser(
        (ModelArguments, DataArguments, TrainingArguments)
    )

    model_args, data_args, training_args = parser.parse_args_into_dataclasses()

    attn_implementation = model_args.attn_implementation
    local_rank = training_args.local_rank
    training_args.data_group = data_args.data_group
    os.makedirs(training_args.output_dir, exist_ok=True)

    if "qwen2.5" in model_args.model_name_or_path.lower():
        model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
            model_args.model_name_or_path,
            cache_dir=training_args.cache_dir,
            attn_implementation=attn_implementation,
            torch_dtype=(torch.bfloat16 if training_args.bf16 else None),
        )
        data_args.image_processor = AutoProcessor.from_pretrained(
            model_args.model_name_or_path,
        ).image_processor
        data_args.model_type = "qwen2.5vl"
    else:
        model = Qwen2VLForConditionalGeneration.from_pretrained(
            model_args.model_name_or_path,
            cache_dir=training_args.cache_dir,
            attn_implementation=attn_implementation,
            torch_dtype=(torch.bfloat16 if training_args.bf16 else None),
        )
        data_args.image_processor = Qwen2VLImageProcessor.from_pretrained(
            model_args.model_name_or_path,
        )
        data_args.model_type = "qwen2vl"

    if data_args.data_flatten:
        replace_qwen2_vl_attention_class()
    model.config.use_cache = False

    if training_args.gradient_checkpointing:
        if hasattr(model, "enable_input_require_grads"):
            model.enable_input_require_grads()
        else:

            def make_inputs_require_grad(module, input, output):
                output.requires_grad_(True)

            model.get_input_embeddings().register_forward_hook(make_inputs_require_grad)

    tokenizer = transformers.AutoTokenizer.from_pretrained(
        model_args.model_name_or_path,
        cache_dir=training_args.cache_dir,
        model_max_length=training_args.model_max_length,
        padding_side="right",
        use_fast=False,
    )
    
    # tokenizer.add_special_tokens({"additional_special_tokens": tokenizer.additional_special_tokens + ["<|embed_start|>"]})
    # tokenizer.add_special_tokens({"additional_special_tokens": tokenizer.additional_special_tokens + ["<|embed_end|>"]})
    # tokenizer.add_tokens(["<|embed_start|>","<|embed_end|>"])
    # # model.resize_token_embeddings(len(tokenizer))
    # embed_start_id = tokenizer.convert_tokens_to_ids("<|embed_start|>")
    # embed_end_id = tokenizer.convert_tokens_to_ids("<|embed_end|>")

    # left_bracket_id = tokenizer.convert_tokens_to_ids("<tool_call>")
    # right_bracket_id = tokenizer.convert_tokens_to_ids("</tool_call>")
    
    # with torch.no_grad():
    #     model.get_input_embeddings().weight[embed_start_id] = model.get_input_embeddings().weight[left_bracket_id]
    #     model.get_input_embeddings().weight[embed_end_id] = model.get_input_embeddings().weight[right_bracket_id]

    # Initialize the embeddings of newly added tokens from the embeddings of existing tokens
    tokenizer.add_tokens(["<gen_emb>"])
    gen_emb_id = tokenizer.convert_tokens_to_ids("<gen_emb>")
    if "qwen2.5" in model_args.model_name_or_path.lower():
        right_tool_id = tokenizer.convert_tokens_to_ids("</tool_call>")
        left_tool_id = tokenizer.convert_tokens_to_ids("<tool_call>")
    else:
        right_tool_id = tokenizer.convert_tokens_to_ids("<|object_ref_end|>")
        left_tool_id = tokenizer.convert_tokens_to_ids("<|object_ref_start|>")
    
    tokenizer.add_tokens(["<disc_emb>"])
    disc_emb_id = tokenizer.convert_tokens_to_ids("<disc_emb>")

    
    embedding_weight = model.get_input_embeddings().weight
    with GatheredParameters([embedding_weight], modifier_rank=0):
        if torch.distributed.get_rank() == 0:
            with torch.no_grad():
                embedding_weight[gen_emb_id] = embedding_weight[right_tool_id].clone()
                embedding_weight[disc_emb_id] = embedding_weight[left_tool_id].clone()

    print("Generative Embedding token id: ", tokenizer.get_vocab()["<gen_emb>"])
    print("Discriminative Embedding token id: ", tokenizer.get_vocab()["<disc_emb>"])
    set_model(model_args, model)

    if model_args.use_lora:
        lora_config = LoraConfig(
            r=model_args.lora_r,  
            lora_alpha=model_args.lora_alpha,  
            target_modules=model_args.lora_target_modules.split(","), 
            lora_dropout=model_args.lora_dropout,  
            init_lora_weights="gaussian",
            use_dora=True,
            inference_mode=False
        )
        # model.model = get_peft_model(model.model, lora_config)
        model = get_peft_model(model, lora_config)

    if torch.distributed.get_rank() == 0:
        model.visual.print_trainable_parameters()
        if model_args.use_lora:
            model.print_trainable_parameters()
        else:
            model.model.print_trainable_parameters()
        # Print the parameters of training
        rank0_print("Trainable parameters:")
        rank0_print(
            f"Trainable parameters: {sum(p.numel() for p in model.parameters() if p.requires_grad)}"
        )

    training_args.gradient_checkpointing_kwargs = {'use_reentrant':False}
    data_module = make_supervised_data_module(tokenizer=tokenizer, data_args=data_args)
    trainer = Trainer(
        model=model, processing_class=tokenizer, args=training_args, **data_module
    )

    if list(pathlib.Path(training_args.output_dir).glob("checkpoint-*")):
        logging.info("checkpoint found, resume training")
        trainer.train(resume_from_checkpoint=True)
    else:
        trainer.train()
    trainer.save_state()
    data_args.image_processor.save_pretrained(training_args.output_dir)
    # with open(os.path.join(training_args.output_dir, "chat_template.json"), "w") as f:
    #     f.write(json.dumps(tokenizer.chat_template))
        
    # tokenizer.save_pretrained(training_args.output_dir)
    # source_path = os.path.join(model_args.model_name_or_path, "chat_template.json")
    # template_path = os.path.join(training_args.output_dir, "chat_template.json")
    # shutil.copy2(source_path, template_path)

    model.config.use_cache = True

    safe_save_model_for_hf_trainer(trainer=trainer, output_dir=training_args.output_dir)
    # trainer.save_model(training_args.output_dir)


if __name__ == "__main__":
    train(attn_implementation="flash_attention_2")
