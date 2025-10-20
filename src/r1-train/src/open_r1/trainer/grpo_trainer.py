# Copyright 2025 The HuggingFace Team. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import os
import textwrap
from collections import defaultdict
from typing import Any, Callable, Optional, Union, Sized

import torch
import torch.utils.data
from torch.nn import functional as F
import transformers
from datasets import Dataset, IterableDataset
from packaging import version
from transformers import (
    AriaForConditionalGeneration,
    AriaProcessor,
    AutoModelForCausalLM,
    AutoModelForSequenceClassification,
    AutoProcessor,
    AutoTokenizer,
    GenerationConfig,
    PreTrainedModel,
    PreTrainedTokenizerBase,
    Qwen2VLForConditionalGeneration,
    Qwen2_5_VLForConditionalGeneration,
    Trainer,
    TrainerCallback,
    is_wandb_available,
)
from transformers.integrations.deepspeed import is_deepspeed_zero3_enabled, is_deepspeed_available
from transformers.utils import is_peft_available

from trl.data_utils import apply_chat_template, is_conversational, maybe_apply_chat_template
from trl.models import create_reference_model, prepare_deepspeed, unwrap_model_for_generation
from trl.trainer.grpo_config import GRPOConfig
from trl.trainer.utils import generate_model_card, get_comet_experiment_url
from trl import GRPOTrainer

from qwen_vl_utils import process_vision_info

from accelerate.utils import is_peft_model, set_seed
import PIL.Image

import copy
from torch.utils.data import Sampler
import warnings

if is_peft_available():
    from peft import PeftConfig, get_peft_model

if is_wandb_available():
    import wandb

from open_r1.vlm_modules.vlm_module import VLMBaseModule
# What we call a reward function is a callable that takes a list of prompts and completions and returns a list of
# rewards. When it's a string, it's a model ID, so it's loaded as a pretrained model.
RewardFunc = Union[str, PreTrainedModel, Callable[[list, list], list[float]]]


class RepeatRandomSampler(Sampler):
    """
    Sampler that repeats the indices of a dataset in a structured manner.

    Args:
        data_source (`Sized`):
            Dataset to sample from.
        mini_repeat_count (`int`):
            Number of times to repeat each index per batch.
        batch_size (`int`, *optional*, defaults to `1`):
            Number of unique indices per batch.
        repeat_count (`int`, *optional*, defaults to `1`):
            Number of times to repeat the full sampling process.
        seed (`int` or `None`, *optional*, defaults to `None`):
            Random seed for reproducibility.
    """

    def __init__(
        self,
        data_source: Sized,
        mini_repeat_count: int,
        batch_size: int = 1,
        repeat_count: int = 1,
        seed: Optional[int] = None,
    ):
        self.data_source = data_source
        self.mini_repeat_count = mini_repeat_count
        self.batch_size = batch_size
        self.repeat_count = repeat_count
        self.num_samples = len(data_source)
        self.seed = seed
        self.generator = torch.Generator()
        if seed is not None:
            self.generator.manual_seed(seed)

    def __iter__(self):
        indexes = torch.randperm(self.num_samples, generator=self.generator).tolist()
        indexes = [indexes[i : i + self.batch_size] for i in range(0, len(indexes), self.batch_size)]
        indexes = [chunk for chunk in indexes if len(chunk) == self.batch_size]

        for chunk in indexes:
            for _ in range(self.repeat_count):
                for index in chunk:
                    for _ in range(self.mini_repeat_count):
                        yield index

    def __len__(self) -> int:
        return self.num_samples * self.mini_repeat_count * self.repeat_count

class RepeatSampler(Sampler):
    """
    Sampler that repeats the indices of a dataset in a structured, fixed order manner.

    Args:
        data_source (`Sized`):
            Dataset to sample from.
        mini_repeat_count (`int`):
            Number of times to repeat each index per batch.
        batch_size (`int`, *optional*, defaults to `1`):
            Number of unique indices per batch.
        repeat_count (`int`, *optional*, defaults to `1`):
            Number of times to repeat the full sampling process.
    """

    def __init__(
        self,
        data_source: Sized,
        mini_repeat_count: int,
        batch_size: int = 1,
        repeat_count: int = 1,
    ):
        self.data_source = data_source
        self.mini_repeat_count = mini_repeat_count
        self.batch_size = batch_size
        self.repeat_count = repeat_count
        self.num_samples = len(data_source)

    def __iter__(self):
        # 顺序采样
        indexes = list(range(self.num_samples))
        # 按batch_size分组
        indexes = [indexes[i : i + self.batch_size] for i in range(0, len(indexes), self.batch_size)]
        indexes = [chunk for chunk in indexes if len(chunk) == self.batch_size]

        for chunk in indexes:
            for _ in range(self.repeat_count):
                for index in chunk:
                    for _ in range(self.mini_repeat_count):
                        yield index

    def __len__(self) -> int:
        return (self.num_samples // self.batch_size) * self.batch_size * self.mini_repeat_count * self.repeat_count

class VLMGRPOTrainer(Trainer):
    """
    Trainer for the Group Relative Policy Optimization (GRPO) method. This algorithm was initially proposed in the
    paper [DeepSeekMath: Pushing the Limits of Mathematical Reasoning in Open Language Models](https://huggingface.co/papers/2402.03300).

    Example:

    ```python
    from datasets import load_dataset
    from trl import GRPOTrainer

    dataset = load_dataset("trl-lib/tldr", split="train")

    trainer = GRPOTrainer(
        model="Qwen/Qwen2-0.5B-Instruct",
        reward_funcs="weqweasdas/RM-Gemma-2B",
        train_dataset=dataset,
    )

    trainer.train()
    ```

    Args:
        model (`Union[str, PreTrainedModel]`):
            Model to be trained. Can be either:

            - A string, being the *model id* of a pretrained model hosted inside a model repo on huggingface.co, or
              a path to a *directory* containing model weights saved using
              [`~transformers.PreTrainedModel.save_pretrained`], e.g., `'./my_model_directory/'`. The model is
              loaded using [`~transformers.AutoModelForCausalLM.from_pretrained`] with the keywork arguments
              in `args.model_init_kwargs`.
            - A [`~transformers.PreTrainedModel`] object. Only causal language models are supported.
        reward_funcs (`Union[RewardFunc, list[RewardFunc]]`):
            Reward functions to be used for computing the rewards. To compute the rewards, we call all the reward
            functions with the prompts and completions and sum the rewards. Can be either:

            - A single reward function, such as:
                - A string: The *model ID* of a pretrained model hosted inside a model repo on huggingface.co, or a
                path to a *directory* containing model weights saved using
                [`~transformers.PreTrainedModel.save_pretrained`], e.g., `'./my_model_directory/'`. The model is loaded
                using [`~transformers.AutoModelForSequenceClassification.from_pretrained`] with `num_labels=1` and the
                keyword arguments in `args.model_init_kwargs`.
                - A [`~transformers.PreTrainedModel`] object: Only sequence classification models are supported.
                - A custom reward function: The function is provided with the prompts and the generated completions,
                  plus any additional columns in the dataset. It should return a list of rewards. For more details, see
                  [Using a custom reward function](#using-a-custom-reward-function).
            - A list of reward functions, where each item can independently be any of the above types. Mixing different
            types within the list (e.g., a string model ID and a custom reward function) is allowed.
        args ([`GRPOConfig`], *optional*, defaults to `None`):
            Configuration for this trainer. If `None`, a default configuration is used.
        train_dataset ([`~datasets.Dataset`] or [`~datasets.IterableDataset`]):
            Dataset to use for training. It must include a column `"prompt"`. Any additional columns in the dataset is
            ignored. The format of the samples can be either:

            - [Standard](dataset_formats#standard): Each sample contains plain text.
            - [Conversational](dataset_formats#conversational): Each sample contains structured messages (e.g., role
              and content).
        eval_dataset ([`~datasets.Dataset`], [`~datasets.IterableDataset`] or `dict[str, Union[Dataset, IterableDataset]]`):
            Dataset to use for evaluation. It must meet the same requirements as `train_dataset`.
        processing_class ([`~transformers.PreTrainedTokenizerBase`], *optional*, defaults to `None`):
            Processing class used to process the data. The padding side must be set to "left". If `None`, the
            processing class is loaded from the model's name with [`~transformers.AutoTokenizer.from_pretrained`].
        reward_processing_classes (`Union[PreTrainedTokenizerBase, list[PreTrainedTokenizerBase]]`, *optional*, defaults to `None`):
            Processing classes corresponding to the reward functions specified in `reward_funcs`. Can be either:

            - A single processing class: Used when `reward_funcs` contains only one reward function.
            - A list of processing classes: Must match the order and length of the reward functions in `reward_funcs`.
            If set to `None`, or if an element of the list corresponding to a [`~transformers.PreTrainedModel`] is
            `None`, the tokenizer for the model is automatically loaded using [`~transformers.AutoTokenizer.from_pretrained`].
            For elements in `reward_funcs` that are custom reward functions (not [`~transformers.PreTrainedModel`]),
            the corresponding entries in `reward_processing_classes` are ignored.
        callbacks (list of [`~transformers.TrainerCallback`], *optional*, defaults to `None`):
            List of callbacks to customize the training loop. Will add those to the list of default callbacks
            detailed in [here](https://huggingface.co/docs/transformers/main_classes/callback).

            If you want to remove one of the default callbacks used, use the [`~transformers.Trainer.remove_callback`]
            method.
        optimizers (`tuple[torch.optim.Optimizer, torch.optim.lr_scheduler.LambdaLR]`, *optional*, defaults to `(None, None)`):
            A tuple containing the optimizer and the scheduler to use. Will default to an instance of [`AdamW`] on your
            model and a scheduler given by [`get_linear_schedule_with_warmup`] controlled by `args`.
        peft_config ([`~peft.PeftConfig`], *optional*, defaults to `None`):
            PEFT configuration used to wrap the model. If `None`, the model is not wrapped.
    """

    def __init__(
        self,
        model: Union[str, PreTrainedModel],
        reward_funcs: Union[RewardFunc, list[RewardFunc]],
        args: GRPOConfig = None,
        vlm_module: VLMBaseModule = None,
        train_dataset: Optional[Union[Dataset, IterableDataset]] = None,
        eval_dataset: Optional[Union[Dataset, IterableDataset, dict[str, Union[Dataset, IterableDataset]]]] = None,
        processing_class: Optional[PreTrainedTokenizerBase] = None,
        reward_processing_classes: Optional[Union[PreTrainedTokenizerBase, list[PreTrainedTokenizerBase]]] = None,
        callbacks: Optional[list[TrainerCallback]] = None,
        optimizers: tuple[Optional[torch.optim.Optimizer], Optional[torch.optim.lr_scheduler.LambdaLR]] = (None, None),
        peft_config: Optional["PeftConfig"] = None,
        freeze_vision_modules: Optional[bool] = False,
        attn_implementation: str = "flash_attention_2",
        torch_dtype: str = "bfloat16",
        **kwargs,
    ):
        # Args
        if args is None:
            model_name = model if isinstance(model, str) else model.config._name_or_path
            model_name = model_name.split("/")[-1]
            args = GRPOConfig(f"{model_name}-GRPO")
        
        self.vlm_module = vlm_module

        # Models
        # Trained model
        model_init_kwargs = args.model_init_kwargs or {}
        # FIXME
        # Remember to modify it in the invernvl
        model_init_kwargs["attn_implementation"] = attn_implementation
        if model_init_kwargs.get("torch_dtype") is None:
            model_init_kwargs["torch_dtype"] = torch_dtype
        
        assert isinstance(model, str), "model must be a string in the current implementation"
        model_id = model
        torch_dtype = model_init_kwargs.get("torch_dtype")
        if isinstance(torch_dtype, torch.dtype) or torch_dtype == "auto" or torch_dtype is None:
            pass  # torch_dtype is already a torch.dtype or "auto" or None
        elif isinstance(torch_dtype, str):  # it's a str, but not "auto"
            torch_dtype = getattr(torch, torch_dtype)
        else:
            raise ValueError(
                "Invalid `torch_dtype` passed to `GRPOConfig`. Expected either 'auto' or a string representing "
                f"a `torch.dtype` (e.g., 'float32'), but got {torch_dtype}."
            )
        model_init_kwargs["use_cache"] = (
            False if args.gradient_checkpointing else model_init_kwargs.get("use_cache")
        )
            # Disable caching if gradient checkpointing is enabled (not supported)
        model_init_kwargs["use_cache"] = (
            False if args.gradient_checkpointing else model_init_kwargs.get("use_cache")
        )
        model_cls = self.vlm_module.get_model_class(model_id, model_init_kwargs)
        model = model_cls.from_pretrained(model_id, **model_init_kwargs)

        # LoRA
        self.vision_modules_keywords = self.vlm_module.get_vision_modules_keywords()
        if peft_config is not None:
            def find_all_linear_names(model, multimodal_keywords):
                cls = torch.nn.Linear
                lora_module_names = set()
                for name, module in model.named_modules():
                    # LoRA is not applied to the vision modules
                    if any(mm_keyword in name for mm_keyword in multimodal_keywords):
                        continue
                    if isinstance(module, cls):
                        lora_module_names.add(name)
                for m in lora_module_names:  # needed for 16-bit
                    if "embed_tokens" in m:
                        lora_module_names.remove(m)
                return list(lora_module_names)
            target_modules = find_all_linear_names(model, self.vision_modules_keywords)
            peft_config.target_modules = target_modules
            model = get_peft_model(model, peft_config)

        # Freeze vision modules
        if freeze_vision_modules:
            print("Freezing vision modules...")
            for n, p in model.named_parameters():
                if any(keyword in n for keyword in self.vision_modules_keywords):
                    p.requires_grad = False

        # Enable gradient checkpointing if requested
        if args.gradient_checkpointing:
            model = self._enable_gradient_checkpointing(model, args)

        # Reference model
        if is_deepspeed_available():
            self.ref_model = model_cls.from_pretrained(model_id, **model_init_kwargs)
        elif peft_config is None:
            # If PEFT configuration is not provided, create a reference model based on the initial model.
            self.ref_model = create_reference_model(model)
        else:
            # If PEFT is used, the reference model is not needed since the adapter can be disabled
            # to revert to the initial model.
            self.ref_model = None

        # Processing class
        if processing_class is None:
            processing_cls = self.vlm_module.get_processing_class()
            processing_class = processing_cls.from_pretrained(model_id, trust_remote_code=model_init_kwargs.get("trust_remote_code", None))
            for processing_keyword in self.vlm_module.get_custom_processing_keywords():
                if processing_keyword in kwargs:
                    setattr(processing_class, processing_keyword, kwargs[processing_keyword])
            if getattr(processing_class, "tokenizer",  None) is not None:
                pad_token_id = processing_class.tokenizer.pad_token_id
                # add embedding token id
                # processing_class.tokenizer.add_special_tokens({
                #     "additional_special_tokens": processing_class.tokenizer.additional_special_tokens + ["<gen_emb>"]
                # })

                # if processing_class.tokenizer has no embedding token
                if "<gen_emb>" not in processing_class.tokenizer.get_vocab():
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
                    from deepspeed.runtime.zero.partition_parameters import GatheredParameters

                    processing_class.tokenizer.add_tokens(["<gen_emb>"])
                    processing_class.tokenizer.add_tokens(["<disc_emb>"])
                    gen_emb_id = processing_class.tokenizer.convert_tokens_to_ids("<gen_emb>")
                    disc_emb_id = processing_class.tokenizer.convert_tokens_to_ids("<disc_emb>")
                    embedding_weight = model.get_input_embeddings().weight
                    if "qwen2.5" in model_id.lower():
                        right_tool_id = processing_class.tokenizer.convert_tokens_to_ids("</tool_call>")
                        left_tool_id = processing_class.tokenizer.convert_tokens_to_ids("<tool_call>")
                    else:
                        right_tool_id = processing_class.tokenizer.convert_tokens_to_ids("<|object_ref_end|>")
                        left_tool_id = processing_class.tokenizer.convert_tokens_to_ids("<|object_ref_start|>")
                    with GatheredParameters([embedding_weight], modifier_rank=0):
                        if torch.distributed.get_rank() == 0:
                            with torch.no_grad():
                                embedding_weight[gen_emb_id] = embedding_weight[right_tool_id].clone()
                                embedding_weight[disc_emb_id] = embedding_weight[left_tool_id].clone()

                                
                processing_class.embed_token_id = processing_class.tokenizer.get_vocab()["<gen_emb>"]
                processing_class.pad_token_id = pad_token_id
                processing_class.eos_token_id = processing_class.tokenizer.eos_token_id
            else:
                assert isinstance(processing_class, PreTrainedTokenizerBase), "processing_class must be an instance of PreTrainedTokenizerBase if it has no tokenizer attribute"
                pad_token_id = processing_class.pad_token_id

        self.vlm_module.post_model_init(model, processing_class)
        self.vlm_module.post_model_init(self.ref_model, processing_class)

        # Reward functions
        if not isinstance(reward_funcs, list):
            reward_funcs = [reward_funcs]
        for i, reward_func in enumerate(reward_funcs):
            if isinstance(reward_func, str):
                reward_funcs[i] = AutoModelForSequenceClassification.from_pretrained(
                    reward_func, num_labels=1, **model_init_kwargs
                )
        self.reward_funcs = reward_funcs

        # Reward processing class
        if reward_processing_classes is None:
            reward_processing_classes = [None] * len(reward_funcs)
        elif not isinstance(reward_processing_classes, list):
            reward_processing_classes = [reward_processing_classes]
        else:
            if len(reward_processing_classes) != len(reward_funcs):
                raise ValueError("The number of reward processing classes must match the number of reward functions.")

        for i, (reward_processing_class, reward_func) in enumerate(zip(reward_processing_classes, reward_funcs)):
            if isinstance(reward_func, PreTrainedModel):
                if reward_processing_class is None:
                    reward_processing_class = AutoTokenizer.from_pretrained(reward_func.config._name_or_path)
                if reward_processing_class.pad_token_id is None:
                    reward_processing_class.pad_token = reward_processing_class.eos_token
                # The reward model computes the reward for the latest non-padded token in the input sequence.
                # So it's important to set the pad token ID to the padding token ID of the processing class.
                reward_func.config.pad_token_id = reward_processing_class.pad_token_id
                reward_processing_classes[i] = reward_processing_class
        self.reward_processing_classes = reward_processing_classes

        # Data collator
        def data_collator(features):  # No data collation is needed in GRPO
            qry = [x['qry'] for x in features]
            pos = [x['pos'] for x in features]
            return {"qry": qry, "pos": pos}

        # Training arguments
        self.max_prompt_length = args.max_prompt_length
        self.max_prompt_length = None
        if args.max_prompt_length is not None:
            warnings.warn("Setting max_prompt_length is currently not supported, it has been set to None")

        self.max_completion_length = args.max_completion_length  # = |o_i| in the GRPO paper
        self.num_generations = args.num_generations  # = G in the GRPO paper
        self.generation_config = GenerationConfig(
            max_new_tokens=self.max_completion_length,
            do_sample=True,  
            temperature=1,
            pad_token_id=pad_token_id,
        )
        if hasattr(self.vlm_module, "get_eos_token_id"): # For InternVL
            self.generation_config.eos_token_id = self.vlm_module.get_eos_token_id(processing_class)
            print(222, self.vlm_module.get_eos_token_id(processing_class))
        self.beta = args.beta
        self.epsilon_low = args.epsilon
        self.epsilon_high = args.epsilon_high if args.epsilon_high is not None else args.epsilon

        # Multi-step
        self.num_iterations = args.num_iterations  # = 𝜇 in the GRPO paper
        # Tracks the number of iterations (forward + backward passes), including those within a gradient accumulation cycle
        self._step = 0
        # Buffer the batch to reuse generated outputs across multiple updates
        self._buffered_qry_inputs = [None] * args.gradient_accumulation_steps
        self._buffered_pos_inputs = [None] * args.gradient_accumulation_steps

        # The trainer estimates the number of FLOPs (floating-point operations) using the number of elements in the
        # input tensor associated with the key "input_ids". However, in GRPO, the sampled data does not include the
        # "input_ids" key. Instead, the available keys is "prompt". As a result, the trainer issues the warning:
        # "Could not estimate the number of tokens of the input, floating-point operations will not be computed." To
        # suppress this warning, we set the "estimate_tokens" key in the model's "warnings_issued" dictionary to True.
        # This acts as a flag to indicate that the warning has already been issued.
        model.warnings_issued["estimate_tokens"] = True

        # Initialize the metrics
        self._metrics = defaultdict(list)

        super().__init__(
            model=model,
            args=args,
            data_collator=data_collator,
            train_dataset=train_dataset,
            eval_dataset=eval_dataset,
            processing_class=processing_class,
            callbacks=callbacks,
            optimizers=optimizers,
        )

        # Check if the per_device_train/eval_batch_size * num processes can be divided by the number of generations
        num_processes = self.accelerator.num_processes
        global_batch_size = args.per_device_train_batch_size * num_processes
        possible_values = [n_gen for n_gen in range(2, global_batch_size + 1) if (global_batch_size) % n_gen == 0]
        if self.num_generations not in possible_values:
            raise ValueError(
                f"The global train batch size ({num_processes} x {args.per_device_train_batch_size}) must be evenly "
                f"divisible by the number of generations per prompt ({self.num_generations}). Given the current train "
                f"batch size, the valid values for the number of generations are: {possible_values}."
            )
        if self.args.eval_strategy != "no":
            global_batch_size = args.per_device_eval_batch_size * num_processes
            possible_values = [n_gen for n_gen in range(2, global_batch_size + 1) if (global_batch_size) % n_gen == 0]
            if self.num_generations not in possible_values:
                raise ValueError(
                    f"The global eval batch size ({num_processes} x {args.per_device_eval_batch_size}) must be evenly "
                    f"divisible by the number of generations per prompt ({self.num_generations}). Given the current "
                    f"eval batch size, the valid values for the number of generations are: {possible_values}."
                )

        # Ensure each process receives a unique seed to prevent duplicate completions when generating with
        # transformers if num_generations exceeds per_device_train_batch_size. We could skip it if we use vLLM, but
        # it's safer to set it in all cases.
        set_seed(args.seed, device_specific=True)

        # Gradient accumulation requires scaled loss. Normally, loss scaling in the parent class depends on whether the
        # model accepts loss-related kwargs. Since we compute our own loss, this check is irrelevant. We set
        # self.model_accepts_loss_kwargs to False to enable scaling.
        self.model_accepts_loss_kwargs = False

        if self.ref_model is not None:
            if self.is_deepspeed_enabled:
                self.ref_model = prepare_deepspeed(self.ref_model, self.accelerator)
            else:
                self.ref_model = self.accelerator.prepare_model(self.ref_model, evaluation_mode=True)

        for i, reward_func in enumerate(self.reward_funcs):
            if isinstance(reward_func, PreTrainedModel):
                self.reward_funcs[i] = self.accelerator.prepare_model(reward_func, evaluation_mode=True)

    def _enable_gradient_checkpointing(self, model: PreTrainedModel, args: GRPOConfig) -> PreTrainedModel:
        """Enables gradient checkpointing for the model."""
        # Ensure use_cache is disabled
        model.config.use_cache = False

        # Enable gradient checkpointing on the base model for PEFT
        if is_peft_model(model):
            model.base_model.gradient_checkpointing_enable()
        # Enable gradient checkpointing for non-PEFT models
        else:
            try:
                model.gradient_checkpointing_enable()
            except:
                # For InternVL; these operations are copied from the original training script of InternVL
                model.language_model.config.use_cache = False
                model.vision_model.gradient_checkpointing = True
                model.vision_model.encoder.gradient_checkpointing = True
                model.language_model._set_gradient_checkpointing()
                # This line is necessary, otherwise the `model.gradient_checkpointing_enable()` will be executed during the training process, leading to an error since InternVL does not support this operation.
                args.gradient_checkpointing = False

        gradient_checkpointing_kwargs = args.gradient_checkpointing_kwargs or {}
        use_reentrant = (
            "use_reentrant" not in gradient_checkpointing_kwargs or gradient_checkpointing_kwargs["use_reentrant"]
        )

        if use_reentrant:
            model.enable_input_require_grads()

        return model
    
    def _set_signature_columns_if_needed(self):
        # If `self.args.remove_unused_columns` is True, non-signature columns are removed.
        # By default, this method sets `self._signature_columns` to the model's expected inputs.
        # In GRPOTrainer, we preprocess data, so using the model's signature columns doesn't work.
        # Instead, we set them to the columns expected by the `training_step` method, hence the override.
        if self._signature_columns is None:
            self._signature_columns = ["prompt"]


    # Get the per-token log probabilities for the completions for the model and the reference model
    def _get_per_token_logps(self, model, input_ids, attention_mask, output_hidden_states=False, **custom_multimodal_inputs):

        output = model(input_ids=input_ids, attention_mask=attention_mask, output_hidden_states=output_hidden_states, **custom_multimodal_inputs)  # (B, L, V)
        if output_hidden_states:
            logits = output.logits
            embeddings = output.hidden_states[-1]
        else:
            logits = output.logits  # (B, L, V)

        logits = logits[:, :-1, :]  # (B, L-1, V), exclude the last logit: it corresponds to the next token pred
        input_ids = input_ids[:, 1:]  # (B, L-1), exclude the first input ID since we don't have logits for it
        # get the hidden states of the <gen_emb> token
        index = input_ids == self.processing_class.embed_token_id
        # print(f"Index: {index}")
        # print(f"Last 10 input_ids: {input_ids[:, -10:]}")
        # print(f"Processing class embed token id: {self.processing_class.embed_token_id}")
        
        if output_hidden_states:
            embeddings = embeddings[:, 1:, :]  # (B, L-1, d), exclude the first embeddings since we don't have input_ids for it
            # output_embeddings shape (B, d)
            output_embeddings = torch.zeros((embeddings.shape[0], embeddings.shape[-1]), device=embeddings.device)
            for i in range(len(input_ids)):
                # if the input exist a <gen_emb> token, use the hidden state of the last <gen_emb> token
                # else use the last hidden state due to the left padding
                # index[i] is a boolean tensor, if it contains at least one True, it means that the <gen_emb> token exists
                # and we can use the last hidden state of the <gen_emb> token
                if index[i].any():
                    all_embeddings = embeddings[i][index[i]]
                    output_embeddings[i] = all_embeddings[-1]
                else:
                    output_embeddings[i] = embeddings[i][-1]

            # normalize the output embeddings
            output_embeddings = torch.nn.functional.normalize(output_embeddings, p=2, dim=-1)

        # Compute the log probabilities for the input tokens. Use a loop to reduce memory peak.
        per_token_logps = []
        for logits_row, input_ids_row in zip(logits, input_ids):
            log_probs = logits_row.log_softmax(dim=-1)
            token_log_prob = torch.gather(log_probs, dim=1, index=input_ids_row.unsqueeze(1)).squeeze(1)
            per_token_logps.append(token_log_prob)
        
        if output_hidden_states:
            return torch.stack(per_token_logps), output_embeddings

        return torch.stack(per_token_logps)


    def _prepare_inputs(self, inputs):
        # Simple pass-through, just like original
        return inputs

    def _get_key_from_inputs(self, x, key):
        ele = x.get(key, None)
        assert ele is not None, f"The key {key} is not found in the input"
        if isinstance(ele, list):
            return [e for e in ele]
        else:
            return [ele]

    def _generate_completions(self, inputs: dict[str, Union[torch.Tensor, Any]], model) -> dict[str, Union[torch.Tensor, Any]]:
        device = self.accelerator.device
        prompts = [x["prompt"] for x in inputs]
        prompts_text = self.vlm_module.prepare_prompt(self.processing_class, inputs)

        image_inputs, video_inputs = process_vision_info(prompts)

        prompt_inputs = self.processing_class(
            text=prompts_text,
            images=image_inputs,
            videos=video_inputs,
            return_tensors="pt",
            padding=True,
            padding_side="left",
            add_special_tokens=False,
        )
        prompt_inputs = super()._prepare_inputs(prompt_inputs)

        prompt_ids, prompt_mask = prompt_inputs["input_ids"], prompt_inputs["attention_mask"]
        # print(f"Prompt IDs shape: {prompt_ids.shape}, Prompt Mask shape: {prompt_mask.shape}")

        # Generate completions
        with unwrap_model_for_generation(model, self.accelerator) as unwrapped_model:
            generate_returned_result = unwrapped_model.generate(
                **{k: v for k, v in prompt_inputs.items() if k not in self.vlm_module.get_non_generate_params()}, 
                generation_config=self.generation_config,
                # use_cache=True,  # Ensure that the model uses cache for generation
            )
            prompt_length = prompt_ids.size(1)
            if not self.vlm_module.is_embeds_input():
                prompt_completion_ids = generate_returned_result
                prompt_ids = prompt_completion_ids[:, :prompt_length]
                completion_ids = prompt_completion_ids[:, prompt_length:]
            else:
                # In this case, the input of the LLM backbone is the embedding of the combination of the image and text prompt
                # So the returned result of the `generate` method only contains the completion ids
                completion_ids = generate_returned_result
                prompt_completion_ids = torch.cat([prompt_ids, completion_ids], dim=1)

        # Mask everything after the first EOS token
        is_eos = completion_ids == self.processing_class.eos_token_id
        eos_idx = torch.full((is_eos.size(0),), is_eos.size(1), dtype=torch.long, device=device)
        eos_idx[is_eos.any(dim=1)] = is_eos.int().argmax(dim=1)[is_eos.any(dim=1)]
        sequence_indices = torch.arange(is_eos.size(1), device=device).expand(is_eos.size(0), -1)
        completion_mask = (sequence_indices <= eos_idx.unsqueeze(1)).int()

        # Concatenate prompt_mask with completion_mask for logit computation
        attention_mask = torch.cat([prompt_mask, completion_mask], dim=1)  # (B, P+C)

        # Get the multimodal inputs
        multimodal_keywords = self.vlm_module.get_custom_multimodal_keywords()
        multimodal_inputs = {k: prompt_inputs[k] if k in prompt_inputs else None for k in multimodal_keywords}
        with torch.no_grad():
            # When using num_iterations == 1, old_per_token_logps == per_token_logps, so we can skip its
            # computation here, and use per_token_logps.detach() instead.
            if self.num_iterations > 1:
                old_per_token_logps = self._get_per_token_logps(
                    model, prompt_completion_ids, attention_mask, **multimodal_inputs
                )
                old_per_token_logps = old_per_token_logps[:, prompt_length - 1:]
            else:
                old_per_token_logps = None

            if self.beta == 0.0:
                ref_per_token_logps = None
            elif self.ref_model is not None:
                ref_per_token_logps = self._get_per_token_logps(
                    self.ref_model, prompt_completion_ids, attention_mask, **multimodal_inputs
                )
            else:
                with self.accelerator.unwrap_model(model).disable_adapter():
                    ref_per_token_logps = self._get_per_token_logps(
                        model, prompt_completion_ids, attention_mask, **multimodal_inputs
                    )

        if ref_per_token_logps is not None:
            ref_per_token_logps = ref_per_token_logps[:, prompt_length - 1:]

        # Decode the generated completions
        completions = self.processing_class.batch_decode(completion_ids, skip_special_tokens=True)
        if is_conversational(inputs[0]):
            completions = [[{"role": "assistant", "content": completion}] for completion in completions]

        return {
            "prompt_ids": prompt_ids,
            "prompt_mask": prompt_mask,
            "completion_ids": completion_ids,
            "completion_mask": completion_mask,
            "old_per_token_logps": old_per_token_logps,
            "ref_per_token_logps": ref_per_token_logps,
            "multimodal_inputs": multimodal_inputs,
            "completions": completions,
        }

    def _score_completions(self, inputs: dict[str, Union[torch.Tensor, Any]], generation_results: dict[str, Union[torch.Tensor, Any]], qry_embeddings, target_embeddings, log_key) -> dict[str, Union[torch.Tensor, Any]]:
        
        device = self.accelerator.device
        prompts = [x["prompt"] for x in inputs]
        prompt_ids = generation_results["prompt_ids"]
        prompt_mask = generation_results["prompt_mask"]
        completion_ids = generation_results["completion_ids"]
        completion_mask = generation_results["completion_mask"]
        old_per_token_logps = generation_results["old_per_token_logps"]
        ref_per_token_logps = generation_results["ref_per_token_logps"]
        multimodal_inputs = generation_results["multimodal_inputs"]
        completions = generation_results["completions"]

        # Compute the rewards
        # No need to duplicate prompts as we're not generating multiple completions per prompt

        rewards_per_func = torch.zeros(len(prompts), len(self.reward_funcs), device=device)
        for i, (reward_func, reward_processing_class) in enumerate(
            zip(self.reward_funcs, self.reward_processing_classes)
        ):
            if isinstance(reward_func, PreTrainedModel):
                if is_conversational(inputs[0]):
                    messages = [{"messages": p + c} for p, c in zip(prompts, completions)]
                    texts = [apply_chat_template(x, reward_processing_class)["text"] for x in messages]
                else:
                    texts = [p + c for p, c in zip(prompts, completions)]
                reward_inputs = reward_processing_class(
                    texts, return_tensors="pt", padding=True, padding_side="right", add_special_tokens=False
                )
                reward_inputs = super()._prepare_inputs(reward_inputs)
                with torch.inference_mode():
                    rewards_per_func[:, i] = reward_func(**reward_inputs).logits[:, 0]  # Shape (B*G,)
            else:
                # Repeat all input columns (but "prompt" and "completion") to match the number of generations
                reward_kwargs = {key: [] for key in inputs[0].keys() if key not in ["prompt", "completion"]}
                for key in reward_kwargs:
                    for example in inputs:
                        # No need to duplicate prompts as we're not generating multiple completions per prompt
                        # reward_kwargs[key].extend([example[key]] * self.num_generations)
                        reward_kwargs[key].extend([example[key]])
                output_reward_func = reward_func(prompts=prompts, completions=completions, **reward_kwargs)
                rewards_per_func[:, i] = torch.tensor(output_reward_func, dtype=torch.float32, device=device)

        # Gather rewards across processes
        rewards_per_func = self.accelerator.gather(rewards_per_func)
        
        # Sum the rewards from all reward functions
        rewards = rewards_per_func.sum(dim=1)

        # new added contrastive reward
        all_qry_embeddings = self.accelerator.gather(qry_embeddings)
        all_target_embeddings = self.accelerator.gather(target_embeddings)

        G = self.num_generations
        N = all_qry_embeddings.size(0) // G

        # Compute the cosine similarity between the embeddings
        logits_per_qry = all_qry_embeddings @ all_target_embeddings.T # (B*D, B*D), D:device
        

        # Calculate the rewards based on the similarity scores
        rewards_embed = torch.zeros(N * G, device=device)
        similarity_diff = torch.zeros(N * G, device=device)
        rewards_threshold = torch.zeros(N * G, device=device)  # Threshold for positive similarity
        threshold = 0.5  # Threshold for positive similarity, can be adjusted
        # Iterate over each query
        for i in range(logits_per_qry.size(0)):
            # Get the group index for the current query
            group_idx = i // G

            # Get the similarity scores for the current query
            qry_similarities = logits_per_qry[i]

            # Get the positive similarities (within the same group)
            positive_similarities = qry_similarities[group_idx * G:(group_idx + 1) * G]

            # Get the negative similarities (outside the current group)
            negative_similarities = torch.cat([
                qry_similarities[:group_idx * G],
                qry_similarities[(group_idx + 1) * G:]
            ])



            # Count the proportion of positive_similarities in the top-G of the entire qry_similarities
            topk_values, topk_indices = torch.topk(qry_similarities, G)
            pos_start = group_idx * G
            pos_end = (group_idx + 1) * G
            num_positives_in_topk = ((topk_indices >= pos_start) & (topk_indices < pos_end)).sum().item()

            # if num_positives_in_topk == G:
            #     rewards_embed[i] = 1.0
            # else:
            #     rewards_embed[i] = 0
            # rewards_embed[i] *= positive_similarities.mean() - negative_similarities.mean()
            
            rewards_embed[i] = num_positives_in_topk / G
            similarity_diff[i] = positive_similarities.mean() - negative_similarities.mean()



        

        rewards_embed = rewards_embed * similarity_diff

        # print("rewards_embed: ", rewards_embed)
        rewards = rewards + rewards_embed


        # Compute grouped-wise rewards
        # Each group consists of num_generations completions for the same prompt
        mean_grouped_rewards = rewards.view(-1, self.num_generations).mean(dim=1)
        std_grouped_rewards = rewards.view(-1, self.num_generations).std(dim=1)
        
        # Normalize the rewards to compute the advantages
        mean_grouped_rewards = mean_grouped_rewards.repeat_interleave(self.num_generations, dim=0)
        std_grouped_rewards = std_grouped_rewards.repeat_interleave(self.num_generations, dim=0)
        advantages = (rewards - mean_grouped_rewards) / (std_grouped_rewards + 1e-4)
        
        # print(f"Advantages: {advantages}")
        # Get only the local slice of advantages
        process_slice = slice(
            self.accelerator.process_index * len(prompts),
            (self.accelerator.process_index + 1) * len(prompts),
        )
        advantages = advantages[process_slice]

        # Log the metrics
        completion_length = self.accelerator.gather_for_metrics(completion_mask.sum(1)).float().mean().item()
        self._metrics[f"{log_key}_completion_length"].append(completion_length)

        # Log the emb rewards
        self._metrics[f"{log_key}_embed_rewards/emb"].append(self.accelerator.gather_for_metrics(rewards_embed).mean().item())

        reward_per_func = self.accelerator.gather_for_metrics(rewards_per_func).mean(0)
        for i, reward_func in enumerate(self.reward_funcs):
            if isinstance(reward_func, PreTrainedModel):
                reward_func_name = reward_func.config._name_or_path.split("/")[-1]
            else:
                reward_func_name = reward_func.__name__
            self._metrics[f"{log_key}_rewards/{reward_func_name}"].append(reward_per_func[i].item())

        self._metrics[f"{log_key}_reward"].append(self.accelerator.gather_for_metrics(rewards).mean().item())

        self._metrics[f"{log_key}_reward_std"].append(self.accelerator.gather_for_metrics(std_grouped_rewards).mean().item())


        return advantages

    def compute_loss(self, model, inputs, return_outputs=False, num_items_in_batch=None):
        if return_outputs:
            raise ValueError("The GRPOTrainer does not support returning outputs")

        # Check if we need to generate new completions or use buffered ones
        if self.state.global_step % self.num_iterations == 0:
            # print("Generating new completions...")
            qry_inputs = self._generate_completions(inputs['qry'], model)
            self._buffered_qry_inputs[self._step % self.args.gradient_accumulation_steps] = qry_inputs

            
            pos_inputs = self._generate_completions(inputs['pos'], model)
            self._buffered_pos_inputs[self._step % self.args.gradient_accumulation_steps] = pos_inputs
        else:
            qry_inputs = self._buffered_qry_inputs[self._step % self.args.gradient_accumulation_steps]
            pos_inputs = self._buffered_pos_inputs[self._step % self.args.gradient_accumulation_steps]
        self._step += 1

        # Get the prepared inputs
        qry_prompt_ids, qry_prompt_mask = qry_inputs["prompt_ids"], qry_inputs["prompt_mask"]
        qry_completion_ids, qry_completion_mask = qry_inputs["completion_ids"], qry_inputs["completion_mask"]
        qry_multimodal_inputs = qry_inputs["multimodal_inputs"]
        
        pos_prompt_ids, pos_prompt_mask = pos_inputs["prompt_ids"], pos_inputs["prompt_mask"]
        pos_completion_ids, pos_completion_mask = pos_inputs["completion_ids"], pos_inputs["completion_mask"]
        pos_multimodal_inputs = pos_inputs["multimodal_inputs"]

        # Concatenate for full sequence
        qry_input_ids = torch.cat([qry_prompt_ids, qry_completion_ids], dim=1)
        qry_attention_mask = torch.cat([qry_prompt_mask, qry_completion_mask], dim=1)

        pos_input_ids = torch.cat([pos_prompt_ids, pos_completion_ids], dim=1)
        pos_attention_mask = torch.cat([pos_prompt_mask, pos_completion_mask], dim=1)

        # Get the current policy's log probabilities
        qry_per_token_logps, qry_embeddings = self._get_per_token_logps(model, qry_input_ids, qry_attention_mask, output_hidden_states=True, **qry_multimodal_inputs)
        pos_per_token_logps, pos_embeddings = self._get_per_token_logps(model, pos_input_ids, pos_attention_mask, output_hidden_states=True, **pos_multimodal_inputs)
        
        # Get rid of the prompt (-1 because of the shift done in get_per_token_logps)
        qry_per_token_logps = qry_per_token_logps[:, qry_prompt_ids.size(1) - 1:]
        pos_per_token_logps = pos_per_token_logps[:, pos_prompt_ids.size(1) - 1:]
        
        # Calculate the advantages
        # print("Calculating advantages...")
        qry_advantages = self._score_completions(inputs['qry'], qry_inputs, qry_embeddings.detach(), pos_embeddings.detach(), "qry")
        pos_advantages = self._score_completions(inputs['pos'], pos_inputs, pos_embeddings.detach(), qry_embeddings.detach(), "pos")
        # print("qry_advantages: ", qry_advantages)
        # print("pos_advantages: ", pos_advantages)
        # print("End calculating advantages...")
        # When using num_iterations == 1, old_per_token_logps == per_token_logps, so we can skip its computation
        # and use per_token_logps.detach() instead
        qry_old_per_token_logps = qry_inputs["old_per_token_logps"] if self.num_iterations > 1 else qry_per_token_logps.detach()
        pos_old_per_token_logps = pos_inputs["old_per_token_logps"] if self.num_iterations > 1 else pos_per_token_logps.detach()

        # Compute the policy ratio and clipped version
        qry_coef_1 = torch.exp(qry_per_token_logps - qry_old_per_token_logps)
        qry_coef_2 = torch.clamp(qry_coef_1, 1 - self.epsilon_low, 1 + self.epsilon_high)
        qry_per_token_loss1 = qry_coef_1 * qry_advantages.unsqueeze(1)
        qry_per_token_loss2 = qry_coef_2 * qry_advantages.unsqueeze(1)
        qry_per_token_loss = -torch.min(qry_per_token_loss1, qry_per_token_loss2)

        pos_coef_1 = torch.exp(pos_per_token_logps - pos_old_per_token_logps)
        pos_coef_2 = torch.clamp(pos_coef_1, 1 - self.epsilon_low, 1 + self.epsilon_high)
        pos_per_token_loss1 = pos_coef_1 * pos_advantages.unsqueeze(1)
        pos_per_token_loss2 = pos_coef_2 * pos_advantages.unsqueeze(1)
        pos_per_token_loss = -torch.min(pos_per_token_loss1, pos_per_token_loss2)

        # Add KL penalty if beta > 0
        if self.beta > 0:
            qry_ref_per_token_logps = qry_inputs["ref_per_token_logps"]
            qry_per_token_kl = torch.exp(qry_ref_per_token_logps - qry_per_token_logps) - (qry_ref_per_token_logps - qry_per_token_logps) - 1
            qry_per_token_loss = qry_per_token_loss + self.beta * qry_per_token_kl

            pos_ref_per_token_logps = pos_inputs["ref_per_token_logps"]
            pos_per_token_kl = torch.exp(pos_ref_per_token_logps - pos_per_token_logps) - (pos_ref_per_token_logps - pos_per_token_logps) - 1
            pos_per_token_loss = pos_per_token_loss + self.beta * pos_per_token_kl

            # Log KL divergence
            qry_mean_kl = ((qry_per_token_kl * qry_completion_mask).sum(dim=1) / qry_completion_mask.sum(dim=1)).mean()
            self._metrics["qry_kl"].append(self.accelerator.gather_for_metrics(qry_mean_kl).mean().item())

            pos_mean_kl = ((pos_per_token_kl * pos_completion_mask).sum(dim=1) / pos_completion_mask.sum(dim=1)).mean()
            self._metrics["pos_kl"].append(self.accelerator.gather_for_metrics(pos_mean_kl).mean().item())
            

        # Compute final loss
        # qry_loss = ((qry_per_token_loss * qry_completion_mask).sum(dim=1) / qry_completion_mask.sum(dim=1)).mean()
        # pos_loss = ((pos_per_token_loss * pos_completion_mask).sum(dim=1) / pos_completion_mask.sum(dim=1)).mean()

        qry_loss = ((qry_per_token_loss * qry_completion_mask).sum(dim=1) / self.max_completion_length).mean()
        pos_loss = ((pos_per_token_loss * pos_completion_mask).sum(dim=1) / self.max_completion_length).mean()



        loss = (qry_loss + pos_loss) / 2
        # loss = qry_loss

        # Log clip ratio
        qry_is_clipped = (qry_per_token_loss1 < qry_per_token_loss2).float()
        qry_clip_ratio = (qry_is_clipped * qry_completion_mask).sum() / qry_completion_mask.sum()
        self._metrics["qry_clip_ratio"].append(self.accelerator.gather_for_metrics(qry_clip_ratio).mean().item())

        pos_is_clipped = (pos_per_token_loss1 < pos_per_token_loss2).float()
        pos_clip_ratio = (pos_is_clipped * pos_completion_mask).sum() / pos_completion_mask.sum()
        self._metrics["pos_clip_ratio"].append(self.accelerator.gather_for_metrics(pos_clip_ratio).mean().item())
        
        # sync
        # self.accelerator.synchronize()
        # print("loss", loss)
        return loss

    def log(self, logs: dict[str, float], start_time: Optional[float] = None) -> None:
        metrics = {key: sum(val) / len(val) for key, val in self._metrics.items()}  # average the metrics
        logs = {**logs, **metrics}
        if version.parse(transformers.__version__) >= version.parse("4.47.0.dev0"):
            super().log(logs, start_time)
        else:  # transformers<=4.46
            super().log(logs)
        self._metrics.clear()

    def create_model_card(
        self,
        model_name: Optional[str] = None,
        dataset_name: Optional[str] = None,
        tags: Union[str, list[str], None] = None,
    ):
        """
        Creates a draft of a model card using the information available to the `Trainer`.

        Args:
            model_name (`str` or `None`, *optional*, defaults to `None`):
                Name of the model.
            dataset_name (`str` or `None`, *optional*, defaults to `None`):
                Name of the dataset used for training.
            tags (`str`, `list[str]` or `None`, *optional*, defaults to `None`):
                Tags to be associated with the model card.
        """
        if not self.is_world_process_zero():
            return

        if hasattr(self.model.config, "_name_or_path") and not os.path.isdir(self.model.config._name_or_path):
            base_model = self.model.config._name_or_path
        else:
            base_model = None

        tags = tags or []
        if isinstance(tags, str):
            tags = [tags]

        if hasattr(self.model.config, "unsloth_version"):
            tags.append("unsloth")

        citation = textwrap.dedent(
            """\
            @article{zhihong2024deepseekmath,
                title        = {{DeepSeekMath: Pushing the Limits of Mathematical Reasoning in Open Language Models}},
                author       = {Zhihong Shao and Peiyi Wang and Qihao Zhu and Runxin Xu and Junxiao Song and Mingchuan Zhang and Y. K. Li and Y. Wu and Daya Guo},
                year         = 2024,
                eprint       = {arXiv:2402.03300},
            """
        )

        model_card = generate_model_card(
            base_model=base_model,
            model_name=model_name,
            hub_model_id=self.hub_model_id,
            dataset_name=dataset_name,
            tags=tags,
            wandb_url=wandb.run.get_url() if is_wandb_available() and wandb.run is not None else None,
            comet_url=get_comet_experiment_url(),
            trainer_name="GRPO",
            trainer_citation=citation,
            paper_title="DeepSeekMath: Pushing the Limits of Mathematical Reasoning in Open Language Models",
            paper_id="2402.03300",
        )

        model_card.save(os.path.join(self.args.output_dir, "README.md"))

    def _get_train_sampler(self) -> Sampler:
        """Returns a sampler that ensures proper data sampling for GRPO training."""
        effective_batch_size = (
            self.args.per_device_train_batch_size
            * self.accelerator.num_processes
            * self.args.gradient_accumulation_steps
        )
        
        # return RepeatRandomSampler(
        #     data_source=self.train_dataset,
        #     mini_repeat_count=self.num_generations,
        #     batch_size=effective_batch_size // self.num_generations,
        #     repeat_count=self.num_iterations,
        #     seed=self.args.seed,
        # )
        
        return RepeatSampler(
            data_source=self.train_dataset,
            mini_repeat_count=self.num_generations,
            batch_size=effective_batch_size // self.num_generations,
            repeat_count=self.num_iterations,
        )

    def _get_eval_sampler(self, eval_dataset) -> Sampler:
        """Returns a sampler for evaluation."""
        return RepeatRandomSampler(
            data_source=eval_dataset,
            mini_repeat_count=self.num_generations,
            seed=self.args.seed,
        )
