import transformers
from dataclasses import dataclass, field
from typing import Dict, Optional, Sequence, List


@dataclass
class ModelArguments:
    model_name_or_path: Optional[str] = field(default="Qwen/Qwen2.5-VL-3B-Instruct")
    tune_mm_llm: bool = field(default=False)
    tune_mm_mlp: bool = field(default=False)
    tune_mm_vision: bool = field(default=False)
    attn_implementation: str = field(default="flash_attention_2")
    use_lora: bool = field(default=False)
    lora_r: int = field(default=8)
    lora_alpha: int = field(default=16)
    lora_dropout: float = field(default=0.1)
    coconut_force_reinit_all_tokens: bool = field(
        default=False,
        metadata={
            "help": "Force re-initialize ALL special token embeddings (including pretrained <gen_emb>/<disc_emb>). "
                    "Use for ablation experiments only."
        },
    )
    lora_use_dora: bool = field(
        default=False,
        metadata={"help": "Enable DoRA variant for LoRA (higher memory usage)."},
    )
    lora_target_modules: str = field(
        default="qkv_proj,o_proj,gate_up_proj,down_proj,k_proj,q_proj,out_proj,v_proj",
        metadata={"help": "lora target modules"}
    )
    latent_moe_enable: bool = field(
        default=False,
        metadata={"help": "Enable latent MoE transition in latent loop."},
    )
    latent_moe_num_experts: int = field(
        default=4,
        metadata={"help": "Number of routed experts in latent MoE."},
    )
    latent_moe_top_k: int = field(
        default=2,
        metadata={"help": "Top-k experts selected by latent MoE router."},
    )
    latent_moe_use_shared_expert: bool = field(
        default=True,
        metadata={"help": "Whether to enable shared expert branch in latent MoE."},
    )
    latent_moe_balance_loss_weight: float = field(
        default=0.0,
        metadata={"help": "Weight of latent MoE balance loss; 0.0 keeps baseline objective."},
    )
    latent_moe_step_embed_max_steps: int = field(
        default=32,
        metadata={"help": "Max number of latent steps covered by step embedding table."},
    )
    latent_moe_context_type: str = field(
        default="prefix_last",
        metadata={
            "help": "Router context source for latent MoE. One of: none, prefix_last, disc."
        },
    )
    latent_moe_expert_dropout: float = field(
        default=0.0,
        metadata={"help": "Dropout probability inside each latent MoE expert MLP."},
    )


@dataclass
class DataArguments:
    dataset_use: str = field(default="")
    video_max_frames: Optional[int] = field(default=8)
    video_min_frames: Optional[int] = field(default=4)
    data_flatten: bool = field(default=False)
    base_interval: int = field(default=2)
    max_pixels: int = field(default=28 * 28 * 576)
    min_pixels: int = field(default=28 * 28 * 16)
    video_max_frame_pixels: int = field(default=32 * 28 * 28)
    video_min_frame_pixels: int = field(default=4 * 28 * 28)
    data_group: bool = field(default=True)
    coconut_sampling_strategy: str = field(
        default="legacy",
        metadata={
            "help": "Sampling strategy for COCONUT training. `legacy` keeps the original sampler; "
                    "`subset_balanced` makes each global batch contain one sub-dataset and spreads "
                    "every sub-dataset across the full curriculum timeline."
        },
    )
    coconut_annotation_path: str = field(
        default="/home/share/yty_data/UME_R1_train/UME-sft-train.jsonl",
        metadata={"help": "Path to UME SFT json/jsonl annotations. Prefer jsonl for lower RAM."},
    )
    coconut_media_root: str = field(
        default="/home/share/yty_data/vlm2vec_train",
        metadata={"help": "Optional media root used to resolve relative image/video paths."},
    )
    coconut_subset_filter: str = field(
        default="CIRR",
        metadata={"help": "Comma-separated dataset_name filters. Default keeps CIRR subset."},
    )
    coconut_use_qry: bool = field(
        default=True,
        metadata={"help": "Whether to use qry side samples."},
    )
    coconut_use_pos: bool = field(
        default=True,
        metadata={"help": "Whether to use pos side samples."},
    )
    coconut_curriculum_stages: str = field(
        default="0,0.25,0.5,0.75,1.0",
        metadata={"help": "Comma-separated replacement ratios for curriculum stages."},
    )
    coconut_final_stage_portion: float = field(
        default=0.5,
        metadata={
            "help": "Portion of total training reserved for the final curriculum stage. "
                    "The remaining portion is evenly split across earlier stages."
        },
    )
    coconut_latent_answer_in_final_half: bool = field(
        default=False,
        metadata={
            "help": "If True, split the final curriculum stage into two halves: keep answer text in the first half, "
                    "then drop answer text in the second half so latent steps are followed by <gen_emb>."
        },
    )
    coconut_final_stage_answer_portion: float = field(
        default=0.5,
        metadata={
            "help": "Portion inside the final curriculum stage where answer text is removed when "
                    "coconut_latent_answer_in_final_half=True. 0.5 means the second half of the final stage."
        },
    )
    coconut_think_segments: int = field(
        default=4,
        metadata={"help": "Number of coarse thought segments (m)."},
    )
    coconut_ct_tokens_per_segment: int = field(
        default=1,
        metadata={"help": "Latent length multiplier per replaced segment (c)."},
    )
    coconut_include_gen_emb_loss: bool = field(
        default=True,
        metadata={"help": "Whether to include <gen_emb> token in CE loss."},
    )
    coconut_gen_contrastive_weight: float = field(
        default=1.0,
        metadata={"help": "Weight for gen embedding contrastive loss."},
    )
    coconut_disc_contrastive_weight: float = field(
        default=1.0,
        metadata={"help": "Weight for disc embedding contrastive loss."},
    )
    coconut_contrastive_logit_scale: float = field(
        default=50.0,
        metadata={"help": "Logit scale used by contrastive loss."},
    )
    coconut_contrastive_cross_device: bool = field(
        default=True,
        metadata={"help": "Whether to gather contrastive pairs across GPUs."},
    )
    coconut_contrastive_local_loss: bool = field(
        default=True,
        metadata={"help": "If True, use local anchors against global negatives."},
    )
    coconut_debug_disc_oracle_pos_from_qry: bool = field(
        default=False,
        metadata={
            "help": "Debug only: replace disc pos reps with detached disc qry reps to verify contrastive label alignment."
        },
    )
    coconut_debug_contrastive_stats: bool = field(
        default=False,
        metadata={
            "help": "Debug only: log contrastive diag/offdiag/top1 stats (for gen/disc) during training."
        },
    )
    coconut_debug_disc_fullseq_alignment: bool = field(
        default=False,
        metadata={
            "help": "Debug only: compare COCONUT prefix disc reps against full-sequence disc reps and log cosine stats."
        },
    )
    coconut_enable_oom_precheck: bool = field(
        default=True,
        metadata={
            "help": "Run per-stage OOM precheck before actual training starts."
        },
    )
    coconut_oom_precheck_batches: int = field(
        default=2,
        metadata={
            "help": "How many mini-batches to probe for each prechecked curriculum stage."
        },
    )
    coconut_oom_precheck_subsets: str = field(
        default="",
        metadata={
            "help": "Comma-separated dataset names to sample from during OOM precheck "
                    "(e.g. 'K700,Video-MME,YouCook2'). Empty means use the training dataloader as-is."
        },
    )


@dataclass
class TrainingArguments(transformers.TrainingArguments):
    cache_dir: Optional[str] = field(default=None)
    optim: str = field(default="adamw_torch")
    model_max_length: int = field(
        default=512,
        metadata={
            "help": "Maximum sequence length. Sequences will be right padded (and possibly truncated)."
        },
    )
    mm_projector_lr: Optional[float] = None
    vision_tower_lr: Optional[float] = None

