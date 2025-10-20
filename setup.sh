# conda create -n vlm-r1 python=3.11 
# conda activate vlm-r1

# Install the packages in open-r1-multimodal .
cd src/r1-train # We edit the grpo_embed.py and grpo_trainer.py in open-r1 repo.
pip install -e ".[dev]"

# Addtional modules
pip install wandb==0.18.3
pip install tensorboardx
pip install transformers==4.49.0
pip install torch==2.6.0
pip install qwen_vl_utils torchvision
pip install qwen-vl-utils==0.0.11
pip install flash-attn==2.7.2 --no-build-isolation
pip install babel
pip install python-Levenshtein
pip install matplotlib
pip install pycocotools
pip install openai
pip install httpx[socks]
pip install peft
pip install einops
pip install accelerate
pip install timm
pip install matplotlib