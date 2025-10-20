from transformers import Qwen2VLForConditionalGeneration,AutoProcessor
from qwen_vl_utils import process_vision_info
import torch

pretrained_path = "release/UME-R1-2B"

# We recommend enabling flash_attention_2 for better acceleration and memory saving, especially in multi-image and video scenarios.
model = Qwen2VLForConditionalGeneration.from_pretrained(
    pretrained_path,
    torch_dtype=torch.bfloat16,
    attn_implementation="flash_attention_2",
    device_map="cuda:0",
)

# default processor
# processor = AutoProcessor.from_pretrained(pretrained_path)

# You can set min_pixels and max_pixels according to your needs, such as a token range of 256-1280, to balance performance and cost.
min_pixels = 768
max_pixels = 2359296
processor = AutoProcessor.from_pretrained(pretrained_path, min_pixels=min_pixels, max_pixels=max_pixels)

messages = [
    {
        "role": "user",
        "content": [
            {
                "type": "image",
                "image": "UME-R1/assets/example.jpg",
            },
            {"type": "text", "text": "Represent the given image with the following question: What is in the image?\n<disc_emb>\n"},
        ],
    }
]

# Preparation for inference
text = processor.apply_chat_template(
    messages, tokenize=False, add_generation_prompt=True
)

image_inputs, video_inputs = process_vision_info(messages)
inputs = processor(
    text=[text],
    images=image_inputs,
    videos=video_inputs,
    padding=True,
    return_tensors="pt",
)
inputs = inputs.to(model.device)

def get_embedding_idx(generated_ids_trimmed, EMBEDDING_TOKEN_ID):

    embedding_idx = []
    # Search from the last token forward
    for i, out_ids in enumerate(generated_ids_trimmed):
        embed_exist = False
        for j in range(len(out_ids) - 1, -1, -1):
            if out_ids[j] == EMBEDDING_TOKEN_ID:
                embedding_idx.append(j)
                embed_exist = True
                break
        if not embed_exist:
            embedding_idx.append(-1)

    return embedding_idx

def normalize_reps(reps):
    # Normalize the representations
    reps = torch.nn.functional.normalize(reps, p=2, dim=-1)
    return reps

output = model(**inputs, output_hidden_states=True, return_dict=True)
hidden_states = output.hidden_states[-1][0]
# print("output.hidden_states shape: ", hidden_states.shape)
embedding_idx = get_embedding_idx(inputs['input_ids'], processor.tokenizer.get_vocab()["<disc_emb>"])

# Get the last hidden state of the <gen_emb> token
embedding_reps = hidden_states[embedding_idx[0]]

# Normalize the representations
embedding_reps = normalize_reps(embedding_reps)


pos_messages = [
    {
        "role": "user",
        "content": [
            {"type": "text", "text": "A cat and a dog\n<disc_emb>\n" },
        ],
    }
]

# Preparation for inference
text = processor.apply_chat_template(
    pos_messages, tokenize=False, add_generation_prompt=True
)
image_inputs, video_inputs = process_vision_info(pos_messages)

inputs = processor(
    text=[text],
    images=image_inputs,
    videos=video_inputs,
    padding=True,
    return_tensors="pt",
)
inputs = inputs.to(model.device)

output = model(**inputs, output_hidden_states=True, return_dict=True)
hidden_states = output.hidden_states[-1][0]
embedding_idx = get_embedding_idx(inputs['input_ids'], processor.tokenizer.get_vocab()["<disc_emb>"])

embedding_reps_pos = hidden_states[embedding_idx[0]]

# Normalize the representations
embedding_reps_pos = normalize_reps(embedding_reps_pos)

# Calculate cosine similarity
similarity = embedding_reps @ embedding_reps_pos.T
print("positive similarity: ", similarity)

neg_messages = [
    {
        "role": "user",
        "content": [
            {"type": "text", "text": "A cat and a tiger\n<disc_emb>\n" },
        ],
    }
]
# Preparation for inference
text = processor.apply_chat_template(
    neg_messages, tokenize=False, add_generation_prompt=True
)
image_inputs, video_inputs = process_vision_info(neg_messages)
inputs = processor(
    text=[text],
    images=image_inputs,
    videos=video_inputs,
    padding=True,
    return_tensors="pt",
)

inputs = inputs.to(model.device)

output = model(**inputs, output_hidden_states=True, return_dict=True)
hidden_states = output.hidden_states[-1][0]
embedding_idx = get_embedding_idx(inputs['input_ids'], processor.tokenizer.get_vocab()["<disc_emb>"])

embedding_reps_neg = hidden_states[embedding_idx[0]]

# Normalize the representations
embedding_reps_neg = normalize_reps(embedding_reps_neg)

# Calculate cosine similarity
similarity = embedding_reps @ embedding_reps_neg.T
print("negative similarity: ", similarity)
