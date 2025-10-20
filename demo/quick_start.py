from transformers import Qwen2VLForConditionalGeneration,AutoProcessor
from qwen_vl_utils import process_vision_info
import torch

model = Qwen2VLForConditionalGeneration.from_pretrained(
    "release/UME-R1-2B",
    torch_dtype=torch.bfloat16,
    attn_implementation="flash_attention_2",
    device_map="cuda:0",
)

processor = AutoProcessor.from_pretrained("release/UME-R1-2B")

prompt = '''Represent the above input text, images, videos, or any combination of the three as embeddings. 
First output the thinking process in <think> </think> tags and then summarize the entire input in a word or sentence. 
Finally, use the <gen_emb> tag to represent the entire input.'''



messages = [
    {
        "role": "user",
        "content": [
            {
                "type": "image",
                "image": "UME-R1/assets/example.jpg",
            },
            {"type": "text", "text": "Represent the given image with the following question: What is in the image?\n<disc_emb>\n" + prompt},
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

# Inference: Generation of the output
generated_output = model.generate(**inputs, max_new_tokens=8192, output_hidden_states=True, return_dict_in_generate=True, use_cache=True)
# Post-process the output
generated_ids = generated_output.sequences
hidden_states = generated_output.hidden_states

generated_ids_trimmed = [
    out_ids[len(in_ids) :] for in_ids, out_ids in zip(inputs.input_ids, generated_ids)
]

def get_embedding_idx(generated_ids_trimmed, EMBEDDING_TOKEN_ID):

    embedding_idx = []
    for i, out_ids in enumerate(generated_ids_trimmed):
        embed_exist = False
        for j in range(len(out_ids) - 1, -1, -1):
            if out_ids[j] == EMBEDDING_TOKEN_ID:
                embedding_idx.append(j + 1)
                embed_exist = True
                break
        if not embed_exist:
            embedding_idx.append(-1)

    return embedding_idx

def normalize_reps(reps):
    reps = torch.nn.functional.normalize(reps, p=2, dim=-1)
    return reps

# Get the last hidden state of the <gen_emb> token
embedding_idx = get_embedding_idx(generated_ids_trimmed, processor.tokenizer.get_vocab()["<gen_emb>"])
embedding_reps = hidden_states[embedding_idx[0]][-1].squeeze(1)

# Normalize the representations
embedding_reps = normalize_reps(embedding_reps)

output_text = processor.batch_decode(
    generated_ids_trimmed, skip_special_tokens=False, clean_up_tokenization_spaces=False
)