from datasets import load_dataset
from tqdm import tqdm
import json
import os
import datasets
import re
import numpy as np
VID_EXTENSIONS = (".mp4", ".avi", ".mov", ".mkv")
IMAGE_EXTENSIONS = (".jpg", ".jpeg", ".png")

def load_frames(frames_dir, filter_func=None):
    """
    Load image frames from a directory, with an optional filter function.
    """
    def natural_sort_key(filename):
        """Extract numbers from filenames for correct sorting."""
        return [int(text) if text.isdigit() else text.lower() for text in re.split(r'(\d+)', filename)]

    results = []
    if not os.path.exists(frames_dir) or not os.path.isdir(frames_dir):
        raise ValueError(f"Frames directory {frames_dir} does not exist or is not a directory.")
        return []
    frame_names = os.listdir(frames_dir)
    frame_names = sorted(frame_names, key=natural_sort_key)
    for frame_name in frame_names:
        ext = os.path.splitext(frame_name)[-1].lower()
        if ext.lower() in IMAGE_EXTENSIONS:
            if filter_func is None or filter_func(frame_name):
                image_path = f"{frames_dir}/{frame_name}"
                results.append(image_path)
    return results

def sample_frames(frames, num_segments):
    duration = len(frames)
    frame_id_array = np.linspace(0, duration-1, num_segments, dtype=int)
    frame_id_list = frame_id_array.tolist()
    last_frame_id = frame_id_list[-1]

    sampled_frames = []
    for frame_idx in frame_id_list:
        try:
            single_frame_path = frames[frame_idx]
        except:
            break
        sampled_frames.append(single_frame_path)
    # If total frame numbers is less than num_segments, append the last images to achieve
    while len(sampled_frames) < num_segments:
        sampled_frames.append(frames[last_frame_id])
    return sampled_frames
    
def process_video_frames(frame_dir, num_frames=None):
    """
    Load and sample frames as input into the model.
    """
    if num_frames == 0:
        return []
    frames = load_frames(frame_dir)
    if num_frames is None:
        return frames
    elif num_frames and num_frames <= len(frames):
        frames = sample_frames(frames, num_segments=num_frames)
    return frames

def process_conversations(conversations):
    query = conversations[0]["value"].replace("\n<video>", '')
    pos_text = conversations[1]["value"]
    return query, pos_text

def process_conversations_for_vret(conversations, prompt):
    caption = conversations[1]["value"]
    query = caption
    if prompt:
        query = prompt + query

    return query

VRET_QRY_PROMPT = "Find a video that contains the following visual content: "
VRET_TGT_PROMPT = "Understand the content of the provided video."

def load_llavahound_caption_dataset(dataset, frame_basedir, save_file, data_mode="caption_retrieval", num_frames=8):
    """
    Load the LLaVA-Hound dataset from a JSON file.
    """
    json_data = []
    count = 0
    for item in tqdm(dataset):
        data_id = item['id']
        conversations = item['conversations']
        video_id = item['video']
        if data_mode == "caption_retrieval":
            query, pos_text = process_conversations(conversations)
            frame_paths = process_video_frames(os.path.join(frame_basedir, video_id), num_frames=num_frames) # list of frame paths
            json_data.append({
                "dataset": "llavahound",
                "dataset_name": "llavahound_caption",
                "qry_text": query,
                "qry_image": "",
                "qry_video": frame_paths,
                "solution": "<embedding>",
                "pos_text": pos_text,
                "pos_image": "",
                "pos_video": [""],
            })
        elif data_mode == "video_retrieval":
            query = process_conversations_for_vret(conversations, prompt=VRET_QRY_PROMPT)
            frame_paths = process_video_frames(os.path.join(frame_basedir, video_id), num_frames=num_frames)
            pos_text = VRET_TGT_PROMPT
            json_data.append({
                "dataset": "llavahound",
                "dataset_name": "llavahound_caption",
                "qry_text": query,
                "qry_image": "",
                "qry_video": [""],
                "solution": "<embedding>",
                "pos_text": pos_text,
                "pos_image": "",
                "pos_video": frame_paths,
            })
        
    if not os.path.exists(os.path.dirname(save_file)):
        os.makedirs(os.path.dirname(save_file))
    with open(save_file, 'w') as file:
        json.dump(json_data, file, indent=4)
    print(f"Dataset saved to {save_file} with {len(json_data)} entries.")


if __name__ == "__main__":
    num_frames = 8  # Number of frames to sample from each video
    dataset_path = "MMEB-train/llavahound/train_video_and_instruction/video_instruction/train/sft/video_240k_caption_15k.jsonl"
    frame_basedir = "MMEB-train/llavahound/train_video_and_instruction/train_300k"
    dataset = datasets.load_dataset("json", split="train", data_files=dataset_path, streaming=False)
    save_file = "MMEB-train/llavahound/llavahound_caption_retrieval_train.json"
    load_llavahound_caption_dataset(dataset, frame_basedir, data_mode="caption_retrieval", num_frames=num_frames, save_file=save_file)

    save_file = "MMEB-train/llavahound/llavahound_video_retrieval_train.json"
    load_llavahound_caption_dataset(dataset, frame_basedir, data_mode="video_retrieval", num_frames=num_frames, save_file=save_file)
