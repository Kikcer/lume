from datasets import load_dataset
from tqdm import tqdm
import json
import os


def load_vidore_dataset(split="train"):
    """
    Load the Vidore dataset from a JSON file.
    
    Args:
        split (str): The split of the dataset to load (default is "train").
        
    """
    dataset = load_dataset("vidore/colpali_train_set", split="train")
    json_data = []
    prompt = "Represent the given image with the following question:"
    count = 0
    for item in tqdm(dataset):
        query = item['query']
        image = item['image']
        image_filename = item['image_filename']
        answer = item['answer']
        query = f"{prompt} {query}"
        
        # save the image to {count}.jpg
        image_path = os.path.join("MMEB-train/vidore/images", f"{count}.jpg")
        if not os.path.exists(os.path.dirname(image_path)):
            os.makedirs(os.path.dirname(image_path))
        
        image.save(image_path)
        json_data.append({
            "dataset": "vidore",
            "dataset_name": "vidore/colpali_train_set",
            "qry_text": query,
            "qry_image": image_path,
            "solution": "<embedding>",
            "pos_text": answer,
            "pos_image": "",
        })

        count += 1
    save_path = 'MMEB-train/vidore/vidore_train.json'
    if not os.path.exists(os.path.dirname(save_path)):
        os.makedirs(os.path.dirname(save_path))

    with open(save_path, 'w') as file:
        json.dump(json_data, file, indent=4)
    print(f"Dataset saved to {save_path} with {len(json_data)} entries.")

    return json_data

        

if __name__ == "__main__":

    load_vidore_dataset()
    print("Dataset loaded and saved successfully.")