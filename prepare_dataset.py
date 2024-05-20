import argparse
import glob
import pickle
import re

from datasets import Dataset
from PIL import Image
from transformers import CLIPProcessor, T5Tokenizer

model_name = "google/flan-t5-base"
tokenizer = T5Tokenizer.from_pretrained(model_name)
processor = CLIPProcessor.from_pretrained("openai/clip-vit-base-patch16")


def extract_train_number(file_path):
    match = re.search(r"_train_(\d+)", file_path)
    if match:
        return int(match.group(1))
    return None


def tokenize_function(example):
    example["labels"] = tokenizer(
        example["caption"], padding="max_length", truncation=True, return_tensors="pt"
    ).input_ids
    return example


def create_dataset(pkl_file_list):
    data_dict = {}
    for pkl_file in pkl_file_list:
        with open(pkl_file, "rb") as file:
            data = pickle.load(file)

        for file_idx, info in data["info"].items():
            if file_idx in data_dict:
                continue
            try:
                image = Image.open(f"../tw_data/conceptual/train/{file_idx}.jpg")
                inputs = processor(images=image, return_tensors="pt")
            except:
                continue

            data_dict[file_idx] = {
                "file_idx": file_idx,
                "processed_image": inputs["pixel_values"],
                "caption": info["caption"],
            }
    data_list = list(data_dict.values())
    print(len(data_list))
    dataset = Dataset.from_list(data_list)
    tokenized_dataset = dataset.map(tokenize_function, batched=True)
    return tokenized_dataset


def main():
    # Create the parser
    parser = argparse.ArgumentParser(description="Description of your program")
    parser.add_argument("-n", "--num", type=int, help="nth number")
    args = parser.parse_args()
    num = int(args.num)
    pkl_file_list = glob.glob("../tw_data/conceptual/conceptual_train_*.pkl")

    # Sort the list using the extracted numbers
    pkl_file_list = sorted(pkl_file_list, key=extract_train_number)

    target_pkl_file_list = pkl_file_list[num * 16 : (num + 1) * 16]
    print(f"target_pkl_file_list: {num * 16} to {(num + 1) * 16}")
    for pkl_file in target_pkl_file_list:
        print(f"{pkl_file}")
    dataset = create_dataset(target_pkl_file_list)

    dataset.save_to_disk(f"dataset/train/{num}")
    print(f"dataset/train/{num}")


if __name__ == "__main__":
    main()
