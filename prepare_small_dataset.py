import glob
import os
import pickle

from datasets import Dataset, DatasetDict
from PIL import Image
from transformers import CLIPProcessor, T5Tokenizer

DATA_PATH = "../tw_data"
MODEL_NAME = "google/flan-t5-base"

tokenizer = T5Tokenizer.from_pretrained(MODEL_NAME)
processor = CLIPProcessor.from_pretrained("openai/clip-vit-base-patch16")


def tokenize_function(example):
    example["labels"] = tokenizer(
        example["caption"], padding="max_length", truncation=True, return_tensors="pt"
    ).input_ids
    return example


pkl_file_list = sorted(glob.glob(os.path.join(DATA_PATH, "conceptual/conceptual_val_*.pkl")))

problem_files = []
data_list = []
for pkl_file in pkl_file_list:
    with open(pkl_file, "rb") as file:
        data = pickle.load(file)

    for file_idx, info in data["info"].items():
        try:
            image = Image.open(os.path.join(DATA_PATH, f"conceptual/val/{file_idx}.jpg"))
            inputs = processor(images=image, return_tensors="pt")
        except:
            problem_files.append(file_idx)
            continue

        data_dict = {"file_idx": file_idx, "processed_image": inputs["pixel_values"], "caption": info["caption"]}
        data_list.append(data_dict)

dataset = Dataset.from_list(data_list)  # make dataset
tokenized_dataset = dataset.map(tokenize_function, batched=True)  # tokenize the caption to labels

train_val_split = tokenized_dataset.train_test_split(test_size=0.2)  # Split into train and validation sets

# Create a new DatasetDict with train and validation sets
tokenized_dataset_dict = DatasetDict({"train": train_val_split["train"], "validation": train_val_split["test"]})

print(f"Number of samples: {len(tokenized_dataset)}")
print(f"Number of problem files: {len(problem_files)}")
print(tokenized_dataset_dict)

tokenized_dataset.save_to_disk("temp/dataset/small")
tokenized_dataset_dict.save_to_disk("temp/dataset/small_split")
