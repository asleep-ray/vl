import os
from datetime import datetime

import pytz
import torch
from datasets import load_from_disk
from peft import LoraConfig, get_peft_model
from transformers import Trainer, TrainingArguments

import model
import utils

timestamp = datetime.now(pytz.timezone("Asia/Seoul")).strftime("%Y%m%d-%H%M%S")  # KST

# Disable MLflow integration
os.environ["DISABLE_MLFLOW_INTEGRATION"] = "TRUE"

# Load the dataset
dataset_dict = load_from_disk("dataset/small_split")

# Load the model
model_name = "google/flan-t5-base"
integrated_model = model.IntegratedModel(model_name)
print("integrated_model is loaded")
utils.print_trainable_model_module(integrated_model)


lora_config = LoraConfig(
    r=8,  # Rank
    lora_alpha=32,
    target_modules=["q", "v", "wi_0", "wi_1"],
    lora_dropout=0.05,
    bias="none",
)
integrated_model.decoder = get_peft_model(integrated_model.decoder, lora_config)
print("lora is applied to the decoder of the model.")
utils.print_trainable_model_module(integrated_model)

output_dir = f"./output/integrated_model-small_split-{timestamp}"

print(f"Output directory: {output_dir}")

training_args = TrainingArguments(
    output_dir=output_dir,
    learning_rate=1e-3,
    num_train_epochs=20,
    weight_decay=0.01,
    logging_steps=1,
    max_steps=500,
    logging_dir=f"./runs/{timestamp}",
    save_steps=50,
    save_total_limit=5,
    per_device_train_batch_size=16,  # Increase the batch size for training
    per_device_eval_batch_size=8,  # Increase the batch size for evaluation
)

trainer = Trainer(
    model=integrated_model,
    args=training_args,
    train_dataset=dataset_dict["train"],
    eval_dataset=dataset_dict["validation"],
)


# Custom save function to avoid safetensors issues
def save_model_custom(output_dir, **kwargs):
    state_dict = trainer.model.state_dict()
    print("Saving model to", output_dir)
    os.makedirs(output_dir, exist_ok=True)
    torch.save(state_dict, os.path.join(output_dir, "pytorch_model.bin"))


trainer.save_model = save_model_custom

trainer.train()
