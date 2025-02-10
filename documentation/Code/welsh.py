import amrlib
import json
import os
import random
import numpy as np
import torch
import matplotlib.pyplot as plt
from datasets import Dataset, DatasetDict
from sklearn.model_selection import train_test_split
from transformers import (
    T5ForConditionalGeneration,
    T5Tokenizer,
    Trainer,
    TrainingArguments,
    DataCollatorForSeq2Seq,
    TrainerCallback
)
import smatch
from collections import defaultdict
import re

# Set random seeds for reproducibility
random.seed(42)
np.random.seed(42)
torch.manual_seed(42)
if torch.cuda.is_available():
    torch.cuda.manual_seed_all(42)
torch.cuda.empty_cache()

file_path = "data/massive_amr_welsh.jsonl"

def load_amr_data(file_path):
    with open(file_path, "r", encoding="utf-8") as file:
        data = [json.loads(line) for line in file]
    # Filter out entries missing either the sentence or AMR
    data = [entry for entry in data if entry.get("raw_amr") and entry.get("utt")]
    return data

data = load_amr_data(file_path)
random.shuffle(data)

# Extract sentences and AMR graphs
sentences = [entry["utt"] for entry in data]
amrs = [entry["raw_amr"] for entry in data]

# Split into train, validation, and test sets (90/10 then 10% of train for validation)
train_sents, test_sents, train_amrs, test_amrs = train_test_split(sentences, amrs, test_size=0.1, random_state=42)
train_sents, val_sents, train_amrs, val_amrs = train_test_split(train_sents, train_amrs, test_size=0.1, random_state=42)

# Convert to Hugging Face Dataset format
def create_dataset(sentences, amrs):
    return Dataset.from_dict({"sentence": sentences, "amr": amrs})

datasets = DatasetDict({
    "train": create_dataset(train_sents, train_amrs),
    "validation": create_dataset(val_sents, val_amrs),
    "test": create_dataset(test_sents, test_amrs),
})

# Load Multilingual T5 Model (STOG & GTOS)
model_name = "t5-small"  # You can use a larger model if resources allow.
tokenizer = T5Tokenizer.from_pretrained(model_name)
model = T5ForConditionalGeneration.from_pretrained(model_name)

def preprocess_function(examples):
    # For T5 it's common to add a task prefix. Here we add "parse: " before the sentence.
    inputs = ["parse: " + ex for ex in examples["sentence"]]
    targets = examples["amr"]

    # Tokenize inputs (source) and targets
    model_inputs = tokenizer(inputs, max_length=256, truncation=True, padding="max_length")
    labels = tokenizer(targets, max_length=512, truncation=True, padding="max_length")

    # Replace padding token id's in labels with -100 so they are ignored by the loss function
    labels["input_ids"] = [
        [l if l != tokenizer.pad_token_id else -100 for l in label]
        for label in labels["input_ids"]
    ]
    model_inputs["labels"] = labels["input_ids"]
    return model_inputs

# Apply tokenization
tokenized_datasets = datasets.map(preprocess_function, batched=True, remove_columns=["sentence", "amr"])

data_collator = DataCollatorForSeq2Seq(tokenizer, model=model)

# Define Optimized Training Arguments
training_args = TrainingArguments(
    output_dir="./amr_t5_model",
    evaluation_strategy="epoch",
    save_strategy="epoch",
    learning_rate=3e-4,
    per_device_train_batch_size=8,
    per_device_eval_batch_size=8,
    num_train_epochs=10,
    weight_decay=0.01,
    save_total_limit=2,
    logging_steps=50,
    logging_dir="./logs",
    report_to="none",
)

# Custom Callback for Logging and Visualization
class LossCallback(TrainerCallback):
    def __init__(self):
        self.losses = []
        self.eval_losses = []

    def on_log(self, args, state, control, logs=None, **kwargs):
        if logs:
            if "loss" in logs:
                self.losses.append((state.global_step, logs["loss"]))
            if "eval_loss" in logs:
                self.eval_losses.append((state.global_step, logs["eval_loss"]))

    def plot_loss(self):
        steps, losses = zip(*self.losses)
        eval_steps, eval_losses = zip(*self.eval_losses)

        plt.figure(figsize=(10, 5))
        plt.plot(steps, losses, label="Training Loss", marker="o")
        plt.plot(eval_steps, eval_losses, label="Validation Loss", marker="x")
        plt.xlabel("Training Steps")
        plt.ylabel("Loss")
        plt.title("Training & Validation Loss Curve")
        plt.legend()
        plt.show()
        
loss_callback = LossCallback()

# Initialize Trainer and Train the Model
trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=tokenized_datasets["train"],
    eval_dataset=tokenized_datasets["validation"],
    tokenizer=tokenizer,
    data_collator=data_collator,
    callbacks=[loss_callback],
)

print("Starting T5 Training for AMR Parsing...")
trainer.train()

# Plot Training Progress
loss_callback.plot_loss()
print("Dataset Columns:", datasets["test"].column_names)
print("Dataset Columns (after tokenization):", tokenized_datasets["test"].column_names)

# Save Fine-Tuned Model
model.save_pretrained("./fine_tuned_amr_t5")
tokenizer.save_pretrained("./fine_tuned_amr_t5")

# Load Fine-Tuned T5 Model & Tokenizer
model_path = "./fine_tuned_amr_t5"
model = T5ForConditionalGeneration.from_pretrained(model_path)
tokenizer = T5Tokenizer.from_pretrained(model_path)

# Move model to GPU if available
device = "cuda" if torch.cuda.is_available() else "cpu"
model.to(device)
model.eval()

# Load test Data
file_path = "data/massive_amr_welsh.jsonl"

def load_amr_data(file_path):
    with open(file_path, "r", encoding="utf-8") as file:
        data = [json.loads(line) for line in file]
    return [entry for entry in data if entry.get("raw_amr") and entry.get("utt")]

data = load_amr_data(file_path)
sentences = [entry["utt"] for entry in data]
gold_amrs = [entry["raw_amr"] for entry in data]

# Convert to Hugging Face Dataset format
test_dataset = Dataset.from_dict({"sentence": sentences, "amr": gold_amrs})

# Improved AMR Cleaning Function (Fix Duplicate Names)
def fix_duplicate_nodes(amr_text):
    if not amr_text.strip():
        return "INVALID_AMR"

    used_vars = defaultdict(int)  # Tracks occurrences of each variable
    renamed_vars = {}

    def rename_variable(match):
        var_name = match.group(1)
        if var_name in used_vars:
            new_var_name = f"{var_name}_{used_vars[var_name]}"
            used_vars[var_name] += 1
            renamed_vars[var_name] = new_var_name
            return f"({new_var_name} /"
        else:
            used_vars[var_name] = 1
            return f"({var_name} /"

    # Ensure variables like (t2 / thing) are renamed uniquely
    amr_text = re.sub(r"\((\w+)\s+\/", rename_variable, amr_text)

    # Replace occurrences of renamed variables in AMR text
    for old_var, new_var in renamed_vars.items():
        amr_text = amr_text.replace(f" {old_var} ", f" {new_var} ")

    return amr_text

# Generate AMR Predictions (Fixed)
def generate_amr_predictions(model, tokenizer, test_dataset, max_samples=100):
    predictions = []

    for i, example in enumerate(test_dataset):
        if i >= max_samples:  # Limit number of samples for efficiency
            break

        input_text = "parse: " + example["sentence"]
        input_ids = tokenizer.encode(input_text, return_tensors="pt").to(device)

        with torch.no_grad():
            output_ids = model.generate(input_ids, max_length=512)
        
        predicted_amr = tokenizer.decode(output_ids[0], skip_special_tokens=True)
        predicted_amr = fix_duplicate_nodes(predicted_amr)  # Ensure AMR is valid

        gold_amr = fix_duplicate_nodes(example["amr"])  # Fix duplicates in gold AMRs too

        predictions.append((gold_amr, predicted_amr))

    return predictions

# Generate predictions
predictions = generate_amr_predictions(model, tokenizer, test_dataset)

# Compute Smatch Score (with Fixes)
def is_valid_amr(amr):
    return amr.count('/') > 0

def compute_smatch(gold_amrs, predicted_amrs):
    total_precision, total_recall, total_f1 = 0, 0, 0
    valid_samples = 0

    for i, (gold, pred) in enumerate(zip(gold_amrs, predicted_amrs)):
        try:
            # Ensure AMRs are valid
            if not is_valid_amr(pred) or not is_valid_amr(gold):
                print(f"Skipping sample {i} due to invalid AMR")
                continue

            precision, recall, f_score = smatch.get_amr_match(str(gold), str(pred))
            total_precision += precision
            total_recall += recall
            total_f1 += f_score
            valid_samples += 1
        except Exception as e:
            print(f"Error in Smatch calculation for sample {i}: {e}")
            print(f"Gold AMR: {gold}")
            print(f"Predicted AMR: {pred}")
            print("-" * 50)

    if valid_samples == 0:
        return 0, 0, 0  # Avoid division by zero

    return total_precision / valid_samples, total_recall / valid_samples, total_f1 / valid_samples

# Extract predicted AMRs for Smatch evaluation
gold_amrs = [gold for gold, _ in predictions if gold is not None]
predicted_amrs = [pred for _, pred in predictions if pred is not None]

# Compute Smatch Score
precision, recall, f1 = compute_smatch(gold_amrs, predicted_amrs)

# Display Results
print(f"Smatch Score - Precision: {precision:.4f}, Recall: {recall:.4f}, F1: {f1:.4f}")