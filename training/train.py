import os
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, Trainer, TrainingArguments
from datasets import load_dataset
from peft import LoraConfig, get_peft_model

BASE_MODEL = "TinyLlama/TinyLlama-1.1B-Chat-v1.0"
OUTPUT_DIR = "slm/ayesha_slm"

os.makedirs(OUTPUT_DIR, exist_ok=True)

device = "mps" if torch.backends.mps.is_available() else "cpu"
print("Using device:", device)

tokenizer = AutoTokenizer.from_pretrained(BASE_MODEL)

model = AutoModelForCausalLM.from_pretrained(BASE_MODEL)
model.to(device)

lora_config = LoraConfig(
    r=8,
    lora_alpha=16,
    target_modules=["q_proj", "v_proj"],
    lora_dropout=0.05,
    task_type="CAUSAL_LM",
)

model = get_peft_model(model, lora_config)
model.print_trainable_parameters()

dataset = load_dataset("json", data_files="data/train.json")

# ✅ FIXED PROMPT FORMAT
def preprocess(example):
    prompt = (
        "### Instruction:\n"
        f"{example['instruction']}\n\n"
        "### Input:\n"
        f"{example['input']}\n\n"
        "### Response:\n"
        f"{example['output']}"
    )

    tokenized = tokenizer(
        prompt,
        truncation=True,
        padding="max_length",
        max_length=512,
    )

    tokenized["labels"] = tokenized["input_ids"].copy()
    return tokenized

dataset = dataset.map(preprocess, remove_columns=dataset["train"].column_names)

training_args = TrainingArguments(
    output_dir=OUTPUT_DIR,
    per_device_train_batch_size=1,
    gradient_accumulation_steps=2,
    num_train_epochs=3,
    learning_rate=2e-4,
    logging_steps=1,
    save_strategy="no",
    report_to="none",
    remove_unused_columns=False,
)

trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=dataset["train"],
)

trainer.train()

model.save_pretrained(OUTPUT_DIR)
tokenizer.save_pretrained(OUTPUT_DIR)

print("✅ LoRA adapter retrained and saved correctly")
