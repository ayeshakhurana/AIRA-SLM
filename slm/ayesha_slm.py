import torch
import os
from transformers import AutoTokenizer, AutoModelForCausalLM
from peft import PeftModel

BASE_MODEL = "TinyLlama/TinyLlama-1.1B-Chat-v1.0"
ADAPTER_PATH = os.path.abspath("slm/ayesha_slm")

device = "mps" if torch.backends.mps.is_available() else "cpu"

tokenizer = AutoTokenizer.from_pretrained(BASE_MODEL)

base_model = AutoModelForCausalLM.from_pretrained(BASE_MODEL)
base_model.to(device)

model = PeftModel.from_pretrained(
    base_model,
    ADAPTER_PATH,
    local_files_only=True
)
model.eval()

def generate_response(user_input: str) -> str:
    prompt = (
        "### Instruction:\n"
        "Answer the question clearly and completely.\n\n"
        "### Input:\n"
        f"{user_input}\n\n"
        "### Response:\n"
    )

    inputs = tokenizer(prompt, return_tensors="pt").to(device)

    with torch.no_grad():
        output = model.generate(
            **inputs,
            max_new_tokens=200,
            temperature=0.4,
            top_p=0.9,
            pad_token_id=tokenizer.eos_token_id
        )

    decoded = tokenizer.decode(output[0], skip_special_tokens=True)
    return decoded.split("### Response:")[-1].strip()
