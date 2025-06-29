import os
import torch
import torch.nn as nn
from transformers import AutoModelForCausalLM, AutoTokenizer
from dotenv import load_dotenv

load_dotenv()
hf_token = os.getenv("HF_TOKEN")

compressed_model_path = "./compressed_model"
quantized_model_path = "./quantized_model"

# Load model and tokenizer
print("Loading compressed model...")
model = AutoModelForCausalLM.from_pretrained(compressed_model_path, torch_dtype=torch.float32)
tokenizer = AutoTokenizer.from_pretrained(compressed_model_path)

# Recursively quantize nn.Linear inside nn.Sequential
def quantize_sequential_linears(module):
    for name, child in module.named_children():
        if isinstance(child, nn.Sequential):
            # Check if it’s 2 Linear layers (as we made during SVD)
            if all(isinstance(layer, nn.Linear) for layer in child):
                quantized_layers = [
                    torch.quantization.quantize_dynamic(layer, {nn.Linear}, dtype=torch.qint8)
                    for layer in child
                ]
                setattr(module, name, nn.Sequential(*quantized_layers))
        else:
            quantize_sequential_linears(child)

print("Applying quantization to compressed SVD layers...")
quantize_sequential_linears(model)

# Save quantized model manually
print("Saving quantized model...")
os.makedirs(quantized_model_path, exist_ok=True)
torch.save(model.state_dict(), os.path.join(quantized_model_path, "pytorch_model.bin"))
tokenizer.save_pretrained(quantized_model_path)

print("Quantized model saved to ./quantized_model")