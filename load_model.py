import os
from dotenv import load_dotenv
from transformers import AutoModelForCausalLM, AutoTokenizer
from torch.nn import Module, Linear 

load_dotenv()
hf_token = os.getenv("HF_TOKEN")

model_id = "katuni4ka/tiny-random-qwen1.5-moe"

print("Loading tokenizer...")
tokenizer = AutoTokenizer.from_pretrained(model_id, token=hf_token)

print("Loading model (may take time)...")
model = AutoModelForCausalLM.from_pretrained(
    model_id,
    device_map="auto",
    torch_dtype="auto",
    token=hf_token
)

print("Model and tokenizer loaded successfully.")

if __name__ == "__main__":
    prompt = "Q: What is a mixture of experts model?\nA:"
    inputs = tokenizer(prompt, return_tensors="pt").to(model.device)
    outputs = model.generate(**inputs, max_new_tokens=50)
    print(tokenizer.decode(outputs[0], skip_special_tokens=True).encode('ascii', 'ignore').decode())
    print("Model architecture summary:\n")
    for name, module in model.named_modules():
        if isinstance(module, (Linear, Module)):  
            try:
                params = sum(p.numel() for p in module.parameters())
                print(f"{name:<80} | {str(type(module)).split('.')[-1][:-2]:<20} | Params: {params}")
            except Exception as e:
                print(f"{name:<80} | {str(type(module)).split('.')[-1][:-2]:<20} | ERROR: {e}")
