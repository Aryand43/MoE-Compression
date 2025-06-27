import torch
import torch.nn as nn
import os
from transformers import AutoModelForCausalLM
from dotenv import load_dotenv

load_dotenv()
hf_token = os.getenv("HF_TOKEN")

# Load directly from Hugging Face with caching
model_id = "katuni4ka/tiny-random-qwen1.5-moe"
model = AutoModelForCausalLM.from_pretrained(
    model_id,
    cache_dir="./cached_model/",
    device_map="cpu",
    torch_dtype=torch.float32,
    token=hf_token
)
print("Model loaded. Starting SVD compression...\n")

# SVD Compression for a Linear layer
def svd_compress_linear_layer(layer: nn.Linear, rank: int = 32):
    with torch.no_grad():
        W = layer.weight.data  # [out_features, in_features]
        U, S, Vh = torch.linalg.svd(W, full_matrices=False)

        U_k = U[:, :rank]
        S_k = S[:rank]
        V_k = Vh[:rank, :]

        first = nn.Linear(V_k.shape[1], rank, bias=False)
        second = nn.Linear(rank, U_k.shape[0], bias=True)

        first.weight.data = V_k
        second.weight.data = (U_k * S_k).T
        second.bias.data = layer.bias.data.clone()

        return nn.Sequential(first, second)

# Replace Linear layers with compressed versions
def compress_model(model, rank=32):
    for name, module in model.named_modules():
        for attr_name in dir(module):
            if not hasattr(module, attr_name):
                continue  # skip if attribute doesn't exist
            attr = getattr(module, attr_name)

            if isinstance(attr, nn.Linear):
                if attr.weight is None:
                    print(f"Skipped {name}.{attr_name}: weight is None")
                    continue
                try:
                    compressed = svd_compress_linear_layer(attr, rank=rank)
                    setattr(module, attr_name, compressed)
                    print(f"Compressed: {name}.{attr_name}")
                except Exception as e:
                    print(f"Skipped {name}.{attr_name}: {e}")
    return model

# Run compression
model = compress_model(model, rank=32)

# Save compressed model
save_path = "./compressed_model/"
os.makedirs(save_path, exist_ok=True)
model.save_pretrained(save_path)
print("\nCompression complete. Saved to ./compressed_model/")
