import os
import time
import torch
import csv
import re
from transformers import AutoModelForCausalLM, AutoTokenizer
from torch.nn.functional import cosine_similarity

def load_model_and_tokenizer(path):
    tokenizer = AutoTokenizer.from_pretrained(path)
    model = AutoModelForCausalLM.from_pretrained(path, torch_dtype=torch.float32).eval()
    return tokenizer, model

def clean_output(output_ids, tokenizer):
    decoded = tokenizer.decode(output_ids, skip_special_tokens=True)
    decoded = decoded.strip()
    decoded = re.sub(r'\s+', ' ', decoded)
    decoded = re.sub(r'[^\x00-\x7F]+', '', decoded)  # Optional: strip non-ASCII
    return decoded

def get_inference_metrics(model, tokenizer, prompt):
    inputs = tokenizer(prompt, return_tensors="pt")
    start_time = time.time()
    with torch.no_grad():
        output = model.generate(**inputs, max_new_tokens=50)
    end_time = time.time()

    generation_time = end_time - start_time
    decoded_output = clean_output(output[0], tokenizer)

    total_params = sum(p.numel() for p in model.parameters())

    with torch.no_grad():
        logits = model(**inputs).logits

    return {
        "output": decoded_output,
        "time": generation_time,
        "params": total_params,
        "logits": logits
    }

def compare_models(original_path, compressed_path, prompt, output_csv="qwen_comparison_metrics.csv"):
    tokenizer_orig, model_orig = load_model_and_tokenizer(original_path)
    tokenizer_comp, model_comp = load_model_and_tokenizer(compressed_path)

    print("Running inference on original model...")
    orig_metrics = get_inference_metrics(model_orig, tokenizer_orig, prompt)

    print("Running inference on quantized model...")
    comp_metrics = get_inference_metrics(model_comp, tokenizer_comp, prompt)

    cosine_sim = cosine_similarity(
        orig_metrics["logits"].flatten(),
        comp_metrics["logits"].flatten(),
        dim=0
    ).item()

    with open(output_csv, mode='w', newline='', encoding='utf-8') as f:
        writer = csv.writer(f)
        writer.writerow([
            "Prompt", "Orig Output", "Comp Output", 
            "Orig Time (s)", "Comp Time (s)",
            "Orig Params", "Comp Params",
            "Cosine Similarity"
        ])
        writer.writerow([
            prompt,
            orig_metrics["output"],
            comp_metrics["output"],
            orig_metrics["time"],
            comp_metrics["time"],
            orig_metrics["params"],
            comp_metrics["params"],
            cosine_sim
        ])

    print(f"Comparison metrics saved to {output_csv}")

if __name__ == "__main__":
    ORIGINAL_MODEL_PATH = "./cached_model"
    
    # COMPRESSED_MODEL_PATH = "./compressed_model"  # <-- SVD version, now commented
    COMPRESSED_MODEL_PATH = "./quantized_model"      # <-- using quantized model instead

    PROMPT = "The future of AI lies mainly in the field of.."

    compare_models(ORIGINAL_MODEL_PATH, COMPRESSED_MODEL_PATH, PROMPT)