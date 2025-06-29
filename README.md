# MoE Model Quantization & Comparison

## Overview
This project performs post-training dynamic quantization on a Mixture-of-Experts (MoE) LLM checkpoint and benchmarks it against the original model in terms of latency, output similarity, parameter count, and storage size.

## Workflow

### 1. **Model Quantization**
- Script: `quantize_model.py`
- Converts `nn.Linear` layers within `nn.Sequential` modules to quantized versions using `torch.quantization.quantize_dynamic`.
- Saves the quantized model to `./quantized_model`.

### 2. **Benchmarking & Evaluation**
- Script: `comparison.py`
- Runs inference on both the original (`./cached_model`) and quantized (`./quantized_model`) models using a static prompt.
- Records:
  - Generated output
  - Inference time
  - Total parameter count
  - Cosine similarity between logits
  - On-disk model size

- Outputs results to `qwen_comparison_metrics.csv`.

## Key Results (Sample)
| Metric                 | Original      | Quantized     |
|------------------------|---------------|----------------|
| Size (MB)              | ~940 MB       | ~52 MB         |
| Parameters             | 9.88M         | 9.88M          |
| Inference Time (s)     | ~2.60         | ~2.67          |
| Cosine Similarity      | ~0.71         | —              |

## Notes
- The quantized model retains the original architecture and parameters, but uses reduced-precision arithmetic for faster and smaller inference.
- Cosine similarity evaluates the semantic closeness of generated logits.