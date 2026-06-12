"""
We use HF for the tokenizer and the architecture/weight-loading ONLY.
We never call .generate() -- the generation loop is ours (engine/generator.py).
The model object is just "weights + a forward pass that returns logits".
"""

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

MODEL_ID = "TinyLlama/TinyLlama-1.1B-Chat-v1.0"


def load_model(dtype: torch.dtype = torch.float16, device: str = "cuda"):
    """Download (first run only), then load model + tokenizer onto the GPU.

    Weights land in the HF cache (~2.2 GB at fp16). On the Pod, point the
    cache at persistent storage first:  export HF_HOME=/workspace/hf_cache
    """
    tokenizer = AutoTokenizer.from_pretrained(MODEL_ID)

    model = AutoModelForCausalLM.from_pretrained(
        MODEL_ID,
        torch_dtype=dtype,   # fp16: 2 bytes/param instead of 4 -> ~2.2 GB
    )
    model.to(device)
    model.eval()             # inference mode: see note below

    n_params = sum(p.numel() for p in model.parameters())
    mem_gb = torch.cuda.memory_allocated(device) / 1e9
    print(f"loaded {MODEL_ID}")
    print(f"  params: {n_params/1e9:.2f}B | dtype: {dtype} | gpu mem: {mem_gb:.2f} GB")

    return model, tokenizer