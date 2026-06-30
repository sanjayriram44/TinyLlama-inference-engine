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

from transformers import BitsAndBytesConfig

def load_quantized_model(bits: int = 4, device: str = "cuda"):
    """Load TinyLlama with bitsandbytes quantization.
    bits=4 -> NF4 (4-bit), bits=8 -> int8. Weights stored quantized,
    matmuls run in fp16 (weight-only quantization)."""
    if bits == 4:
        qcfg = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_quant_type="nf4",          # NF4: the good 4-bit format
            bnb_4bit_compute_dtype=torch.float16, # math in fp16
            bnb_4bit_use_double_quant=True,       # quantize the quant constants too
        )
    elif bits == 8:
        qcfg = BitsAndBytesConfig(load_in_8bit=True)
    else:
        raise ValueError("bits must be 4 or 8")

    tokenizer = AutoTokenizer.from_pretrained(MODEL_ID)
    model = AutoModelForCausalLM.from_pretrained(
        MODEL_ID, quantization_config=qcfg, device_map=device,
    )
    model.eval()

    mem_gb = torch.cuda.memory_allocated(device) / 1e9
    print(f"loaded {MODEL_ID} @ {bits}-bit | gpu mem: {mem_gb:.2f} GB")
    return model, tokenizer