import torch

@torch.inference_mode()
def perplexity(model, tokenizer, text: str) -> float:
    """Lower = better. How 'surprised' the model is by the text.
    The single number that quantifies quantization's quality cost."""
    ids = tokenizer(text, return_tensors="pt").input_ids.to(model.device)
    out = model(ids, labels=ids)          # HF computes mean cross-entropy loss
    return torch.exp(out.loss).item()      # perplexity = exp(loss)