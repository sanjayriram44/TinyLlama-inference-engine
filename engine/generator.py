"""
Deliberately unoptimized:
  - no KV cache: every step re-feeds the ENTIRE sequence and recomputes
    K/V for all past tokens (use_cache=False)
  - batch size 1, greedy decoding only
  - always generates exactly max_new_tokens (no EOS early-exit, so every
    benchmark run does identical work)

Records a timestamp after every emitted token so the harness can compute
TTFT (first timestamp ~ prefill cost) and TPOT (gaps between the rest
~ per-step decode cost).
"""

import time

import torch


@torch.inference_mode()    # no autograd graph: inference-only memory/speed
def naive_generate(model, input_ids: torch.Tensor, max_new_tokens: int):
    """Greedy-generate max_new_tokens from a (1, T) prompt.

    Returns:
        ids:         (1, T + max_new_tokens) -- prompt + generated tokens
        token_times: list of seconds-since-start, one entry per new token
    """
    ids = input_ids
    token_times = []

    # CUDA is asynchronous: Python queues kernels and races ahead, so the
    # clock must never be read while the GPU still has queued work --
    # otherwise we'd measure kernel LAUNCH time, not execution time.
    torch.cuda.synchronize()
    start = time.perf_counter()

    for _ in range(max_new_tokens):
        # NAIVE: full forward over the whole growing sequence, no cache.
        # Step t re-processes all T+t tokens -- this is why TPOT will CLIMB.
        logits = model(ids, use_cache=False).logits  #this inovkes the nn.module's __call__method.       # (1, T_cur, 32000)

        # Only the last position asks a live question; greedy = argmax.
        next_id = logits[:, -1, :].argmax(dim=-1, keepdim=True)   # (1, 1)

        ids = torch.cat([ids, next_id], dim=1)             # (1, T_cur + 1)

        torch.cuda.synchronize()   # drain the GPU before reading the clock
        token_times.append(time.perf_counter() - start)

    return ids, token_times