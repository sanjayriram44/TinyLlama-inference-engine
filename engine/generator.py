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



@torch.inference_mode()
def cached_generate(model, input_ids: torch.Tensor, max_new_tokens: int):
    """KV-cache generation (Flavor A: HF's built-in cache).

    Prefill the whole prompt ONCE, then feed only the single new token plus
    the carried-forward cache each step. Same (model, ids, n) -> (ids, times)
    contract as naive_generate, so the harness uses it unchanged.

    Must produce output IDENTICAL to naive_generate (lossless) -- the golden
    check is the proof.
    """
    token_times = []

    torch.cuda.synchronize()
    start = time.perf_counter()

    # ---- PREFILL: full prompt once, cache on ----
    out = model(input_ids, use_cache=True)
    past = out.past_key_values                       # the filled KV cache
    next_id = out.logits[:, -1, :].argmax(dim=-1, keepdim=True)   # (1, 1)
    ids = torch.cat([input_ids, next_id], dim=1)

    torch.cuda.synchronize()
    token_times.append(time.perf_counter() - start)  # ~ TTFT (prefill cost)

    # ---- DECODE: feed ONLY the new token + the cache, each step ----
    # -1 because prefill already produced the first token.
    for _ in range(max_new_tokens - 1):
        out = model(next_id, past_key_values=past, use_cache=True)  # feed (1,1) + cache
        past = out.past_key_values                   # carry the UPDATED cache forward
        next_id = out.logits[:, -1, :].argmax(dim=-1, keepdim=True)  # logits are (1,1,V)
        ids = torch.cat([ids, next_id], dim=1)

        torch.cuda.synchronize()
        token_times.append(time.perf_counter() - start)

    return ids, token_times