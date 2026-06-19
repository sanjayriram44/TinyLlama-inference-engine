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

@torch.inference_mode()
def batched_generate(model, tokenizer, prompts: list[str], max_new_tokens: int):
    """Static batching with KV cache: generate for many prompts at once.

    Pads the batch to a rectangle, builds the attention mask so padding is
    ignored, and runs ONE KV-cache decode loop over all sequences together.
    The whole point: decode streams the 2.2GB of weights once per step
    regardless of batch size, so B sequences cost ~the same as 1 -> throughput.

    Static = the batch runs until ALL sequences hit max_new_tokens (no early
    eviction / refill -- that's continuous batching, the next step).

    Returns:
        sequences:   list[str], decoded text per prompt (padding stripped)
        token_times: list of cumulative seconds-since-start, one per decode step
    """
    # ---- LEFT-pad the batch ----
    # Left-padding (not right) is the key trick for batched generation: it puts
    # every prompt's LAST real token at the same final column. The decode loop
    # always reads logits[:, -1, :], so all sequences' "next token" lines up in
    # one clean slice -- no per-row bookkeeping for where each prompt ends.
    tokenizer.padding_side = "left"
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token   # Llama has no pad token by default

    enc = tokenizer(prompts, return_tensors="pt", padding=True).to(model.device)
    ids = enc.input_ids                 # (B, T_pad)
    attn = enc.attention_mask           # (B, T_pad): 1 = real token, 0 = padding

    B = ids.shape[0]
    token_times = []

    torch.cuda.synchronize()
    start = time.perf_counter()

    # ---- PREFILL: whole padded batch once, cache on ----
    out = model(ids, attention_mask=attn, use_cache=True)
    past = out.past_key_values
    next_ids = out.logits[:, -1, :].argmax(dim=-1, keepdim=True)   # (B, 1)
    generated = [next_ids]                                          # collect per step

    torch.cuda.synchronize()
    token_times.append(time.perf_counter() - start)

    # ---- DECODE: feed one new token PER SEQUENCE + the cache ----
    for _ in range(max_new_tokens - 1):
        # The mask must keep growing: every new token is real (a 1), appended
        # to the right. Forgetting to extend the mask is THE classic batched-
        # generation bug -- attention silently sees the wrong shape/positions.
        attn = torch.cat([attn, torch.ones((B, 1), device=model.device, dtype=attn.dtype)], dim=1)

        out = model(next_ids, attention_mask=attn, past_key_values=past, use_cache=True)
        past = out.past_key_values
        next_ids = out.logits[:, -1, :].argmax(dim=-1, keepdim=True)   # (B, 1)
        generated.append(next_ids)

        torch.cuda.synchronize()
        token_times.append(time.perf_counter() - start)

    # ---- assemble per-sequence outputs ----
    gen = torch.cat(generated, dim=1)                  # (B, max_new_tokens)
    full = torch.cat([ids, gen], dim=1)                # prompt + generated
    sequences = tokenizer.batch_decode(full, skip_special_tokens=True)

    return sequences, token_times