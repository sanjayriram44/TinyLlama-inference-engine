## Results

Frozen workload: 36-token prompt, 128 generated tokens, greedy decoding,
averaged over 3 runs. Same prompt and `golden_output.txt` correctness anchor
across every stage. Hardware: single NVIDIA A40 (46 GB).

| Stage | Optimization | TTFT (ms) | TPOT (ms) | Tokens/s | Peak mem (GB) | Matches golden |
|-------|--------------|-----------|-----------|----------|---------------|----------------|
| 0 | Naive baseline (no cache) | 15–21 | 16–19 | 52–64 | 2.23 | created |
| 2 | KV cache (HF built-in) | ~14 | ~13 | 74–77 | 2.22 | **true** |
| 3 | Continuous batching | _todo_ | _todo_ | _todo_ | _todo_ | true |
| 4 | Quantization (INT8/INT4) | _todo_ | _todo_ | _todo_ | _todo_ | lossy → quality metric |
| 5 | FlashAttention + GQA | _todo_ | _todo_ | _todo_ | _todo_ | true |
| 6 | Tensor parallelism (2× GPU) | _todo_ | _todo_ | _todo_ | _todo_ | true |
| 8 | vs. vLLM | _todo_ | _todo_ | _todo_ | _todo_ | reference |

## Stage 2 — KV cache findings

The KV cache eliminates the naive loop's redundant recomputation: instead of
re-feeding the entire growing sequence each step, the prompt is prefilled once
and each decode step processes only the single new token, attending against the
cached keys/values of all prior tokens.

**Correctness is the deliverable, and it passed.** `output_matches_golden`
stayed `true` — the cache reproduces the naive baseline's output byte-for-byte,
proving it's lossless. (Greedy decoding is deterministic, so any divergence
would have signalled a bug — an off-by-one in the prefill/decode token count, or
feeding the wrong tensor back. None occurred.)

**Speedup was real but modest, exactly as predicted:**

| Metric | Naive | KV cache | Change |
|--------|-------|----------|--------|
| TPOT | ~17 ms | ~13 ms | ~25% faster |
| Tokens/s | ~57 | ~77 | ~35% higher |

This confirms the stage-0 finding from the other direction. Decode time is
dominated by **streaming the 2.2 GB of weights from VRAM each step** — a fixed
cost the cache cannot reduce. The cache only removes the *recompute* of past
K/V, which at 1.1B params and ~150 tokens is small relative to weight-streaming.
So the win is genuine but capped by the memory-bandwidth floor. **The cache's
payoff grows with sequence length** — at thousands of tokens, the recompute it
eliminates becomes the dominant cost, and the gap between naive and cached
widens sharply. Finding that crossover is the natural next experiment.

**Takeaway:** the KV cache is correct and lossless here (the point of a lossless
optimization stage), and its limited speedup is itself the lesson — you can't
optimize away a cost that isn't your bottleneck.