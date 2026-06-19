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

## Stage 3 — Batching findings

Static batching with KV cache: many prompts (varied lengths, left-padded) run
through one decode loop together. Metric shifts from per-token latency to
**aggregate throughput** (total tokens across the batch / wall time).

### Throughput sweep (A40, 128 tokens/seq, 3 runs)

| Batch | Throughput (tok/s) | Per-seq (tok/s) | Wall time (s) | Peak mem (GB) |
|-------|--------------------|-----------------|---------------|---------------|
| 1 | 60.1 | 60.1 | 2.13 | 2.21 |
| 2 | 132.8 | 66.4 | 1.93 | 2.22 |
| 4 | 267.9 | 67.0 | 1.91 | 2.23 |
| 8 | 532.7 | 66.6 | 1.92 | 2.25 |
| 16 | 1057.2 | 66.1 | 1.94 | 2.29 |
| 32 | 2134.1 | 66.7 | 1.92 | 2.37 |

### The finding: batching is nearly free here, and the compute wall is far away

Throughput scaled **~35× across a 32× batch increase** — almost perfectly
linear. The proof is the wall-time column: batch 32 did **32× the work in the
same ~1.9s** as batch 1. This is the whole thesis of batching, confirmed:
because decode streams the 2.2 GB of weights once per step regardless of batch
size, stacking sequences fills otherwise-idle compute for free. Per-sequence
speed held flat (~66 tok/s) — no user paid a latency penalty.

**The predicted throughput "bend" (compute saturation) did not appear** — and
that's the real result. Evidence the GPU was never saturated: `nvidia-smi`
showed util at only 51–70% even at batch 32, and peak memory used just 2.37 GB
of the A40's 46 GB. A 1.1B model on a 46 GB card has enormous headroom; the
memory-bound → compute-bound transition lies far beyond batch 32. **Follow-up:**
rerun with batch sizes 64–512 to find the bend (where throughput stops scaling
linearly and per-seq tok/s starts dropping) — that crossover is the experiment's
true conclusion.